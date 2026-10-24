"""Build model inputs: encoded user/item tables, the purchase sequence with history pointers, and text embeddings.

Usage:
    python -m src.features --sample-frac 0.05 --no-text   # quick local run
    python -m src.features                                # full data (run on Azure ML)

Output directory (default data/features[/sample_<frac>]):
    users.npz        per customer_idx: user_vocab, age_bucket, club_status, news_frequency
    items.npz        per item row (articles.csv order): article_id, item_vocab, categorical codes, train_count
    seq_*.npy        all purchases, deduplicated per (customer, day, article), sorted by (customer, day):
                     seq_customer, seq_item (item row), seq_day, seq_user_start, seq_hist_end
    item_text_emb.npy  [n_items, 384] float16 sentence embeddings of prod_name + detail_desc
    meta.json        vocab sizes, constants and category names

History of sequence row r = rows [max(seq_user_start[r], seq_hist_end[r] - HIST_LEN), seq_hist_end[r]):
same customer, strictly earlier days. The 1-D layout lets workers memory-map it instead of copying.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load
from src.split import MIN_ITEM_PURCHASES, MIN_USER_PURCHASES, TRAIN_END
from src.tracking import peak_rss_gb

HIST_LEN = 20
DAY0 = pd.Timestamp("2018-09-20")  # first day in the dataset -> day 0

# ID conventions shared with the model: 0 = padding (history slots), 1 = rare/unknown, 2+ = real IDs.
PAD_ID, RARE_ID, FIRST_ID = 0, 1, 2

AGE_BINS = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65]  # bucket 0 = unknown age

USER_CATEGORICALS = {"club_member_status": "club_status", "fashion_news_frequency": "news_frequency"}
ITEM_CATEGORICALS = ["product_group_name", "colour_group_name", "product_type_name", "index_group_name"]


def day_number(ts) -> np.ndarray:
    days = np.asarray(pd.to_datetime(ts), dtype="datetime64[D]") - np.datetime64(DAY0.date(), "D")
    return days.astype(np.int16)


def encode_category(values: pd.Series) -> tuple[np.ndarray, list[str]]:
    """Codes 1..K in sorted order of the category names; missing -> 0."""
    names = sorted(values.dropna().unique().tolist())
    codes = pd.Categorical(values, categories=names).codes.astype(np.int16) + 1
    return codes, names


def vocab_ids(counts: np.ndarray, min_count: int) -> np.ndarray:
    """Entities with count > min_count get ids 2, 3, ...; everything else shares RARE_ID."""
    ids = np.full(len(counts), RARE_ID, dtype=np.int32)
    frequent = counts > min_count
    ids[frequent] = FIRST_ID + np.arange(frequent.sum(), dtype=np.int32)
    return ids


def build_sequence(tx: pd.DataFrame, item_row_of: pd.Index) -> dict[str, np.ndarray]:
    cust = tx["customer_idx"].to_numpy()
    item = item_row_of.get_indexer(tx["article_id"].to_numpy()).astype(np.int32)
    if (item < 0).any():
        raise ValueError("transactions contain article_ids missing from articles.csv")
    day = day_number(tx["t_dat"])

    order = np.lexsort((item, day, cust))
    cust, item, day = cust[order], item[order], day[order]
    # Drop repeat purchases of the same article by the same customer on the same day.
    keep = np.ones(len(cust), dtype=bool)
    keep[1:] = (cust[1:] != cust[:-1]) | (day[1:] != day[:-1]) | (item[1:] != item[:-1])
    cust, item, day = cust[keep], item[keep], day[keep]

    idx = np.arange(len(cust), dtype=np.int32)
    new_user = np.ones(len(cust), dtype=bool)
    new_user[1:] = cust[1:] != cust[:-1]
    new_day = new_user.copy()
    new_day[1:] |= day[1:] != day[:-1]
    # Running maximum of each group's first index = start of the current user / (user, day) group.
    user_start = np.maximum.accumulate(np.where(new_user, idx, 0)).astype(np.int32)
    hist_end = np.maximum.accumulate(np.where(new_day, idx, 0)).astype(np.int32)
    return {
        "seq_customer": cust,
        "seq_item": item,
        "seq_day": day,
        "seq_user_start": user_start,
        "seq_hist_end": hist_end,
    }


def build(sample_frac: float | None, out: Path, with_text: bool) -> dict:
    start = time.perf_counter()
    customers = load("customers", sample_frac)
    articles = load("articles", sample_frac)
    tx = load("transactions", sample_frac, columns=["t_dat", "customer_idx", "article_id"])

    item_row_of = pd.Index(articles["article_id"])
    seq = build_sequence(tx, item_row_of)
    del tx
    train_end_day = int(day_number([TRAIN_END])[0])
    is_train = seq["seq_day"] <= train_end_day

    # Training-window counts drive the vocab thresholds and the logQ correction.
    n_customers = int(customers["customer_idx"].max()) + 1  # dense index over the full customer table
    user_counts = np.bincount(seq["seq_customer"][is_train], minlength=n_customers)
    item_counts = np.bincount(seq["seq_item"][is_train], minlength=len(articles)).astype(np.int32)

    meta = {"hist_len": HIST_LEN, "day0": str(DAY0.date()), "train_end_day": train_end_day,
            "pad_id": PAD_ID, "rare_id": RARE_ID, "sample_frac": sample_frac, "categories": {}}

    # Users: one row per customer_idx (sampled customers keep their global index; the rest stay unknown).
    users = {
        "user_vocab": vocab_ids(user_counts, MIN_USER_PURCHASES),
        "age_bucket": np.zeros(n_customers, dtype=np.int16),
    }
    rows = customers["customer_idx"].to_numpy()
    age = customers["age"].to_numpy()
    users["age_bucket"][rows] = np.where(np.isnan(age), 0, np.digitize(age, AGE_BINS) + 1)
    for col, key in USER_CATEGORICALS.items():
        values = customers[col].astype("string").str.upper()  # raw data mixes 'None' and 'NONE'
        codes, names = encode_category(values)
        users[key] = np.zeros(n_customers, dtype=np.int16)
        users[key][rows] = codes
        meta["categories"][key] = names

    items = {
        "article_id": articles["article_id"].to_numpy(),
        "item_vocab": vocab_ids(item_counts, MIN_ITEM_PURCHASES),
        "train_count": item_counts,
    }
    for col in ITEM_CATEGORICALS:
        items[col], meta["categories"][col] = encode_category(articles[col].astype("string"))

    meta["vocab_sizes"] = {
        "user_vocab": int(users["user_vocab"].max()) + 1,
        "item_vocab": int(items["item_vocab"].max()) + 1,
        "age_bucket": len(AGE_BINS) + 2,
        **{k: len(meta["categories"][k]) + 1 for k in meta["categories"]},
    }
    hist_lens = np.minimum(seq["seq_hist_end"] - seq["seq_user_start"], HIST_LEN)[is_train]
    meta["stats"] = {
        "sequence_rows": len(seq["seq_item"]),
        "train_examples": int(is_train.sum()),
        "users_with_own_id": int((users["user_vocab"] >= FIRST_ID).sum()),
        "items_with_own_id": int((items["item_vocab"] >= FIRST_ID).sum()),
        "train_examples_with_empty_history_pct": round(100 * float((hist_lens == 0).mean()), 2),
        "train_examples_with_full_history_pct": round(100 * float((hist_lens == HIST_LEN).mean()), 2),
        "train_examples_rare_item_pct": round(100 * float((items["item_vocab"][seq["seq_item"][is_train]] == RARE_ID).mean()), 2),
    }

    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "users.npz", **users)
    np.savez(out / "items.npz", **items)
    for name, array in seq.items():
        np.save(out / f"{name}.npy", array)

    if with_text:
        from src.text_embeddings import embed_articles

        t0 = time.perf_counter()
        emb = embed_articles(articles)
        np.save(out / "item_text_emb.npy", emb)
        meta["text"] = {"dim": emb.shape[1], "seconds": round(time.perf_counter() - t0, 1)}

    meta["stats"]["seconds"] = round(time.perf_counter() - start, 1)
    meta["stats"]["peak_rss_gb"] = round(peak_rss_gb(), 2)
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-frac", type=float, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=None, help="where feature files go")
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks"), help="where the small stats JSON goes")
    parser.add_argument("--no-text", action="store_true", help="skip the sentence-transformer step")
    parser.add_argument("--check", action="store_true", help="run src.check_dataset on the result")
    parser.add_argument("--publish-dir", type=Path, default=None, help="copy the finished features here")
    args = parser.parse_args()

    default_dir = Path("data/features") / ("" if args.sample_frac is None else f"sample_{args.sample_frac}")
    features_dir = args.artifacts_dir or default_dir
    meta = build(args.sample_frac, features_dir, with_text=not args.no_text)
    summary = {k: meta[k] for k in ["vocab_sizes", "stats"] + (["text"] if "text" in meta else [])}
    print(json.dumps(summary, indent=2))

    suffix = "" if args.sample_frac is None else f"_sample{args.sample_frac}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"features{suffix}.json").write_text(json.dumps(summary, indent=2) + "\n")

    if args.check:
        from src.check_dataset import run_checks

        name = "full" if args.sample_frac is None else f"sample{args.sample_frac}"
        run_checks(features_dir, args.out_dir, name)
    if args.publish_dir is not None:
        # Build and check on local disk, then copy once to the (network-mounted) job output.
        shutil.copytree(features_dir, args.publish_dir, dirs_exist_ok=True)
