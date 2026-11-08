"""Candidate generation for the ranking stage: several sources merged into one pool per customer and week.

Every source uses only purchases strictly before the target week:
    repeat           the customer's distinct past purchases, newest first
    two_tower        the blended two-tower ranking: cosine(user, item) + alpha * log(1 + 7-day sales)
    popularity       the most-sold items of the last 7 days (the same list for everyone)
    bought_together  items bought in the same basket (customer + day) as the customer's recent purchases,
                     counted over the last `pair_weeks` weeks
    same_product     other colours/variants (same product_code) of the customer's recent purchases, best sellers first

A ranker can only reorder what the pool contains, so the report measures coverage: the share of each customer's
target-week purchases that appear in a source (or the merged pool) at a given depth.

Usage:
    python -m src.candidates --features-dir data/features/sample_0.05 --processed-dir data/processed \
        --checkpoint models/sample-12wk/best.pt --alpha 0.05
    python -m infra.submit_job src.candidates --requirements requirements-job-serve.txt \
        --data hm-features:1 --extra-data hm-processed:1 --experiment phase9-ranking -- \
        --features-dir '{data}' --processed-dir '{extra}' --run-name full-main-6ep
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.blend_eval import load_model
from src.dataset import FeatureStore
from src.evaluate import WeekEvaluator
from src.split import TEST, VAL
from src.tracking import peak_rss_gb

SOURCES = ["repeat", "two_tower", "popularity", "bought_together", "same_product"]
# Share of the pool each source may fill, in this order; two_tower fills whatever is left.
QUOTAS = {"repeat": 0.15, "bought_together": 0.2, "same_product": 0.1, "popularity": 0.1}
DEPTHS = [12, 50, 100, 200, 500]
BUDGETS = [100, 200, 500]


def product_codes(store: FeatureStore, processed_dir: Path) -> np.ndarray:
    """product_code per item row."""
    articles = pd.read_parquet(processed_dir / "articles.parquet", columns=["article_id", "product_code"])
    order = pd.DataFrame({"article_id": store.items["article_id"]})
    return order.merge(articles, on="article_id", how="left", validate="one_to_one")["product_code"].to_numpy()


def week_rows(ev: WeekEvaluator, days: int) -> pd.DataFrame:
    """Sequence rows (customer, day, item) in the `days` before the target week."""
    s = ev.store.seq
    day = np.asarray(s["seq_day"])
    keep = (day < ev.start_day) & (day >= ev.start_day - days)
    return pd.DataFrame({"customer": np.asarray(s["seq_customer"])[keep], "day": day[keep],
                         "item": np.asarray(s["seq_item"])[keep]})


def top_per_group(df: pd.DataFrame, group: str, item: str, score: str, n: int) -> pd.DataFrame:
    df = df.sort_values([group, score], ascending=[True, False], kind="stable")
    return df[df.groupby(group).cumcount() < n]


def to_lists(df: pd.DataFrame, users: np.ndarray, user_col: str, item_col: str) -> list[np.ndarray]:
    """Per user (in `users` order) the items of `df`, keeping df's order within a user."""
    grouped = {u: g[item_col].to_numpy() for u, g in df.groupby(user_col, sort=False)}
    empty = np.empty(0, dtype=np.int64)
    return [grouped.get(u, empty).astype(np.int64) for u in users]


def pair_table(ev: WeekEvaluator, weeks: int, top_m: int = 20, max_basket: int = 30) -> pd.DataFrame:
    """For each item, its `top_m` most frequent basket partners (seed, partner, count)."""
    rows = week_rows(ev, 7 * weeks)
    rows["basket"] = rows["customer"].to_numpy(np.int64) << 16 | rows["day"].to_numpy(np.int64)
    size = rows.groupby("basket")["item"].transform("size")
    rows = rows.loc[size <= max_basket, ["basket", "item"]].drop_duplicates()
    pairs = rows.merge(rows, on="basket", suffixes=("_a", "_b"))
    pairs = pairs[pairs["item_a"] != pairs["item_b"]]
    counts = pairs.groupby(["item_a", "item_b"]).size().rename("count").reset_index()
    counts = top_per_group(counts, "item_a", "item_b", "count", top_m)
    return counts.rename(columns={"item_a": "seed", "item_b": "partner"})


def seeds(repeat_lists: list[np.ndarray], users: np.ndarray, n_seeds: int) -> pd.DataFrame:
    """(user, seed, recency) for each user's `n_seeds` most recent distinct purchases; recency 0 = newest."""
    lengths = np.array([min(len(r), n_seeds) for r in repeat_lists])
    return pd.DataFrame({
        "user": np.repeat(users, lengths),
        "seed": np.concatenate([r[:n_seeds] for r in repeat_lists]) if lengths.sum() else np.empty(0, np.int64),
        "recency": np.concatenate([np.arange(n) for n in lengths]) if lengths.sum() else np.empty(0, np.int64),
    })


def bought_together(user_seeds: pd.DataFrame, pairs: pd.DataFrame, users: np.ndarray, n: int) -> list[np.ndarray]:
    """Partners of the user's recent items, scored by co-purchase count (newer seeds weigh more)."""
    joined = user_seeds.merge(pairs, on="seed")
    joined["score"] = joined["count"] / (1.0 + joined["recency"])
    scored = joined.groupby(["user", "partner"], sort=False)["score"].sum().reset_index()
    return to_lists(top_per_group(scored, "user", "partner", "score", n), users, "user", "partner")


def same_product(user_seeds: pd.DataFrame, codes: np.ndarray, sales: np.ndarray, users: np.ndarray,
                 n: int) -> list[np.ndarray]:
    """Other variants of the user's recent items, best sellers first."""
    variants = pd.DataFrame({"code": codes, "variant": np.arange(len(codes)), "sales": sales})
    joined = user_seeds.assign(code=codes[user_seeds["seed"].to_numpy()]).merge(variants, on="code")
    joined = joined[joined["variant"] != joined["seed"]]
    joined["score"] = joined["sales"] - 1e-3 * joined["recency"]  # ties: variants of newer purchases first
    scored = joined.groupby(["user", "variant"], sort=False)["score"].max().reset_index()
    return to_lists(top_per_group(scored, "user", "variant", "score", n), users, "user", "variant")


def generate(ev: WeekEvaluator, model, codes: np.ndarray, alpha: float, depth: int = 500, n_seeds: int = 10,
             pair_weeks: int = 4, device: torch.device = torch.device("cpu")) -> dict[str, list[np.ndarray]]:
    """Every source's ranked list per evaluated user (ev.users order)."""
    prior = ev.recent_popularity(7)
    sales_7d = np.expm1(prior)
    repeat = ev.repeat_candidates(max_items=depth)
    user_seeds = seeds(repeat, ev.users, n_seeds)
    two_tower = ev.recommend(model, device, k=depth, prior=prior, prior_weight=alpha)
    popular = np.argsort(-sales_7d, kind="stable")[:depth]
    return {
        "repeat": repeat,
        "two_tower": list(two_tower),
        "popularity": [popular] * len(ev.users),
        "bought_together": bought_together(user_seeds, pair_table(ev, pair_weeks), ev.users, depth),
        "same_product": same_product(user_seeds, codes, sales_7d, ev.users, depth),
    }


def merge(lists: dict[str, list[np.ndarray]], budget: int) -> list[np.ndarray]:
    """Per user: each quota source up to its share of the budget, then two_tower fills the rest (deduplicated)."""
    pools = []
    for u in range(len(lists["two_tower"])):
        pool: dict[int, None] = {}
        for source, share in QUOTAS.items():
            for item in lists[source][u][:int(share * budget)].tolist():
                pool.setdefault(item)
        for item in lists["two_tower"][u].tolist():
            if len(pool) >= budget:
                break
            pool.setdefault(item)
        pools.append(np.fromiter(pool, dtype=np.int64, count=len(pool)))
    return pools


def coverage(actual: list[set], pools: list[np.ndarray], depth: int | None = None) -> dict:
    """Mean per-user share of purchases in the pool, micro share over all purchases, and mean pool size."""
    per_user, hits, total, sizes = [], 0, 0, []
    for a, pool in zip(actual, pools):
        p = set((pool if depth is None else pool[:depth]).tolist())
        h = len(a & p)
        per_user.append(h / len(a))
        hits, total, sizes = hits + h, total + len(a), sizes + [len(p)]
    return {"recall": round(float(np.mean(per_user)), 4), "micro_recall": round(hits / total, 4),
            "mean_size": round(float(np.mean(sizes)), 1)}


def report(ev: WeekEvaluator, lists: dict[str, list[np.ndarray]]) -> dict:
    actual = [ev.actual[u] for u in ev.users]
    result = {"users": len(ev.users), "purchases": int(sum(len(a) for a in actual))}
    seen_before = ev.store.items["train_count"] > 0  # bought at some point in the training window
    result["new_item_purchase_share"] = round(
        float(np.mean([not seen_before[i] for a in actual for i in a])), 4)
    result["sources"] = {s: {d: coverage(actual, lists[s], d) for d in DEPTHS} for s in SOURCES}

    # Purchases found by exactly one source at full depth: what each source adds that the others miss.
    found = {s: [set(lst.tolist()) & a for lst, a in zip(lists[s], actual)] for s in SOURCES}
    result["only_source_purchases"] = {
        s: int(sum(len(f - set().union(*(found[o][u] for o in SOURCES if o != s))) for u, f in enumerate(found[s])))
        for s in SOURCES
    }
    union = [np.fromiter(set().union(*(set(lists[s][u].tolist()) for s in SOURCES)), dtype=np.int64)
             for u in range(len(actual))]
    result["union_all_sources"] = coverage(actual, union)
    result["merged"] = {b: coverage(actual, merge(lists, b)) for b in BUDGETS}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-dir", required=True, type=Path)
    parser.add_argument("--processed-dir", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--run-name", help="training run in MLflow whose best checkpoint to download")
    parser.add_argument("--alpha", type=float, default=None, help="default: best on val in the repeat-blend results")
    parser.add_argument("--max-users", type=int, default=None, help="subsample users per week (faster)")
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks"))
    args = parser.parse_args()
    start = time.perf_counter()

    if args.run_name:
        from src.export_embeddings import find_run
        from src.repeat_eval import fetch_checkpoint
        checkpoint, run_name = fetch_checkpoint(find_run(args.run_name), Path("models")), args.run_name
    else:
        checkpoint, run_name = args.checkpoint, args.checkpoint.parent.name
    alpha = args.alpha
    if alpha is None:
        alpha = json.loads((Path("benchmarks") / f"repeat_blend_{run_name}.json").read_text())["best_alpha_on_val"]

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    store = FeatureStore(args.features_dir, use_text=ck["model_config"]["use_text"])
    model = load_model(checkpoint, store, torch.device("cpu"))
    codes = product_codes(store, args.processed_dir)

    result = {"checkpoint": run_name, "alpha": alpha, "quotas": QUOTAS, "weeks": {}}
    for week in (VAL, TEST):
        ev = WeekEvaluator(store, week, max_users=args.max_users)
        lists = generate(ev, model, codes, alpha)
        result["weeks"][week.name] = report(ev, lists)
        r = result["weeks"][week.name]
        print(f"{week.name}: union {r['union_all_sources']}, merged {r['merged']} "
              f"({time.perf_counter() - start:.0f}s)", flush=True)
    result["seconds"] = round(time.perf_counter() - start, 1)
    result["peak_rss_gb"] = round(peak_rss_gb(), 2)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"candidates_{run_name}.json").write_text(json.dumps(result, indent=2) + "\n")
