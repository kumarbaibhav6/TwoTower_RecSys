"""Export the serving bundle (see src/serving.py) for a trained checkpoint, as of the start of the test week.

The cutoff is the test week's first day, so what is served is exactly what was evaluated: the bundle is checked
against the offline evaluation (same user vectors, same lists, same test MAP@12) before it is published.
(repeat, alpha) come from the val-week sweep in benchmarks/repeat_blend_<run>.json.
Each export needs a new --register version (twotower-serve:2 is the full-main-6ep bundle; :1 is an empty failed run).

Usage:
    python -m src.export_embeddings --features-dir data/features/sample_0.05 --processed-dir data/processed \
        --checkpoint models/sample-12wk/best.pt --repeat 6 --alpha 0.05 --publish-dir models/serve/sample-12wk
    python -m infra.submit_job src.export_embeddings --requirements requirements-job-serve.txt \
        --data hm-features:1 --extra-data hm-processed:1 --register twotower-serve:3 -- \
        --features-dir '{data}' --processed-dir '{extra}' --run-name full-main-6ep --publish-dir '{artifacts}'
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch

from src.blend_eval import load_model
from src.dataset import FeatureStore
from src.evaluate import TOP_K, WeekEvaluator
from src.repeat_eval import fetch_checkpoint
from src.serving import ARTICLE_FIELDS, MAX_REPEAT, ServingBundle
from src.split import TEST
from src.tracking import git_sha, peak_rss_gb

TRAINING_EXPERIMENT = "phase5-training"


def find_run(run_name: str) -> str:
    """Most recent training run with this name that uploaded a best checkpoint."""
    runs = mlflow.search_runs(experiment_names=[TRAINING_EXPERIMENT], output_format="list",
                              filter_string=f"tags.mlflow.runName = '{run_name}'", order_by=["start_time DESC"])
    runs = [r for r in runs if "best_checkpoint" in r.data.tags]
    if not runs:
        raise SystemExit(f"no run named {run_name!r} with a best_checkpoint tag in {TRAINING_EXPERIMENT}")
    return runs[0].info.run_id


def recent_unique_items(seq_customer: np.ndarray, seq_item: np.ndarray, before: np.ndarray, n_customers: int,
                        max_items: int) -> np.ndarray:
    """[n_customers, max_items] distinct items bought in the `before` rows, newest first, -1 padded.

    Rows are sorted by (customer, day), so row position is time order (ties within a day keep row order), which
    matches WeekEvaluator.repeat_candidates.
    """
    out = np.full((n_customers, max_items), -1, dtype=np.int32)
    pos = np.flatnonzero(before)
    if len(pos) == 0:
        return out
    cust, item = seq_customer[pos].astype(np.int64), seq_item[pos].astype(np.int64)
    pair = cust << 20 | item  # item rows < 2**20
    order = np.lexsort((pos, pair))
    pair_sorted = pair[order]
    keep = order[np.r_[pair_sorted[1:] != pair_sorted[:-1], True]]  # last (newest) row of each (customer, item)
    cust, item, pos = cust[keep], item[keep], pos[keep]
    order = np.lexsort((-pos, cust))  # by customer, newest first
    cust, item = cust[order], item[order]
    starts = np.r_[0, np.flatnonzero(np.diff(cust)) + 1]
    rank = np.arange(len(cust)) - np.repeat(starts, np.diff(np.r_[starts, len(cust)]))
    first = rank < max_items
    out[cust[first], rank[first]] = item[first]
    return out


def user_state(store: FeatureStore, cutoff_day: int) -> dict[str, np.ndarray]:
    """Per customer, the history inputs as of the cutoff (purchases strictly before it)."""
    s = store.seq
    n_customers = len(store.users["user_vocab"])
    customers = np.arange(n_customers, dtype=np.int64)
    day = np.asarray(s["seq_day"])
    seq_customer = np.asarray(s["seq_customer"])
    keys = seq_customer.astype(np.int64) << 16 | day.astype(np.int64)
    user_start = np.searchsorted(keys, customers << 16, side="left")
    hist_end = np.searchsorted(keys, customers << 16 | cutoff_day, side="left")
    del keys
    _, rows, valid = store.history(hist_end, user_start)
    n_prior = hist_end - user_start
    last_day = np.where(n_prior > 0, day[np.maximum(hist_end - 1, 0)], 0)
    return {
        "user_hist_rows": np.where(valid, rows, -1).astype(np.int32),
        "user_n_prior": n_prior.astype(np.int32),
        "user_last_day": last_day.astype(np.int16),
        "user_repeat": recent_unique_items(seq_customer, np.asarray(s["seq_item"]), day < cutoff_day,
                                           n_customers, MAX_REPEAT),
    }


def write_catalog(store: FeatureStore, processed_dir: Path, out: Path) -> None:
    articles = pd.read_parquet(processed_dir / "articles.parquet", columns=ARTICLE_FIELDS)
    order = pd.DataFrame({"article_id": store.items["article_id"]})
    articles = order.merge(articles, on="article_id", how="left", validate="one_to_one")
    for col in ARTICLE_FIELDS[1:]:
        articles[col] = articles[col].astype("string").fillna("")
    articles.to_parquet(out / "articles.parquet", index=False)

    customers = pd.read_parquet(processed_dir / "customers.parquet", columns=["customer_idx", "customer_id"])
    customers = customers.sort_values("customer_idx")
    n_customers = len(store.users["user_vocab"])
    assert (customers["customer_idx"].to_numpy()[:n_customers] == np.arange(n_customers)).all()
    np.save(out / "customer_ids.npy", customers["customer_id"].to_numpy()[:n_customers].astype("S64"))


def write_bundle(store: FeatureStore, model, model_config: dict, week: WeekEvaluator, processed_dir: Path, out: Path,
                 run_name: str, repeat: int, alpha: float, days: int = 7) -> dict:
    """Write every bundle file for recommending as of `week`'s first day; returns the config."""
    cpu = torch.device("cpu")
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "model_config": model_config}, out / "model.pt")
    np.save(out / "item_emb.npy", week.item_vectors(model, cpu).numpy())
    np.save(out / "prior.npy", week.recent_popularity(days))
    for name in ("items.npz", "users.npz"):
        shutil.copy(store.dir / name, out / name)
    for name, array in user_state(store, week.start_day).items():
        np.save(out / f"{name}.npy", array)
    write_catalog(store, processed_dir, out)
    config = {
        "run_name": run_name, "git_sha": git_sha(), "cutoff_date": str(week.week.start.date()),
        "cutoff_day": week.start_day, "repeat": repeat, "alpha": alpha, "prior_days": days, "index": {"type": "flat"},
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return config


def check_parity(bundle_dir: Path, test: WeekEvaluator, model, repeat: int, alpha: float, days: int,
                 chunk: int = 4096) -> dict:
    """The bundle must give the offline evaluation's user vectors, lists and metrics."""
    bundle = ServingBundle(bundle_dir)
    cpu = torch.device("cpu")
    sample = slice(0, min(10_000, len(test.users)))
    reference_vecs = test.user_vectors(model, cpu)[sample].numpy()
    vec_diff = float(np.abs(bundle.user_vectors(test.users[sample]) - reference_vecs).max())

    start = time.perf_counter()
    served = np.concatenate([bundle.recommend(test.users[i:i + chunk], k=TOP_K, repeat=repeat)
                             for i in range(0, len(test.users), chunk)])
    seconds = time.perf_counter() - start
    reference = test.recommend(model, cpu, prior=test.recent_popularity(days), prior_weight=alpha, repeat=repeat)
    return {
        "users": len(test.users),
        "max_user_vector_diff": vec_diff,
        "identical_lists_pct": round(100 * float((served == reference).all(axis=1).mean()), 3),
        "identical_top12_pct": round(100 * float((served[:, :12] == reference[:, :12]).all(axis=1).mean()), 3),
        "served_metrics": test.evaluate(None, cpu, top=served),
        "reference_metrics": test.evaluate(None, cpu, top=reference),
        "served_seconds": round(seconds, 1),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-dir", required=True, type=Path)
    parser.add_argument("--processed-dir", required=True, type=Path, help="articles.parquet and customers.parquet")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--run-name", help="training run in MLflow whose best checkpoint to download")
    parser.add_argument("--repeat", type=int, default=None, help="default: best on val in the repeat-blend results")
    parser.add_argument("--alpha", type=float, default=None, help="default: best on val in the repeat-blend results")
    parser.add_argument("--days", type=int, default=7, help="popularity prior window")
    parser.add_argument("--publish-dir", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks"))
    args = parser.parse_args()
    start = time.perf_counter()

    run_name = args.run_name or args.checkpoint.parent.name
    if args.run_name:
        checkpoint = fetch_checkpoint(find_run(args.run_name), Path("models"))
    else:
        checkpoint = args.checkpoint
    repeat, alpha = args.repeat, args.alpha
    if repeat is None or alpha is None:
        blend = json.loads((Path("benchmarks") / f"repeat_blend_{run_name}.json").read_text())
        repeat = blend["best_repeat_on_val"] if repeat is None else repeat
        alpha = blend["best_alpha_on_val"] if alpha is None else alpha
    print(f"run {run_name}: repeat {repeat}, alpha {alpha}", flush=True)

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    store = FeatureStore(args.features_dir, use_text=ck["model_config"]["use_text"])
    model = load_model(checkpoint, store, torch.device("cpu"))
    test = WeekEvaluator(store, TEST)

    config = write_bundle(store, model, ck["model_config"], test, args.processed_dir, args.publish_dir,
                          run_name=run_name, repeat=repeat, alpha=alpha, days=args.days)
    out = args.publish_dir
    print(f"bundle written to {out} ({time.perf_counter() - start:.0f}s)", flush=True)

    parity = check_parity(out, test, model, repeat, alpha, args.days)
    config["test_metrics"] = {k: parity["served_metrics"][k] for k in ("map@12", "recall@12", "recall@100")}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    sizes_mb = {p.name: round(p.stat().st_size / 1e6, 1) for p in sorted(out.iterdir())}
    result = {**config, "parity": parity, "bundle_mb": sizes_mb, "bundle_total_mb": round(sum(sizes_mb.values()), 1),
              "seconds": round(time.perf_counter() - start, 1), "peak_rss_gb": round(peak_rss_gb(), 2)}
    print(json.dumps({k: v for k, v in result.items() if k != "bundle_mb"}, indent=2), flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"serve_bundle_{run_name}.json").write_text(json.dumps(result, indent=2) + "\n")
