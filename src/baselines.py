"""Non-learned baselines, evaluated on the val and test weeks.

Usage:
    python -m src.baselines --sample-frac 0.05   # quick local run
    python -m src.baselines                      # full data (run on Azure ML: infra/submit_job.py)

Writes benchmarks/baselines[_sample<frac>].json and logs one MLflow child run per baseline.
"""

import argparse
import json
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.metrics import evaluate
from src.split import TRAIN_END, WEEKS, Week, ground_truth, load_transactions, sparsity_report
from src.tracking import git_sha, init_mlflow, peak_rss_gb

N_CANDIDATES = 100  # long enough for Recall@100
POPULARITY_DAYS = 7


def popularity(history_end: pd.Timestamp, k: int, sample_frac: float | None) -> np.ndarray:
    """Top-k best sellers of the last POPULARITY_DAYS before the target week."""
    start = history_end - pd.Timedelta(days=POPULARITY_DAYS - 1)
    tx = load_transactions(start, history_end, ["article_id"], sample_frac)
    return tx["article_id"].value_counts().index[:k].to_numpy()


def repeat_purchase(
    users: np.ndarray, history_end: pd.Timestamp, k: int, fallback: np.ndarray, sample_frac: float | None
) -> dict:
    """Each user's previously bought articles, most recent first, padded with `fallback`."""
    tx = load_transactions(
        end=history_end, columns=["t_dat", "customer_idx", "article_id"], sample_frac=sample_frac, customers=users
    )
    recent = tx.sort_values(["customer_idx", "t_dat"], ascending=[True, False], kind="stable").drop_duplicates(
        ["customer_idx", "article_id"]
    )
    recent = recent[recent.groupby("customer_idx").cumcount() < k]
    # Rows are grouped by customer, so split the article array wherever the customer changes.
    codes, items = recent["customer_idx"].to_numpy(), recent["article_id"].to_numpy()
    bounds = np.flatnonzero(np.diff(codes)) + 1
    own_by_user = dict(zip(codes[np.r_[0, bounds]], np.split(items, bounds))) if len(codes) else {}

    preds = {}
    for user in users:
        own = own_by_user.get(user, np.empty(0, dtype=np.int32))
        fill = fallback[~np.isin(fallback, own)][: k - len(own)]
        preds[user] = np.concatenate([own, fill]).tolist()
    return preds


def run(sample_frac: float | None) -> dict:
    results = {}
    for week in WEEKS.values():
        start = time.perf_counter()
        actual = ground_truth(week, sample_frac)
        actual_by_user = {u: set(items) for u, items in actual.items()}
        users = actual.index.to_numpy()

        pop = popularity(week.history_end, N_CANDIDATES, sample_frac)
        pop_list = pop.tolist()
        results.setdefault("popularity", {})[week.name] = evaluate(actual_by_user, {u: pop_list for u in users})
        repeat = repeat_purchase(users, week.history_end, N_CANDIDATES, pop, sample_frac)
        results.setdefault("repeat_purchase", {})[week.name] = evaluate(actual_by_user, repeat)
        print(f"{week.name}: {len(users):,} users evaluated in {time.perf_counter() - start:.1f}s")
    return results


def sparsity(sample_frac: float | None) -> dict:
    train = load_transactions(end=TRAIN_END, columns=["customer_idx", "article_id"], sample_frac=sample_frac)
    eval_users = {w.name: ground_truth(w, sample_frac).index.to_numpy() for w in WEEKS.values()}
    return sparsity_report(train, eval_users)


def log_to_mlflow(results: dict, sparsity_stats: dict, sample_frac: float | None) -> None:
    with mlflow.start_run(run_name="baselines") as parent:
        mlflow.log_params({"sample_frac": sample_frac or 1.0, "git_sha": git_sha()})
        mlflow.log_dict(sparsity_stats, "sparsity_report.json")
        for baseline, by_week in results.items():
            with mlflow.start_run(run_name=baseline, nested=True):
                mlflow.log_params({"model": baseline, "sample_frac": sample_frac or 1.0, "git_sha": git_sha()})
                for week, metrics in by_week.items():
                    for name, value in metrics.items():
                        # MLflow metric names can't contain '@'.
                        mlflow.log_metric(f"{week}_{name.replace('@', '_at_')}", value)
        mlflow.log_metric("peak_rss_gb", peak_rss_gb())
    print(f"mlflow run: {parent.info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-frac", type=float, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks"))
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    results = run(args.sample_frac)
    sparsity_stats = sparsity(args.sample_frac)
    output = {"results": results, "sparsity": sparsity_stats, "peak_rss_gb": round(peak_rss_gb(), 2)}
    print(json.dumps(output, indent=2))

    suffix = "" if args.sample_frac is None else f"_sample{args.sample_frac}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"baselines{suffix}.json"
    out.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {out}")

    if not args.no_mlflow:
        init_mlflow("phase2-baselines")
        log_to_mlflow(results, sparsity_stats, args.sample_frac)
