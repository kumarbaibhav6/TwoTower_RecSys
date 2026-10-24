"""Print dataset statistics and check the memory footprint of the processed transactions.

Usage:
    python -m src.data_stats [--sample-frac 0.05]

Writes benchmarks/data_stats.json (full data only).
"""

import argparse
import json
import os
from pathlib import Path

import psutil

from src.data_loader import load

MEMORY_BUDGET_GB = 2.0


def rss_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def describe(counts) -> dict:
    q = counts.quantile([0.25, 0.5, 0.75, 0.9, 0.99])
    return {
        "mean": round(float(counts.mean()), 2),
        "p25": float(q[0.25]),
        "median": float(q[0.5]),
        "p75": float(q[0.75]),
        "p90": float(q[0.9]),
        "p99": float(q[0.99]),
        "max": int(counts.max()),
    }


def compute_stats(sample_frac: float | None) -> dict:
    rss_before = rss_gb()
    tx = load("transactions", sample_frac)
    rss_after = rss_gb()
    customers = load("customers", sample_frac, columns=["customer_idx"])
    articles = load("articles", sample_frac, columns=["article_id"])

    per_user = tx.groupby("customer_idx").size()
    per_item = tx.groupby("article_id").size()
    return {
        "transactions": len(tx),
        "duplicate_rows": int(tx.duplicated().sum()),
        "customers_total": len(customers),
        "customers_with_purchases": int(tx["customer_idx"].nunique()),
        "articles_total": len(articles),
        "articles_with_purchases": int(tx["article_id"].nunique()),
        "date_min": str(tx["t_dat"].min().date()),
        "date_max": str(tx["t_dat"].max().date()),
        "purchases_per_user": describe(per_user),
        "purchases_per_item": describe(per_item),
        "users_gt5_purchases": int((per_user > 5).sum()),
        "items_gt20_purchases": int((per_item > 20).sum()),
        "memory": {
            "dataframe_gb": round(tx.memory_usage(deep=True).sum() / 1e9, 3),
            "rss_increase_gb": round(rss_after - rss_before, 3),
            "dtypes": {col: str(dtype) for col, dtype in tx.dtypes.items()},
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-frac", type=float, default=None)
    args = parser.parse_args()

    stats = compute_stats(args.sample_frac)
    print(json.dumps(stats, indent=2))

    mem = stats["memory"]["dataframe_gb"]
    status = "OK" if mem < MEMORY_BUDGET_GB else "OVER BUDGET"
    print(f"\ntransactions in memory: {mem:.2f} GB (budget {MEMORY_BUDGET_GB} GB) -> {status}")

    if args.sample_frac is None:
        out = Path("benchmarks/data_stats.json")
        out.write_text(json.dumps(stats, indent=2) + "\n")
        print(f"wrote {out}")
