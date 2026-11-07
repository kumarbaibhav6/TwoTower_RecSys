"""ANN benchmark on a serving bundle: recall@k against exact search vs. single-query latency, for Flat / IVF / HNSW.

Queries are real user vectors (random customers with a purchase history) searched against the blended item vectors
the API uses, so recall is measured on the served ranking. Latency is one query at a time on one thread, the way
the API calls the index. With --apply, the fastest setting (by p99) that reaches --min-recall is written into the
bundle (index.faiss + config.json); exact search is always a candidate.

Usage:
    python -m src.build_index --bundle models/serve/full-main-6ep
    python -m src.build_index --bundle models/serve/full-main-6ep --apply
"""

import argparse
import json
import platform
import time
from pathlib import Path

import faiss
import numpy as np
import torch

from src.serving import ServingBundle, augmented_queries, make_index

SETTINGS = (
    [("flat", {})]
    + [("ivf", {"nlist": nlist, "nprobe": nprobe}) for nlist in (256, 1024) for nprobe in (1, 4, 16, 64)]
    + [("hnsw", {"m": 32, "ef_search": ef}) for ef in (16, 32, 64, 128, 256)]
)


def recall_vs_exact(found: np.ndarray, exact: np.ndarray) -> float:
    hits = [len(np.intersect1d(f[f >= 0], e)) for f, e in zip(found, exact)]
    return float(np.sum(hits) / exact.size)


def latency_ms(index: faiss.Index, queries: np.ndarray, k: int) -> np.ndarray:
    times = np.empty(len(queries))
    for i in range(len(queries)):
        start = time.perf_counter()
        index.search(queries[i:i + 1], k)
        times[i] = time.perf_counter() - start
    return times * 1000


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--queries", type=int, default=10_000, help="for recall")
    parser.add_argument("--latency-queries", type=int, default=2_000)
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    faiss.omp_set_num_threads(1)

    bundle = ServingBundle(args.bundle)
    rng = np.random.default_rng(0)
    with_history = np.flatnonzero(bundle.n_prior > 0)
    customers = np.sort(rng.choice(with_history, min(args.queries, len(with_history)), replace=False))
    queries = augmented_queries(bundle.user_vectors(customers))
    items = bundle.item_vectors
    exact = make_index(items, "flat").search(queries, args.k)[1]

    results = []
    for kind, params in SETTINGS:
        start = time.perf_counter()
        index = make_index(items, kind, **params)
        build_s = time.perf_counter() - start
        found = index.search(queries, args.k)[1]
        times = latency_ms(index, queries[:args.latency_queries], args.k)
        row = {"type": kind, **params, "recall_at_k": round(recall_vs_exact(found, exact), 4),
               "p50_ms": round(float(np.percentile(times, 50)), 3), "p99_ms": round(float(np.percentile(times, 99)), 3),
               "build_s": round(build_s, 2)}
        results.append(row)
        print(row, flush=True)

    eligible = [r for r in results if r["recall_at_k"] >= args.min_recall]
    chosen = min(eligible, key=lambda r: r["p99_ms"])
    result = {"bundle": bundle.config["run_name"], "items": len(items), "dim": items.shape[1], "k": args.k,
              "queries": len(queries), "latency_queries": min(args.latency_queries, len(queries)), "threads": 1,
              "machine": f"{platform.system()} {platform.machine()}", "min_recall": args.min_recall,
              "chosen": chosen, "results": results}
    print(f"chosen (fastest p99 with recall >= {args.min_recall}): {chosen}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"ann_tradeoff_{bundle.config['run_name']}.json").write_text(json.dumps(result, indent=2) + "\n")

    if args.apply:
        params = {k: v for k, v in chosen.items() if k in ("type", "nlist", "nprobe", "m", "ef_search")}
        index_path = args.bundle / "index.faiss"
        if params["type"] == "flat":
            index_path.unlink(missing_ok=True)
        else:
            faiss.write_index(make_index(items, params["type"], **params), str(index_path))
        config = {**bundle.config, "index": params}
        (args.bundle / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        print(f"applied {params} to {args.bundle}")
