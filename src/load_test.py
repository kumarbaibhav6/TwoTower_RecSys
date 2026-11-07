"""Load test for the recommendation API: concurrent POST /recommend at several concurrency levels.

Each level runs closed-loop: `concurrency` workers each send their next request as soon as the last one returns,
over random customers with a purchase history. Reports client-side latency (p50/p95/p99), throughput and the
server-side stage timings the API returns.

Usage:
    python -m src.load_test --url http://localhost:8000 --label local
    python -m src.load_test --url https://<app>.azurecontainerapps.io --label aca-1cpu
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import numpy as np

LEVELS = [1, 8, 32, 64]


async def run_level(client: httpx.AsyncClient, url: str, customers: list[str], concurrency: int,
                    n_requests: int, k: int) -> dict:
    latencies, server_ms, errors = [], [], 0
    queue = iter(np.random.default_rng(concurrency).choice(customers, n_requests).tolist())

    async def worker():
        nonlocal errors
        for cid in queue:
            start = time.perf_counter()
            try:
                r = await client.post(f"{url}/recommend", json={"customer_id": cid, "k": k})
                r.raise_for_status()
                latencies.append((time.perf_counter() - start) * 1000)
                server_ms.append(r.json()["timing_ms"]["total"])
            except httpx.HTTPError:
                errors += 1

    start = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    wall = time.perf_counter() - start
    lat = np.array(latencies)
    return {
        "concurrency": concurrency, "requests": n_requests, "errors": errors,
        "throughput_rps": round(len(lat) / wall, 1),
        **{f"p{q}_ms": round(float(np.percentile(lat, q)), 1) for q in (50, 95, 99)},
        "server_mean_ms": round(float(np.mean(server_ms)), 2),
    }


async def main(args) -> dict:
    limits = httpx.Limits(max_connections=max(args.levels), max_keepalive_connections=max(args.levels))
    async with httpx.AsyncClient(timeout=30, limits=limits) as client:
        health = (await client.get(f"{args.url}/health")).json()
        customers = (await client.get(f"{args.url}/customers/sample", params={"n": 1000})).json()["customer_ids"]
        await run_level(client, args.url, customers, 4, 50, args.k)  # warm-up, not reported
        levels = []
        for c in args.levels:
            levels.append(await run_level(client, args.url, customers, c, max(args.requests, 10 * c), args.k))
            print(levels[-1], flush=True)
    return {"label": args.label, "url": args.url, "k": args.k, "model": health["model"], "index": health["index"],
            "levels": levels}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--label", required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=LEVELS)
    parser.add_argument("--requests", type=int, default=500, help="per level (at least 10 per worker)")
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks"))
    args = parser.parse_args()
    result = asyncio.run(main(args))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"load_test_{args.label}.json").write_text(json.dumps(result, indent=2) + "\n")
