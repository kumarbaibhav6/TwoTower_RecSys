"""Phase 3 verification: batch shapes, a leakage audit, and one full DataLoader pass with flat memory.

Usage:
    python -m src.check_dataset --features-dir data/features/sample_0.05
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import psutil

from src.dataset import FeatureStore, TwoTowerDataset, make_loader
from src.features import HIST_LEN
from src.tracking import peak_rss_gb

LEAKAGE_SAMPLE = 200_000


def rss_gb() -> float:
    """Resident memory of this process and its DataLoader workers."""
    proc = psutil.Process(os.getpid())
    return sum(p.memory_info().rss for p in [proc, *proc.children(recursive=True)]) / 1e9


def leakage_audit(store: FeatureStore, ds: TwoTowerDataset, seed: int = 0) -> dict:
    """For random examples, every history position must be the same customer on a strictly earlier day."""
    rng = np.random.default_rng(seed)
    rows = rng.choice(ds.rows, size=min(LEAKAGE_SAMPLE, len(ds)), replace=False)
    s = store.seq
    hist_end, user_start = s["seq_hist_end"][rows].astype(np.int64), s["seq_user_start"][rows].astype(np.int64)
    positions = hist_end[:, None] - HIST_LEN + np.arange(HIST_LEN)
    valid = positions >= user_start[:, None]
    pos = np.where(valid, positions, rows[:, None])  # invalid slots compare the row with itself (ignored)
    same_customer = s["seq_customer"][pos] == s["seq_customer"][rows][:, None]
    earlier_day = s["seq_day"][pos] < s["seq_day"][rows][:, None]
    violations = int((valid & ~(same_customer & earlier_day)).sum())
    # Completeness: history must reach back to the user's first purchase or fill all HIST_LEN slots.
    prior_days = hist_end - user_start
    complete = valid.sum(1) == np.minimum(prior_days, HIST_LEN)
    return {"examples_checked": len(rows), "history_slots_checked": int(valid.sum()),
            "violations": violations, "incomplete_histories": int((~complete).sum())}


def full_pass(ds: TwoTowerDataset, batch_size: int, num_workers: int) -> dict:
    loader = make_loader(ds, batch_size=batch_size, num_workers=num_workers)
    rss, seen, start = [], 0, time.perf_counter()
    shapes = None
    for i, batch in enumerate(loader):
        if shapes is None:
            shapes = {f"{group}.{k}": list(v.shape) for group in ("user", "item") for k, v in batch[group].items()}
            shapes["item_row"] = list(batch["item_row"].shape)
            shapes["item_log_q"] = list(batch["item_log_q"].shape)
        seen += len(batch["item_row"])
        if i % max(1, len(loader) // 20) == 0:
            rss.append(round(rss_gb(), 3))
    seconds = time.perf_counter() - start
    # After warm-up (first 25% of samples), memory should not keep climbing.
    warm = rss[len(rss) // 4:]
    return {
        "batches": len(loader), "examples": seen, "seconds": round(seconds, 1),
        "examples_per_second": int(seen / seconds), "batch_shapes": shapes,
        "rss_gb_samples": rss, "rss_growth_after_warmup_gb": round(warm[-1] - warm[0], 3),
    }


def run_checks(features_dir: Path, out_dir: Path, name: str | None, batch_size: int = 2048,
               num_workers: int = 2) -> dict:
    store = FeatureStore(features_dir)
    ds = TwoTowerDataset(store)
    result = {
        "features_dir": str(features_dir), "train_examples": len(ds),
        "text_features": store.text_emb is not None, "batch_size": batch_size,
        "num_workers": num_workers, "leakage": leakage_audit(store, ds),
        "full_pass": full_pass(ds, batch_size, num_workers),
    }
    result["peak_rss_gb_main_process"] = round(peak_rss_gb(), 2)
    print(json.dumps(result, indent=2))

    suffix = f"_{name}" if name else ""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"dataset_check{suffix}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks"))
    parser.add_argument("--name", default=None, help="output file suffix, e.g. sample0.05")
    args = parser.parse_args()

    run_checks(args.features_dir, args.out_dir, args.name, args.batch_size, args.num_workers)
