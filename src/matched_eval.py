"""Matched baseline: how much does the two-tower model add on top of repeat purchases + popularity?

Every list puts the customer's `repeat` most recent past purchases first (identical slots, same code) and fills the
rest from one of three rankings:
    popularity   item sales in the 7 days before the week (the same for everyone)
    two_tower    cosine(user, item) alone
    blend        cosine(user, item) + alpha * log(1 + 7-day sales), the deployed ranking
So at equal `repeat`, any MAP@12 gap between rows comes from the fill ranking alone.

Reported per week (val, test):
    metrics for each ranking at each repeat in REPEATS
    paired bootstrap 95% intervals (customers resampled with replacement) for blend - popularity, two_tower -
        popularity and blend - two_tower, at the blend's repeat ("matched") and with each ranking at its own best
        repeat on val ("tuned", so the baseline is not handicapped by a repeat chosen for the blend)
    MAP@12 by number of distinct past purchases, where the fill ranking decides 12 - min(repeat, n) slots
    seen-items diagnostic: the same comparison with each customer's purchases cut to items bought at least once in
        training (customers left with none are dropped), to test whether articles new since the model's training
        cutoff explain a gap between weeks. Lists are unchanged, so a new item in a list now counts as a miss.

A checkpoint trained through val (--train-through-val) has seen the val week, so only test is scored, with the
(alpha, repeat) chosen on val for an earlier model: pass --alpha/--repeat, or they are read from
benchmarks/repeat_blend_<run>.json.

Usage:
    python -m src.matched_eval --features-dir data/features/sample_0.05 --checkpoint models/sample-12wk/best.pt
    python -m infra.submit_job src.matched_eval --requirements requirements-job-serve.txt \
        --data hm-features:1 --experiment phase5-blend -- --features-dir '{data}' --run-name full-main-6ep
    ... -- --features-dir '{data}' --run-name full-through-val --alpha 0.05 --repeat 6
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.blend_eval import load_model
from src.dataset import FeatureStore
from src.evaluate import TOP_K, WeekEvaluator, merge_repeat
from src.metrics import average_precision_at_k
from src.split import TEST, VAL
from src.tracking import peak_rss_gb

RANKINGS = ["popularity", "two_tower", "blend"]
REPEATS = [0, 3, 6, 12]
PAIRS = [("blend", "popularity"), ("two_tower", "popularity"), ("blend", "two_tower")]
# Distinct past purchases: (low, high, label), inclusive.
HISTORY_BUCKETS = [(0, 0, "0"), (1, 5, "1-5"), (6, 20, "6-20"), (21, np.inf, "21+")]
SUMMARY_KEYS = ("map@12", "recall@12", "recall@100", "no_history_map@12", "new_item_recall@100")


def ranked_lists(ev: WeekEvaluator, model, device, prior: np.ndarray, alpha: float, depth: int) -> dict:
    """[n_users, depth] item rows per ranking, best first."""
    scores = ev.scores_fn(model, device)
    popular = np.argsort(-prior, kind="stable")[:depth]
    return {
        "popularity": np.broadcast_to(popular, (len(ev.users), depth)),
        "two_tower": ev.recommend(model, device, k=depth, scores=scores),
        "blend": ev.recommend(model, device, k=depth, prior=prior, prior_weight=alpha, scores=scores),
    }


def per_user_ap(actual: list[set], top: np.ndarray) -> np.ndarray:
    return np.array([average_precision_at_k(a, row.tolist()) for a, row in zip(actual, top)])


def new_item_recall_at(ev: WeekEvaluator, top: np.ndarray, k: int) -> float:
    """Share of (customer, new article) purchases found in the top k, micro-averaged."""
    hits = total = 0
    for u, row in zip(ev.users, top):
        new = {i for i in ev.actual[u] if ev.new_items[i]}
        total += len(new)
        hits += len(new & set(row[:k].tolist()))
    return round(hits / total, 5) if total else 0.0


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 0, chunk: int = 50) -> dict:
    """Mean of a - b and its 95% interval, resampling users (paired: the same users for both)."""
    rng = np.random.default_rng(seed)
    diffs, ratios = [], []
    for _ in range(0, n_boot, chunk):
        idx = rng.integers(0, len(a), size=(chunk, len(a)))
        mean_a, mean_b = a[idx].mean(axis=1), b[idx].mean(axis=1)
        diffs.append(mean_a - mean_b)
        ratios.append(mean_a / mean_b - 1)
    diffs, ratios = np.concatenate(diffs)[:n_boot], np.concatenate(ratios)[:n_boot]
    return {
        "a": round(float(a.mean()), 5), "b": round(float(b.mean()), 5),
        "diff": round(float(a.mean() - b.mean()), 5),
        "diff_ci95": [round(float(x), 5) for x in np.percentile(diffs, [2.5, 97.5])],
        "relative": round(float(a.mean() / b.mean() - 1), 4),
        "relative_ci95": [round(float(x), 4) for x in np.percentile(ratios, [2.5, 97.5])],
        "share_of_resamples_a_not_better": round(float(np.mean(diffs <= 0)), 4),
    }


def by_history(ap: dict, n_past: np.ndarray, repeat: int) -> dict:
    out = {}
    for low, high, label in HISTORY_BUCKETS:
        mask = (n_past >= low) & (n_past <= high)
        out[label] = {"users": int(mask.sum()), **{r: round(float(ap[(r, repeat)][mask].mean()), 5)
                                                   for r in RANKINGS if mask.any()}}
    return out


def run_week(ev: WeekEvaluator, model, device, alpha: float, days: int) -> tuple[dict, dict, dict, np.ndarray]:
    """Summary metrics and per-user AP for every (ranking, repeat), per-user AP on seen items only (customers with
    at least one seen-item purchase), and distinct past purchases per user."""
    past = ev.repeat_candidates(max_items=10**9)
    ranked = ranked_lists(ev, model, device, ev.recent_popularity(days), alpha, TOP_K + max(REPEATS))
    actual = [ev.actual[u] for u in ev.users]
    seen = [{i for i in a if not ev.new_items[i]} for a in actual]
    has_seen = np.array([bool(s) for s in seen])
    metrics, ap, ap_seen = {}, {}, {}
    for ranking in RANKINGS:
        for repeat in REPEATS:
            top = merge_repeat(ranked[ranking], past, repeat)
            r = ev.evaluate(None, device, top=top)
            metrics[(ranking, repeat)] = {k: round(r[k], 5) for k in SUMMARY_KEYS}
            metrics[(ranking, repeat)]["new_item_recall@12"] = new_item_recall_at(ev, top, 12)
            ap[(ranking, repeat)] = per_user_ap(actual, top)
            ap_seen[(ranking, repeat)] = per_user_ap([s for s, h in zip(seen, has_seen) if h], top[has_seen])
            metrics[(ranking, repeat)]["seen_items_map@12"] = round(float(ap_seen[(ranking, repeat)].mean()), 5)
    return metrics, ap, ap_seen, np.array([len(p) for p in past])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-dir", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--run-name", help="training run in MLflow whose best checkpoint to download")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--max-users", type=int, default=None, help="subsample users per week (faster)")
    parser.add_argument("--alpha", type=float, default=None, help="default: best on val in the repeat-blend results")
    parser.add_argument("--repeat", type=int, default=None, help="default: best on val in the repeat-blend results")
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks"))
    args = parser.parse_args()
    start = time.perf_counter()

    if args.run_name:
        from src.export_embeddings import find_run
        from src.repeat_eval import fetch_checkpoint
        checkpoint, run_name = fetch_checkpoint(find_run(args.run_name), Path("models")), args.run_name
    else:
        checkpoint, run_name = args.checkpoint, args.checkpoint.parent.name
    alpha, blend_repeat = args.alpha, args.repeat
    if alpha is None or blend_repeat is None:
        blend_cfg = json.loads((Path("benchmarks") / f"repeat_blend_{run_name}.json").read_text())
        alpha = blend_cfg["best_alpha_on_val"] if alpha is None else alpha
        blend_repeat = blend_cfg["best_repeat_on_val"] if blend_repeat is None else blend_repeat

    device = torch.device("cpu")
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    through_val = ck.get("train_config", {}).get("train_through_val", False)
    store = FeatureStore(args.features_dir, use_text=ck["model_config"]["use_text"])
    model = load_model(checkpoint, store, device)  # restores the checkpoint's training window and vocab

    weeks = {}
    for week in ((TEST,) if through_val else (VAL, TEST)):
        ev = WeekEvaluator(store, week, max_users=args.max_users)
        weeks[week.name] = (ev, *run_week(ev, model, device, alpha, args.days))
        print(f"{week.name}: {len(ev.users):,} users ({time.perf_counter() - start:.0f}s)", flush=True)

    if through_val:  # no clean val week to tune on: every ranking uses the given repeat
        best_repeat = {r: blend_repeat for r in RANKINGS}
    else:
        val_metrics = weeks[VAL.name][1]
        best_repeat = {r: max(REPEATS, key=lambda rp: val_metrics[(r, rp)]["map@12"]) for r in RANKINGS}
    result = {"checkpoint": run_name, "train_through_val": through_val, "alpha": alpha, "blend_repeat": blend_repeat,
              "best_repeat_on_val": None if through_val else best_repeat, "weeks": {}}
    for name, (ev, metrics, ap, ap_seen, n_past) in weeks.items():
        purchases = [i for u in ev.users for i in ev.actual[u]]
        result["weeks"][name] = {
            "users": len(ev.users),
            "new_item_purchase_share": round(float(np.mean(ev.new_items[purchases])), 4),
            "seen_items_users": len(ap_seen[(RANKINGS[0], REPEATS[0])]),
            "metrics": {f"{r},repeat={rp}": m for (r, rp), m in metrics.items()},
            "matched": {f"{a}-{b}": paired_bootstrap(ap[(a, blend_repeat)], ap[(b, blend_repeat)], args.n_boot)
                        for a, b in PAIRS},
            "tuned": {f"{a}-{b}": paired_bootstrap(ap[(a, best_repeat[a])], ap[(b, best_repeat[b])], args.n_boot)
                      for a, b in PAIRS},
            "map@12_by_history": by_history(ap, n_past, blend_repeat),
            "seen_items_matched": {
                f"{a}-{b}": paired_bootstrap(ap_seen[(a, blend_repeat)], ap_seen[(b, blend_repeat)], args.n_boot)
                for a, b in PAIRS},
            "seen_items_no_repeat": {
                f"{a}-{b}": paired_bootstrap(ap_seen[(a, 0)], ap_seen[(b, 0)], args.n_boot) for a, b in PAIRS},
        }
        m = result["weeks"][name]["matched"]["blend-popularity"]
        print(f"{name} matched blend vs popularity: MAP@12 {m['a']:.4f} vs {m['b']:.4f}, "
              f"relative {m['relative']:+.1%} (95% CI {m['relative_ci95'][0]:+.1%} to {m['relative_ci95'][1]:+.1%})",
              flush=True)
        m = result["weeks"][name]["seen_items_matched"]["blend-popularity"]
        print(f"{name} seen items only: MAP@12 {m['a']:.4f} vs {m['b']:.4f}, relative {m['relative']:+.1%} "
              f"(95% CI {m['relative_ci95'][0]:+.1%} to {m['relative_ci95'][1]:+.1%})", flush=True)
    result["seconds"] = round(time.perf_counter() - start, 1)
    result["peak_rss_gb"] = round(peak_rss_gb(), 2)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"matched_baseline_{run_name}.json").write_text(json.dumps(result, indent=2) + "\n")
