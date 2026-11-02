"""Repeat-purchase candidates + model + popularity prior.

Each user's `repeat` most recent past purchases go first; the rest of the list is the blend ranking
cosine(user, item) + alpha * log(1 + item sales in the last 7 days). (repeat, alpha) is chosen on the val
week by MAP@12 and applied once to the test week.

Usage:
    python -m src.repeat_eval --features-dir data/features/full --checkpoint models/full-no-item-id/best.pt
    python -m src.repeat_eval --features-dir ... --mlflow-run <run id>   # fetch checkpoint/epoch_N/best.pt
"""

import argparse
import json
import time
from pathlib import Path

import mlflow
import torch

from src.blend_eval import load_model
from src.dataset import FeatureStore
from src.evaluate import TOP_K, WeekEvaluator, merge_repeat
from src.split import TEST, VAL
from src.tracking import git_sha, init_mlflow, peak_rss_gb

REPEATS = [0, 3, 6, 12, 24, 100]  # 100 = every past purchase (the repeat-purchase baseline's list length)
ALPHAS = [0.02, 0.05, 0.1, 0.2, 1.0, 5.0]  # 5.0 is near pure popularity: repeat 100 + alpha 5 ~ the baseline
SUMMARY_KEYS = ("map@12", "recall@12", "recall@100", "no_history_map@12", "new_item_recall@100")


def ranked_lists(ev: WeekEvaluator, model, device, scores, prior, alpha: float):
    """Blend ranking with enough extra columns for any number of repeat slots."""
    return ev.recommend(model, device, k=TOP_K + max(REPEATS), prior=prior, prior_weight=alpha, scores=scores)


def fetch_checkpoint(run_id: str, dest: Path) -> Path:
    client = mlflow.MlflowClient()
    path = client.get_run(run_id).data.tags["best_checkpoint"]
    name = client.get_run(run_id).data.tags["mlflow.runName"]
    out = dest / name
    out.mkdir(parents=True, exist_ok=True)
    local = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=path, dst_path=str(out)))
    return local


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-dir", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--mlflow-run", help="training run id whose best checkpoint to download")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks"))
    args = parser.parse_args()

    init_mlflow("phase5-blend")
    checkpoint = args.checkpoint or fetch_checkpoint(args.mlflow_run, Path("models"))
    run_name = (args.checkpoint.parent.name if args.checkpoint else
                mlflow.MlflowClient().get_run(args.mlflow_run).data.tags["mlflow.runName"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck_cfg = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_config"]
    store = FeatureStore(args.features_dir, use_text=ck_cfg["use_text"])
    model = load_model(checkpoint, store, device)

    start = time.perf_counter()
    val = WeekEvaluator(store, VAL)
    val_scores, val_prior, val_repeat = val.scores_fn(model, device), val.recent_popularity(args.days), val.repeat_candidates()
    sweep = {}
    for alpha in ALPHAS:
        ranked = ranked_lists(val, model, device, val_scores, val_prior, alpha)
        for repeat in REPEATS:
            r = val.evaluate(None, device, top=merge_repeat(ranked, val_repeat, repeat))
            sweep[(repeat, alpha)] = r
            print(f"repeat {repeat:<3} alpha {alpha:<5} val MAP@12 {r['map@12']:.4f}  Recall@100 {r['recall@100']:.4f}"
                  f"  ({time.perf_counter() - start:.0f}s)", flush=True)
    best_repeat, best_alpha = max(sweep, key=lambda key: sweep[key]["map@12"])

    test = WeekEvaluator(store, TEST)
    test_scores, test_prior, test_repeat = test.scores_fn(model, device), test.recent_popularity(args.days), test.repeat_candidates()
    ranked = ranked_lists(test, model, device, test_scores, test_prior, best_alpha)
    result = {
        "checkpoint": run_name, "prior_days": args.days, "best_repeat_on_val": best_repeat, "best_alpha_on_val": best_alpha,
        "val_sweep": {f"repeat={rp},alpha={a}": {k: r[k] for k in SUMMARY_KEYS} for (rp, a), r in sweep.items()},
        "test_blend_no_repeat": test.evaluate(None, device, top=ranked[:, :TOP_K]),
        "test_repeat_blend": test.evaluate(None, device, top=merge_repeat(ranked, test_repeat, best_repeat)),
        "seconds": round(time.perf_counter() - start, 1), "peak_rss_gb": round(peak_rss_gb(), 2),
    }
    print(json.dumps({k: result[k] for k in ["best_repeat_on_val", "best_alpha_on_val", "test_blend_no_repeat",
                                             "test_repeat_blend"]}, indent=2))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"repeat_blend_{run_name}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")

    with mlflow.start_run(run_name=f"repeat-blend-{run_name}"):
        mlflow.log_params({"checkpoint": run_name, "prior_days": args.days, "best_repeat": best_repeat,
                           "best_alpha": best_alpha, "git_sha": git_sha()})
        for split in ("test_blend_no_repeat", "test_repeat_blend"):
            mlflow.log_metrics({f"{split}_{k.replace('@', '_at_')}": v for k, v in result[split].items()})
        mlflow.log_dict(result, out.name)
