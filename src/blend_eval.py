"""Popularity-prior blend: score = cosine(user, item) + alpha * log(1 + item sales in the last 7 days).

alpha is chosen on the val week only and then applied once to the test week. alpha = 0 is the pure model.

Usage:
    python -m src.blend_eval --features-dir data/features/full --checkpoint models/full-main/best.pt
"""

import argparse
import json
from pathlib import Path

import mlflow
import torch

from src.dataset import FeatureStore
from src.evaluate import WeekEvaluator
from src.model import ModelConfig, TwoTowerModel, build_model
from src.split import TEST, VAL
from src.tracking import init_mlflow

ALPHAS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0]  # large alpha -> pure popularity


def load_model(path: Path, store: FeatureStore, device: torch.device) -> TwoTowerModel:
    ck = torch.load(path, map_location=device, weights_only=False)
    return build_model(ModelConfig(**ck["model_config"]), store, device, ck["model"]).eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck_cfg = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model_config"]
    store = FeatureStore(args.features_dir, use_text=ck_cfg["use_text"])
    model = load_model(args.checkpoint, store, device)

    val = WeekEvaluator(store, VAL)
    val_scores, val_prior = val.scores_fn(model, device), val.recent_popularity(args.days)
    sweep = {}
    for alpha in ALPHAS:
        r = val.evaluate(model, device, prior=val_prior, prior_weight=alpha, scores=val_scores)
        sweep[alpha] = r
        print(f"alpha {alpha:<6} val MAP@12 {r['map@12']:.4f}  Recall@100 {r['recall@100']:.4f}", flush=True)
    best_alpha = max(sweep, key=lambda a: sweep[a]["map@12"])

    test = WeekEvaluator(store, TEST)
    test_prior = test.recent_popularity(args.days)
    test_scores = test.scores_fn(model, device)
    result = {
        "checkpoint": str(args.checkpoint), "prior_days": args.days, "best_alpha_on_val": best_alpha,
        "val_sweep": {str(a): {k: v for k, v in r.items() if k in ("map@12", "recall@12", "recall@100")} for a, r in sweep.items()},
        "test_pure_model": test.evaluate(model, device, scores=test_scores),
        "test_blend": test.evaluate(model, device, prior=test_prior, prior_weight=best_alpha, scores=test_scores),
    }
    print(json.dumps({k: result[k] for k in ["best_alpha_on_val", "test_pure_model", "test_blend"]}, indent=2))
    out = args.out or Path("benchmarks") / f"blend_{args.checkpoint.parent.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    init_mlflow("phase5-blend")
    with mlflow.start_run(run_name=f"blend-{args.checkpoint.parent.name}"):
        mlflow.log_params({"checkpoint": args.checkpoint.parent.name, "prior_days": args.days, "best_alpha": best_alpha})
        for split in ("test_pure_model", "test_blend"):
            mlflow.log_metrics({f"{split}_{k.replace('@', '_at_')}": v for k, v in result[split].items()})
        mlflow.log_dict(result, out.name)
