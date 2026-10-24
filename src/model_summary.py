"""Phase 4 verification: a real batch through the model, parameter counts and a forward/backward timing.

Usage:
    python -m src.model_summary --features-dir data/features/sample_0.05
"""

import argparse
import json
import time
from pathlib import Path

import torch

from src.dataset import FeatureStore, TwoTowerDataset
from src.model import ModelConfig, TwoTowerModel, parameter_counts

# fp32 weights + gradients + two AdamW moment buffers
BYTES_PER_TRAINED_PARAM = 4 * 4


def summarize(model: TwoTowerModel) -> dict:
    counts = parameter_counts(model)
    counts["training_state_gb"] = round(counts["total"] * BYTES_PER_TRAINED_PARAM / 1e9, 2)
    return counts


def timed_step(model: TwoTowerModel, batch: dict, repeats: int = 3) -> float:
    model.train()
    seconds = []
    for _ in range(repeats + 1):  # first run is warm-up
        start = time.perf_counter()
        u, i = model(batch)
        loss = torch.nn.functional.cross_entropy(model.score(u, i), torch.arange(len(u)))
        loss.backward()
        model.zero_grad(set_to_none=True)
        seconds.append(time.perf_counter() - start)
    return round(1000 * sum(seconds[1:]) / repeats, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--full-meta", type=Path, default=Path("benchmarks/features.json"),
                        help="full-data vocab sizes, for the full-size parameter count")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/model_summary.json"))
    args = parser.parse_args()

    torch.manual_seed(0)
    store = FeatureStore(args.features_dir)
    if store.text_emb is None:
        raise SystemExit(f"{args.features_dir} has no item_text_emb.npy; download it with infra.download_asset")
    ds = TwoTowerDataset(store)
    batch = ds.__getitems__(list(range(args.batch_size)))

    model = TwoTowerModel(ModelConfig.from_meta(store.meta)).eval()
    with torch.no_grad():
        u, i = model(batch)
        scores = model.score(u, i)
    result = {
        "config": {k: v for k, v in model.cfg.to_dict().items() if k != "vocab_sizes"},
        "batch_size": args.batch_size,
        "user_vectors": list(u.shape),
        "item_vectors": list(i.shape),
        "scores": list(scores.shape),
        "vector_norms_min_max": [round(float(x), 4) for x in (u.norm(dim=-1).min(), i.norm(dim=-1).max())],
        "untrained_diagonal_minus_offdiagonal_score": round(
            float(scores.diag().mean() - (scores.sum() - scores.diag().sum()) / (len(u) ** 2 - len(u))), 4
        ),
        "cpu_forward_backward_ms": timed_step(model, batch),
        "params_sample_vocab": summarize(model),
    }

    full_vocab = json.loads(args.full_meta.read_text())["vocab_sizes"]
    for name, overrides in {"full": {}, "full_no_user_id": {"use_user_id": False}}.items():
        result[f"params_{name}"] = summarize(TwoTowerModel(ModelConfig(vocab_sizes=full_vocab, **overrides)))

    print(json.dumps(result, indent=2))
    args.out.write_text(json.dumps(result, indent=2) + "\n")
