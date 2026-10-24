"""Train the two-tower model with in-batch negatives; select the checkpoint on validation MAP@12.

Usage:
    python -m src.train --features-dir data/features/sample_0.05 --epochs 3          # local smoke test
    python -m src.train --features-dir data/features/full --run-name full-baseline   # Colab GPU
    python -m src.train ... --no-logq | --no-history | --no-text | --no-user-id      # ablations

Logs to MLflow (experiment phase5-training), writes models/<run-name>/best.pt and
benchmarks/train_<run-name>.json.
"""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import numpy as np
import torch

from src.dataset import FeatureStore, TwoTowerDataset, make_loader
from src.evaluate import WeekEvaluator
from src.loss import in_batch_softmax_loss
from src.model import ModelConfig, TwoTowerModel, parameter_counts
from src.split import TEST, VAL
from src.tracking import git_sha, init_mlflow, peak_rss_gb


@dataclass
class TrainConfig:
    features_dir: str
    run_name: str = "two-tower"
    epochs: int = 10
    batch_size: int = 2048
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_frac: float = 0.03
    grad_clip: float = 1.0
    patience: int = 2  # epochs without val MAP@12 improvement before stopping
    num_workers: int = 2
    max_steps_per_epoch: int | None = None  # for quick tests
    val_max_users: int | None = None  # subsample val users for faster per-epoch evaluation
    train_weeks: int | None = None  # only purchases in the last N weeks of the training window are targets
    log_every: int = 100
    seed: int = 0
    use_logq: bool = True
    mask_accidental_hits: bool = True
    use_history: bool = True
    use_text: bool = True
    use_user_id: bool = True
    temperature: float = 0.05
    learn_temperature: bool = False


def move(batch: dict, device: torch.device) -> dict:
    return {k: move(v, device) if isinstance(v, dict) else v.to(device, non_blocking=True) for k, v in batch.items()}


def make_optimizer(model: TwoTowerModel, cfg: TrainConfig) -> torch.optim.Optimizer:
    """No weight decay on embeddings, norms and biases: decay would shrink rarely-updated ID vectors."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        is_embedding = "emb" in name and "proj" not in name
        (no_decay if is_embedding or param.ndim < 2 else decay).append(param)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr,
    )


def lr_lambda(total_steps: int, warmup_steps: int):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return fn


def log_metrics(prefix: str, metrics: dict, step: int) -> None:
    mlflow.log_metrics({f"{prefix}_{k.replace('@', '_at_')}": float(v) for k, v in metrics.items()}, step=step)


def train(cfg: TrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    store = FeatureStore(cfg.features_dir, use_text=cfg.use_text)
    train_end_day = store.meta["train_end_day"]
    min_day = None if cfg.train_weeks is None else train_end_day - 7 * cfg.train_weeks + 1
    dataset = TwoTowerDataset(store, min_day=min_day)
    loader = make_loader(dataset, cfg.batch_size, num_workers=cfg.num_workers, seed=cfg.seed)
    val_eval = WeekEvaluator(store, VAL, max_users=cfg.val_max_users, seed=cfg.seed)

    model_cfg = ModelConfig.from_meta(
        store.meta, use_history=cfg.use_history, use_text=cfg.use_text, use_user_id=cfg.use_user_id,
        temperature=cfg.temperature, learn_temperature=cfg.learn_temperature,
    )
    model = TwoTowerModel(model_cfg).to(device)
    optimizer = make_optimizer(model, cfg)
    steps_per_epoch = min(len(loader), cfg.max_steps_per_epoch or len(loader))
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda(total_steps, max(1, int(cfg.warmup_frac * total_steps)))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = Path("models") / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    best = {"map@12": -1.0, "epoch": -1}
    history, step, stale = [], 0, 0

    mlflow.log_params({**{k: v for k, v in asdict(cfg).items() if k != "features_dir"},
                       "git_sha": git_sha(), "device": device.type, "train_examples": len(dataset),
                       "torch_version": torch.__version__,
                       "gpu": torch.cuda.get_device_name() if device.type == "cuda" else "none",
                       "params_total": parameter_counts(model)["total"],
                       "features_sample_frac": store.meta.get("sample_frac") or 1.0})

    for epoch in range(cfg.epochs):
        model.train()
        epoch_start, seen, running = time.perf_counter(), 0, []
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= steps_per_epoch:
                break
            batch = move(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                user_vecs, item_vecs = model(batch)
            loss, stats = in_batch_softmax_loss(
                user_vecs, item_vecs, batch["item_row"], batch["item_log_q"], model.temperature,
                use_logq=cfg.use_logq, mask_accidental_hits=cfg.mask_accidental_hits,
            )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            step += 1
            seen += len(batch["item_row"])
            running.append(loss.item())
            if step % cfg.log_every == 0:
                mlflow.log_metrics({"train_loss": float(np.mean(running[-cfg.log_every:])),
                                    "in_batch_accuracy": stats["in_batch_accuracy"],
                                    "accidental_hits_per_row": stats["accidental_hits_per_row"],
                                    "grad_norm": float(grad_norm), "lr": scheduler.get_last_lr()[0],
                                    "temperature": float(model.temperature)}, step=step)
            if step % (cfg.log_every * 10) == 0:
                rate = seen / (time.perf_counter() - epoch_start)
                eta = (steps_per_epoch - batch_idx - 1) * cfg.batch_size / rate
                print(f"  epoch {epoch} step {batch_idx + 1}/{steps_per_epoch} | loss "
                      f"{np.mean(running[-cfg.log_every:]):.4f} | {rate:,.0f} ex/s | epoch ETA {eta / 60:.1f} min", flush=True)

        train_seconds = time.perf_counter() - epoch_start
        eval_start = time.perf_counter()
        val = val_eval.evaluate(model, device)
        record = {"epoch": epoch, "train_loss": float(np.mean(running)), "train_seconds": round(train_seconds, 1),
                  "examples_per_second": int(seen / train_seconds), "eval_seconds": round(time.perf_counter() - eval_start, 1),
                  **{f"val_{k}": v for k, v in val.items()}}
        history.append(record)
        log_metrics("val", val, step)
        mlflow.log_metrics({"epoch_train_loss": record["train_loss"], "epoch_seconds": train_seconds,
                            "examples_per_second": record["examples_per_second"]}, step=step)
        print(f"epoch {epoch}: loss {record['train_loss']:.4f} | val MAP@12 {val['map@12']:.4f} "
              f"Recall@100 {val['recall@100']:.4f} | {record['examples_per_second']:,} ex/s", flush=True)

        if val["map@12"] > best["map@12"]:
            best = {"map@12": val["map@12"], "epoch": epoch}
            torch.save({"model": model.state_dict(), "model_config": model_cfg.to_dict(), "train_config": asdict(cfg),
                        "epoch": epoch, "val": val}, out_dir / "best.pt")
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                print(f"early stopping: no val MAP@12 improvement for {cfg.patience} epochs")
                break

    # Final numbers: the best checkpoint on the full val week and, once, on the test week.
    checkpoint = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    final = {"val": WeekEvaluator(store, VAL).evaluate(model, device),
             "test": WeekEvaluator(store, TEST).evaluate(model, device)}
    log_metrics("best_val", final["val"], step)
    log_metrics("test", final["test"], step)
    mlflow.log_metric("peak_rss_gb", peak_rss_gb())

    result = {"run_name": cfg.run_name, "best_epoch": best["epoch"], "config": asdict(cfg),
              "params": parameter_counts(model), "history": history, **final}
    return result


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--run-name", default="two-tower")
    for name, default in [("epochs", 10), ("batch-size", 2048), ("num-workers", 2), ("patience", 2),
                          ("max-steps-per-epoch", None), ("val-max-users", None), ("train-weeks", None), ("log-every", 100), ("seed", 0)]:
        parser.add_argument(f"--{name}", type=int, default=default)
    for name, default in [("lr", 1e-3), ("weight-decay", 0.01), ("temperature", 0.05)]:
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.add_argument("--learn-temperature", action="store_true")
    for switch in ["logq", "history", "text", "user-id", "mask-accidental-hits"]:
        parser.add_argument(f"--no-{switch}", action="store_true")
    args = vars(parser.parse_args())
    flags = {"use_logq": not args.pop("no_logq"), "use_history": not args.pop("no_history"),
             "use_text": not args.pop("no_text"), "use_user_id": not args.pop("no_user_id"),
             "mask_accidental_hits": not args.pop("no_mask_accidental_hits")}
    return TrainConfig(**args, **flags)


if __name__ == "__main__":
    cfg = parse_args()
    init_mlflow("phase5-training")
    with mlflow.start_run(run_name=cfg.run_name):
        result = train(cfg)
        mlflow.log_dict(result, f"train_{cfg.run_name}.json")  # so Colab results reach the repo via MLflow
    print(json.dumps({k: result[k] for k in ["run_name", "best_epoch", "val", "test"]}, indent=2))
    Path("benchmarks").mkdir(exist_ok=True)
    Path(f"benchmarks/train_{cfg.run_name}.json").write_text(json.dumps(result, indent=2) + "\n")
