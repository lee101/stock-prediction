"""Baseline: train directly with same LR as finetune (no pretraining).
This isolates whether pretraining helps vs just using a lower LR."""
from __future__ import annotations
import json
import time
from pathlib import Path

from binanceneural.config import TrainingConfig, DatasetConfig
from binanceneural.data import BinanceHourlyDataModule
from binanceneural.trainer import BinanceHourlyTrainer
from experiments.pretrain_finetune.run import evaluate_selector, TARGET_SYMBOLS, RESULTS_DIR


def train_direct(
    target_symbol: str,
    *,
    epochs: int = 20,
    learning_rate: float = 1e-4,
    sequence_length: int = 96,
    dry_train_steps: int = 300,
    run_name: str = "baseline",
) -> Path:
    data_cfg = DatasetConfig(
        symbol=target_symbol,
        data_root=Path("trainingdatahourly/crypto"),
        forecast_cache_root=Path("binanceneural/forecast_cache"),
        sequence_length=sequence_length,
        forecast_horizons=(1,),
        cache_only=True,
    )
    data = BinanceHourlyDataModule(data_cfg)
    train_cfg = TrainingConfig(
        epochs=epochs,
        batch_size=16,
        sequence_length=sequence_length,
        run_name=run_name,
        dry_train_steps=dry_train_steps,
        learning_rate=learning_rate,
    )
    trainer = BinanceHourlyTrainer(train_cfg, data)
    artifacts = trainer.train()
    if artifacts.best_checkpoint is None:
        raise RuntimeError("No checkpoint saved")
    return artifacts.best_checkpoint


def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    results = {"experiment": "baseline_low_lr", "timestamp": ts}
    checkpoints = {}

    for lr_label, lr in [("1e-4", 1e-4), ("5e-5", 5e-5)]:
        print(f"\n{'='*60}")
        print(f"Baseline LR={lr_label}")
        ckpts = {}
        for target in TARGET_SYMBOLS:
            name = f"baseline_lr{lr_label}_{target.lower()}_{ts}"
            print(f"  Training {target}...")
            ckpt = train_direct(target, epochs=20, learning_rate=lr, run_name=name)
            ckpts[target] = ckpt
            results[f"{target}_lr{lr_label}_ckpt"] = str(ckpt)

        print(f"Selector eval for LR={lr_label}...")
        metrics = evaluate_selector(ckpts, tag=f"baseline_lr{lr_label}_{ts}")
        results[f"selector_lr{lr_label}"] = metrics
        ret = metrics.get("total_return", "N/A")
        print(f"  => return={ret}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"baseline_lr_{ts}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
