"""Pretrain on similar-vol crypto pairs, then finetune on target symbols.

Experiment: does pretraining on extra symbols improve selector PnL?
Baseline: ft30 direct training (BTC ep26=29.6x, ETH ep30=190.9x, SOL ep30=211.5x)
Selector baseline: 2478x (70d val)
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from binanceneural.config import TrainingConfig, DatasetConfig
from binanceneural.data import MultiSymbolDataModule, BinanceHourlyDataModule
from binanceneural.trainer import BinanceHourlyTrainer

RESULTS_DIR = Path(__file__).parent / "results"

PRETRAIN_POOLS = {
    "highvol": ["DOGEUSD", "UNIUSD", "LINKUSD", "AVAXUSD", "AAVEUSD"],
    "midvol": ["LTCUSD", "LINKUSD", "AVAXUSD", "DOTUSD"],
    "all_alt": ["DOGEUSD", "UNIUSD", "LINKUSD", "AVAXUSD", "AAVEUSD", "LTCUSD"],
    "cross_target": [],  # special: pretrain on the other 2 targets
}

TARGET_SYMBOLS = ["BTCUSD", "ETHUSD", "SOLUSD"]


def _make_data_config(
    symbol: str,
    sequence_length: int = 96,
    cache_only: bool = True,
    forecast_horizons: tuple = (1,),
) -> DatasetConfig:
    return DatasetConfig(
        symbol=symbol,
        data_root=Path("trainingdatahourly/crypto"),
        forecast_cache_root=Path("binanceneural/forecast_cache"),
        sequence_length=sequence_length,
        forecast_horizons=forecast_horizons,
        cache_only=cache_only,
    )


def pretrain(
    pool: List[str],
    target_symbol: str,
    *,
    epochs: int = 15,
    run_name: str = "pretrain",
    sequence_length: int = 96,
    dry_train_steps: int = 300,
) -> Path:
    data_cfg = _make_data_config(target_symbol, sequence_length=sequence_length)
    data = MultiSymbolDataModule(pool, data_cfg)
    train_cfg = TrainingConfig(
        epochs=epochs,
        batch_size=16,
        sequence_length=sequence_length,
        run_name=run_name,
        dry_train_steps=dry_train_steps,
    )
    trainer = BinanceHourlyTrainer(train_cfg, data)
    artifacts = trainer.train()
    if artifacts.best_checkpoint is None:
        raise RuntimeError("No checkpoint saved during pretraining")
    print(f"Pretrain done: {artifacts.best_checkpoint}")
    return artifacts.best_checkpoint


def finetune(
    target_symbol: str,
    pretrain_checkpoint: Path,
    *,
    epochs: int = 20,
    run_name: str = "finetune",
    sequence_length: int = 96,
    dry_train_steps: int = 300,
    learning_rate: float = 1e-4,
) -> Path:
    data_cfg = _make_data_config(target_symbol, sequence_length=sequence_length)
    data = BinanceHourlyDataModule(data_cfg)
    train_cfg = TrainingConfig(
        epochs=epochs,
        batch_size=16,
        sequence_length=sequence_length,
        run_name=run_name,
        dry_train_steps=dry_train_steps,
        learning_rate=learning_rate,
        preload_checkpoint_path=pretrain_checkpoint,
    )
    trainer = BinanceHourlyTrainer(train_cfg, data)
    artifacts = trainer.train()
    if artifacts.best_checkpoint is None:
        raise RuntimeError("No checkpoint saved during finetuning")
    print(f"Finetune done: {artifacts.best_checkpoint}")
    return artifacts.best_checkpoint


def evaluate_selector(checkpoints: Dict[str, Path], tag: str = "") -> dict:
    """Run selector sim and return metrics."""
    from binanceexp1.run_multiasset_selector import main as _selector_main
    import sys
    ckpt_str = ",".join(f"{s}={p}" for s, p in checkpoints.items())
    old_argv = sys.argv
    sys.argv = [
        "run_multiasset_selector",
        "--symbols", ",".join(checkpoints.keys()),
        "--checkpoints", ckpt_str,
        "--forecast-horizons", "1",
        "--cache-only",
        "--default-intensity", "5.0",
        "--risk-weight", "0.0",
        "--min-edge", "0.0",
        "--max-hold-hours", "6",
        "--allow-reentry-same-bar",
        "--offset-map", "BTCUSD=0.0,ETHUSD=0.0003,SOLUSD=0.0005",
    ]
    import io
    from contextlib import redirect_stdout
    f = io.StringIO()
    try:
        with redirect_stdout(f):
            _selector_main()
    except SystemExit:
        pass
    finally:
        sys.argv = old_argv
    output = f.getvalue()
    metrics = {}
    for line in output.strip().split("\n"):
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            try:
                metrics[key] = float(val.strip())
            except ValueError:
                metrics[key] = val.strip()
    if tag:
        metrics["tag"] = tag
    return metrics


def run_experiment(
    pool_name: str,
    pretrain_epochs: int = 15,
    finetune_epochs: int = 20,
    finetune_lr: float = 1e-4,
) -> dict:
    ts = time.strftime("%Y%m%d_%H%M%S")
    results = {"pool": pool_name, "pretrain_epochs": pretrain_epochs,
               "finetune_epochs": finetune_epochs, "finetune_lr": finetune_lr, "timestamp": ts}

    checkpoints: Dict[str, Path] = {}
    for target in TARGET_SYMBOLS:
        if pool_name == "cross_target":
            pool = [s for s in TARGET_SYMBOLS if s != target]
        else:
            pool = PRETRAIN_POOLS[pool_name]
        results["pool_symbols"] = pool

        pretrain_name = f"pt_{pool_name}_{target.lower()}_{ts}"
        finetune_name = f"ft_{pool_name}_{target.lower()}_{ts}"

        print(f"\n{'='*60}")
        print(f"Pretraining for {target} on {pool}")
        pt_ckpt = pretrain(
            pool + [target], target,
            epochs=pretrain_epochs, run_name=pretrain_name,
        )

        print(f"Finetuning {target}")
        ft_ckpt = finetune(
            target, pt_ckpt,
            epochs=finetune_epochs, run_name=finetune_name,
            learning_rate=finetune_lr,
        )
        checkpoints[target] = ft_ckpt
        results[f"{target}_checkpoint"] = str(ft_ckpt)

    print(f"\n{'='*60}")
    print("Running selector simulation...")
    selector_metrics = evaluate_selector(checkpoints, tag=f"pt_{pool_name}_{ts}")
    results["selector"] = selector_metrics
    print(f"Selector: total_return={selector_metrics.get('total_return', 'N/A')}, "
          f"sortino={selector_metrics.get('sortino', 'N/A')}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"pt_{pool_name}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved: {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="highvol", choices=list(PRETRAIN_POOLS.keys()))
    parser.add_argument("--pretrain-epochs", type=int, default=15)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    args = parser.parse_args()
    run_experiment(args.pool, args.pretrain_epochs, args.finetune_epochs, args.finetune_lr)


if __name__ == "__main__":
    main()
