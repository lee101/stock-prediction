#!/usr/bin/env python3
"""Sweep BTC LoRA hyperparameters with MAE consistency tracking."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

DEFAULT_DATA_PATH = Path("trainingdata/csv/BTC.csv")
DEFAULT_OUTPUT_ROOT = Path("chronos2_finetuned")
DEFAULT_RESULTS_PATH = Path("hyperparams/btc_lora_sweep_results.json")


@dataclass
class SweepConfig:
    context_length: int = 512
    prediction_length: int = 24
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_steps: int = 1000
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    preaug: str = "baseline"
    val_hours: int = 168
    test_hours: int = 168


@dataclass
class ConsistencyMetrics:
    mae_mean: float
    mae_std: float
    mae_max: float
    mae_p90: float
    mae_percent_mean: float
    mae_percent_std: float
    mae_percent_max: float
    count: int


def compute_windowed_mae(
    pipeline: Any,
    df: pd.DataFrame,
    context_length: int,
    prediction_length: int,
    start_idx: int,
    end_idx: int,
) -> ConsistencyMetrics:
    """Compute MAE across sliding windows with consistency stats."""
    from chronos2_trainer import _median_quantile_index

    if start_idx < context_length:
        start_idx = context_length

    target_cols = ["open", "high", "low", "close"]
    close_idx = target_cols.index("close")

    quantiles = getattr(pipeline, "quantiles", None)
    q_index = _median_quantile_index(list(quantiles)) if quantiles else 0

    window_maes: List[float] = []
    window_mae_pcts: List[float] = []

    for idx in range(start_idx, end_idx, prediction_length):
        context = df.iloc[idx - context_length : idx]
        future = df.iloc[idx : idx + prediction_length]
        if len(context) < context_length or len(future) < 1:
            continue

        inputs = context[target_cols].to_numpy(dtype=np.float32).T
        try:
            preds = pipeline.predict([inputs], prediction_length=len(future), batch_size=1)
        except Exception:
            continue
        if not preds:
            continue

        pred_tensor = preds[0].detach().cpu().numpy()
        pred_vals = pred_tensor[:, q_index, :].T

        actual = future[target_cols].to_numpy(dtype=np.float32)
        actual_close = actual[:, close_idx]
        pred_close = pred_vals[:, close_idx]

        mae = float(np.mean(np.abs(actual_close - pred_close)))
        mae_pct = float(np.mean(np.abs(actual_close - pred_close) / np.abs(actual_close + 1e-8)) * 100)

        window_maes.append(mae)
        window_mae_pcts.append(mae_pct)

    if not window_maes:
        return ConsistencyMetrics(
            mae_mean=float("inf"), mae_std=0, mae_max=float("inf"), mae_p90=float("inf"),
            mae_percent_mean=float("inf"), mae_percent_std=0, mae_percent_max=float("inf"),
            count=0
        )

    return ConsistencyMetrics(
        mae_mean=float(np.mean(window_maes)),
        mae_std=float(np.std(window_maes)),
        mae_max=float(np.max(window_maes)),
        mae_p90=float(np.percentile(window_maes, 90)),
        mae_percent_mean=float(np.mean(window_mae_pcts)),
        mae_percent_std=float(np.std(window_mae_pcts)),
        mae_percent_max=float(np.max(window_mae_pcts)),
        count=len(window_maes),
    )


def run_single_config(cfg: SweepConfig, data_path: Path, output_root: Path) -> dict:
    """Train and evaluate a single configuration."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from chronos2_trainer import TrainerConfig, run_finetune

    run_name = f"BTC_lora_ctx{cfg.context_length}_lr{cfg.learning_rate:.0e}_r{cfg.lora_r}_{cfg.preaug}_{time.strftime('%Y%m%d_%H%M%S')}"

    trainer_cfg = TrainerConfig(
        symbol="BTC",
        data_root=data_path.parent,
        output_root=output_root,
        context_length=cfg.context_length,
        prediction_length=cfg.prediction_length,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        num_steps=cfg.num_steps,
        val_hours=cfg.val_hours,
        test_hours=cfg.test_hours,
        finetune_mode="lora",
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        save_name=run_name,
    )

    report = run_finetune(trainer_cfg)

    result = {
        "config": asdict(cfg),
        "run_name": run_name,
        "val_mae_percent": report.val_metrics.mae_percent,
        "val_mae": report.val_metrics.mae,
        "test_mae_percent": report.test_metrics.mae_percent,
        "test_mae": report.test_metrics.mae,
        "output_dir": report.output_dir,
    }

    return result


SWEEP_GRID = {
    "context_length": [256, 512, 1024],
    "learning_rate": [5e-5, 1e-4, 2e-4],
    "lora_r": [8, 16, 32],
    "num_steps": [500, 1000, 2000],
}


def expand_grid(base: SweepConfig) -> List[SweepConfig]:
    keys = sorted(SWEEP_GRID.keys())
    configs = []
    for values in product(*[SWEEP_GRID[k] for k in keys]):
        params = dict(zip(keys, values))
        cfg_dict = asdict(base)
        cfg_dict.update(params)
        configs.append(SweepConfig(**cfg_dict))
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--quick", action="store_true", help="Run reduced grid")
    parser.add_argument("--single", action="store_true", help="Run single config only")
    args = parser.parse_args()

    args.results_path.parent.mkdir(parents=True, exist_ok=True)

    base_cfg = SweepConfig()
    if args.single:
        configs = [base_cfg]
    elif args.quick:
        configs = [
            SweepConfig(context_length=256, learning_rate=1e-4, lora_r=16, num_steps=500),
            SweepConfig(context_length=512, learning_rate=1e-4, lora_r=16, num_steps=1000),
            SweepConfig(context_length=1024, learning_rate=5e-5, lora_r=32, num_steps=1000),
        ]
    else:
        configs = expand_grid(base_cfg)

    logger.info("Running {} configurations", len(configs))

    results = []
    for idx, cfg in enumerate(configs, 1):
        logger.info("[{}/{}] ctx={} lr={:.0e} r={}", idx, len(configs), cfg.context_length, cfg.learning_rate, cfg.lora_r)
        try:
            result = run_single_config(cfg, args.data_path, args.output_root)
            results.append(result)
            logger.info("  val_mae%={:.4f} test_mae%={:.4f}", result["val_mae_percent"], result["test_mae_percent"])
        except Exception as e:
            logger.error("  Failed: {}", e)
            results.append({"config": asdict(cfg), "error": str(e)})

    with open(args.results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to {}", args.results_path)

    valid = [r for r in results if "error" not in r]
    if valid:
        best = min(valid, key=lambda r: r["val_mae_percent"])
        logger.info("Best: val_mae%={:.4f} ctx={} lr={}", best["val_mae_percent"], best["config"]["context_length"], best["config"]["learning_rate"])


if __name__ == "__main__":
    main()
