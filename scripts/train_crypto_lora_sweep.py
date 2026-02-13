#!/usr/bin/env python3
"""Train crypto LoRAs with hyperparameter sweep for best MAE on validation."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from loguru import logger
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from preaug import get_augmentation

DEFAULT_OUTPUT_ROOT = Path("chronos2_finetuned")
DEFAULT_RESULTS_DIR = Path("hyperparams/crypto_lora_sweep")


@dataclass
class TrainConfig:
    symbol: str = "SOLUSD"
    context_length: int = 128
    prediction_length: int = 24
    batch_size: int = 32
    learning_rate: float = 5e-5
    num_steps: int = 1000
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    preaug: str = "percent_change"
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

    def consistency_score(self) -> float:
        return self.mae_percent_mean + 0.5 * self.mae_percent_std + 0.3 * (self.mae_percent_max - self.mae_percent_mean)


def load_hourly_frame(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


def split_data(df: pd.DataFrame, val_hours: int, test_hours: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = len(df) - (val_hours + test_hours)
    val_end = len(df) - test_hours
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def compute_consistency_metrics(
    pipeline: Any,
    df: pd.DataFrame,
    context_length: int,
    prediction_length: int,
    start_idx: int,
    end_idx: int,
    preaug_name: str = "baseline",
) -> ConsistencyMetrics:
    if start_idx < context_length:
        start_idx = context_length
    if start_idx >= end_idx - prediction_length:
        return ConsistencyMetrics(
            mae_mean=float("inf"), mae_std=0, mae_max=float("inf"), mae_p90=float("inf"),
            mae_percent_mean=float("inf"), mae_percent_std=0, mae_percent_max=float("inf"), count=0
        )

    target_cols = ["open", "high", "low", "close"]
    close_idx = target_cols.index("close")
    quantiles = getattr(pipeline, "quantiles", None)
    q_index = int(np.argmin([abs(float(q) - 0.5) for q in quantiles])) if quantiles else 0

    window_maes, window_mae_pcts = [], []
    step = max(1, prediction_length // 2)

    for idx in range(start_idx, end_idx - prediction_length, step):
        context = df.iloc[idx - context_length : idx]
        future = df.iloc[idx : idx + prediction_length]
        if len(context) < context_length or len(future) < prediction_length:
            continue

        ctx_data = context.copy()
        aug = get_augmentation(preaug_name) if preaug_name != "baseline" else None
        if aug:
            ctx_data = aug.transform_dataframe(ctx_data)

        inputs = ctx_data[target_cols].to_numpy(dtype=np.float32).T
        try:
            preds = pipeline.predict([inputs], prediction_length=prediction_length, batch_size=1)
        except Exception as e:
            logger.warning("Prediction failed: {}", e)
            continue
        if not preds:
            continue

        pred_tensor = preds[0].detach().cpu().numpy()
        pred_vals = pred_tensor[:, q_index, :].T

        if aug:
            pred_vals = aug.inverse_transform_predictions(pred_vals, context, columns=target_cols)

        actual = future[target_cols].to_numpy(dtype=np.float32)
        actual_close = actual[:, close_idx]
        pred_close = pred_vals[:, close_idx]

        mae = float(np.mean(np.abs(actual_close - pred_close)))
        mae_pct = float(np.mean(np.abs(actual_close - pred_close) / np.clip(np.abs(actual_close), 1e-8, None)) * 100)

        window_maes.append(mae)
        window_mae_pcts.append(mae_pct)

    if not window_maes:
        return ConsistencyMetrics(
            mae_mean=float("inf"), mae_std=0, mae_max=float("inf"), mae_p90=float("inf"),
            mae_percent_mean=float("inf"), mae_percent_std=0, mae_percent_max=float("inf"), count=0
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


def train_and_evaluate(cfg: TrainConfig, data_path: Path, output_root: Path) -> Dict[str, Any]:
    from chronos2_trainer import _load_pipeline, _fit_pipeline, _save_pipeline

    df = load_hourly_frame(data_path)
    logger.info("Loaded {} rows for {}", len(df), cfg.symbol)
    train_df, val_df, test_df = split_data(df, cfg.val_hours, cfg.test_hours)

    augmentation = get_augmentation(cfg.preaug)
    train_aug = augmentation.transform_dataframe(train_df.copy())
    val_aug = augmentation.transform_dataframe(val_df.copy())

    logger.info("Data split: train={} val={} test={}", len(train_df), len(val_df), len(test_df))

    pipeline = _load_pipeline("amazon/chronos-2", "cuda", None)

    target_cols = ["open", "high", "low", "close"]
    train_inputs = [{"target": train_aug[target_cols].to_numpy(dtype=np.float32).T}]
    val_inputs = [{"target": val_aug[target_cols].to_numpy(dtype=np.float32).T}]

    run_name = f"{cfg.symbol}_lora_{cfg.preaug}_ctx{cfg.context_length}_lr{cfg.learning_rate:.0e}_r{cfg.lora_r}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = output_root / run_name

    class FakeConfig:
        def __init__(self, c):
            self.context_length = c.context_length
            self.prediction_length = c.prediction_length
            self.batch_size = c.batch_size
            self.learning_rate = c.learning_rate
            self.num_steps = c.num_steps
            self.finetune_mode = "lora"
            self.lora_r = c.lora_r
            self.lora_alpha = c.lora_alpha
            self.lora_dropout = c.lora_dropout
            self.lora_targets = ("q", "k", "v", "o")
            self.merge_lora = True

    finetuned = _fit_pipeline(pipeline, train_inputs, val_inputs, FakeConfig(cfg), output_dir)
    _save_pipeline(finetuned, output_dir, "finetuned-ckpt")

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    val_start = len(train_df)
    val_end = len(train_df) + len(val_df)
    test_start = val_end
    test_end = len(full_df)

    val_metrics = compute_consistency_metrics(finetuned, full_df, cfg.context_length, cfg.prediction_length, val_start, val_end, cfg.preaug)
    test_metrics = compute_consistency_metrics(finetuned, full_df, cfg.context_length, cfg.prediction_length, test_start, test_end, cfg.preaug)

    return {
        "config": asdict(cfg),
        "run_name": run_name,
        "output_dir": str(output_dir),
        "val": asdict(val_metrics),
        "test": asdict(test_metrics),
        "val_consistency_score": val_metrics.consistency_score(),
        "test_consistency_score": test_metrics.consistency_score(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("trainingdatahourly/crypto"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--prediction-length", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--preaug", type=str, default="percent_change")
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    data_path = args.data_root / f"{args.symbol}.csv"
    if not data_path.exists():
        logger.error("Data file not found: {}", data_path)
        return

    cfg = TrainConfig(
        symbol=args.symbol,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        lora_r=args.lora_r,
        preaug=args.preaug,
    )

    logger.info("Training {} LoRA: ctx={} preaug={} lr={:.0e}", cfg.symbol, cfg.context_length, cfg.preaug, cfg.learning_rate)

    result = train_and_evaluate(cfg, data_path, args.output_root)

    result_path = args.results_dir / f"{result['run_name']}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Results: {}", result_path)
    logger.info("Val MAE%: {:.4f} (+/-{:.4f})", result["val"]["mae_percent_mean"], result["val"]["mae_percent_std"])
    logger.info("Val consistency: {:.4f}", result["val_consistency_score"])
    logger.info("Test MAE%: {:.4f}", result["test"]["mae_percent_mean"])


if __name__ == "__main__":
    main()
