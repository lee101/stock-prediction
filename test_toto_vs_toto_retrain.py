#!/usr/bin/env python3
"""
Compare base Toto predictions against the calibrated Toto variant.

The calibration parameters are produced by ``tototraining/train_calibrated_toto.py``.
This script reports MAE on absolute prices as well as the corresponding returns.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from src.models.toto_wrapper import TotoPipeline
from src.models.toto_aggregation import aggregate_quantile_plus_std

DATA_PATH = Path("trainingdata") / "BTCUSD.csv"
CALIBRATION_FILE = Path("tototraining") / "artifacts" / "calibrated_toto.json"

EVAL_POINTS = 64
TOTO_MODEL_ID = "Datadog/Toto-Open-Base-1.0"
NUM_SAMPLES = 4096
SAMPLES_PER_BATCH = 512
QUANTILE = 0.15
STD_SCALE = 0.15
MIN_CONTEXT = 192


def _load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Expected dataset at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if "timestamp" not in df.columns or "close" not in df.columns:
        raise KeyError("Dataset must include 'timestamp' and 'close' columns.")
    return df.sort_values("timestamp").reset_index(drop=True)


def _load_calibration() -> Tuple[float, float]:
    if not CALIBRATION_FILE.exists():
        raise FileNotFoundError(
            f"Calibration file not found at {CALIBRATION_FILE}. "
            "Run tototraining/train_calibrated_toto.py first."
        )
    with CALIBRATION_FILE.open("r") as fp:
        payload = json.load(fp)
    return float(payload["scale"]), float(payload["bias"])


def _collect_predictions(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    close = df["close"].to_numpy(dtype=np.float64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = TotoPipeline.from_pretrained(
        model_id=TOTO_MODEL_ID,
        device_map=device,
    )

    preds = []
    actuals = []
    start = max(MIN_CONTEXT, len(close) - EVAL_POINTS)
    for end in range(start, len(close)):
        context = close[:end].astype(np.float32)
        forecast = pipeline.predict(
            context=context,
            prediction_length=1,
            num_samples=NUM_SAMPLES,
            samples_per_batch=SAMPLES_PER_BATCH,
        )
        samples = forecast[0].samples if hasattr(forecast[0], "samples") else forecast[0]
        aggregated = aggregate_quantile_plus_std(
            samples,
            quantile=QUANTILE,
            std_scale=STD_SCALE,
        )
        preds.append(float(np.atleast_1d(aggregated)[0]))
        actuals.append(close[end])

    return np.asarray(preds, dtype=np.float64), np.asarray(actuals, dtype=np.float64), close[start - 1]


def _compute_return_errors(preds: np.ndarray, actuals: np.ndarray, prev_price: float) -> np.ndarray:
    prev = prev_price
    errors = []
    for pred, actual in zip(preds, actuals):
        pred_return = (pred - prev) / prev
        actual_return = (actual - prev) / prev
        errors.append(pred_return - actual_return)
        prev = actual
    return np.asarray(errors, dtype=np.float64)


def main() -> None:
    df = _load_dataset()
    scale, bias = _load_calibration()
    preds, actuals, prev_price = _collect_predictions(df)

    calibrated = scale * preds + bias

    base_mae = np.mean(np.abs(actuals - preds))
    calib_mae = np.mean(np.abs(actuals - calibrated))

    base_return_errors = _compute_return_errors(preds, actuals, prev_price)
    calib_return_errors = _compute_return_errors(calibrated, actuals, prev_price)
    base_return_mae = np.mean(np.abs(base_return_errors))
    calib_return_mae = np.mean(np.abs(calib_return_errors))

    print("=== Toto vs Calibrated Toto (Horizon=1) ===")
    print(f"Evaluation points: {len(preds)}")
    print(f"Base Toto price MAE:        {base_mae:.6f}")
    print(f"Calibrated Toto price MAE:  {calib_mae:.6f}")
    print(f"Base Toto return MAE:       {base_return_mae:.6f}")
    print(f"Calibrated Toto return MAE: {calib_return_mae:.6f}")
    print(f"Calibration parameters -> scale: {scale:.6f}, bias: {bias:.6f}")


if __name__ == "__main__":
    main()
