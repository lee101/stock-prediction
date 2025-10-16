#!/usr/bin/env python3
"""
Lightweight calibration procedure for the Toto forecaster.

The script fits an affine calibration (scale + bias) that maps the base Toto
prediction to the observed closing price on a historical window.  The
calibration is stored under ``tototraining/artifacts/calibrated_toto.json`` and
can be reused by downstream evaluation scripts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.toto_wrapper import TotoPipeline
from src.models.toto_aggregation import aggregate_quantile_plus_std

DATA_PATH = Path("trainingdata") / "BTCUSD.csv"
ARTIFACT_PATH = Path("tototraining") / "artifacts"
CALIBRATION_FILE = ARTIFACT_PATH / "calibrated_toto.json"

TOTO_MODEL_ID = "Datadog/Toto-Open-Base-1.0"
TOTO_NUM_SAMPLES = 4096
TOTO_SAMPLES_PER_BATCH = 512
TOTO_QUANTILE = 0.15
TOTO_STD_SCALE = 0.15
MIN_CONTEXT = 192
TRAIN_SPLIT = 0.8


def _prepare_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Expected dataset at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if "timestamp" not in df.columns or "close" not in df.columns:
        raise KeyError("Dataset must contain 'timestamp' and 'close' columns.")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _gather_predictions(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    close = df["close"].to_numpy(dtype=np.float64)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = TotoPipeline.from_pretrained(
        model_id=TOTO_MODEL_ID,
        device_map=device,
    )

    preds = []
    actuals = []
    for end in range(MIN_CONTEXT, len(close)):
        context = close[:end].astype(np.float32)
        forecast = pipeline.predict(
            context=context,
            prediction_length=1,
            num_samples=TOTO_NUM_SAMPLES,
            samples_per_batch=TOTO_SAMPLES_PER_BATCH,
        )
        samples = forecast[0].samples if hasattr(forecast[0], "samples") else forecast[0]
        aggregated = aggregate_quantile_plus_std(
            samples,
            quantile=TOTO_QUANTILE,
            std_scale=TOTO_STD_SCALE,
        )
        preds.append(float(np.atleast_1d(aggregated)[0]))
        actuals.append(close[end])

    return np.asarray(preds, dtype=np.float64), np.asarray(actuals, dtype=np.float64)


def _fit_affine(preds: np.ndarray, actuals: np.ndarray) -> Tuple[float, float]:
    X = np.vstack([preds, np.ones_like(preds)]).T
    solution, *_ = np.linalg.lstsq(X, actuals, rcond=None)
    scale, bias = solution
    return float(scale), float(bias)


def _evaluate(preds: np.ndarray, actuals: np.ndarray, scale: float, bias: float) -> Tuple[float, float]:
    calibrated = scale * preds + bias
    mae = np.mean(np.abs(actuals - calibrated))
    base_mae = np.mean(np.abs(actuals - preds))
    return base_mae, mae


def main() -> None:
    df = _prepare_data()
    preds, actuals = _gather_predictions(df)

    split_idx = int(len(preds) * TRAIN_SPLIT)
    train_preds, val_preds = preds[:split_idx], preds[split_idx:]
    train_actuals, val_actuals = actuals[:split_idx], actuals[split_idx:]

    scale, bias = _fit_affine(train_preds, train_actuals)
    train_base_mae, train_calib_mae = _evaluate(train_preds, train_actuals, scale, bias)
    val_base_mae, val_calib_mae = _evaluate(val_preds, val_actuals, scale, bias)

    ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": TOTO_MODEL_ID,
        "num_samples": TOTO_NUM_SAMPLES,
        "samples_per_batch": TOTO_SAMPLES_PER_BATCH,
        "quantile": TOTO_QUANTILE,
        "std_scale": TOTO_STD_SCALE,
        "scale": scale,
        "bias": bias,
        "train_base_mae": train_base_mae,
        "train_calibrated_mae": train_calib_mae,
        "val_base_mae": val_base_mae,
        "val_calibrated_mae": val_calib_mae,
        "min_context": MIN_CONTEXT,
    }
    with CALIBRATION_FILE.open("w") as fp:
        json.dump(payload, fp, indent=2)

    print("=== Toto Calibration Summary ===")
    print(f"Training samples: {len(train_preds)}, Validation samples: {len(val_preds)}")
    print(f"Scale: {scale:.6f}, Bias: {bias:.6f}")
    print(f"Train MAE (base -> calibrated): {train_base_mae:.6f} -> {train_calib_mae:.6f}")
    print(f"Val   MAE (base -> calibrated): {val_base_mae:.6f} -> {val_calib_mae:.6f}")
    print(f"Saved calibration to {CALIBRATION_FILE}")


if __name__ == "__main__":
    main()
