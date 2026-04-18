#!/usr/bin/env python3
"""Bench the XGB pipeline end-to-end: feature-build + train + predict, CPU vs GPU.

Usage:
    .venv/bin/python scripts/bench_xgb_gpu.py --n-symbols 100 --n-estimators 400 \
        --max-depth 5 --learning-rate 0.03

Reports per-stage wall times. The point is not micro-perf; it's to make
it obvious which stage dominates so we know whether it's worth moving
features to GPU.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xgbnew.features import DAILY_FEATURE_COLS, build_features_for_symbol
from xgbnew.model import XGBStockModel


def load_symbols(data_root: Path, n: int) -> list[str]:
    syms = sorted(p.stem for p in (data_root / "train").glob("*.csv"))
    return syms[:n]


def read_ohlcv(data_root: Path, symbol: str) -> pd.DataFrame | None:
    path = data_root / "train" / f"{symbol}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    ts_col = "timestamp" if "timestamp" in df.columns else "date"
    df["timestamp"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"])
    if len(df) < 60:
        return None
    return df


def build_features(data_root: Path, symbols: list[str]) -> pd.DataFrame:
    parts = []
    for sym in symbols:
        df = read_ohlcv(data_root, sym)
        if df is None:
            continue
        feat = build_features_for_symbol(df, symbol=sym)
        feat = feat.dropna(subset=DAILY_FEATURE_COLS[:5])
        parts.append(feat)
    if not parts:
        raise RuntimeError("no features built — check --data-root")
    return pd.concat(parts, ignore_index=True)


def time_train(feat_df: pd.DataFrame, *, device: str | None, xgb_params: dict) -> tuple[XGBStockModel, float]:
    t0 = time.time()
    model = XGBStockModel(device=device, **xgb_params)
    model.fit(feat_df, DAILY_FEATURE_COLS, verbose=False)
    t1 = time.time()
    return model, t1 - t0


def time_predict(model: XGBStockModel, feat_df: pd.DataFrame) -> float:
    t0 = time.time()
    _ = model.predict_scores(feat_df)
    t1 = time.time()
    return t1 - t0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(REPO / "trainingdata"))
    p.add_argument("--n-symbols", type=int, default=100)
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=0.03)
    args = p.parse_args()

    data_root = Path(args.data_root)
    syms = load_symbols(data_root, args.n_symbols)
    print(f"[bench] using {len(syms)} symbols from {data_root}")

    # Feature build (CPU — baseline)
    t0 = time.time()
    feat_df = build_features(data_root, syms)
    t1 = time.time()
    print(f"[bench] feature build (CPU/pandas): {t1 - t0:.2f}s  "
          f"({len(feat_df):,} rows, {(t1 - t0) / len(syms) * 1000:.1f} ms/symbol)")

    xgb_params = dict(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        random_state=42,
    )

    # Train CPU
    model_cpu, cpu_train_s = time_train(feat_df, device=None, xgb_params=xgb_params)
    print(f"[bench] XGB train (CPU):            {cpu_train_s:.2f}s")

    # Train GPU
    model_gpu, gpu_train_s = time_train(feat_df, device="cuda", xgb_params=xgb_params)
    print(f"[bench] XGB train (GPU):            {gpu_train_s:.2f}s  "
          f"(speedup {cpu_train_s / max(gpu_train_s, 1e-6):.2f}×)")

    # Predict CPU and GPU
    cpu_pred_s = time_predict(model_cpu, feat_df)
    gpu_pred_s = time_predict(model_gpu, feat_df)
    print(f"[bench] XGB predict (CPU booster):  {cpu_pred_s:.3f}s")
    print(f"[bench] XGB predict (GPU booster):  {gpu_pred_s:.3f}s")

    # Total wall-clock summary
    cpu_total = (t1 - t0) + cpu_train_s + cpu_pred_s
    gpu_total = (t1 - t0) + gpu_train_s + gpu_pred_s
    print(
        f"\n[bench] end-to-end wall (CPU features + CPU train): {cpu_total:.2f}s"
        f"\n[bench] end-to-end wall (CPU features + GPU train): {gpu_total:.2f}s"
        f"\n[bench] stage shares @GPU-train: features={100 * (t1 - t0) / gpu_total:.0f}%"
        f"  train={100 * gpu_train_s / gpu_total:.0f}%"
        f"  predict={100 * gpu_pred_s / gpu_total:.0f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
