from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple


def load_ohlcv(symbol: str, data_root: Path) -> pd.DataFrame:
    path = data_root / f"{symbol}.csv"
    df = pd.read_csv(path, parse_dates=["date"] if "date" in pd.read_csv(path, nrows=0).columns else [0])
    for col in ["close", "high", "low", "volume"]:
        if col not in df.columns:
            alt = [c for c in df.columns if col in c.lower()]
            if alt:
                df[col] = df[alt[0]]
    df = df.sort_values(df.columns[0]).reset_index(drop=True)
    return df


def compute_features(df: pd.DataFrame) -> np.ndarray:
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    vol = df["volume"].values.astype(np.float64) if "volume" in df.columns else np.ones(len(df))

    n = len(close)
    feats = []

    for period in [1, 4, 12, 24]:
        ret = np.zeros(n)
        ret[period:] = (close[period:] - close[:-period]) / np.maximum(close[:-period], 1e-8)
        feats.append(np.clip(ret, -0.5, 0.5))

    hl_range = (high - low) / np.maximum(close, 1e-8)
    feats.append(np.clip(hl_range, 0, 0.5))

    vol_ma = pd.Series(vol).rolling(24, min_periods=1).mean().values
    vol_std = pd.Series(vol).rolling(24, min_periods=1).std().fillna(1).values
    vol_std = np.maximum(vol_std, 1e-8)
    vol_z = (vol - vol_ma) / vol_std
    feats.append(np.clip(vol_z, -3, 3))

    close_pos = np.zeros(n)
    denom = high - low
    valid = denom > 1e-8
    close_pos[valid] = (close[valid] - low[valid]) / denom[valid]
    feats.append(close_pos)

    volatility = pd.Series(close).pct_change().rolling(12, min_periods=1).std().fillna(0).values
    feats.append(np.clip(volatility * 100, 0, 10))

    hour_sin = np.zeros(n)
    hour_cos = np.zeros(n)
    if "date" in df.columns:
        hours = pd.to_datetime(df["date"]).dt.hour.values
        hour_sin = np.sin(2 * np.pi * hours / 24)
        hour_cos = np.cos(2 * np.pi * hours / 24)
    feats.append(hour_sin)
    feats.append(hour_cos)

    result = np.stack(feats, axis=-1).astype(np.float32)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result


def load_market_data(
    symbols: List[str],
    data_root: Path,
    validation_days: float = 70.0,
) -> Dict:
    all_close, all_high, all_low, all_features = [], [], [], []
    min_len = None

    dfs = {}
    for sym in symbols:
        df = load_ohlcv(sym, data_root)
        dfs[sym] = df
        if min_len is None or len(df) < min_len:
            min_len = len(df)

    val_bars = int(validation_days * 24)
    train_end = min_len - val_bars
    assert train_end > 100, f"Not enough training data: {train_end} bars"

    n_features = None
    for sym in symbols:
        df = dfs[sym].iloc[:min_len].copy()
        feats = compute_features(df)
        if n_features is None:
            n_features = feats.shape[1]
        all_close.append(df["close"].values[:min_len].astype(np.float32))
        all_high.append(df["high"].values[:min_len].astype(np.float32))
        all_low.append(df["low"].values[:min_len].astype(np.float32))
        all_features.append(feats[:min_len])

    n_symbols = len(symbols)
    n_bars = min_len

    close = np.zeros((n_symbols, n_bars), dtype=np.float32)
    high = np.zeros((n_symbols, n_bars), dtype=np.float32)
    low = np.zeros((n_symbols, n_bars), dtype=np.float32)
    features = np.zeros((n_symbols, n_bars, n_features), dtype=np.float32)

    for i in range(n_symbols):
        close[i] = all_close[i]
        high[i] = all_high[i]
        low[i] = all_low[i]
        features[i] = all_features[i]

    return {
        "close": np.ascontiguousarray(close),
        "high": np.ascontiguousarray(high),
        "low": np.ascontiguousarray(low),
        "features": np.ascontiguousarray(features),
        "n_symbols": n_symbols,
        "n_bars": n_bars,
        "n_features": n_features,
        "train_end": train_end,
        "val_start": train_end,
        "symbols": symbols,
    }
