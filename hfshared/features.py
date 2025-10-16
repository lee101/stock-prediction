#!/usr/bin/env python3
"""Shared feature engineering and normalization helpers.

Centralized to keep hftraining and hfinference in sync.
"""
from __future__ import annotations

from typing import List, Optional
import numpy as np
import pandas as pd


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    return out


def training_feature_columns_list() -> List[str]:
    return [
        'open', 'high', 'low', 'close', 'volume',
        'ma_5', 'ma_10', 'ma_20', 'ma_50',
        'ema_5', 'ema_10', 'ema_20', 'ema_50',
        'rsi', 'macd', 'macd_signal', 'macd_histogram',
        'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
        'price_change', 'price_change_2', 'price_change_5',
        'high_low_ratio', 'close_open_ratio',
        'volume_ratio', 'volatility', 'volatility_ratio',
        'resistance_distance', 'support_distance',
    ]


def compute_training_style_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate hftraining.data_utils.StockDataProcessor feature engineering.

    Returns a DataFrame with OHLCV plus indicators in a canonical order
    (subset filtered to available columns).
    """
    df = standardize_column_names(df)
    for base in ['open', 'high', 'low', 'close']:
        if base not in df.columns:
            df[base] = np.nan
    if 'volume' not in df.columns:
        df['volume'] = 0.0

    # Moving averages / EMAs
    for window in [5, 10, 20, 50]:
        df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
        df[f'ema_{window}'] = df['close'].ewm(span=window).mean()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']

    # Bollinger Bands
    rolling_mean = df['close'].rolling(window=20).mean()
    rolling_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = rolling_mean + (rolling_std * 2)
    df['bb_lower'] = rolling_mean - (rolling_std * 2)
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = (df['close'] - df['bb_lower']) / df['bb_width']

    # Price-based features
    df['price_change'] = df['close'].pct_change()
    df['price_change_2'] = df['close'].pct_change(periods=2)
    df['price_change_5'] = df['close'].pct_change(periods=5)

    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_ratio'] = df['close'] / df['open']

    # Volume features
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    # Volatility + supports
    df['volatility'] = df['close'].rolling(window=20).std()
    df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(window=60).mean()
    df['resistance'] = df['high'].rolling(window=20).max()
    df['support'] = df['low'].rolling(window=20).min()
    df['resistance_distance'] = (df['resistance'] - df['close']) / df['close']
    df['support_distance'] = (df['close'] - df['support']) / df['close']

    cols = training_feature_columns_list()
    sel = [c for c in cols if c in df.columns]
    return df[sel].ffill().bfill().fillna(0.0)


def compute_compact_features(data: pd.DataFrame, feature_mode: str = 'auto', use_pct_change: bool = False) -> np.ndarray:
    """Compact OHLC/OHLCV features with optional percent change transform."""
    df = standardize_column_names(data)
    cols = list(df.columns)
    col_map = {c.lower(): c for c in cols}

    if feature_mode == 'ohlc':
        need = ['open', 'high', 'low', 'close']
    elif feature_mode == 'ohlcv':
        need = ['open', 'high', 'low', 'close', 'volume']
    else:
        need = ['open', 'high', 'low', 'close'] + (['volume'] if 'volume' in col_map else [])

    chosen = [col_map[k] for k in need if k in col_map]
    if len(chosen) < 4:
        base = [c for c in cols[:5]] if len(cols) >= 4 else cols
        chosen = base[:4]

    out = df[chosen].copy()
    if feature_mode == 'ohlcv' and len(chosen) == 4:
        out['__volume__'] = 0.0

    out = out.ffill().bfill().fillna(0.0)
    if use_pct_change:
        out = out.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out.values.astype(np.float32)


def zscore_per_window(features: np.ndarray) -> np.ndarray:
    mu = features.mean(axis=0)
    sigma = features.std(axis=0) + 1e-8
    return (features - mu) / sigma


def normalize_with_scaler(
    features: np.ndarray,
    scaler,
    feature_names: List[str],
    df_for_recompute: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """Normalize using training scaler; optionally recompute features to match ordering."""
    feats = np.asarray(features, dtype=np.float32)
    if df_for_recompute is not None and feature_names:
        feats_df = compute_training_style_features(df_for_recompute)
        for col in feature_names:
            if col not in feats_df.columns:
                feats_df[col] = 0.0
        feats_df = feats_df[feature_names]
        feats = feats_df.values.astype(np.float32)
    return scaler.transform(feats)


def denormalize_with_scaler(
    value: float,
    scaler,
    feature_names: List[str],
    column_name: str = 'close',
    default_index: int = 3,
) -> float:
    try:
        if feature_names and column_name in feature_names:
            idx = feature_names.index(column_name)
        else:
            idx = default_index
        mu = float(scaler.mean_[idx])
        std = float(getattr(scaler, 'scale_', None)[idx]) if hasattr(scaler, 'scale_') else float(np.sqrt(scaler.var_[idx]))
        return float(value) * std + mu
    except Exception:
        return float(value)

