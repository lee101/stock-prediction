"""Enhanced dataset and feature engineering for Bags.fm neural trading.

Adds:
- Volatility features (rolling std of returns)
- Momentum features (multi-horizon returns)
- RSI-like feature (up moves ratio)
- Price position relative to range
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureNormalizerV2:
    """Feature normalization helper (mean/std)."""

    mean: np.ndarray
    std: np.ndarray
    version: str = "v2"

    def transform(self, features: np.ndarray) -> np.ndarray:
        std = np.where(self.std == 0, 1.0, self.std)
        return (features - self.mean) / std

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "FeatureNormalizerV2":
        return cls(
            mean=np.array(payload["mean"], dtype=np.float32),
            std=np.array(payload["std"], dtype=np.float32),
            version=payload.get("version", "v2"),
        )


def load_ohlc_dataframe(
    path: Path,
    mint: Optional[str] = None,
    dedupe: bool = True,
) -> pd.DataFrame:
    """Load OHLC data from CSV and optionally filter by token mint."""
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Missing timestamp column in {path}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if mint:
        df = df[df["token_mint"] == mint]

    if dedupe:
        df = df.drop_duplicates(subset=["timestamp"], keep="last")

    df = df.sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        raise ValueError("No OHLC rows found for the requested mint")

    return df


def build_window_features_v2(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> np.ndarray:
    """Build enhanced features from OHLC window.

    Features per bar:
    1. Return: close[t] / close[t-1] - 1
    2. Range pct: high / low - 1
    3. Open-close return: close / open - 1
    4. High-close: (high - close) / close (upper wick)
    5. Close-low: (close - low) / close (lower wick)

    Aggregate features (at end):
    6. Volatility: std of returns over window
    7. Momentum 5-bar: return over last 5 bars
    8. Momentum 10-bar: return over last 10 bars
    9. Momentum 20-bar: return over last 20 bars
    10. RSI-like: ratio of positive returns
    11. Price position: where close is in recent high-low range
    12. Trend strength: correlation of closes with time
    """
    context_bars = len(closes)

    # Per-bar features (same as v1)
    returns = np.zeros(context_bars, dtype=np.float32)
    if context_bars > 1:
        returns[1:] = closes[1:] / closes[:-1] - 1.0

    range_pct = highs / np.maximum(lows, 1e-12) - 1.0
    oc_return = closes / np.maximum(opens, 1e-12) - 1.0

    # New per-bar features
    high_close = (highs - closes) / np.maximum(closes, 1e-12)
    close_low = (closes - lows) / np.maximum(closes, 1e-12)

    # Aggregate features
    agg_features = []

    # Volatility (std of returns)
    volatility = np.std(returns) if len(returns) > 1 else 0.0
    agg_features.append(volatility)

    # Multi-horizon momentum
    for lookback in [5, 10, 20]:
        if context_bars >= lookback:
            momentum = closes[-1] / closes[-lookback] - 1.0
        else:
            momentum = 0.0
        agg_features.append(momentum)

    # RSI-like (ratio of positive returns)
    positive_returns = np.sum(returns > 0)
    total_returns = np.sum(returns != 0)
    rsi_like = positive_returns / max(total_returns, 1)
    agg_features.append(rsi_like)

    # Price position in range (0 = at low, 1 = at high)
    range_high = np.max(highs)
    range_low = np.min(lows)
    if range_high > range_low:
        price_position = (closes[-1] - range_low) / (range_high - range_low)
    else:
        price_position = 0.5
    agg_features.append(price_position)

    # Trend strength (simplified: average return)
    trend_strength = np.mean(returns)
    agg_features.append(trend_strength)

    # Combine all features
    per_bar_features = np.concatenate([returns, range_pct, oc_return, high_close, close_low])
    all_features = np.concatenate([per_bar_features, np.array(agg_features, dtype=np.float32)])

    return all_features.astype(np.float32)


def build_features_and_targets_v2(
    df: pd.DataFrame,
    context_bars: int,
    horizon: int,
    cost_bps: float,
    min_return: float,
    size_scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build supervised features and targets with enhanced features.

    Returns:
        features: (N, feature_dim)
        signal_targets: (N,)
        size_targets: (N,)
        timestamps: (N,)
    """
    if context_bars < 2:
        raise ValueError("context_bars must be >= 2")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if size_scale <= 0:
        raise ValueError("size_scale must be > 0")

    opens = df["open"].to_numpy(dtype=np.float32)
    highs = df["high"].to_numpy(dtype=np.float32)
    lows = df["low"].to_numpy(dtype=np.float32)
    closes = df["close"].to_numpy(dtype=np.float32)
    timestamps = df["timestamp"].to_numpy()

    max_index = len(df) - horizon
    features = []
    signal_targets = []
    size_targets = []
    label_times = []

    cost_return = cost_bps / 10000.0

    for idx in range(context_bars, max_index):
        start = idx - context_bars
        window_features = build_window_features_v2(
            opens[start:idx],
            highs[start:idx],
            lows[start:idx],
            closes[start:idx],
        )

        current_close = closes[idx]
        future_close = closes[idx + horizon]
        future_return = future_close / max(current_close, 1e-12) - 1.0
        net_return = future_return - cost_return

        signal = 1.0 if net_return > min_return else 0.0
        size_target = max(net_return, 0.0) / size_scale
        size_target = float(np.clip(size_target, 0.0, 1.0))

        features.append(window_features)
        signal_targets.append(signal)
        size_targets.append(size_target)
        label_times.append(timestamps[idx])

    if not features:
        raise ValueError("Not enough rows to build features/targets")

    return (
        np.stack(features),
        np.array(signal_targets, dtype=np.float32),
        np.array(size_targets, dtype=np.float32),
        np.array(label_times),
    )


def fit_normalizer_v2(features: np.ndarray) -> FeatureNormalizerV2:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    return FeatureNormalizerV2(mean=mean.astype(np.float32), std=std.astype(np.float32))
