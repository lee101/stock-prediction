"""Dataset and feature engineering for Bags v2.

Improvements over v1:
- Better feature organization (per-bar vs aggregate)
- More robust normalization
- Time-based features
- Walk-forward data splitting
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureNormalizerV2:
    """Robust feature normalization with clipping."""

    mean: np.ndarray
    std: np.ndarray
    clip_std: float = 5.0  # Clip outliers at N std deviations

    def transform(self, features: np.ndarray) -> np.ndarray:
        std = np.where(self.std < 1e-8, 1.0, self.std)
        normalized = (features - self.mean) / std
        # Clip extreme values
        return np.clip(normalized, -self.clip_std, self.clip_std)

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "clip_std": self.clip_std,
            "version": "v2",
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "FeatureNormalizerV2":
        return cls(
            mean=np.array(payload["mean"], dtype=np.float32),
            std=np.array(payload["std"], dtype=np.float32),
            clip_std=payload.get("clip_std", 5.0),
        )


def load_ohlc_dataframe(
    path: Path,
    mint: Optional[str] = None,
    dedupe: bool = True,
) -> pd.DataFrame:
    """Load OHLC data from CSV."""
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
        raise ValueError("No OHLC rows found")

    return df


def build_window_features_v2(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> np.ndarray:
    """Build features for a window of OHLC bars.

    Returns features organized as:
    - Per-bar features (5 per bar): returns, range_pct, oc_return, upper_wick, lower_wick
    - Aggregate features (7): volatility, momentum_5, momentum_10, momentum_20, rsi_like, price_position, trend

    Total: context_bars * 5 + 7
    """
    context_bars = len(closes)

    # Per-bar features (5 features each)
    # 1. Returns
    returns = np.zeros(context_bars, dtype=np.float32)
    if context_bars > 1:
        returns[1:] = closes[1:] / np.maximum(closes[:-1], 1e-12) - 1.0

    # 2. Range percentage (high/low - 1)
    range_pct = highs / np.maximum(lows, 1e-12) - 1.0

    # 3. Open-close return
    oc_return = closes / np.maximum(opens, 1e-12) - 1.0

    # 4. Upper wick (high - close) / close
    upper_wick = (highs - closes) / np.maximum(closes, 1e-12)

    # 5. Lower wick (close - low) / close
    lower_wick = (closes - lows) / np.maximum(closes, 1e-12)

    # Stack per-bar features: shape (context_bars, 5)
    per_bar = np.stack([returns, range_pct, oc_return, upper_wick, lower_wick], axis=1)
    per_bar_flat = per_bar.flatten()  # (context_bars * 5,)

    # Aggregate features (7)
    agg_features = []

    # 1. Volatility (std of returns)
    volatility = np.std(returns) if len(returns) > 1 else 0.0
    agg_features.append(volatility)

    # 2-4. Multi-horizon momentum
    for lookback in [5, 10, 20]:
        if context_bars >= lookback:
            momentum = closes[-1] / np.maximum(closes[-lookback], 1e-12) - 1.0
        else:
            momentum = 0.0
        agg_features.append(momentum)

    # 5. RSI-like (ratio of positive returns)
    positive_returns = np.sum(returns > 0)
    total_returns = np.sum(returns != 0)
    rsi_like = positive_returns / max(total_returns, 1)
    agg_features.append(rsi_like)

    # 6. Price position in range (0 = at low, 1 = at high)
    range_high = np.max(highs)
    range_low = np.min(lows)
    if range_high > range_low:
        price_position = (closes[-1] - range_low) / (range_high - range_low)
    else:
        price_position = 0.5
    agg_features.append(price_position)

    # 7. Trend strength (average return)
    trend_strength = np.mean(returns)
    agg_features.append(trend_strength)

    # Combine: per_bar_flat + agg_features
    all_features = np.concatenate([per_bar_flat, np.array(agg_features, dtype=np.float32)])

    return all_features.astype(np.float32)


def build_features_and_targets(
    df: pd.DataFrame,
    context_bars: int = 32,
    horizon: int = 3,
    cost_bps: float = 130.0,
    min_return: float = 0.002,
    size_scale: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build supervised dataset.

    Args:
        df: OHLC dataframe
        context_bars: Number of bars for context window
        horizon: Bars ahead for target
        cost_bps: Trading cost in basis points
        min_return: Minimum return to consider positive
        size_scale: Scale for position size target

    Returns:
        features: (N, feature_dim)
        signal_targets: (N,) binary
        size_targets: (N,) 0-1 scaled
        timestamps: (N,)
    """
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

        # Binary signal: 1 if profitable after costs
        signal = 1.0 if net_return > min_return else 0.0

        # Size target: scaled return (0-1)
        size_target = max(net_return, 0.0) / size_scale
        size_target = float(np.clip(size_target, 0.0, 1.0))

        features.append(window_features)
        signal_targets.append(signal)
        size_targets.append(size_target)
        label_times.append(timestamps[idx])

    if not features:
        raise ValueError("Not enough data to build features")

    return (
        np.stack(features),
        np.array(signal_targets, dtype=np.float32),
        np.array(size_targets, dtype=np.float32),
        np.array(label_times),
    )


def fit_normalizer(features: np.ndarray, clip_std: float = 5.0) -> FeatureNormalizerV2:
    """Fit normalizer on training data."""
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    return FeatureNormalizerV2(
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        clip_std=clip_std,
    )


def walk_forward_split(
    n_samples: int,
    n_splits: int = 5,
    train_pct: float = 0.7,
    gap_bars: int = 10,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate walk-forward train/val splits.

    More realistic than random split - always train on past, validate on future.

    Args:
        n_samples: Total number of samples
        n_splits: Number of folds
        train_pct: Percentage of each fold for training
        gap_bars: Gap between train and val to prevent leakage

    Returns:
        List of (train_indices, val_indices) tuples
    """
    splits = []
    fold_size = n_samples // n_splits

    for i in range(n_splits):
        # Each fold starts further into the data
        start = i * (fold_size // 2)
        end = min(start + fold_size, n_samples)

        train_end = start + int((end - start) * train_pct)
        val_start = train_end + gap_bars

        if val_start >= end:
            continue

        train_idx = np.arange(start, train_end)
        val_idx = np.arange(val_start, end)

        splits.append((train_idx, val_idx))

    return splits


def expanding_window_split(
    n_samples: int,
    initial_train_pct: float = 0.5,
    val_pct: float = 0.1,
    step_pct: float = 0.1,
    gap_bars: int = 10,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate expanding window splits.

    Training window grows over time, more realistic for time series.

    Args:
        n_samples: Total samples
        initial_train_pct: Starting training percentage
        val_pct: Validation window percentage
        step_pct: How much to expand each iteration
        gap_bars: Gap between train and val

    Returns:
        List of (train_indices, val_indices) tuples
    """
    splits = []
    train_end_pct = initial_train_pct

    while train_end_pct + val_pct < 1.0:
        train_end = int(n_samples * train_end_pct)
        val_start = train_end + gap_bars
        val_end = int(n_samples * (train_end_pct + val_pct))

        if val_start >= val_end or val_end > n_samples:
            break

        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_end)

        splits.append((train_idx, val_idx))
        train_end_pct += step_pct

    return splits
