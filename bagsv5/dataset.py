"""Dataset utilities for BagsV5 multi-token training."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class FeatureNormalizer:
    """Online feature normalizer with running statistics."""

    def __init__(self):
        self.mean = None
        self.std = None
        self.n_samples = 0

    def fit(self, X: np.ndarray):
        """Fit normalizer to data."""
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8
        self.n_samples = len(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data."""
        if self.mean is None:
            return X
        return (X - self.mean) / self.std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def to_dict(self) -> Dict:
        return {
            'mean': self.mean.tolist() if self.mean is not None else None,
            'std': self.std.tolist() if self.std is not None else None,
            'n_samples': self.n_samples,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'FeatureNormalizer':
        norm = cls()
        norm.mean = np.array(d['mean']) if d['mean'] else None
        norm.std = np.array(d['std']) if d['std'] else None
        norm.n_samples = d.get('n_samples', 0)
        return norm


def load_ohlc_dataframe(path: Path, mint: Optional[str] = None) -> pd.DataFrame:
    """Load OHLC data, optionally filtering by token mint."""
    df = pd.read_csv(path)

    if mint:
        df = df[df['token_mint'].str.startswith(mint[:8])]

    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def load_multi_token_data(path: Path, exclude_mints: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """Load OHLC data for multiple tokens."""
    df = pd.read_csv(path)

    # Exclude specified mints
    if exclude_mints:
        for mint in exclude_mints:
            df = df[~df['token_mint'].str.startswith(mint[:8])]

    # Group by token
    token_dfs = {}
    for symbol in df['token_symbol'].unique():
        if symbol == 'SOL':  # Skip SOL as it's the quote currency
            continue
        token_df = df[df['token_symbol'] == symbol].copy()
        token_df = token_df.sort_values('timestamp').reset_index(drop=True)
        if len(token_df) >= 50:  # Minimum bars required
            token_dfs[symbol] = token_df

    return token_dfs


def build_bar_features(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> np.ndarray:
    """Build per-bar features from OHLC data.

    Features (5 per bar):
    - Return (close/open - 1)
    - High relative (high/open - 1)
    - Low relative (low/open - 1)
    - Range (high - low) / open
    - Body ratio (close - open) / (high - low + eps)
    """
    eps = 1e-10

    returns = closes / (opens + eps) - 1
    high_rel = highs / (opens + eps) - 1
    low_rel = lows / (opens + eps) - 1
    ranges = (highs - lows) / (opens + eps)
    body_ratio = (closes - opens) / (highs - lows + eps)

    features = np.stack([returns, high_rel, low_rel, ranges, body_ratio], axis=-1)
    return features.astype(np.float32)


def build_aggregate_features(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> np.ndarray:
    """Build aggregate features for entire context window.

    Features (7):
    - Total return
    - Volatility (std of returns)
    - Max drawdown
    - Trend strength
    - Recent momentum (last 10 bars)
    - Volume trend (if available)
    - Mean reversion signal
    """
    eps = 1e-10
    n = len(closes)

    # Returns
    returns = closes[1:] / (closes[:-1] + eps) - 1 if n > 1 else np.array([0.0])

    # Total return
    total_return = closes[-1] / (closes[0] + eps) - 1 if n > 0 else 0.0

    # Volatility
    volatility = np.std(returns) if len(returns) > 1 else 0.0

    # Max drawdown
    cummax = np.maximum.accumulate(closes)
    drawdowns = (cummax - closes) / (cummax + eps)
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

    # Trend strength (correlation with linear trend)
    if n > 2:
        x = np.arange(n)
        corr = np.corrcoef(x, closes)[0, 1]
        trend_strength = corr if not np.isnan(corr) else 0.0
    else:
        trend_strength = 0.0

    # Recent momentum (last 10 bars or available)
    lookback = min(10, n)
    if lookback > 1:
        recent_return = closes[-1] / (closes[-lookback] + eps) - 1
    else:
        recent_return = 0.0

    # Price position in range
    price_range = highs.max() - lows.min()
    if price_range > eps:
        price_position = (closes[-1] - lows.min()) / price_range
    else:
        price_position = 0.5

    # Mean reversion signal (distance from moving average)
    ma = np.mean(closes)
    mean_rev = (closes[-1] - ma) / (ma + eps)

    features = np.array([
        total_return,
        volatility,
        max_dd,
        trend_strength,
        recent_return,
        price_position,
        mean_rev,
    ], dtype=np.float32)

    return features


class MultiTokenDataset(Dataset):
    """Dataset for training on multiple BAGS tokens."""

    def __init__(
        self,
        token_dfs: Dict[str, pd.DataFrame],
        context_length: int = 96,
        horizon: int = 3,
        cost_bps: float = 130.0,
        min_return: float = 0.002,
        size_scale: float = 0.02,
    ):
        self.context_length = context_length
        self.horizon = horizon
        self.cost_bps = cost_bps
        self.min_return = min_return
        self.size_scale = size_scale

        # Build samples from all tokens
        self.samples = []
        for symbol, df in token_dfs.items():
            self._add_samples_from_df(df, symbol)

        logger.info(f"Built {len(self.samples)} samples from {len(token_dfs)} tokens")

    def _add_samples_from_df(self, df: pd.DataFrame, symbol: str):
        """Add training samples from a single token's data."""
        opens = df['open'].to_numpy(dtype=np.float32)
        highs = df['high'].to_numpy(dtype=np.float32)
        lows = df['low'].to_numpy(dtype=np.float32)
        closes = df['close'].to_numpy(dtype=np.float32)

        # Valid indices: need context_length bars before and horizon bars after
        for idx in range(self.context_length, len(df) - self.horizon):
            self.samples.append({
                'symbol': symbol,
                'idx': idx,
                'opens': opens,
                'highs': highs,
                'lows': lows,
                'closes': closes,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        data_idx = sample['idx']
        start = data_idx - self.context_length

        # Build features
        bar_features = build_bar_features(
            sample['opens'][start:data_idx],
            sample['highs'][start:data_idx],
            sample['lows'][start:data_idx],
            sample['closes'][start:data_idx],
        )

        agg_features = build_aggregate_features(
            sample['opens'][start:data_idx],
            sample['highs'][start:data_idx],
            sample['lows'][start:data_idx],
            sample['closes'][start:data_idx],
        )

        # Build targets
        cost_return = self.cost_bps / 10000.0
        current_close = sample['closes'][data_idx - 1]
        future_close = sample['closes'][data_idx + self.horizon]
        future_return = future_close / max(current_close, 1e-12) - 1.0
        net_return = future_return - cost_return

        signal_target = 1.0 if net_return > self.min_return else 0.0
        size_target = float(np.clip(max(net_return, 0.0) / self.size_scale, 0.0, 1.0))

        return {
            'bar_features': torch.tensor(bar_features, dtype=torch.float32),
            'agg_features': torch.tensor(agg_features, dtype=torch.float32),
            'signal_target': torch.tensor(signal_target, dtype=torch.float32),
            'size_target': torch.tensor(size_target, dtype=torch.float32),
            'symbol': sample['symbol'],
        }


class SingleTokenDataset(Dataset):
    """Dataset for single token fine-tuning/evaluation."""

    def __init__(
        self,
        df: pd.DataFrame,
        context_length: int = 96,
        horizon: int = 3,
        cost_bps: float = 130.0,
        min_return: float = 0.002,
        size_scale: float = 0.02,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ):
        self.context_length = context_length
        self.horizon = horizon
        self.cost_bps = cost_bps
        self.min_return = min_return
        self.size_scale = size_scale

        self.opens = df['open'].to_numpy(dtype=np.float32)
        self.highs = df['high'].to_numpy(dtype=np.float32)
        self.lows = df['low'].to_numpy(dtype=np.float32)
        self.closes = df['close'].to_numpy(dtype=np.float32)

        # Valid indices
        min_idx = start_idx if start_idx else context_length
        max_idx = end_idx if end_idx else len(df) - horizon

        self.valid_indices = list(range(max(min_idx, context_length), max_idx))
        logger.info(f"Built {len(self.valid_indices)} samples")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data_idx = self.valid_indices[idx]
        start = data_idx - self.context_length

        bar_features = build_bar_features(
            self.opens[start:data_idx],
            self.highs[start:data_idx],
            self.lows[start:data_idx],
            self.closes[start:data_idx],
        )

        agg_features = build_aggregate_features(
            self.opens[start:data_idx],
            self.highs[start:data_idx],
            self.lows[start:data_idx],
            self.closes[start:data_idx],
        )

        cost_return = self.cost_bps / 10000.0
        current_close = self.closes[data_idx - 1]
        future_close = self.closes[data_idx + self.horizon]
        future_return = future_close / max(current_close, 1e-12) - 1.0
        net_return = future_return - cost_return

        signal_target = 1.0 if net_return > self.min_return else 0.0
        size_target = float(np.clip(max(net_return, 0.0) / self.size_scale, 0.0, 1.0))

        return {
            'bar_features': torch.tensor(bar_features, dtype=torch.float32),
            'agg_features': torch.tensor(agg_features, dtype=torch.float32),
            'signal_target': torch.tensor(signal_target, dtype=torch.float32),
            'size_target': torch.tensor(size_target, dtype=torch.float32),
        }
