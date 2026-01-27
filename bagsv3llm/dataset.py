"""Dataset and feature engineering for BagsV3LLM.

Handles:
- Pre-training data from trainingdata/ (stock/crypto OHLC)
- Fine-tuning data from bagstraining/ (CODEX)
- Chronos2 forecast feature integration
- Normalizers and data splits
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


@dataclass
class FeatureNormalizerV3:
    """Robust feature normalization with per-feature statistics."""

    mean: np.ndarray
    std: np.ndarray
    clip_std: float = 5.0

    def transform(self, features: np.ndarray) -> np.ndarray:
        std = np.where(self.std < 1e-8, 1.0, self.std)
        normalized = (features - self.mean) / std
        return np.clip(normalized, -self.clip_std, self.clip_std)

    def inverse_transform(self, normalized: np.ndarray) -> np.ndarray:
        std = np.where(self.std < 1e-8, 1.0, self.std)
        return normalized * std + self.mean

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "clip_std": self.clip_std,
            "version": "v3",
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "FeatureNormalizerV3":
        return cls(
            mean=np.array(payload["mean"], dtype=np.float32),
            std=np.array(payload["std"], dtype=np.float32),
            clip_std=payload.get("clip_std", 5.0),
        )


def load_pretraining_data(
    data_dir: Path = Path("trainingdata"),
    min_rows_per_symbol: int = 300,
    max_symbols: Optional[int] = None,
) -> pd.DataFrame:
    """Load pre-training data from CSV files in trainingdata/.

    Args:
        data_dir: Directory containing CSV files
        min_rows_per_symbol: Minimum rows required for a symbol
        max_symbols: Maximum number of symbols to load (for testing)

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume, symbol
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise ValueError(f"Data directory not found: {data_dir}")

    # Find all CSV files
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")

    logger.info(f"Found {len(csv_files)} CSV files in {data_dir}")

    all_data = []
    symbols_loaded = 0

    for csv_file in csv_files:
        if max_symbols and symbols_loaded >= max_symbols:
            break

        try:
            df = pd.read_csv(csv_file)

            # Standardize column names
            df.columns = df.columns.str.lower().str.strip()

            # Check required columns
            required = ["timestamp", "open", "high", "low", "close"]
            missing = [col for col in required if col not in df.columns]
            if missing:
                logger.debug(f"Skipping {csv_file}: missing columns {missing}")
                continue

            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])

            # Get symbol from file name or column
            if "symbol" in df.columns:
                symbol = df["symbol"].iloc[0]
            else:
                symbol = csv_file.stem

            df["symbol"] = symbol

            if len(df) < min_rows_per_symbol:
                logger.debug(f"Skipping {symbol}: only {len(df)} rows")
                continue

            # Ensure volume column exists
            if "volume" not in df.columns:
                df["volume"] = 0.0

            # Select relevant columns
            df = df[["timestamp", "open", "high", "low", "close", "volume", "symbol"]]
            df = df.sort_values("timestamp").reset_index(drop=True)

            all_data.append(df)
            symbols_loaded += 1
            logger.debug(f"Loaded {symbol}: {len(df)} rows")

        except Exception as e:
            logger.warning(f"Error loading {csv_file}: {e}")
            continue

    if not all_data:
        raise ValueError(f"No valid data loaded from {data_dir}")

    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"Loaded {len(combined)} total rows from {symbols_loaded} symbols")

    return combined


def load_ohlc_dataframe(
    path: Path,
    mint: Optional[str] = None,
    dedupe: bool = True,
) -> pd.DataFrame:
    """Load OHLC data from CSV (for fine-tuning on CODEX)."""
    df = pd.read_csv(path)

    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()

    if "timestamp" not in df.columns:
        raise ValueError(f"Missing timestamp column in {path}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if mint and "token_mint" in df.columns:
        df = df[df["token_mint"] == mint]

    if dedupe:
        df = df.drop_duplicates(subset=["timestamp"], keep="last")

    df = df.sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        raise ValueError("No OHLC rows found")

    return df


def build_bar_features(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> np.ndarray:
    """Build per-bar features (5 features each).

    Features:
    1. returns: close-to-close return
    2. range_pct: (high - low) / low
    3. oc_return: (close - open) / open
    4. upper_wick: (high - close) / close
    5. lower_wick: (close - low) / close
    """
    n = len(closes)

    # 1. Returns
    returns = np.zeros(n, dtype=np.float32)
    if n > 1:
        returns[1:] = closes[1:] / np.maximum(closes[:-1], 1e-12) - 1.0

    # 2. Range percentage
    range_pct = highs / np.maximum(lows, 1e-12) - 1.0

    # 3. Open-close return
    oc_return = closes / np.maximum(opens, 1e-12) - 1.0

    # 4. Upper wick
    upper_wick = (highs - closes) / np.maximum(closes, 1e-12)

    # 5. Lower wick
    lower_wick = (closes - lows) / np.maximum(closes, 1e-12)

    # Stack: (n, 5)
    features = np.stack([returns, range_pct, oc_return, upper_wick, lower_wick], axis=1)
    return features.astype(np.float32)


def build_aggregate_features(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> np.ndarray:
    """Build aggregate features (7 features).

    Features:
    1. volatility: std of returns
    2-4. momentum_5/10/20: multi-horizon momentum
    5. rsi_like: ratio of positive returns
    6. price_position: position in range (0=low, 1=high)
    7. trend_strength: average return
    """
    returns = np.zeros(len(closes), dtype=np.float32)
    if len(closes) > 1:
        returns[1:] = closes[1:] / np.maximum(closes[:-1], 1e-12) - 1.0

    features = []

    # 1. Volatility
    volatility = np.std(returns) if len(returns) > 1 else 0.0
    features.append(volatility)

    # 2-4. Multi-horizon momentum
    for lookback in [5, 10, 20]:
        if len(closes) >= lookback:
            momentum = closes[-1] / np.maximum(closes[-lookback], 1e-12) - 1.0
        else:
            momentum = 0.0
        features.append(momentum)

    # 5. RSI-like
    positive_returns = np.sum(returns > 0)
    total_returns = np.sum(returns != 0)
    rsi_like = positive_returns / max(total_returns, 1)
    features.append(rsi_like)

    # 6. Price position
    range_high = np.max(highs)
    range_low = np.min(lows)
    if range_high > range_low:
        price_position = (closes[-1] - range_low) / (range_high - range_low)
    else:
        price_position = 0.5
    features.append(price_position)

    # 7. Trend strength
    trend_strength = np.mean(returns)
    features.append(trend_strength)

    return np.array(features, dtype=np.float32)


def build_chronos_features(
    predicted_prices: Optional[List[float]] = None,
    predicted_p10: Optional[List[float]] = None,
    predicted_p90: Optional[List[float]] = None,
    current_price: float = 0.0,
    num_horizons: int = 3,
) -> np.ndarray:
    """Build Chronos2 forecast features (12 features).

    For each of 3 horizons: predicted_return, uncertainty, upside, downside
    Total: 3 * 4 = 12 features
    """
    features = []

    for h in range(num_horizons):
        if predicted_prices and h < len(predicted_prices) and current_price > 0:
            pred_price = predicted_prices[h]
            pred_return = pred_price / current_price - 1.0

            # Uncertainty (p90 - p10 spread)
            if predicted_p10 and predicted_p90 and h < len(predicted_p10):
                spread = (predicted_p90[h] - predicted_p10[h]) / current_price
                upside = predicted_p90[h] / current_price - 1.0
                downside = predicted_p10[h] / current_price - 1.0
            else:
                spread = 0.1  # Default uncertainty
                upside = pred_return + 0.05
                downside = pred_return - 0.05
        else:
            # No forecast available - use zeros
            pred_return = 0.0
            spread = 0.1
            upside = 0.0
            downside = 0.0

        features.extend([pred_return, spread, upside, downside])

    return np.array(features, dtype=np.float32)


class PretrainingDataset(Dataset):
    """Dataset for pre-training on stock/crypto data."""

    def __init__(
        self,
        df: pd.DataFrame,
        context_length: int = 256,
        horizon: int = 3,
        cost_bps: float = 50.0,  # Lower cost for stocks
        min_return: float = 0.001,
        size_scale: float = 0.02,
        mask_ratio: float = 0.15,
    ):
        self.context_length = context_length
        self.horizon = horizon
        self.cost_bps = cost_bps
        self.min_return = min_return
        self.size_scale = size_scale
        self.mask_ratio = mask_ratio

        # Build samples per symbol
        self.samples = []
        symbols = df["symbol"].unique()

        for symbol in symbols:
            symbol_df = df[df["symbol"] == symbol].sort_values("timestamp")
            opens = symbol_df["open"].to_numpy(dtype=np.float32)
            highs = symbol_df["high"].to_numpy(dtype=np.float32)
            lows = symbol_df["low"].to_numpy(dtype=np.float32)
            closes = symbol_df["close"].to_numpy(dtype=np.float32)

            max_idx = len(symbol_df) - horizon
            for idx in range(context_length, max_idx):
                start = idx - context_length
                self.samples.append({
                    "symbol": symbol,
                    "opens": opens[start:idx],
                    "highs": highs[start:idx],
                    "lows": lows[start:idx],
                    "closes": closes[start:idx],
                    "future_close": closes[idx + horizon],
                    "current_close": closes[idx - 1],
                })

        logger.info(f"Built {len(self.samples)} pre-training samples from {len(symbols)} symbols")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Build bar features (context_length, 5)
        bar_features = build_bar_features(
            sample["opens"],
            sample["highs"],
            sample["lows"],
            sample["closes"],
        )

        # Build aggregate features (7,)
        agg_features = build_aggregate_features(
            sample["opens"],
            sample["highs"],
            sample["lows"],
            sample["closes"],
        )

        # Chronos features - zeros for pre-training (no forecasts)
        chronos_features = np.zeros(
            (self.context_length, 12), dtype=np.float32
        )

        # Build targets
        cost_return = self.cost_bps / 10000.0
        future_return = sample["future_close"] / max(sample["current_close"], 1e-12) - 1.0
        net_return = future_return - cost_return

        signal_target = 1.0 if net_return > self.min_return else 0.0
        size_target = float(np.clip(max(net_return, 0.0) / self.size_scale, 0.0, 1.0))

        # Create random mask for reconstruction
        mask = np.random.rand(self.context_length) < self.mask_ratio

        return {
            "bar_features": torch.tensor(bar_features, dtype=torch.float32),
            "chronos_features": torch.tensor(chronos_features, dtype=torch.float32),
            "agg_features": torch.tensor(agg_features, dtype=torch.float32),
            "signal_target": torch.tensor(signal_target, dtype=torch.float32),
            "size_target": torch.tensor(size_target, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
        }


class FinetuningDataset(Dataset):
    """Dataset for fine-tuning on CODEX with Chronos2 forecasts."""

    def __init__(
        self,
        df: pd.DataFrame,
        context_length: int = 256,
        horizon: int = 3,
        cost_bps: float = 130.0,
        min_return: float = 0.002,
        size_scale: float = 0.02,
        chronos_wrapper=None,  # Optional Chronos2OHLCWrapper
    ):
        self.context_length = context_length
        self.horizon = horizon
        self.cost_bps = cost_bps
        self.min_return = min_return
        self.size_scale = size_scale
        self.chronos_wrapper = chronos_wrapper

        self.opens = df["open"].to_numpy(dtype=np.float32)
        self.highs = df["high"].to_numpy(dtype=np.float32)
        self.lows = df["low"].to_numpy(dtype=np.float32)
        self.closes = df["close"].to_numpy(dtype=np.float32)
        self.timestamps = df["timestamp"].to_numpy()

        # Build valid indices
        max_idx = len(df) - horizon
        self.valid_indices = list(range(context_length, max_idx))

        logger.info(f"Built {len(self.valid_indices)} fine-tuning samples")

    def _get_chronos_features(self, idx: int) -> np.ndarray:
        """Get Chronos2 forecast features for a given index."""
        if self.chronos_wrapper is None:
            return np.zeros((self.context_length, 12), dtype=np.float32)

        try:
            # Build context dataframe for Chronos
            start = idx - self.context_length
            context_df = pd.DataFrame({
                "timestamp": self.timestamps[start:idx],
                "open": self.opens[start:idx],
                "high": self.highs[start:idx],
                "low": self.lows[start:idx],
                "close": self.closes[start:idx],
            })

            from bagsfm.config import TokenConfig
            token = TokenConfig(mint="CODEX", symbol="CODEX")

            batch = self.chronos_wrapper.predict_ohlc(
                context_df,
                symbol="CODEX",
                prediction_length=3,
                context_length=min(512, len(context_df)),
            )

            q50 = batch.quantile_frames.get(0.5)
            q10 = batch.quantile_frames.get(0.1)
            q90 = batch.quantile_frames.get(0.9)

            if q50 is not None and not q50.empty:
                predicted_prices = q50["close"].tolist()
                predicted_p10 = q10["close"].tolist() if q10 is not None else None
                predicted_p90 = q90["close"].tolist() if q90 is not None else None
            else:
                predicted_prices = None
                predicted_p10 = None
                predicted_p90 = None

            current_price = self.closes[idx - 1]

            # Build chronos features for each position in context
            # For simplicity, use same forecast for last N positions
            base_features = build_chronos_features(
                predicted_prices=predicted_prices,
                predicted_p10=predicted_p10,
                predicted_p90=predicted_p90,
                current_price=current_price,
            )

            # Broadcast to all context positions (could be improved)
            chronos_features = np.zeros((self.context_length, 12), dtype=np.float32)
            chronos_features[-32:] = base_features  # Only last 32 bars get forecast

            return chronos_features

        except Exception as e:
            logger.debug(f"Chronos forecast failed: {e}")
            return np.zeros((self.context_length, 12), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data_idx = self.valid_indices[idx]
        start = data_idx - self.context_length

        # Build bar features
        bar_features = build_bar_features(
            self.opens[start:data_idx],
            self.highs[start:data_idx],
            self.lows[start:data_idx],
            self.closes[start:data_idx],
        )

        # Build aggregate features
        agg_features = build_aggregate_features(
            self.opens[start:data_idx],
            self.highs[start:data_idx],
            self.lows[start:data_idx],
            self.closes[start:data_idx],
        )

        # Get Chronos features (cached or computed)
        chronos_features = self._get_chronos_features(data_idx)

        # Build targets
        cost_return = self.cost_bps / 10000.0
        current_close = self.closes[data_idx - 1]
        future_close = self.closes[data_idx + self.horizon]
        future_return = future_close / max(current_close, 1e-12) - 1.0
        net_return = future_return - cost_return

        signal_target = 1.0 if net_return > self.min_return else 0.0
        size_target = float(np.clip(max(net_return, 0.0) / self.size_scale, 0.0, 1.0))

        return {
            "bar_features": torch.tensor(bar_features, dtype=torch.float32),
            "chronos_features": torch.tensor(chronos_features, dtype=torch.float32),
            "agg_features": torch.tensor(agg_features, dtype=torch.float32),
            "signal_target": torch.tensor(signal_target, dtype=torch.float32),
            "size_target": torch.tensor(size_target, dtype=torch.float32),
        }


def fit_normalizer(
    dataset: Dataset,
    max_samples: int = 10000,
) -> Tuple[FeatureNormalizerV3, FeatureNormalizerV3, FeatureNormalizerV3]:
    """Fit normalizers on dataset.

    Returns:
        bar_normalizer, chronos_normalizer, agg_normalizer
    """
    bar_samples = []
    chronos_samples = []
    agg_samples = []

    indices = np.random.choice(
        len(dataset),
        min(max_samples, len(dataset)),
        replace=False
    )

    for idx in indices:
        sample = dataset[int(idx)]
        bar_samples.append(sample["bar_features"].numpy())
        chronos_samples.append(sample["chronos_features"].numpy())
        agg_samples.append(sample["agg_features"].numpy())

    # Stack and compute statistics
    bar_arr = np.concatenate(bar_samples, axis=0)  # (N*context, 5)
    chronos_arr = np.concatenate(chronos_samples, axis=0)  # (N*context, 12)
    agg_arr = np.stack(agg_samples, axis=0)  # (N, 7)

    bar_normalizer = FeatureNormalizerV3(
        mean=bar_arr.mean(axis=0).astype(np.float32),
        std=bar_arr.std(axis=0).astype(np.float32),
    )

    chronos_normalizer = FeatureNormalizerV3(
        mean=chronos_arr.mean(axis=0).astype(np.float32),
        std=chronos_arr.std(axis=0).astype(np.float32),
    )

    agg_normalizer = FeatureNormalizerV3(
        mean=agg_arr.mean(axis=0).astype(np.float32),
        std=agg_arr.std(axis=0).astype(np.float32),
    )

    return bar_normalizer, chronos_normalizer, agg_normalizer


def create_dataloaders(
    dataset: Dataset,
    batch_size: int = 32,
    train_split: float = 0.8,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders with time-based split."""
    split_idx = int(len(dataset) * train_split)

    train_indices = list(range(split_idx))
    val_indices = list(range(split_idx, len(dataset)))

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
