"""Data loading for ChronosPnL Trader.

Loads hourly stock data with:
- Market hours filtering (9:30 AM - 4:00 PM ET)
- Feature engineering (returns, volatility, time encoding)
- PnL history tracking for Chronos2 forecasting
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from chronospnltrader.config import DataConfig

NEW_YORK = ZoneInfo("America/New_York")


# Features for ChronosPnL model
HOURLY_FEATURES: Tuple[str, ...] = (
    # Returns at multiple scales
    "return_1h",
    "return_4h",
    "return_24h",
    # Volatility & range
    "volatility_24h",
    "range_pct",
    "volume_z",
    # Cyclical time encoding
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    # Chronos multi-scale forecasts
    "chronos_1h_close_delta",
    "chronos_1h_high_delta",
    "chronos_1h_low_delta",
    "chronos_2h_close_delta",
    "chronos_2h_high_delta",
    "chronos_2h_low_delta",
    "chronos_4h_close_delta",
    "chronos_4h_high_delta",
    "chronos_4h_low_delta",
    # PnL-specific features
    "rolling_pnl_7h",  # ~1 trading day PnL
    "rolling_pnl_35h",  # ~5 trading days PnL
    "rolling_pnl_70h",  # ~10 trading days PnL
    "pnl_volatility",
    "win_rate_24h",
)


def is_market_hours(ts: datetime) -> bool:
    """Check if timestamp is during NYSE regular trading hours."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    ts_ny = ts.astimezone(NEW_YORK)

    # Weekday check
    if ts_ny.weekday() >= 5:
        return False

    # Market hours: 9:30 AM - 4:00 PM ET
    market_open = time(9, 30)
    market_close = time(16, 0)
    ts_time = ts_ny.time()

    return market_open <= ts_time < market_close


def filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataframe to only include bars during market hours."""
    if "timestamp" not in df.columns:
        return df

    mask = df["timestamp"].apply(is_market_hours)
    return df[mask].reset_index(drop=True)


@dataclass
class FeatureNormalizer:
    """Z-score normalization for features."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, matrix: np.ndarray) -> "FeatureNormalizer":
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        return (matrix - self.mean) / self.std

    def to_dict(self) -> Dict[str, List[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, payload: Dict[str, List[float]]) -> "FeatureNormalizer":
        mean = np.asarray(payload["mean"], dtype=np.float32)
        std = np.asarray(payload["std"], dtype=np.float32)
        return cls(mean=mean, std=std)


class ChronosPnLDataset(Dataset):
    """Dataset with sequence features, lookahead, and PnL history."""

    def __init__(
        self,
        frame: pd.DataFrame,
        features: np.ndarray,
        sequence_length: int = 168,
        lookahead_hours: int = 24,
        pnl_history_hours: int = 210,
        *,
        chronos_high: np.ndarray,
        chronos_low: np.ndarray,
        pnl_history: np.ndarray,
    ) -> None:
        if len(frame) != len(features):
            raise ValueError("Feature matrix must align with base frame")

        min_required = sequence_length + lookahead_hours
        if len(frame) < min_required:
            raise ValueError(
                f"Frame ({len(frame)}) shorter than sequence + lookahead ({min_required})"
            )

        self.frame = frame.reset_index(drop=True)
        self.features = features.astype(np.float32)
        self.seq_len = int(sequence_length)
        self.lookahead = int(lookahead_hours)
        self.pnl_history_len = int(pnl_history_hours)

        # Price data
        self.opens = frame["open"].to_numpy(dtype=np.float32)
        self.highs = frame["high"].to_numpy(dtype=np.float32)
        self.lows = frame["low"].to_numpy(dtype=np.float32)
        self.closes = frame["close"].to_numpy(dtype=np.float32)
        self.reference_close = frame["reference_close"].to_numpy(dtype=np.float32)

        # Chronos forecasts
        self.c_high = chronos_high.astype(np.float32)
        self.c_low = chronos_low.astype(np.float32)

        # PnL history (simulated or actual)
        self.pnl_history = pnl_history.astype(np.float32)

    def __len__(self) -> int:
        return len(self.frame) - self.seq_len - self.lookahead + 1

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq_end = idx + self.seq_len
        lookahead_end = seq_end + self.lookahead

        # Get PnL history ending at current position
        pnl_start = max(0, seq_end - self.pnl_history_len)
        pnl_hist = self.pnl_history[pnl_start:seq_end]
        # Pad if needed
        if len(pnl_hist) < self.pnl_history_len:
            pad_len = self.pnl_history_len - len(pnl_hist)
            pnl_hist = np.concatenate([np.zeros(pad_len, dtype=np.float32), pnl_hist])

        return {
            # Input features (sequence_length bars)
            "features": torch.from_numpy(self.features[idx:seq_end]),
            "reference_close": torch.from_numpy(self.reference_close[idx:seq_end]),
            "chronos_high": torch.from_numpy(self.c_high[idx:seq_end]),
            "chronos_low": torch.from_numpy(self.c_low[idx:seq_end]),
            # Current bar for trade decision
            "current_close": torch.tensor(self.closes[seq_end - 1], dtype=torch.float32),
            "current_high": torch.tensor(self.highs[seq_end - 1], dtype=torch.float32),
            "current_low": torch.tensor(self.lows[seq_end - 1], dtype=torch.float32),
            # Lookahead for simulation
            "future_opens": torch.from_numpy(self.opens[seq_end:lookahead_end]),
            "future_highs": torch.from_numpy(self.highs[seq_end:lookahead_end]),
            "future_lows": torch.from_numpy(self.lows[seq_end:lookahead_end]),
            "future_closes": torch.from_numpy(self.closes[seq_end:lookahead_end]),
            # PnL history for Chronos2 forecasting
            "pnl_history": torch.from_numpy(pnl_hist),
        }


class ChronosPnLDataModule:
    """Data module for ChronosPnL trading."""

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.feature_columns = HOURLY_FEATURES
        self.frame = self._prepare_frame()

        if len(self.frame) < config.min_history_hours:
            raise ValueError(
                f"Insufficient history ({len(self.frame)} rows, "
                f"minimum {config.min_history_hours})."
            )

        # Split into train/val
        val_hours = int(config.validation_hours)
        min_train = config.sequence_length + config.lookahead_hours + 1

        if val_hours > 0 and len(self.frame) > (val_hours + min_train):
            split_idx = len(self.frame) - val_hours
        else:
            split_idx = int(len(self.frame) * (1 - config.val_fraction))

        split_idx = max(split_idx, min_train)

        train_frame = self.frame.iloc[:split_idx].reset_index(drop=True)
        val_frame = self.frame.iloc[split_idx - config.sequence_length:].reset_index(drop=True)

        # Fit normalizer on training data
        feature_cols = list(self.feature_columns)
        train_features = train_frame[feature_cols].to_numpy(dtype=np.float32)
        self.normalizer = FeatureNormalizer.fit(train_features)

        norm_train = self.normalizer.transform(train_features)
        norm_val = self.normalizer.transform(
            val_frame[feature_cols].to_numpy(dtype=np.float32)
        )

        # Create PnL history (initially zeros, updated during training)
        train_pnl = np.zeros(len(train_frame), dtype=np.float32)
        val_pnl = np.zeros(len(val_frame), dtype=np.float32)

        self.train_dataset = ChronosPnLDataset(
            train_frame,
            norm_train,
            config.sequence_length,
            config.lookahead_hours,
            config.pnl_history_hours,
            chronos_high=train_frame["chronos_high"].to_numpy(dtype=np.float32),
            chronos_low=train_frame["chronos_low"].to_numpy(dtype=np.float32),
            pnl_history=train_pnl,
        )
        self.val_dataset = ChronosPnLDataset(
            val_frame,
            norm_val,
            config.sequence_length,
            config.lookahead_hours,
            config.pnl_history_hours,
            chronos_high=val_frame["chronos_high"].to_numpy(dtype=np.float32),
            chronos_low=val_frame["chronos_low"].to_numpy(dtype=np.float32),
            pnl_history=val_pnl,
        )

    def update_pnl_history(self, pnl_values: np.ndarray, dataset: str = "train") -> None:
        """Update PnL history in dataset after simulation."""
        if dataset == "train":
            self.train_dataset.pnl_history[:len(pnl_values)] = pnl_values
        else:
            self.val_dataset.pnl_history[:len(pnl_values)] = pnl_values

    def train_dataloader(self, batch_size: int, num_workers: int = 0) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self, batch_size: int, num_workers: int = 0) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
        )

    def _prepare_frame(self) -> pd.DataFrame:
        """Load and prepare price data with features."""
        price_frame = self._load_price_history()
        forecast_frame = self._load_forecasts()

        if forecast_frame.empty:
            merged = price_frame.copy()
        else:
            merged = price_frame.merge(
                forecast_frame, on=["timestamp", "symbol"], how="left"
            )

        merged = merged.sort_values("timestamp").reset_index(drop=True)

        # Filter to market hours
        if self.config.market_hours_only:
            merged = filter_market_hours(merged)
            if len(merged) == 0:
                raise ValueError("No data remaining after market hours filtering")

        # Add features
        enriched = self._add_features(merged)

        # Fill Chronos NaN with 0
        chronos_cols = [c for c in self.feature_columns if "chronos" in c]
        for col in chronos_cols:
            if col in enriched.columns:
                enriched[col] = enriched[col].fillna(0.0)

        # Fill PnL features with 0
        pnl_cols = [c for c in self.feature_columns if "pnl" in c or "win_rate" in c]
        for col in pnl_cols:
            if col in enriched.columns:
                enriched[col] = enriched[col].fillna(0.0)

        # Drop rows with NaN in other features
        other_cols = [
            c for c in self.feature_columns
            if "chronos" not in c and "pnl" not in c and "win_rate" not in c
        ]
        enriched = enriched.dropna(subset=other_cols)

        return enriched.reset_index(drop=True)

    def _load_price_history(self) -> pd.DataFrame:
        """Load hourly OHLCV data."""
        symbol = self.config.symbols[0] if self.config.symbols else None
        if symbol is None:
            symbols = self.config.get_all_symbols()
            if not symbols:
                raise FileNotFoundError(f"No CSV files in {self.config.data_root}")
            symbol = symbols[0]

        path = self.config.data_root / f"{symbol.upper()}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset {path}")

        frame = pd.read_csv(path)
        frame.columns = [col.lower() for col in frame.columns]
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame["symbol"] = symbol.upper()

        cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        available_cols = [c for c in cols if c in frame.columns]

        return frame[available_cols].sort_values("timestamp").reset_index(drop=True)

    def _load_forecasts(self) -> pd.DataFrame:
        """Load Chronos2 forecast cache if available."""
        symbol = self.config.symbols[0] if self.config.symbols else None
        if symbol is None:
            return pd.DataFrame()

        cache_path = self.config.forecast_cache_dir / f"{symbol.upper()}.parquet"
        if not cache_path.exists():
            return pd.DataFrame()

        frame = pd.read_parquet(cache_path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame["symbol"] = frame["symbol"].str.upper()
        return frame

    def _add_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw OHLCV data."""
        frame = frame.copy()
        close = frame["close"]

        # Reference close
        frame["reference_close"] = close.shift(1).ffill()

        # Returns
        frame["return_1h"] = close.pct_change(1).shift(1)
        frame["return_4h"] = close.pct_change(4).shift(1)
        frame["return_24h"] = close.pct_change(24).shift(1)

        # Volatility
        frame["volatility_24h"] = close.pct_change(1).rolling(24).std().shift(1)

        # Range
        frame["range_pct"] = (
            (frame["high"] - frame["low"]) / frame["close"].clip(lower=1e-6)
        ).shift(1)

        # Volume z-score
        frame["volume_z"] = self._zscore(frame["volume"], window=168).shift(1)

        # Time encoding
        frame["hour_sin"], frame["hour_cos"] = self._cycle_features(
            frame["timestamp"].dt.hour, period=24
        )
        frame["dow_sin"], frame["dow_cos"] = self._cycle_features(
            frame["timestamp"].dt.dayofweek, period=7
        )

        # Chronos features
        prev_close = close.shift(1).ffill()
        base = prev_close.clip(lower=1e-6)

        # Default Chronos predictions to previous close
        for col in ["predicted_close_p50", "predicted_high_p50", "predicted_low_p50"]:
            if col not in frame.columns:
                frame[col] = prev_close

        frame["chronos_1h_close_delta"] = (frame["predicted_close_p50"] - prev_close) / base
        frame["chronos_1h_high_delta"] = (frame["predicted_high_p50"] - prev_close) / base
        frame["chronos_1h_low_delta"] = (frame["predicted_low_p50"] - prev_close) / base

        # 2h and 4h forecasts
        for suffix in ["2h", "4h"]:
            for metric in ["close", "high", "low"]:
                col = f"predicted_{metric}_p50_{suffix}"
                if col not in frame.columns:
                    frame[col] = prev_close
                frame[f"chronos_{suffix}_{metric}_delta"] = (frame[col] - prev_close) / base

        # Chronos high/low for price decoding
        frame["chronos_high"] = frame["predicted_high_p50"].ffill().fillna(close)
        frame["chronos_low"] = frame["predicted_low_p50"].ffill().fillna(close)

        # PnL features (will be computed from simulation, initially 0)
        frame["rolling_pnl_7h"] = 0.0
        frame["rolling_pnl_35h"] = 0.0
        frame["rolling_pnl_70h"] = 0.0
        frame["pnl_volatility"] = 0.0
        frame["win_rate_24h"] = 0.5  # Start at 50%

        return frame

    @staticmethod
    def _zscore(series: pd.Series, window: int) -> pd.Series:
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        rolling_std = rolling_std.replace(0.0, np.nan)
        return (series - rolling_mean) / rolling_std

    @staticmethod
    def _cycle_features(values: pd.Series, period: int) -> Tuple[pd.Series, pd.Series]:
        radians = 2 * math.pi * values / period
        return np.sin(radians), np.cos(radians)


class MultiSymbolDataModule:
    """Data module for training on multiple stock symbols."""

    def __init__(self, symbols: Sequence[str], config: DataConfig) -> None:
        cleaned: List[str] = []
        seen: set = set()
        for symbol in symbols:
            if not symbol:
                continue
            token = symbol.upper()
            if token not in seen:
                cleaned.append(token)
                seen.add(token)

        if not cleaned:
            raise ValueError("At least one symbol is required.")

        self.symbols = cleaned
        self.target_symbol = cleaned[0]
        self.base_config = config
        self.modules: Dict[str, ChronosPnLDataModule] = {}
        self.failed_symbols: List[str] = []

        for symbol in self.symbols:
            symbol_config = DataConfig(
                symbols=(symbol,),
                data_root=config.data_root,
                forecast_cache_dir=config.forecast_cache_dir,
                sequence_length=config.sequence_length,
                lookahead_hours=config.lookahead_hours,
                pnl_history_hours=config.pnl_history_hours,
                validation_hours=config.validation_hours,
                min_history_hours=config.min_history_hours,
                val_fraction=config.val_fraction,
                market_hours_only=config.market_hours_only,
            )
            try:
                module = ChronosPnLDataModule(symbol_config)
                self.modules[symbol] = module
            except (FileNotFoundError, ValueError) as e:
                print(f"Skipping {symbol}: {e}")
                self.failed_symbols.append(symbol)
                continue

        if not self.modules:
            raise ValueError("No valid symbols found.")

        print(f"Loaded {len(self.modules)} symbols, skipped {len(self.failed_symbols)}")

        if self.target_symbol not in self.modules:
            self.target_symbol = next(iter(self.modules.keys()))

        target_module = self.modules[self.target_symbol]
        self.normalizer = target_module.normalizer
        self.feature_columns = target_module.feature_columns
        self.frame = target_module.frame
        self.val_dataset = target_module.val_dataset

        train_datasets = [mod.train_dataset for mod in self.modules.values()]
        self.train_dataset = ConcatDataset(train_datasets)

        print(f"Total training samples: {len(self.train_dataset)}")

    def train_dataloader(self, batch_size: int, num_workers: int = 0) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self, batch_size: int, num_workers: int = 0) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
        )


def get_all_stock_symbols(data_root: Path = Path("trainingdatahourly/stocks")) -> List[str]:
    """Get all available stock symbols from data directory."""
    if not data_root.exists():
        return []
    return sorted([f.stem for f in data_root.glob("*.csv")])
