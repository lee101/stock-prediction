from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from alpaca.data import TimeFrame, TimeFrameUnit
from torch.utils.data import DataLoader, Dataset

from alpaca_wrapper import download_symbol_history

from .config import DatasetConfig

DEFAULT_FEATURES: Tuple[str, ...] = (
    "return_1h",
    "return_4h",
    "return_24h",
    "volatility_24h",
    "range_pct",
    "volume_z",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "chronos_close_delta",
    "chronos_high_delta",
    "chronos_low_delta",
)


@dataclass
class FeatureNormalizer:
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
    def from_dict(cls, payload: Dict[str, Iterable[float]]) -> "FeatureNormalizer":
        mean = np.asarray(list(payload["mean"]), dtype=np.float32)
        std = np.asarray(list(payload["std"]), dtype=np.float32)
        return cls(mean=mean, std=std)


class HourlyCryptoDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        features: np.ndarray,
        sequence_length: int,
        *,
        chronos_high: np.ndarray,
        chronos_low: np.ndarray,
    ) -> None:
        if len(frame) != len(features):
            raise ValueError("Feature matrix must align with base frame")
        if len(frame) < sequence_length:
            raise ValueError("Frame shorter than sequence length")
        self.frame = frame.reset_index(drop=True)
        self.features = features.astype(np.float32)
        self.seq_len = int(sequence_length)
        self.highs = frame["high"].to_numpy(dtype=np.float32)
        self.lows = frame["low"].to_numpy(dtype=np.float32)
        self.closes = frame["close"].to_numpy(dtype=np.float32)
        self.reference_close = frame["reference_close"].to_numpy(dtype=np.float32)
        self.c_high = chronos_high.astype(np.float32)
        self.c_low = chronos_low.astype(np.float32)

    def __len__(self) -> int:
        return len(self.frame) - self.seq_len + 1

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx
        end = idx + self.seq_len
        return {
            "features": torch.from_numpy(self.features[start:end]),
            "high": torch.from_numpy(self.highs[start:end]),
            "low": torch.from_numpy(self.lows[start:end]),
            "close": torch.from_numpy(self.closes[start:end]),
            "reference_close": torch.from_numpy(self.reference_close[start:end]),
            "chronos_high": torch.from_numpy(self.c_high[start:end]),
            "chronos_low": torch.from_numpy(self.c_low[start:end]),
        }


class HourlyCryptoDataModule:
    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self.feature_columns = tuple(config.feature_columns or DEFAULT_FEATURES)
        self.frame = self._prepare_frame()
        if len(self.frame) < config.min_history_hours:
            raise ValueError(
                f"Insufficient hourly history ({len(self.frame)} rows, minimum {config.min_history_hours})."
            )
        val_hours = int(max(0, config.validation_days) * 24)
        if val_hours > 0 and len(self.frame) > (val_hours + config.sequence_length):
            split_idx = len(self.frame) - val_hours
        else:
            split_idx = int(len(self.frame) * (1 - config.val_fraction))
        split_idx = max(split_idx, config.sequence_length + 1)
        train_frame = self.frame.iloc[:split_idx].reset_index(drop=True)
        val_frame = self.frame.iloc[split_idx - config.sequence_length :].reset_index(drop=True)
        feature_cols = list(self.feature_columns)
        train_features = train_frame[feature_cols].to_numpy(dtype=np.float32)
        self.normalizer = FeatureNormalizer.fit(train_features)
        norm_train = self.normalizer.transform(train_features)
        norm_val = self.normalizer.transform(val_frame[feature_cols].to_numpy(dtype=np.float32))

        self.train_dataset = HourlyCryptoDataset(
            train_frame,
            norm_train,
            config.sequence_length,
            chronos_high=train_frame["chronos_high"].to_numpy(dtype=np.float32),
            chronos_low=train_frame["chronos_low"].to_numpy(dtype=np.float32),
        )
        self.val_dataset = HourlyCryptoDataset(
            val_frame,
            norm_val,
            config.sequence_length,
            chronos_high=val_frame["chronos_high"].to_numpy(dtype=np.float32),
            chronos_low=val_frame["chronos_low"].to_numpy(dtype=np.float32),
        )

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

    # ------------------------------------------------------------------
    def _prepare_frame(self) -> pd.DataFrame:
        price_frame = self._load_price_history()
        forecast_frame = self._load_forecasts()
        merged = price_frame.merge(forecast_frame, on=["timestamp", "symbol"], how="inner")
        merged = merged.sort_values("timestamp").reset_index(drop=True)
        enriched = self._add_features(merged)
        enriched = enriched.dropna(subset=list(self.feature_columns))
        return enriched.reset_index(drop=True)

    def _load_price_history(self) -> pd.DataFrame:
        path = self.config.data_root / f"{self.config.symbol.upper()}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing hourly dataset {path}")
        frame = pd.read_csv(path)
        frame.columns = [col.lower() for col in frame.columns]
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame["symbol"] = self.config.symbol.upper()
        frame = self._refresh_with_latest_history(frame, path)
        cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        return frame[cols].sort_values("timestamp").reset_index(drop=True)

    def _refresh_with_latest_history(self, frame: pd.DataFrame, path: Path) -> pd.DataFrame:
        hours = max(0, int(self.config.refresh_hours))
        if hours <= 0:
            return frame
        now = datetime.now(timezone.utc)
        if frame.empty:
            start = now - timedelta(hours=hours)
        else:
            last_ts = pd.to_datetime(frame["timestamp"].max(), utc=True, errors="coerce")
            if pd.isna(last_ts):
                last_ts = now - timedelta(hours=hours)
            start = last_ts - timedelta(hours=4)
        try:
            live = download_symbol_history(
                symbol=self.config.symbol,
                start=start,
                end=now,
                include_latest=True,
                timeframe=TimeFrame(1, TimeFrameUnit.Hour),
            )
        except Exception:
            return frame
        if live.empty:
            return frame
        live = live.reset_index().rename(columns=str.lower)
        live["timestamp"] = pd.to_datetime(live["timestamp"], utc=True)
        combined = (
            pd.concat([frame, live], ignore_index=True)
            .drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        combined.to_csv(path, index=False)
        return combined

    def _load_forecasts(self) -> pd.DataFrame:
        cache_path = self.config.forecast_cache_dir / f"{self.config.symbol.upper()}.parquet"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"No Chronos forecast cache found at {cache_path}. Run DailyChronosForecastManager.ensure_latest first."
            )
        frame = pd.read_parquet(cache_path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame["symbol"] = frame["symbol"].str.upper()
        keep_cols = [
            "timestamp",
            "symbol",
            "predicted_close_p50",
            "predicted_high_p50",
            "predicted_low_p50",
        ]
        return frame[keep_cols]

    def _add_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        close = frame["close"]
        frame["reference_close"] = close.shift(1).ffill()
        frame["return_1h"] = close.pct_change(1).shift(1)
        frame["return_4h"] = close.pct_change(4).shift(1)
        frame["return_24h"] = close.pct_change(24).shift(1)
        frame["volatility_24h"] = close.pct_change(1).rolling(24).std().shift(1)
        frame["range_pct"] = ((frame["high"] - frame["low"]) / frame["close"].clip(lower=1e-6)).shift(1)
        frame["volume_z"] = self._zscore(frame["volume"], window=168).shift(1)
        frame["hour_sin"], frame["hour_cos"] = self._cycle_features(frame["timestamp"].dt.hour, period=24)
        frame["dow_sin"], frame["dow_cos"] = self._cycle_features(frame["timestamp"].dt.dayofweek, period=7)
        for col in ("predicted_close_p50", "predicted_high_p50", "predicted_low_p50"):
            frame[col] = frame[col].ffill()
        prev_close = frame["close"].shift(1)
        prev_close = prev_close.ffill()
        base = prev_close.clip(lower=1e-6)
        frame["chronos_close_delta"] = (frame["predicted_close_p50"] - prev_close) / base
        frame["chronos_high_delta"] = (frame["predicted_high_p50"] - prev_close) / base
        frame["chronos_low_delta"] = (frame["predicted_low_p50"] - prev_close) / base
        frame["chronos_high"] = frame["predicted_high_p50"].ffill().fillna(frame["close"])
        frame["chronos_low"] = frame["predicted_low_p50"].ffill().fillna(frame["close"])
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


class MultiSymbolDataset(Dataset):
    """Concatenates multiple HourlyCryptoDatasets for multi-pair training."""

    def __init__(self, datasets: List[HourlyCryptoDataset]) -> None:
        self.datasets = datasets
        self.lengths = [len(ds) for ds in datasets]
        self.cumulative_lengths = [0] + list(np.cumsum(self.lengths))

    def __len__(self) -> int:
        return sum(self.lengths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        for i, (start, end) in enumerate(zip(self.cumulative_lengths[:-1], self.cumulative_lengths[1:])):
            if start <= idx < end:
                return self.datasets[i][idx - start]
        raise IndexError(f"Index {idx} out of range")


class MultiSymbolDataModule:
    """Data module for training on multiple crypto pairs."""

    def __init__(self, symbols: Sequence[str], config: DatasetConfig) -> None:
        """Initialize multi-symbol data module.

        Args:
            symbols: List of symbols to train on (e.g., ["BTCUSD", "ETHUSD", "UNIUSD"])
            config: Base dataset config (will be overridden per symbol)
        """
        self.symbols = list(symbols)
        self.base_config = config
        self.modules: Dict[str, HourlyCryptoDataModule] = {}
        self.normalizers: Dict[str, FeatureNormalizer] = {}

        # Create data module for each symbol
        for symbol in self.symbols:
            symbol_config = DatasetConfig(
                symbol=symbol,
                data_root=config.data_root,
                forecast_cache_dir=config.forecast_cache_dir,
                sequence_length=config.sequence_length,
                val_fraction=config.val_fraction,
                min_history_hours=config.min_history_hours,
                max_feature_lookback_hours=config.max_feature_lookback_hours,
                feature_columns=config.feature_columns,
                refresh_hours=config.refresh_hours,
                validation_days=config.validation_days,
            )
            module = HourlyCryptoDataModule(symbol_config)
            self.modules[symbol] = module
            self.normalizers[symbol] = module.normalizer

        # Concatenate all training datasets
        train_datasets = [mod.train_dataset for mod in self.modules.values()]
        self.train_dataset = MultiSymbolDataset(train_datasets)

        # Use first symbol for validation (or target symbol if specified)
        target_symbol = self.symbols[0] if len(self.symbols) > 0 else "LINKUSD"
        self.val_dataset = self.modules[target_symbol].val_dataset
        self.normalizer = self.modules[target_symbol].normalizer

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
