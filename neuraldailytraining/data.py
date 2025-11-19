from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import DailyDatasetConfig

LOGGER = logging.getLogger(__name__)

DEFAULT_FEATURES: Tuple[str, ...] = (
    "close",
    "predicted_close",
    "predicted_high",
    "predicted_low",
    "predicted_close_p10",
    "predicted_close_p90",
    "chronos_quantile_spread",
    "chronos_move_pct",
    "chronos_volatility_pct",
    "atr_pct_14",
    "range_pct",
    "volume_z",
    "day_sin",
    "day_cos",
    "chronos_close_delta",
    "chronos_high_delta",
    "chronos_low_delta",
    "asset_class",
)

FORECAST_COLUMNS = (
    "predicted_close",
    "predicted_high",
    "predicted_low",
    "predicted_close_p10",
    "predicted_close_p90",
    "forecast_move_pct",
    "forecast_volatility_pct",
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


class SymbolFrameBuilder:
    """Reusable helper that prepares per-symbol frames."""

    def __init__(self, config: DailyDatasetConfig, feature_columns: Sequence[str]) -> None:
        self.config = config
        self.feature_columns = tuple(feature_columns)

    def build(self, symbol: str) -> pd.DataFrame:
        prices = self._load_price_history(symbol)
        forecasts = self._load_forecasts(symbol)
        merged = prices.merge(forecasts, on=["symbol", "date"], how="left")
        for column in FORECAST_COLUMNS:
            if column not in merged:
                merged[column] = np.nan
        merged.sort_values("date", inplace=True)
        merged = merged.fillna(method="ffill")
        merged[["predicted_high", "predicted_low", "predicted_close"]] = merged[
            ["predicted_high", "predicted_low", "predicted_close"]
        ].fillna(merged["close"], axis=0)
        merged["forecast_move_pct"] = merged["forecast_move_pct"].fillna(0.0)
        merged["forecast_volatility_pct"] = merged["forecast_volatility_pct"].fillna(0.0)
        asset_flag = 1.0 if symbol.upper().endswith("-USD") else 0.0
        enriched = self._add_features(merged, asset_flag=asset_flag)
        enriched = enriched.dropna(subset=list(self.feature_columns))
        return enriched.reset_index(drop=True)

    def _load_price_history(self, symbol: str) -> pd.DataFrame:
        safe = symbol.replace("/", "-")
        path = Path(self.config.data_root) / f"{safe}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing daily dataset for {symbol}: {path}")
        frame = pd.read_csv(path)
        lower_cols = {col: col.lower() for col in frame.columns}
        frame.rename(columns=lower_cols, inplace=True)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        if self.config.start_date:
            start_ts = pd.to_datetime(self.config.start_date, utc=True)
            frame = frame[frame["timestamp"] >= start_ts]
        if self.config.end_date:
            end_ts = pd.to_datetime(self.config.end_date, utc=True)
            frame = frame[frame["timestamp"] <= end_ts]
        frame["symbol"] = symbol
        frame["date"] = frame["timestamp"].dt.floor("D")
        keep = ["timestamp", "date", "symbol", "open", "high", "low", "close", "volume"]
        return frame[keep].sort_values("date").reset_index(drop=True)

    def _load_forecasts(self, symbol: str) -> pd.DataFrame:
        candidates = [
            f"{symbol}.parquet",
            f"{symbol.replace('/', '_')}.parquet",
            f"{symbol.replace('-', '_')}.parquet",
            f"{symbol.replace('-', '').replace('/', '')}.parquet",
        ]
        path = None
        for name in candidates:
            candidate = Path(self.config.forecast_cache_dir) / name
            if candidate.exists():
                path = candidate
                break
        if path is None:
            LOGGER.warning(
                "No forecast cache for %s under %s; using fallback zeros.",
                symbol,
                self.config.forecast_cache_dir,
            )
            fallback_date = pd.Timestamp("1970-01-01", tz="UTC")
            return pd.DataFrame({"symbol": [symbol], "date": [fallback_date]})
        frame = pd.read_parquet(path)
        norm = {col: col.lower() for col in frame.columns}
        frame.rename(columns=norm, inplace=True)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame["date"] = frame["timestamp"].dt.floor("D")
        frame["symbol"] = symbol
        cols = ["symbol", "date", *[col for col in FORECAST_COLUMNS if col in frame.columns]]
        return frame[cols].sort_values("date").reset_index(drop=True)

    def _add_features(self, frame: pd.DataFrame, *, asset_flag: float) -> pd.DataFrame:
        work = frame.copy()
        work["return_1d"] = work["close"].pct_change().shift(1)
        work["return_5d"] = work["close"].pct_change(5).shift(1)
        work["return_21d"] = work["close"].pct_change(21).shift(1)
        work["volatility_5d"] = work["close"].pct_change().rolling(5).std().shift(1)
        work["volatility_21d"] = work["close"].pct_change().rolling(21).std().shift(1)
        work["range_pct"] = ((work["high"] - work["low"]) / work["close"].clip(lower=1e-6)).shift(1)
        work["volume_z"] = self._zscore(work["volume"], window=60).shift(1)
        work["day_sin"], work["day_cos"] = self._cycle_features(work["date"].dt.dayofweek, period=7)
        base = work["close"].shift(1).fillna(method="ffill").clip(lower=1e-6)
        work["chronos_close_delta"] = (work["predicted_close"] - base) / base
        work["chronos_high_delta"] = (work["predicted_high"] - base) / base
        work["chronos_low_delta"] = (base - work["predicted_low"]) / base
        work["chronos_high"] = work["predicted_high"].fillna(work["close"])
        work["chronos_low"] = work["predicted_low"].fillna(work["close"])
        work["reference_close"] = work["close"]
        work["chronos_move_pct"] = work["forecast_move_pct"].fillna(0.0)
        work["chronos_volatility_pct"] = work["forecast_volatility_pct"].fillna(0.0)
        if "predicted_close_p10" not in work.columns:
            work["predicted_close_p10"] = work["predicted_close"]
        if "predicted_close_p90" not in work.columns:
            work["predicted_close_p90"] = work["predicted_close"]
        work["predicted_close_p10"] = work["predicted_close_p10"].fillna(work["predicted_close"])
        work["predicted_close_p90"] = work["predicted_close_p90"].fillna(work["predicted_close"])
        work["chronos_quantile_spread"] = (
            work["predicted_close_p90"] - work["predicted_close_p10"]
        ).fillna(0.0)
        work["atr_pct_14"] = self._atr_percent(work).shift(1)
        work["asset_class"] = float(asset_flag)
        work.replace([np.inf, -np.inf], 0.0, inplace=True)
        return work

    @staticmethod
    def _zscore(series: pd.Series, window: int) -> pd.Series:
        mean = series.rolling(window, min_periods=1).mean()
        std = series.rolling(window, min_periods=1).std()
        std = std.replace(0.0, np.nan)
        return (series - mean) / std

    @staticmethod
    def _cycle_features(values: pd.Series, period: int) -> Tuple[pd.Series, pd.Series]:
        radians = 2 * math.pi * values / period
        return np.sin(radians), np.cos(radians)

    @staticmethod
    def _atr_percent(frame: pd.DataFrame, window: int = 14) -> pd.Series:
        close = frame["close"]
        prev_close = close.shift(1)
        components = pd.concat(
            [
                frame["high"] - frame["low"],
                (frame["high"] - prev_close).abs(),
                (frame["low"] - prev_close).abs(),
            ],
            axis=1,
        )
        tr = components.max(axis=1).fillna(0.0)
        atr = tr.rolling(window, min_periods=1).mean()
        return atr / close.clip(lower=1e-6)


class DailySymbolDataset(Dataset):
    """Sliding-window dataset for a single symbol."""

    def __init__(
        self,
        frame: pd.DataFrame,
        features: np.ndarray,
        sequence_length: int,
        *,
        asset_flag: float,
    ) -> None:
        if len(frame) != len(features):
            raise ValueError("Feature matrix must align with base frame length.")
        if len(frame) < sequence_length:
            raise ValueError("Frame shorter than sequence length.")
        self.frame = frame.reset_index(drop=True)
        self.features = features.astype(np.float32)
        self.seq_len = int(sequence_length)
        self.asset_flag = float(asset_flag)
        self.high = frame["high"].to_numpy(dtype=np.float32)
        self.low = frame["low"].to_numpy(dtype=np.float32)
        self.close = frame["close"].to_numpy(dtype=np.float32)
        self.reference_close = frame["reference_close"].to_numpy(dtype=np.float32)
        self.chronos_high = frame["chronos_high"].to_numpy(dtype=np.float32)
        self.chronos_low = frame["chronos_low"].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.frame) - self.seq_len + 1

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx
        end = idx + self.seq_len
        return {
            "features": torch.from_numpy(self.features[start:end]),
            "high": torch.from_numpy(self.high[start:end]),
            "low": torch.from_numpy(self.low[start:end]),
            "close": torch.from_numpy(self.close[start:end]),
            "reference_close": torch.from_numpy(self.reference_close[start:end]),
            "chronos_high": torch.from_numpy(self.chronos_high[start:end]),
            "chronos_low": torch.from_numpy(self.chronos_low[start:end]),
            "asset_class": torch.tensor(self.asset_flag, dtype=torch.float32),
        }


class MultiSymbolDataset(Dataset):
    """Concatenates multiple DailySymbolDatasets for joint training."""

    def __init__(self, datasets: List[DailySymbolDataset]) -> None:
        if not datasets:
            raise ValueError("At least one dataset is required.")
        self.datasets = datasets
        self.lengths = [len(ds) for ds in datasets]
        self.offsets = np.cumsum([0, *self.lengths])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        dataset_idx = np.searchsorted(self.offsets, idx, side="right") - 1
        base = self.offsets[dataset_idx]
        return self.datasets[dataset_idx][idx - base]


class DailyDataModule:
    """Prepares multi-symbol datasets for neural policy training."""

    def __init__(self, config: DailyDatasetConfig) -> None:
        self.config = config
        self.sequence_length = config.sequence_length
        self.feature_columns = tuple(config.feature_columns or DEFAULT_FEATURES)
        self.symbols = [symbol.upper() for symbol in config.symbols]
        if not self.symbols:
            raise ValueError("No symbols configured for daily training.")
        self._builder = SymbolFrameBuilder(config, self.feature_columns)
        self._symbol_frames: Dict[str, pd.DataFrame] = {}
        self.train_dataset: Optional[MultiSymbolDataset] = None
        self.val_dataset: Optional[MultiSymbolDataset] = None
        self.normalizer: Optional[FeatureNormalizer] = None
        self._prepare()

    # ------------------------------------------------------------------
    def train_dataloader(self, batch_size: int, num_workers: int = 0) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("DailyDataModule not initialised.")
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )

    def val_dataloader(self, batch_size: int, num_workers: int = 0) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("DailyDataModule not initialised.")
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )

    # ------------------------------------------------------------------
    def _prepare(self) -> None:
        symbol_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        train_matrices: List[np.ndarray] = []
        train_datasets: List[DailySymbolDataset] = []
        val_datasets: List[DailySymbolDataset] = []

        for symbol in self.symbols:
            frame = self._build_symbol_frame(symbol)
            if len(frame) < max(self.config.min_history_days, self.sequence_length + 5):
                raise ValueError(f"{symbol} only has {len(frame)} rows; need >= {self.config.min_history_days}.")
            self._symbol_frames[symbol] = frame
            train_frame, val_frame = self._split_frame(frame)
            train_features = train_frame[list(self.feature_columns)].to_numpy(dtype=np.float32)
            asset_flag = float(frame["asset_class"].iloc[0]) if not frame.empty else 0.0
            symbol_data[symbol] = {"train": train_frame, "val": val_frame, "asset_flag": asset_flag}
            train_matrices.append(train_features)

        stacked = np.concatenate(train_matrices, axis=0)
        self.normalizer = FeatureNormalizer.fit(stacked)

        for symbol, frames in symbol_data.items():
            train_frame = frames["train"]
            val_frame = frames["val"]
            train_norm = self.normalizer.transform(train_frame[list(self.feature_columns)].to_numpy(dtype=np.float32))
            val_norm = self.normalizer.transform(val_frame[list(self.feature_columns)].to_numpy(dtype=np.float32))
            asset_flag = frames.get("asset_flag", 0.0)
            train_datasets.append(DailySymbolDataset(train_frame, train_norm, self.sequence_length, asset_flag=asset_flag))
            val_datasets.append(DailySymbolDataset(val_frame, val_norm, self.sequence_length, asset_flag=asset_flag))

        self.train_dataset = MultiSymbolDataset(train_datasets)
        self.val_dataset = MultiSymbolDataset(val_datasets)

    def _split_frame(self, frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        total = len(frame)

        # Special case: validation_days=0 means use ALL data for training
        if self.config.validation_days == 0:
            # Use full frame for training, create minimal validation set from end
            train = frame.reset_index(drop=True)
            # Validation set is just the last sequence (needed to keep trainer happy)
            val = frame.iloc[-(self.sequence_length + 1):].reset_index(drop=True)
            return train, val

        min_tail = max(self.sequence_length + 1, self.config.validation_days)
        dyn_tail = int(total * self.config.val_fraction)
        tail = max(min_tail, dyn_tail)
        tail = min(tail, total - self.sequence_length - 1)
        if tail <= 0:
            raise ValueError("Validation tail is non-positive; reduce sequence length or validation_days.")
        split_idx = total - tail
        train = frame.iloc[:split_idx].reset_index(drop=True)
        val = frame.iloc[split_idx - self.sequence_length :].reset_index(drop=True)
        return train, val

    def _build_symbol_frame(self, symbol: str) -> pd.DataFrame:
        return self._builder.build(symbol)


__all__ = [
    "DailyDataModule",
    "DailySymbolDataset",
    "MultiSymbolDataset",
    "FeatureNormalizer",
    "SymbolFrameBuilder",
]
