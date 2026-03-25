from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from RLgpt.config import DailyPlanDataConfig
from binanceneural.config import DatasetConfig
from binanceneural.data import BinanceHourlyDataModule

BASE_DAILY_FEATURES: tuple[str, ...] = (
    "open_gap_pct",
    "prev_return_24h",
    "prev_volatility_24h",
    "prev_range_pct",
    "prev_volume_z",
)


def build_daily_feature_names(horizons: Iterable[int]) -> List[str]:
    names = list(BASE_DAILY_FEATURES)
    for horizon in horizons:
        suffix = f"_h{int(horizon)}"
        names.extend(
            [
                f"prev_chronos_close_delta{suffix}",
                f"prev_chronos_high_delta{suffix}",
                f"prev_chronos_low_delta{suffix}",
            ]
        )
    return names


@dataclass(frozen=True)
class TensorNormalizer:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, features: torch.Tensor) -> "TensorNormalizer":
        if features.ndim != 3:
            raise ValueError(f"Expected [days, assets, features], got shape {tuple(features.shape)}")
        mean = features.mean(dim=(0, 1))
        std = features.std(dim=(0, 1), unbiased=False)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        return cls(mean=mean, std=std)

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.mean) / self.std

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "mean": [float(value) for value in self.mean.detach().cpu().tolist()],
            "std": [float(value) for value in self.std.detach().cpu().tolist()],
        }


@dataclass(frozen=True)
class DailyPlanTensors:
    symbols: tuple[str, ...]
    feature_names: tuple[str, ...]
    days: tuple[pd.Timestamp, ...]
    features: torch.Tensor
    daily_anchor: torch.Tensor
    prev_close: torch.Tensor
    hourly_open: torch.Tensor
    hourly_high: torch.Tensor
    hourly_low: torch.Tensor
    hourly_close: torch.Tensor
    hourly_mask: torch.Tensor

    def __len__(self) -> int:
        return int(self.features.shape[0])

    @property
    def num_assets(self) -> int:
        return int(self.features.shape[1])

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[2])

    def slice(self, start: int, end: int | None) -> "DailyPlanTensors":
        return DailyPlanTensors(
            symbols=self.symbols,
            feature_names=self.feature_names,
            days=self.days[start:end],
            features=self.features[start:end],
            daily_anchor=self.daily_anchor[start:end],
            prev_close=self.prev_close[start:end],
            hourly_open=self.hourly_open[start:end],
            hourly_high=self.hourly_high[start:end],
            hourly_low=self.hourly_low[start:end],
            hourly_close=self.hourly_close[start:end],
            hourly_mask=self.hourly_mask[start:end],
        )

    def train_val_split(self, validation_days: int) -> tuple["DailyPlanTensors", "DailyPlanTensors"]:
        if len(self) < 2:
            raise ValueError("At least two aligned trading days are required for train/validation split.")
        val_days = max(1, int(validation_days))
        if val_days >= len(self):
            val_days = 1
        split_idx = len(self) - val_days
        if split_idx <= 0:
            raise ValueError("Validation split leaves no training days.")
        return self.slice(0, split_idx), self.slice(split_idx, None)

    def with_features(self, features: torch.Tensor) -> "DailyPlanTensors":
        return replace(self, features=features)


class DailyPlanTensorDataset(Dataset):
    def __init__(self, bundle: DailyPlanTensors) -> None:
        self.bundle = bundle

    def __len__(self) -> int:
        return len(self.bundle)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "features": self.bundle.features[idx],
            "daily_anchor": self.bundle.daily_anchor[idx],
            "prev_close": self.bundle.prev_close[idx],
            "hourly_open": self.bundle.hourly_open[idx],
            "hourly_high": self.bundle.hourly_high[idx],
            "hourly_low": self.bundle.hourly_low[idx],
            "hourly_close": self.bundle.hourly_close[idx],
            "hourly_mask": self.bundle.hourly_mask[idx],
        }


def load_symbol_hourly_feature_frame(symbol: str, config: DailyPlanDataConfig) -> pd.DataFrame:
    dataset_config = DatasetConfig(
        symbol=str(symbol).upper(),
        data_root=config.data_root,
        forecast_cache_root=config.forecast_cache_root,
        forecast_horizons=tuple(int(value) for value in config.forecast_horizons),
        sequence_length=max(8, int(config.sequence_length)),
        min_history_hours=max(24, int(config.min_history_hours)),
        max_feature_lookback_hours=max(24, int(config.max_feature_lookback_hours)),
        validation_days=max(1, int(config.validation_days)),
        cache_only=bool(config.cache_only),
    )
    module = BinanceHourlyDataModule(dataset_config)
    return module.frame.copy()


def prepare_daily_plan_tensors(config: DailyPlanDataConfig) -> DailyPlanTensors:
    symbols = _normalize_symbols(config.symbols)
    if not symbols:
        raise ValueError("At least one symbol is required.")

    feature_names = tuple(build_daily_feature_names(config.forecast_horizons))
    per_symbol_records: dict[str, Dict[pd.Timestamp, dict[str, np.ndarray | float]]] = {}
    for symbol in symbols:
        frame = load_symbol_hourly_feature_frame(symbol, config)
        per_symbol_records[symbol] = _build_symbol_daily_records(
            frame,
            horizons=config.forecast_horizons,
            min_bars_per_day=config.min_bars_per_day,
        )
        if not per_symbol_records[symbol]:
            raise ValueError(f"Symbol {symbol} produced no daily plan records.")

    common_days = sorted(set.intersection(*(set(records.keys()) for records in per_symbol_records.values())))
    if len(common_days) < 2:
        raise ValueError("Need at least two common trading days across all symbols.")

    max_bars = max(
        len(per_symbol_records[symbol][day]["hourly_open"]) for symbol in symbols for day in common_days
    )
    num_days = len(common_days)
    num_assets = len(symbols)
    num_features = len(feature_names)

    features = np.zeros((num_days, num_assets, num_features), dtype=np.float32)
    daily_anchor = np.zeros((num_days, num_assets), dtype=np.float32)
    prev_close = np.zeros((num_days, num_assets), dtype=np.float32)
    hourly_open = np.zeros((num_days, max_bars, num_assets), dtype=np.float32)
    hourly_high = np.zeros((num_days, max_bars, num_assets), dtype=np.float32)
    hourly_low = np.zeros((num_days, max_bars, num_assets), dtype=np.float32)
    hourly_close = np.zeros((num_days, max_bars, num_assets), dtype=np.float32)
    hourly_mask = np.zeros((num_days, max_bars, num_assets), dtype=np.float32)

    for day_idx, day in enumerate(common_days):
        for asset_idx, symbol in enumerate(symbols):
            record = per_symbol_records[symbol][day]
            day_features = np.asarray(record["features"], dtype=np.float32)
            features[day_idx, asset_idx, :] = day_features
            daily_anchor[day_idx, asset_idx] = float(record["daily_anchor"])
            prev_close[day_idx, asset_idx] = float(record["prev_close"])

            bars = len(record["hourly_open"])
            hourly_open[day_idx, :bars, asset_idx] = np.asarray(record["hourly_open"], dtype=np.float32)
            hourly_high[day_idx, :bars, asset_idx] = np.asarray(record["hourly_high"], dtype=np.float32)
            hourly_low[day_idx, :bars, asset_idx] = np.asarray(record["hourly_low"], dtype=np.float32)
            hourly_close[day_idx, :bars, asset_idx] = np.asarray(record["hourly_close"], dtype=np.float32)
            hourly_mask[day_idx, :bars, asset_idx] = 1.0

    return DailyPlanTensors(
        symbols=symbols,
        feature_names=feature_names,
        days=tuple(common_days),
        features=torch.from_numpy(features),
        daily_anchor=torch.from_numpy(daily_anchor),
        prev_close=torch.from_numpy(prev_close),
        hourly_open=torch.from_numpy(hourly_open),
        hourly_high=torch.from_numpy(hourly_high),
        hourly_low=torch.from_numpy(hourly_low),
        hourly_close=torch.from_numpy(hourly_close),
        hourly_mask=torch.from_numpy(hourly_mask),
    )


def _normalize_symbols(symbols: Iterable[str]) -> tuple[str, ...]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        token = str(symbol or "").strip().upper()
        if not token or token in seen:
            continue
        seen.add(token)
        cleaned.append(token)
    return tuple(cleaned)


def _build_symbol_daily_records(
    frame: pd.DataFrame,
    *,
    horizons: Iterable[int],
    min_bars_per_day: int,
) -> Dict[pd.Timestamp, dict[str, np.ndarray | float]]:
    required = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "return_24h",
        "volatility_24h",
        "range_pct",
        "volume_z",
    }
    for horizon in horizons:
        suffix = f"_h{int(horizon)}"
        required.update(
            {
                f"chronos_close_delta{suffix}",
                f"chronos_high_delta{suffix}",
                f"chronos_low_delta{suffix}",
            }
        )
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns for daily plan data: {missing}")

    normalized = frame.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True)
    normalized = normalized.sort_values("timestamp").reset_index(drop=True)
    normalized["day"] = normalized["timestamp"].dt.floor("D")
    grouped = [group.reset_index(drop=True) for _, group in normalized.groupby("day", sort=True)]

    records: Dict[pd.Timestamp, dict[str, np.ndarray | float]] = {}
    for prev_group, day_group in zip(grouped[:-1], grouped[1:]):
        if len(day_group) < max(1, int(min_bars_per_day)):
            continue
        prev_row = prev_group.iloc[-1]
        first_row = day_group.iloc[0]

        prev_close = float(prev_row["close"])
        daily_anchor = float(first_row["open"])
        if not np.isfinite(prev_close) or not np.isfinite(daily_anchor) or prev_close <= 0 or daily_anchor <= 0:
            continue

        feature_values = [
            (daily_anchor - prev_close) / prev_close,
            float(prev_row["return_24h"]),
            float(prev_row["volatility_24h"]),
            float(prev_row["range_pct"]),
            float(prev_row["volume_z"]),
        ]
        for horizon in horizons:
            suffix = f"_h{int(horizon)}"
            feature_values.extend(
                [
                    float(prev_row[f"chronos_close_delta{suffix}"]),
                    float(prev_row[f"chronos_high_delta{suffix}"]),
                    float(prev_row[f"chronos_low_delta{suffix}"]),
                ]
            )
        features = np.asarray(feature_values, dtype=np.float32)
        if not np.all(np.isfinite(features)):
            continue

        hourly = day_group[["open", "high", "low", "close"]].to_numpy(dtype=np.float32)
        if not np.isfinite(hourly).all():
            continue

        records[pd.Timestamp(first_row["day"])] = {
            "features": features,
            "daily_anchor": daily_anchor,
            "prev_close": prev_close,
            "hourly_open": hourly[:, 0],
            "hourly_high": hourly[:, 1],
            "hourly_low": hourly[:, 2],
            "hourly_close": hourly[:, 3],
        }
    return records
