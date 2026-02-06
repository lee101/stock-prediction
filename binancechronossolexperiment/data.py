from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from binanceneural.data import (
    BinanceHourlyDataset,
    FeatureNormalizer,
    build_default_feature_columns,
    build_feature_frame,
)

from .forecasts import build_forecast_bundle


@dataclass(frozen=True)
class SplitConfig:
    val_days: int = 20
    test_days: int = 10


class ChronosSolDataModule:
    def __init__(
        self,
        *,
        symbol: str,
        data_root: Path,
        forecast_cache_root: Path,
        forecast_horizons: Sequence[int],
        context_hours: int,
        quantile_levels: Sequence[float],
        batch_size: int,
        model_id: str,
        sequence_length: int,
        split_config: SplitConfig,
        max_feature_lookback_hours: int = 24 * 7,
        min_history_hours: int = 24 * 30,
        max_history_days: Optional[int] = None,
        feature_columns: Optional[Sequence[str]] = None,
        cache_only: bool = False,
        preaugmentation_dirs: Optional[Sequence[Path]] = None,
        device_map: str = "cuda",
    ) -> None:
        self.symbol = symbol.upper()
        self.data_root = Path(data_root)
        self.forecast_cache_root = Path(forecast_cache_root)
        self.forecast_horizons = tuple(int(h) for h in forecast_horizons)
        self.context_hours = int(context_hours)
        self.quantile_levels = tuple(float(q) for q in quantile_levels)
        self.batch_size = int(batch_size)
        self.model_id = str(model_id)
        self.sequence_length = int(sequence_length)
        self.split_config = split_config
        self.max_feature_lookback_hours = int(max_feature_lookback_hours)
        self.min_history_hours = int(min_history_hours)
        self.max_history_days = max_history_days
        self.cache_only = bool(cache_only)
        self.preaugmentation_dirs = tuple(preaugmentation_dirs) if preaugmentation_dirs else None
        self.device_map = device_map

        if feature_columns is None:
            self.feature_columns = tuple(build_default_feature_columns(self.forecast_horizons))
        else:
            self.feature_columns = tuple(feature_columns)

        frame = self._prepare_frame()
        if len(frame) < self.min_history_hours:
            raise ValueError(
                f"Insufficient hourly history ({len(frame)} rows, minimum {self.min_history_hours})."
            )

        (
            train_frame,
            val_frame,
            test_frame,
            self.val_window_start,
            self.test_window_start,
        ) = split_frame_by_days(
            frame,
            sequence_length=self.sequence_length,
            val_days=self.split_config.val_days,
            test_days=self.split_config.test_days,
        )
        self.train_frame = train_frame
        self.val_frame = val_frame
        self.test_frame = test_frame
        self.full_frame = frame

        feature_cols = list(self.feature_columns)
        train_features = train_frame[feature_cols].to_numpy(dtype=np.float32)
        self.normalizer = FeatureNormalizer.fit(train_features)

        norm_train = self.normalizer.transform(train_features)
        norm_val = self.normalizer.transform(val_frame[feature_cols].to_numpy(dtype=np.float32))
        norm_test = self.normalizer.transform(test_frame[feature_cols].to_numpy(dtype=np.float32))

        primary_horizon = int(self.forecast_horizons[0])
        self.train_dataset = BinanceHourlyDataset(
            train_frame,
            norm_train,
            self.sequence_length,
            primary_horizon=primary_horizon,
        )
        self.val_dataset = BinanceHourlyDataset(
            val_frame,
            norm_val,
            self.sequence_length,
            primary_horizon=primary_horizon,
        )
        self.test_dataset = BinanceHourlyDataset(
            test_frame,
            norm_test,
            self.sequence_length,
            primary_horizon=primary_horizon,
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

    def test_dataloader(self, batch_size: int, num_workers: int = 0) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
        )

    # ------------------------------------------------------------------
    def _prepare_frame(self) -> pd.DataFrame:
        price_frame = self._load_price_history()
        end_ts = pd.to_datetime(price_frame["timestamp"], utc=True, errors="coerce").max()
        start_ts = None
        if self.max_history_days and self.max_history_days > 0:
            lookback_hours = int(self.max_history_days) * 24 + max(0, int(self.max_feature_lookback_hours))
            if end_ts is not None and not pd.isna(end_ts) and lookback_hours > 0:
                start_ts = end_ts - pd.Timedelta(hours=float(lookback_hours))

        forecast_frame = self._load_forecasts(start=start_ts, end=end_ts)
        merged = price_frame.merge(forecast_frame, on=["timestamp", "symbol"], how="inner")
        merged = merged.sort_values("timestamp").reset_index(drop=True)
        enriched = build_feature_frame(
            merged,
            horizons=self.forecast_horizons,
            max_lookback=self.max_feature_lookback_hours,
        )
        enriched = enriched.dropna(subset=list(self.feature_columns))
        if self.max_history_days:
            max_rows = int(self.max_history_days) * 24
            if max_rows > 0 and len(enriched) > max_rows:
                enriched = enriched.iloc[-max_rows:].copy()
        return enriched.reset_index(drop=True)

    def _load_price_history(self) -> pd.DataFrame:
        path = self.data_root / f"{self.symbol}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing hourly dataset {path}")
        frame = pd.read_csv(path)
        frame.columns = [col.lower() for col in frame.columns]
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame["symbol"] = self.symbol
        cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        return frame[cols].sort_values("timestamp").reset_index(drop=True)

    def _load_forecasts(
        self,
        *,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
    ) -> pd.DataFrame:
        return build_forecast_bundle(
            symbol=self.symbol,
            data_root=self.data_root,
            cache_root=self.forecast_cache_root,
            horizons=self.forecast_horizons,
            context_hours=self.context_hours,
            quantile_levels=self.quantile_levels,
            batch_size=self.batch_size,
            model_id=self.model_id,
            device_map=self.device_map,
            cache_only=self.cache_only,
            start=start,
            end=end,
            preaugmentation_dirs=self.preaugmentation_dirs,
        )


def split_frame_by_days(
    frame: pd.DataFrame,
    *,
    sequence_length: int,
    val_days: int,
    test_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    if val_days <= 0 or test_days <= 0:
        raise ValueError("val_days and test_days must be positive")
    if "timestamp" not in frame.columns:
        raise KeyError("Frame must include a timestamp column")

    frame = frame.sort_values("timestamp").reset_index(drop=True)
    total_hours = len(frame)
    val_hours = int(val_days * 24)
    test_hours = int(test_days * 24)
    required = sequence_length + val_hours + test_hours + 1
    if total_hours <= required:
        raise ValueError(
            f"Not enough history ({total_hours} rows) for val_days={val_days}, test_days={test_days}, "
            f"sequence_length={sequence_length}."
        )

    test_start = total_hours - test_hours
    val_start = test_start - val_hours
    if val_start <= sequence_length:
        raise ValueError("Not enough history for requested splits after sequence overlap.")

    train_frame = frame.iloc[:val_start].copy()
    val_frame = frame.iloc[val_start - sequence_length : test_start].copy()
    test_frame = frame.iloc[test_start - sequence_length :].copy()

    val_start_ts = frame.iloc[val_start]["timestamp"]
    test_start_ts = frame.iloc[test_start]["timestamp"]
    return train_frame, val_frame, test_frame, val_start_ts, test_start_ts


__all__ = ["ChronosSolDataModule", "SplitConfig", "split_frame_by_days"]
