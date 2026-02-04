from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from binanceexp1.data import (
    BinanceExp1Dataset,
    FeatureNormalizer,
    build_default_feature_columns,
    build_feature_frame,
)
from binanceneural.forecasts import build_forecast_bundle
from src.fees import get_fee_for_symbol
from src.symbol_utils import is_crypto_symbol

from .config import DatasetConfig


@dataclass
class AssetMeta:
    symbol: str
    asset_class: str
    maker_fee: float
    periods_per_year: float


class AlpacaHourlyDataModule:
    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        if config.feature_columns is None:
            self.feature_columns = tuple(build_default_feature_columns(config))
        else:
            self.feature_columns = tuple(config.feature_columns)
        self.primary_horizon = int(config.forecast_horizons[0])
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

        self.train_dataset = BinanceExp1Dataset(
            train_frame,
            norm_train,
            config.sequence_length,
            primary_horizon=self.primary_horizon,
        )
        self.val_dataset = BinanceExp1Dataset(
            val_frame,
            norm_val,
            config.sequence_length,
            primary_horizon=self.primary_horizon,
        )

        self.asset_meta = self._build_asset_meta(self.frame, config.symbol)

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
        price_frame = self._load_price_history()
        forecast_frame = self._load_forecasts()
        merged = price_frame.merge(forecast_frame, on=["timestamp", "symbol"], how="inner")
        merged = merged.sort_values("timestamp").reset_index(drop=True)
        enriched = build_feature_frame(
            merged,
            config=self.config,
        )
        enriched = enriched.dropna(subset=list(self.feature_columns))
        return enriched.reset_index(drop=True)

    def _load_price_history(self) -> pd.DataFrame:
        data_root = self.config.resolved_data_root()
        path = Path(data_root) / f"{self.config.symbol.upper()}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing hourly dataset {path}")
        frame = pd.read_csv(path)
        frame.columns = [col.lower() for col in frame.columns]
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        if "symbol" not in frame.columns:
            frame["symbol"] = self.config.symbol.upper()
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
        cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        return frame[cols].sort_values("timestamp").reset_index(drop=True)

    def _load_forecasts(self) -> pd.DataFrame:
        horizons = tuple(int(h) for h in self.config.forecast_horizons)
        return build_forecast_bundle(
            symbol=self.config.symbol.upper(),
            data_root=Path(self.config.resolved_data_root()),
            cache_root=Path(self.config.forecast_cache_root),
            horizons=horizons,
            context_hours=24 * 14,
            quantile_levels=(0.1, 0.5, 0.9),
            batch_size=128,
            cache_only=self.config.cache_only,
        )

    @staticmethod
    def _build_asset_meta(frame: pd.DataFrame, symbol: str) -> AssetMeta:
        symbol = symbol.upper()
        asset_class = "crypto" if is_crypto_symbol(symbol) else "stock"
        maker_fee = float(get_fee_for_symbol(symbol))
        periods_per_year = _infer_periods_per_year(frame["timestamp"], asset_class)
        return AssetMeta(
            symbol=symbol,
            asset_class=asset_class,
            maker_fee=maker_fee,
            periods_per_year=periods_per_year,
        )


class MultiSymbolDataset:
    def __init__(self, datasets: List[BinanceExp1Dataset]) -> None:
        self.datasets = datasets
        self.lengths = [len(ds) for ds in datasets]
        self.cumulative_lengths = [0] + list(np.cumsum(self.lengths))

    def __len__(self) -> int:
        return sum(self.lengths)

    def __getitem__(self, idx: int):
        for i, (start, end) in enumerate(zip(self.cumulative_lengths[:-1], self.cumulative_lengths[1:])):
            if start <= idx < end:
                return self.datasets[i][idx - start]
        raise IndexError(f"Index {idx} out of range")


class AlpacaMultiSymbolDataModule:
    def __init__(self, symbols: Sequence[str], config: DatasetConfig) -> None:
        cleaned: List[str] = []
        seen: set[str] = set()
        for symbol in symbols:
            if not symbol:
                continue
            token = symbol.upper()
            if token not in seen:
                cleaned.append(token)
                seen.add(token)
        if not cleaned:
            raise ValueError("At least one symbol is required for multi-symbol training.")

        target_symbol = (config.symbol or cleaned[0]).upper()
        if target_symbol not in seen:
            cleaned.insert(0, target_symbol)
            seen.add(target_symbol)
        else:
            cleaned = [target_symbol] + [sym for sym in cleaned if sym != target_symbol]

        self.symbols = cleaned
        self.target_symbol = target_symbol
        self.base_config = config
        self.modules: Dict[str, AlpacaHourlyDataModule] = {}
        self.normalizers: Dict[str, FeatureNormalizer] = {}
        self.asset_meta_by_symbol: Dict[str, AssetMeta] = {}

        for symbol in self.symbols:
            symbol_config = DatasetConfig(
                symbol=symbol,
                data_root=config.data_root,
                forecast_cache_root=config.forecast_cache_root,
                forecast_horizons=config.forecast_horizons,
                sequence_length=config.sequence_length,
                val_fraction=config.val_fraction,
                min_history_hours=config.min_history_hours,
                max_feature_lookback_hours=config.max_feature_lookback_hours,
                feature_columns=config.feature_columns,
                refresh_hours=config.refresh_hours,
                validation_days=config.validation_days,
                cache_only=config.cache_only,
                moving_average_windows=config.moving_average_windows,
                ema_windows=config.ema_windows,
                atr_windows=config.atr_windows,
                volume_z_window=config.volume_z_window,
                volume_shock_window=config.volume_shock_window,
                volume_spike_z=config.volume_spike_z,
                trend_windows=config.trend_windows,
                drawdown_windows=config.drawdown_windows,
                vol_regime_short=config.vol_regime_short,
                vol_regime_long=config.vol_regime_long,
                rsi_window=config.rsi_window,
                allow_mixed_asset_class=config.allow_mixed_asset_class,
            )
            module = AlpacaHourlyDataModule(symbol_config)
            self.modules[symbol] = module
            self.normalizers[symbol] = module.normalizer
            self.asset_meta_by_symbol[symbol] = module.asset_meta

        asset_classes = {meta.asset_class for meta in self.asset_meta_by_symbol.values()}
        if len(asset_classes) > 1 and not config.allow_mixed_asset_class:
            raise ValueError(
                "Mixed asset classes detected. Set allow_mixed_asset_class=True if you intend to mix crypto + stocks."
            )

        train_datasets = [mod.train_dataset for mod in self.modules.values()]
        self.train_dataset = MultiSymbolDataset(train_datasets)

        target_module = self.modules[self.target_symbol]
        self.val_dataset = target_module.val_dataset
        self.normalizer = target_module.normalizer
        self.feature_columns = target_module.feature_columns
        self.frame = target_module.frame

        self.asset_meta = self.asset_meta_by_symbol[self.target_symbol]
        self.periods_per_year = _weighted_periods_per_year(self.asset_meta_by_symbol, train_datasets)
        self.maker_fee = float(self.asset_meta.maker_fee)
        self.asset_class = self.asset_meta.asset_class

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


def _infer_periods_per_year(timestamps: Iterable[pd.Timestamp], asset_class: str) -> float:
    if asset_class == "crypto":
        return float(24 * 365)
    ts = pd.to_datetime(pd.Series(timestamps), utc=True, errors="coerce")
    ts = ts.dropna()
    if ts.empty:
        return float(252 * 7)
    try:
        ts_local = ts.dt.tz_convert("America/New_York")
    except Exception:
        ts_local = ts
    counts = ts_local.dt.date.value_counts()
    avg_bars = float(counts.mean()) if not counts.empty else 0.0
    if not math.isfinite(avg_bars) or avg_bars <= 0:
        avg_bars = 7.0
    avg_bars = min(max(avg_bars, 1.0), 24.0)
    return float(avg_bars * 252)


def _weighted_periods_per_year(asset_meta: Dict[str, AssetMeta], datasets: List[BinanceExp1Dataset]) -> float:
    total = 0.0
    weighted = 0.0
    for symbol, dataset in zip(asset_meta.keys(), datasets):
        count = float(len(dataset))
        meta = asset_meta[symbol]
        weighted += meta.periods_per_year * count
        total += count
    if total <= 0:
        return float(next(iter(asset_meta.values())).periods_per_year)
    return float(weighted / total)


__all__ = [
    "AlpacaHourlyDataModule",
    "AlpacaMultiSymbolDataModule",
    "AssetMeta",
    "FeatureNormalizer",
    "build_default_feature_columns",
    "build_feature_frame",
]
