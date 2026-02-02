from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from binanceneural.forecasts import build_forecast_bundle

from .config import DatasetConfig

BASE_FEATURES: Tuple[str, ...] = (
    "return_1h",
    "return_4h",
    "return_24h",
    "log_return_1h",
    "volatility_24h",
    "volatility_168h",
    "range_pct",
    "close_open_pct",
    "high_close_pct",
    "low_close_pct",
    "volume_z",
    "volume_change_1h",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
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


class BinanceExp1Dataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        features: np.ndarray,
        sequence_length: int,
        *,
        primary_horizon: int,
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
        self.primary_horizon = int(primary_horizon)
        self.chronos_high = frame[f"predicted_high_p50_h{self.primary_horizon}"].to_numpy(dtype=np.float32)
        self.chronos_low = frame[f"predicted_low_p50_h{self.primary_horizon}"].to_numpy(dtype=np.float32)

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
            "chronos_high": torch.from_numpy(self.chronos_high[start:end]),
            "chronos_low": torch.from_numpy(self.chronos_low[start:end]),
        }


class BinanceExp1DataModule:
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
        path = Path(self.config.data_root) / f"{self.config.symbol.upper()}.csv"
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
            data_root=Path(self.config.data_root),
            cache_root=Path(self.config.forecast_cache_root),
            horizons=horizons,
            context_hours=24 * 14,
            quantile_levels=(0.1, 0.5, 0.9),
            batch_size=128,
            cache_only=self.config.cache_only,
        )


class MultiSymbolDataset(Dataset):
    def __init__(self, datasets: List[BinanceExp1Dataset]) -> None:
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
        self.modules: Dict[str, BinanceExp1DataModule] = {}
        self.normalizers: Dict[str, FeatureNormalizer] = {}

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
            )
            module = BinanceExp1DataModule(symbol_config)
            self.modules[symbol] = module
            self.normalizers[symbol] = module.normalizer

        train_datasets = [mod.train_dataset for mod in self.modules.values()]
        self.train_dataset = MultiSymbolDataset(train_datasets)

        target_module = self.modules[self.target_symbol]
        self.val_dataset = target_module.val_dataset
        self.normalizer = target_module.normalizer
        self.feature_columns = target_module.feature_columns
        self.frame = target_module.frame

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

def build_default_feature_columns(config: DatasetConfig) -> List[str]:
    columns = list(BASE_FEATURES)
    for window in config.trend_windows:
        columns.append(f"trend_{int(window)}h")
    for window in config.drawdown_windows:
        columns.append(f"drawdown_{int(window)}h")
    columns.append(f"vol_regime_{int(config.vol_regime_short)}_{int(config.vol_regime_long)}")
    columns.append(f"volume_shock_{int(config.volume_shock_window)}h")
    columns.append("volume_spike")
    for window in config.moving_average_windows:
        columns.append(f"ma_delta_{int(window)}h")
    for window in config.ema_windows:
        columns.append(f"ema_delta_{int(window)}h")
    for window in config.atr_windows:
        columns.append(f"atr_pct_{int(window)}h")
    columns.append(f"rsi_{int(config.rsi_window)}h")
    for horizon in config.forecast_horizons:
        suffix = f"_h{int(horizon)}"
        columns.extend(
            [
                f"chronos_close_delta{suffix}",
                f"chronos_high_delta{suffix}",
                f"chronos_low_delta{suffix}",
            ]
        )
    return columns


def build_feature_frame(frame: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    frame = frame.copy()
    frame["reference_close"] = frame["close"].astype(float)
    close = frame["close"].astype(float)
    open_ = frame["open"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    volume = frame["volume"].astype(float)

    frame["return_1h"] = close.pct_change(1)
    frame["return_4h"] = close.pct_change(4)
    frame["return_24h"] = close.pct_change(24)
    frame["log_return_1h"] = np.log(close).diff()
    frame["volatility_24h"] = frame["return_1h"].rolling(24).std()
    frame["volatility_168h"] = frame["return_1h"].rolling(168).std()

    frame["range_pct"] = (high - low).abs() / close.replace(0.0, np.nan)
    frame["close_open_pct"] = (close - open_) / open_.replace(0.0, np.nan)
    frame["high_close_pct"] = (high - close) / close.replace(0.0, np.nan)
    frame["low_close_pct"] = (close - low) / close.replace(0.0, np.nan)

    frame["volume_z"] = _zscore(volume, window=max(2, int(config.volume_z_window)))
    frame["volume_change_1h"] = volume.pct_change(1)

    for window in config.trend_windows:
        window = int(window)
        frame[f"trend_{window}h"] = close / close.shift(window).replace(0.0, np.nan) - 1.0

    for window in config.drawdown_windows:
        window = int(window)
        rolling_max = close.rolling(window).max()
        frame[f"drawdown_{window}h"] = close / rolling_max.replace(0.0, np.nan) - 1.0

    vol_short = int(config.vol_regime_short)
    vol_long = int(config.vol_regime_long)
    if vol_short <= 1 or vol_long <= 1:
        raise ValueError("vol_regime_short and vol_regime_long must be >= 2.")
    if vol_short == 24:
        short_vol = frame["volatility_24h"]
    else:
        short_vol = frame["return_1h"].rolling(vol_short).std()
    if vol_long == 168:
        long_vol = frame["volatility_168h"]
    else:
        long_vol = frame["return_1h"].rolling(vol_long).std()
    frame[f"vol_regime_{vol_short}_{vol_long}"] = short_vol / long_vol.replace(0.0, np.nan)

    shock_window = max(2, int(config.volume_shock_window))
    volume_mean = volume.rolling(shock_window).mean()
    frame[f"volume_shock_{shock_window}h"] = volume / volume_mean.replace(0.0, np.nan) - 1.0
    frame["volume_spike"] = (frame["volume_z"] > float(config.volume_spike_z)).astype(float)

    hours = frame["timestamp"].dt.hour
    dow = frame["timestamp"].dt.dayofweek
    frame["hour_sin"], frame["hour_cos"] = _cycle_features(hours, 24)
    frame["dow_sin"], frame["dow_cos"] = _cycle_features(dow, 7)

    for window in config.moving_average_windows:
        ma = close.rolling(window).mean()
        frame[f"ma_delta_{int(window)}h"] = (close - ma) / ma.replace(0.0, np.nan)

    for window in config.ema_windows:
        ema = close.ewm(span=window, adjust=False).mean()
        frame[f"ema_delta_{int(window)}h"] = (close - ema) / ema.replace(0.0, np.nan)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    for window in config.atr_windows:
        atr = tr.rolling(window).mean()
        frame[f"atr_pct_{int(window)}h"] = atr / close.replace(0.0, np.nan)

    frame[f"rsi_{int(config.rsi_window)}h"] = _rsi(close, window=int(config.rsi_window))

    for horizon in config.forecast_horizons:
        suffix = f"_h{int(horizon)}"
        close_col = f"predicted_close_p50{suffix}"
        high_col = f"predicted_high_p50{suffix}"
        low_col = f"predicted_low_p50{suffix}"
        if close_col not in frame.columns:
            continue
        ref = frame["reference_close"].replace(0.0, np.nan)
        frame[f"chronos_close_delta{suffix}"] = (frame[close_col] - ref) / ref
        if high_col in frame.columns:
            frame[f"chronos_high_delta{suffix}"] = (frame[high_col] - ref) / ref
        if low_col in frame.columns:
            frame[f"chronos_low_delta{suffix}"] = (frame[low_col] - ref) / ref

    # Replace infinities so downstream normalization does not create NaNs.
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)

    return frame


def _zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    rolling_std = rolling_std.replace(0.0, np.nan)
    return (series - rolling_mean) / rolling_std


def _cycle_features(values: pd.Series, period: int) -> Tuple[pd.Series, pd.Series]:
    radians = 2 * math.pi * values / period
    return np.sin(radians), np.cos(radians)


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # If there are no losses in the window, RSI should be 100.
    rsi = rsi.where(avg_loss > 0, 100.0)
    return rsi


__all__ = [
    "BinanceExp1DataModule",
    "FeatureNormalizer",
    "MultiSymbolDataModule",
    "build_default_feature_columns",
    "build_feature_frame",
]
