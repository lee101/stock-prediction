"""Dataset with both daily envelope and hourly forecasts for clamped trading."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from suiclampedwithin.clamped_forecaster import aggregate_hourly_to_daily, load_hourly_data


@dataclass
class DualForecastConfig:
    symbol: str
    data_root: Path
    forecast_cache: Path
    sequence_length: int = 48
    hourly_horizons: Tuple[int, ...] = (1, 4, 24)
    daily_horizon: int = 1
    train_ratio: float = 0.7
    val_ratio: float = 0.15


BASE_FEATURES = (
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
)

DAILY_FEATURES = (
    "daily_return_1d",
    "daily_range_pct",
    "daily_volatility_7d",
)


def compute_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute base trading features from OHLCV data."""
    out = df.copy()
    out["return_1h"] = out["close"].pct_change(1)
    out["return_4h"] = out["close"].pct_change(4)
    out["return_24h"] = out["close"].pct_change(24)
    out["volatility_24h"] = out["close"].pct_change().rolling(24).std()
    out["range_pct"] = (out["high"] - out["low"]) / out["close"].clip(lower=1e-8)
    vol_mean = out["volume"].rolling(24).mean()
    vol_std = out["volume"].rolling(24).std().clip(lower=1e-8)
    out["volume_z"] = (out["volume"] - vol_mean) / vol_std

    ts = pd.to_datetime(out["timestamp"])
    out["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
    out["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
    return out


def compute_daily_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily-level features."""
    out = daily_df.copy()
    out["daily_return_1d"] = out["close"].pct_change(1)
    out["daily_range_pct"] = (out["high"] - out["low"]) / out["close"].clip(lower=1e-8)
    out["daily_volatility_7d"] = out["close"].pct_change().rolling(7).std()
    return out


def merge_daily_to_hourly(hourly_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """Merge daily features into hourly frame by date."""
    hourly = hourly_df.copy()
    hourly["date"] = pd.to_datetime(hourly["timestamp"]).dt.date
    daily = daily_df.copy()
    daily["date"] = pd.to_datetime(daily["timestamp"]).dt.date
    daily_cols = ["date"] + [c for c in DAILY_FEATURES if c in daily.columns]
    daily_cols += ["daily_high", "daily_low", "daily_close_pred"] if "daily_high" in daily.columns else []
    merged = hourly.merge(daily[daily_cols], on="date", how="left")
    return merged.drop(columns=["date"])


class DualForecastDataset(Dataset):
    """Dataset with both hourly and daily forecast features."""

    def __init__(
        self,
        frame: pd.DataFrame,
        feature_matrix: np.ndarray,
        sequence_length: int,
        primary_horizon: int = 4,
    ):
        self.frame = frame.reset_index(drop=True)
        self.features = feature_matrix.astype(np.float32)
        self.seq_len = sequence_length
        self.primary_horizon = primary_horizon

        self.highs = frame["high"].to_numpy(dtype=np.float32)
        self.lows = frame["low"].to_numpy(dtype=np.float32)
        self.closes = frame["close"].to_numpy(dtype=np.float32)
        self.reference_close = frame["close"].to_numpy(dtype=np.float32)

        # Hourly forecasts
        h = primary_horizon
        self.chronos_high = frame.get(f"predicted_high_p50_h{h}", frame["high"]).to_numpy(dtype=np.float32)
        self.chronos_low = frame.get(f"predicted_low_p50_h{h}", frame["low"]).to_numpy(dtype=np.float32)

        # Daily envelope (if available)
        self.daily_high = frame.get("daily_high", frame["high"]).to_numpy(dtype=np.float32)
        self.daily_low = frame.get("daily_low", frame["low"]).to_numpy(dtype=np.float32)

        # Clamped forecasts (hourly clamped within daily bounds)
        self.clamped_high = np.minimum(self.chronos_high, self.daily_high)
        self.clamped_low = np.maximum(self.chronos_low, self.daily_low)

    def __len__(self) -> int:
        return max(0, len(self.frame) - self.seq_len + 1)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start, end = idx, idx + self.seq_len
        return {
            "features": torch.from_numpy(self.features[start:end]),
            "high": torch.from_numpy(self.highs[start:end]),
            "low": torch.from_numpy(self.lows[start:end]),
            "close": torch.from_numpy(self.closes[start:end]),
            "reference_close": torch.from_numpy(self.reference_close[start:end]),
            "chronos_high": torch.from_numpy(self.chronos_high[start:end]),
            "chronos_low": torch.from_numpy(self.chronos_low[start:end]),
            "daily_high": torch.from_numpy(self.daily_high[start:end]),
            "daily_low": torch.from_numpy(self.daily_low[start:end]),
            "clamped_high": torch.from_numpy(self.clamped_high[start:end]),
            "clamped_low": torch.from_numpy(self.clamped_low[start:end]),
        }


class DualForecastDataModule:
    """Data module for training with clamped dual forecasts."""

    def __init__(self, config: DualForecastConfig):
        self.config = config
        self.hourly_df: Optional[pd.DataFrame] = None
        self.daily_df: Optional[pd.DataFrame] = None
        self.feature_cols: List[str] = []
        self.normalizer_mean: Optional[np.ndarray] = None
        self.normalizer_std: Optional[np.ndarray] = None

    def prepare(self) -> None:
        """Load and prepare data."""
        cfg = self.config

        # Load hourly data
        self.hourly_df = load_hourly_data(cfg.data_root, cfg.symbol)
        self.hourly_df = compute_base_features(self.hourly_df)

        # Aggregate and compute daily features
        self.daily_df = aggregate_hourly_to_daily(self.hourly_df)
        self.daily_df = compute_daily_features(self.daily_df)

        # Merge daily into hourly
        self.hourly_df = merge_daily_to_hourly(self.hourly_df, self.daily_df)

        # Load forecast caches if available
        self._load_forecast_caches()

        # Build feature columns
        self.feature_cols = list(BASE_FEATURES)
        for c in DAILY_FEATURES:
            if c in self.hourly_df.columns:
                self.feature_cols.append(c)

        # Add forecast ratio features
        for h in cfg.hourly_horizons:
            col = f"fc_ratio_h{h}"
            pred_col = f"predicted_close_p50_h{h}"
            if pred_col in self.hourly_df.columns:
                self.hourly_df[col] = (
                    self.hourly_df[pred_col] / self.hourly_df["close"].clip(lower=1e-8) - 1
                )
                self.feature_cols.append(col)

        # Daily envelope ratio
        if "daily_high" in self.hourly_df.columns:
            self.hourly_df["daily_envelope_ratio"] = (
                (self.hourly_df["daily_high"] - self.hourly_df["daily_low"])
                / self.hourly_df["close"].clip(lower=1e-8)
            )
            self.feature_cols.append("daily_envelope_ratio")

        # Fill NaN and normalize
        self.hourly_df = self.hourly_df.fillna(0)

    def _load_forecast_caches(self) -> None:
        """Load pre-computed Chronos forecasts from cache."""
        cfg = self.config
        for h in cfg.hourly_horizons:
            cache_file = cfg.forecast_cache / f"h{h}" / f"{cfg.symbol}.parquet"
            if cache_file.exists():
                fc = pd.read_parquet(cache_file)
                fc["timestamp"] = pd.to_datetime(fc["timestamp"])
                self.hourly_df = self.hourly_df.merge(
                    fc[["timestamp", "predicted_close_p50", "predicted_high_p50", "predicted_low_p50"]].rename(
                        columns={
                            "predicted_close_p50": f"predicted_close_p50_h{h}",
                            "predicted_high_p50": f"predicted_high_p50_h{h}",
                            "predicted_low_p50": f"predicted_low_p50_h{h}",
                        }
                    ),
                    on="timestamp",
                    how="left",
                )

    def get_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return train/val/test splits."""
        df = self.hourly_df.dropna(subset=self.feature_cols).reset_index(drop=True)
        n = len(df)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
        return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

    def build_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and normalize feature matrix."""
        mat = df[self.feature_cols].to_numpy(dtype=np.float32)
        if self.normalizer_mean is None:
            self.normalizer_mean = mat.mean(axis=0)
            self.normalizer_std = mat.std(axis=0)
            self.normalizer_std = np.where(self.normalizer_std < 1e-6, 1.0, self.normalizer_std)
        return (mat - self.normalizer_mean) / self.normalizer_std

    def build_datasets(self) -> Tuple[DualForecastDataset, DualForecastDataset, DualForecastDataset]:
        """Build train/val/test datasets."""
        train_df, val_df, test_df = self.get_splits()
        train_feat = self.build_feature_matrix(train_df)
        val_feat = self.build_feature_matrix(val_df)
        test_feat = self.build_feature_matrix(test_df)
        seq_len = self.config.sequence_length
        return (
            DualForecastDataset(train_df, train_feat, seq_len),
            DualForecastDataset(val_df, val_feat, seq_len),
            DualForecastDataset(test_df, test_feat, seq_len),
        )
