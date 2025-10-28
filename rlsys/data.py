"""Data loading and feature engineering utilities."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import torch

from .config import DataConfig


@dataclass(slots=True)
class PreparedData:
    features: torch.Tensor
    targets: torch.Tensor
    index: pd.Index


def _compute_log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices).diff().fillna(0.0)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def prepare_features(df: pd.DataFrame, config: DataConfig) -> PreparedData:
    missing = [col for col in config.feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    features = df.loc[:, config.feature_columns].astype(float).copy()
    target = df[config.target_column].astype(float).copy()

    if config.normalize_returns:
        returns = _compute_log_returns(target)
        features["return"] = returns
    else:
        returns = pd.Series(np.zeros(len(df)), index=df.index)
        features["return"] = returns

    if config.include_technical_indicators:
        fast, slow = config.ema_periods
        features["ema_fast"] = _ema(target, fast)
        features["ema_slow"] = _ema(target, slow)
        features["ema_ratio"] = features["ema_fast"] / (features["ema_slow"] + 1e-12)
        features["volatility"] = returns.rolling(window=config.window_size).std().fillna(0.0)

    features = features.fillna(method="ffill").fillna(method="bfill")
    tensor_features = torch.tensor(features.values, dtype=torch.float32)
    tensor_targets = torch.tensor(target.values, dtype=torch.float32)
    return PreparedData(features=tensor_features, targets=tensor_targets, index=df.index)


@lru_cache(maxsize=8)
def sliding_windows(length: int, window_size: int) -> Tuple[slice, ...]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if window_size > length:
        raise ValueError("window_size cannot exceed sequence length")
    return tuple(slice(start, start + window_size) for start in range(length - window_size))


def make_training_batches(data: PreparedData, window_size: int) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    windows = sliding_windows(len(data.features), window_size + 1)
    for window in windows:
        obs = data.features[window.start : window.stop - 1]
        target = data.targets[window.stop - 1]
        yield obs, target
