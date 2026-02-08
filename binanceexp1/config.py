from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple


@dataclass
class DatasetConfig:
    symbol: str = "BTCUSD"
    data_root: Path = Path("trainingdatahourly") / "crypto"
    forecast_cache_root: Path = Path("binanceneural") / "forecast_cache"
    forecast_horizons: Tuple[int, ...] = (1, 4, 12, 24)
    sequence_length: int = 96
    val_fraction: float = 0.15
    min_history_hours: int = 24 * 30
    max_feature_lookback_hours: int = 24 * 100
    feature_columns: Optional[Sequence[str]] = None
    refresh_hours: int = 0
    validation_days: int = 70
    cache_only: bool = False
    moving_average_windows: Tuple[int, ...] = (24 * 7, 24 * 25, 24 * 99)
    ema_windows: Tuple[int, ...] = (24, 168)
    atr_windows: Tuple[int, ...] = (24, 168)
    volume_z_window: int = 168
    volume_shock_window: int = 24
    volume_spike_z: float = 2.0
    trend_windows: Tuple[int, ...] = (168,)
    drawdown_windows: Tuple[int, ...] = (168,)
    vol_regime_short: int = 24
    vol_regime_long: int = 168
    rsi_window: int = 14


@dataclass
class ExperimentConfig:
    context_lengths: Tuple[int, ...] = (64, 96, 192)
    context_strides: Tuple[int, ...] = (1,)
    trim_ratio: float = 0.2


__all__ = ["DatasetConfig", "ExperimentConfig"]
