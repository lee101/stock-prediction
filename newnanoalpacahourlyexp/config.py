from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

from src.symbol_utils import is_crypto_symbol


@dataclass
class DatasetConfig:
    symbol: str = "BTCUSD"
    data_root: Optional[Path] = None
    forecast_cache_root: Path = Path("binanceneural") / "forecast_cache"
    forecast_horizons: Tuple[int, ...] = (1, 24)
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
    allow_mixed_asset_class: bool = False
    allow_short: bool = False
    long_only_symbols: Tuple[str, ...] = ()
    short_only_symbols: Tuple[str, ...] = ()

    def resolved_data_root(self) -> Path:
        if self.data_root is not None:
            return Path(self.data_root)
        return Path("trainingdatahourly/crypto") if is_crypto_symbol(self.symbol) else Path("trainingdatahourly/stocks")


@dataclass
class ExperimentConfig:
    context_lengths: Tuple[int, ...] = (64, 96, 192)
    context_strides: Tuple[int, ...] = (1,)
    trim_ratio: float = 0.2


__all__ = ["DatasetConfig", "ExperimentConfig"]
