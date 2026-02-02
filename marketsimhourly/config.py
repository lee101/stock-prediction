"""Configuration dataclasses for hourly market simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass
class DataConfigHourly:
    """Configuration for hourly data loading."""

    stock_symbols: Tuple[str, ...] = (
        "AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA", "AMD",
        "NFLX", "AVGO", "ADBE", "CRM", "COST", "COIN", "SHOP",
    )
    crypto_symbols: Tuple[str, ...] = (
        "BTCUSD", "ETHUSD", "BNBUSD", "SOLUSD", "UNIUSD",
    )

    data_root: Path = field(default_factory=lambda: Path("trainingdatahourly"))
    forecast_cache_dir: Path = field(default_factory=lambda: Path("strategytraining/forecast_cache_hourly"))

    start_date: date = field(default_factory=lambda: date(2025, 1, 1))
    end_date: date = field(default_factory=lambda: date(2025, 12, 31))

    context_hours: int = 512

    @property
    def all_symbols(self) -> Tuple[str, ...]:
        return self.stock_symbols + self.crypto_symbols


@dataclass
class ForecastConfigHourly:
    """Configuration for Chronos2 hourly forecasting."""

    model_id: str = "amazon/chronos-2"
    device_map: str = "cuda"
    prediction_length: int = 1
    context_length: int = 512
    quantile_levels: Tuple[float, ...] = (0.1, 0.5, 0.9)
    batch_size: int = 128

    use_multivariate: bool = True
    use_cross_learning: bool = False
    cross_learning_min_batch: int = 2
    cross_learning_group_by_asset_type: bool = True
    cross_learning_chunk_size: Optional[int] = None

    frequency: str = "hourly"
    use_preaugmentation: bool = True
    preaugmentation_dirs: Tuple[str, ...] = (
        "preaugstrategies/chronos2/hourly",
        "preaugstrategies/best/hourly",
        "preaugstrategies/chronos2",
        "preaugstrategies/best",
    )


@dataclass
class SimulationConfigHourly:
    """Configuration for hourly simulation."""

    top_n: int = 1
    initial_cash: float = 100_000.0

    maker_fee: float = 0.0008
    taker_fee: float = 0.001
    slippage: float = 0.0005

    leverage: float = 1.0
    margin_rate_annual: float = 0.0625
    leverage_stocks_only: bool = True

    equal_weight: bool = True
    max_position_size: float = 1.0

    min_predicted_return: float = 0.0

    max_hold_hours: int = 0
    max_hold_hours_per_symbol: Optional[Dict[str, int]] = None

    leverage_soft_cap: float = 0.0
    leverage_penalty_rate: float = 0.0
    hold_penalty_start_hours: int = 0
    hold_penalty_rate: float = 0.0

    trading_hours_per_year: int = 24 * 365

    @property
    def hourly_margin_rate(self) -> float:
        return self.margin_rate_annual / (365 * 24)
