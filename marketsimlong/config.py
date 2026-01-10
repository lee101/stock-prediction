"""Configuration dataclasses for long-term daily market simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Tuple, Optional, Literal


@dataclass
class DataConfigLong:
    """Configuration for data loading."""

    # Symbols to trade
    stock_symbols: Tuple[str, ...] = (
        "AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA", "AMD",
        "NFLX", "AVGO", "ADBE", "CRM", "COST", "COIN", "SHOP",
    )
    crypto_symbols: Tuple[str, ...] = (
        "BTCUSD", "ETHUSD", "BNBUSD", "SOLUSD", "UNIUSD",
    )

    # Data paths
    data_root: Path = field(default_factory=lambda: Path("trainingdata/train"))
    forecast_cache_dir: Path = field(default_factory=lambda: Path("strategytraining/forecast_cache"))

    # Date range for simulation
    start_date: date = field(default_factory=lambda: date(2025, 1, 1))
    end_date: date = field(default_factory=lambda: date(2025, 12, 31))

    # Historical context for Chronos forecasting
    context_days: int = 512  # Days of history to use for prediction

    @property
    def all_symbols(self) -> Tuple[str, ...]:
        """Return all symbols (stocks + crypto)."""
        return self.stock_symbols + self.crypto_symbols


@dataclass
class ForecastConfigLong:
    """Configuration for Chronos2 forecasting."""

    model_id: str = "amazon/chronos-2"
    device_map: str = "cuda"
    prediction_length: int = 1  # Predict 1 day ahead
    context_length: int = 512
    quantile_levels: Tuple[float, ...] = (0.1, 0.5, 0.9)
    batch_size: int = 128

    # Multivariate forecasting (better for stocks)
    use_multivariate: bool = True

    # Pre-augmentation strategies
    use_preaugmentation: bool = True
    preaugmentation_dirs: Tuple[str, ...] = (
        "preaugstrategies/chronos2",
        "preaugstrategies/best",
    )


@dataclass
class SimulationConfigLong:
    """Configuration for long-term daily simulation."""

    # Trading parameters
    top_n: int = 1  # Number of top symbols to buy each day (1, 2, or 3)
    initial_cash: float = 100_000.0

    # Fee structure
    maker_fee: float = 0.0008  # 0.08% per trade
    taker_fee: float = 0.001  # 0.1% (for market orders)
    slippage: float = 0.0005  # 0.05% estimated slippage

    # Leverage (for stocks only - crypto typically 1x)
    leverage: float = 1.0  # 1.0 = no leverage, 2.0 = 2x leverage
    margin_rate_annual: float = 0.0625  # 6.25% annual margin interest rate
    leverage_stocks_only: bool = True  # Only apply leverage to stocks, not crypto

    # Trading calendar
    stock_trading_days_per_year: int = 252
    crypto_trading_days_per_year: int = 365

    # Position sizing
    equal_weight: bool = True  # Equal weight across top N
    max_position_size: float = 1.0  # Max fraction of portfolio per position

    # Strategy variant
    strategy: Literal["top_n_daily", "momentum_rebalance"] = "top_n_daily"

    # Risk controls
    min_predicted_return: float = 0.0  # Minimum predicted return to enter position
    max_daily_loss: float = 0.10  # Stop trading if daily loss exceeds 10%

    @property
    def total_cost_per_trade(self) -> float:
        """Total round-trip cost per trade (entry + exit)."""
        return 2 * (self.maker_fee + self.slippage)

    @property
    def daily_margin_rate(self) -> float:
        """Daily margin interest rate."""
        return self.margin_rate_annual / 365


@dataclass
class TuningConfigLong:
    """Configuration for Chronos2 hyperparameter tuning."""

    # Tuning objective
    metric: Literal["mae", "mape", "rmse", "directional_accuracy"] = "mae"

    # Search space
    context_lengths: Tuple[int, ...] = (256, 512, 1024)
    prediction_lengths: Tuple[int, ...] = (1, 3, 5, 7)
    use_multivariate_options: Tuple[bool, ...] = (True, False)

    # Pre-augmentation strategies to try
    preaug_strategies: Tuple[str, ...] = (
        "baseline",
        "log_diff",
        "zscore",
        "pct_change",
        "winsorize",
    )

    # Validation
    val_days: int = 60  # Days to use for validation
    n_trials: int = 50  # Number of Optuna trials

    # Output
    output_dir: Path = field(default_factory=lambda: Path("hyperparams/chronos2_long"))
    save_best: bool = True
