"""Configuration for PnL Forecast meta-strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Tuple, Optional


@dataclass
class DataConfigPnL:
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

    # Date range for simulation
    start_date: date = field(default_factory=lambda: date(2025, 1, 1))
    end_date: date = field(default_factory=lambda: date(2025, 12, 31))

    # Historical context for Chronos forecasting
    context_days: int = 512  # Days of history for OHLC forecast

    @property
    def all_symbols(self) -> Tuple[str, ...]:
        """Return all symbols (stocks + crypto)."""
        return self.stock_symbols + self.crypto_symbols


@dataclass
class ForecastConfigPnL:
    """Configuration for Chronos2 forecasting."""

    model_id: str = "amazon/chronos-2"
    device_map: str = "cuda"
    prediction_length: int = 1  # Predict 1 day ahead
    context_length: int = 512
    quantile_levels: Tuple[float, ...] = (0.1, 0.5, 0.9)
    batch_size: int = 128

    # Multivariate forecasting (better for stocks)
    use_multivariate: bool = True

    # PnL forecasting context
    pnl_context_days: int = 7  # Days of PnL history for forecasting


@dataclass
class StrategyConfigPnL:
    """Configuration for buy-low/sell-high strategy generation."""

    # Trading fee structure (per side)
    stock_fee_bp: float = 3.0  # 3 basis points = 0.03%
    crypto_fee_bp: float = 8.0  # 8 basis points = 0.08%

    # Threshold constraints
    max_threshold_pct: float = 0.5  # Max 0.5% threshold on each side
    min_threshold_pct: float = 0.0  # Min threshold (can be 0)

    # Minimum spread between buy and sell (must cover round-trip fees)
    # buy_threshold + sell_threshold must be > min_spread
    # For stocks: 3bp * 2 = 6bp = 0.06%
    # For crypto: 8bp * 2 = 16bp = 0.16%

    # Strategy grid resolution
    threshold_step_pct: float = 0.05  # 5bp steps for threshold grid

    @property
    def stock_min_spread_pct(self) -> float:
        """Minimum spread for stocks (round-trip fees) as decimal (e.g., 0.0006 = 6bp)."""
        return self.stock_fee_bp * 2 / 10000  # Convert bp to decimal

    @property
    def crypto_min_spread_pct(self) -> float:
        """Minimum spread for crypto (round-trip fees) as decimal (e.g., 0.0016 = 16bp)."""
        return self.crypto_fee_bp * 2 / 10000  # Convert bp to decimal

    def get_min_spread_pct(self, is_crypto: bool) -> float:
        """Get minimum spread for asset type as decimal."""
        return self.crypto_min_spread_pct if is_crypto else self.stock_min_spread_pct

    def get_fee_pct(self, is_crypto: bool) -> float:
        """Get fee percentage for asset type as decimal."""
        return (self.crypto_fee_bp / 10000) if is_crypto else (self.stock_fee_bp / 10000)


@dataclass
class SimulationConfigPnL:
    """Configuration for PnL forecast simulation."""

    # Initial capital per symbol
    initial_cash: float = 10_000.0

    # Strategy history requirements
    min_pnl_history_days: int = 7  # Minimum days of PnL history before forecasting
    max_pnl_history_days: int = 30  # Maximum days of PnL history to use

    # Position sizing
    position_size_pct: float = 1.0  # Use 100% of allocated capital

    # Trade execution assumptions
    use_limit_orders: bool = True  # Assume limit orders at threshold prices
    partial_fills_allowed: bool = True  # Allow partial day fills

    # Slippage model
    slippage_pct: float = 0.0  # No slippage for limit orders at threshold

    # Output
    output_dir: Path = field(default_factory=lambda: Path("pnlforecast_results"))
    save_detailed_results: bool = True

    # Progress
    log_interval_days: int = 20


@dataclass
class FullConfigPnL:
    """Full configuration combining all sub-configs."""

    data: DataConfigPnL = field(default_factory=DataConfigPnL)
    forecast: ForecastConfigPnL = field(default_factory=ForecastConfigPnL)
    strategy: StrategyConfigPnL = field(default_factory=StrategyConfigPnL)
    simulation: SimulationConfigPnL = field(default_factory=SimulationConfigPnL)
