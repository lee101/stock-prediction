"""Bags.fm Solana trading bot module.

This module provides:
- Bags.fm API wrapper for quotes and swaps
- Price data collection and OHLC aggregation
- Chronos2 forecasting for Solana tokens
- Market simulation for backtesting
- Live trading bot with cost modeling
"""

from .config import (
    BagsConfig,
    DataConfig,
    ForecastConfig,
    SimulationConfig,
    TradingConfig,
    TokenConfig,
    SOL_MINT,
    USDC_MINT,
)
from .bags_api import BagsAPIClient, SwapResult, QuoteResponse
from .data_collector import DataCollector, OHLCBar
from .forecaster import TokenForecaster, TokenForecast
from .simulator import (
    MarketSimulator,
    SimulationResult,
    forecast_threshold_strategy,
    build_forecast_cache,
)
from .trader import BagsTrader

__all__ = [
    # Config
    "BagsConfig",
    "DataConfig",
    "ForecastConfig",
    "SimulationConfig",
    "TradingConfig",
    "TokenConfig",
    "SOL_MINT",
    "USDC_MINT",
    # API
    "BagsAPIClient",
    "SwapResult",
    "QuoteResponse",
    # Data
    "DataCollector",
    "OHLCBar",
    # Forecasting
    "TokenForecaster",
    "TokenForecast",
    # Simulation
    "MarketSimulator",
    "SimulationResult",
    "forecast_threshold_strategy",
    "build_forecast_cache",
    # Trading
    "BagsTrader",
]
