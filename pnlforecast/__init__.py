"""PnL Forecast meta-strategy module.

This module implements a meta-strategy that:
1. Uses Chronos2 to forecast OHLC for each symbol
2. Generates buy-at-low/sell-at-high strategies with various thresholds
3. Simulates each strategy's historical PnL over 7+ days
4. Uses Chronos2 to forecast the PnL curve of each strategy
5. Selects the best strategy based on forecasted next-day PnL
"""

from .config import (
    DataConfigPnL,
    ForecastConfigPnL,
    StrategyConfigPnL,
    SimulationConfigPnL,
)
from .strategy import StrategyThresholds, generate_threshold_strategies
from .simulator import StrategySimulator, StrategyPnLResult
from .selector import PnLForecaster, StrategySelector
from .backtester import PnLForecastBacktester, BacktestResult

__all__ = [
    "DataConfigPnL",
    "ForecastConfigPnL",
    "StrategyConfigPnL",
    "SimulationConfigPnL",
    "StrategyThresholds",
    "generate_threshold_strategies",
    "StrategySimulator",
    "StrategyPnLResult",
    "PnLForecaster",
    "StrategySelector",
    "PnLForecastBacktester",
    "BacktestResult",
]
