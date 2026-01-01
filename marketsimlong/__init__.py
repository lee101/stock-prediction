"""
marketsimlong - Long-term daily market simulation.

This module implements a naive daily trading strategy that:
1. Uses Chronos2 forecasts to predict next-day close prices for all symbols
2. Each trading day, buys the top N symbols with highest predicted % growth
3. Closes all positions at end of day, buys new top N the next day
4. Handles stock (252 trading days) vs crypto (365 trading days) calendars
5. Simulates over an entire year to measure strategy performance
"""

from .config import (
    SimulationConfigLong,
    DataConfigLong,
    ForecastConfigLong,
    TuningConfigLong,
)
from .simulator import (
    LongTermDailySimulator,
    SimulationResult,
    DayResult,
    TradeRecord,
)

__all__ = [
    "SimulationConfigLong",
    "DataConfigLong",
    "ForecastConfigLong",
    "TuningConfigLong",
    "LongTermDailySimulator",
    "SimulationResult",
    "DayResult",
    "TradeRecord",
]
