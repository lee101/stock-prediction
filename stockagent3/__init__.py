"""Stockagent3: two-stage LLM portfolio allocator with Chronos2 forecasts."""

from .agent import (
    PortfolioAllocation,
    SymbolAllocation,
    TradePlan,
    TradePosition,
    generate_trade_plan,
)

try:  # pragma: no cover - optional dependency
    from .forecaster import Chronos2Forecast, Chronos2Forecaster
except Exception:  # pragma: no cover
    Chronos2Forecast = None  # type: ignore[assignment]
    Chronos2Forecaster = None  # type: ignore[assignment]

__all__ = [
    "PortfolioAllocation",
    "SymbolAllocation",
    "TradePlan",
    "TradePosition",
    "Chronos2Forecast",
    "Chronos2Forecaster",
    "generate_trade_plan",
]
