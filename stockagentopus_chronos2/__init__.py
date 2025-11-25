"""Claude Opus trading agent with Chronos2 neural forecasting integration."""

from .agent import (
    OpusPlanResult,
    OpusPlanStep,
    OpusReplanResult,
    generate_opus_plan,
    simulate_opus_plan,
    simulate_opus_replanning,
)
from .forecaster import Chronos2Forecast, generate_chronos2_forecasts
from .models import TradingPlanOutput, TradingInstruction

__all__ = [
    "OpusPlanResult",
    "OpusPlanStep",
    "OpusReplanResult",
    "generate_opus_plan",
    "simulate_opus_plan",
    "simulate_opus_replanning",
    "Chronos2Forecast",
    "generate_chronos2_forecasts",
    "TradingPlanOutput",
    "TradingInstruction",
]
