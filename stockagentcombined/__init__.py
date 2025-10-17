"""Public exports for the combined Toto/Kronos toolchain."""

from importlib import import_module
from typing import Any

__all__ = [
    "CombinedForecastGenerator",
    "CombinedForecast",
    "ModelForecast",
    "ErrorBreakdown",
    "SimulationConfig",
    "CombinedPlanBuilder",
    "build_daily_plans",
    "run_simulation",
]

_FORECASTER_SYMBOLS = {
    "CombinedForecastGenerator",
    "CombinedForecast",
    "ModelForecast",
    "ErrorBreakdown",
}

_PLAN_SYMBOLS = {
    "CombinedPlanBuilder",
    "SimulationConfig",
    "build_daily_plans",
}

def __getattr__(name: str) -> Any:
    if name in _FORECASTER_SYMBOLS:
        module = import_module("stockagentcombined.forecaster")
        return getattr(module, name)
    if name in _PLAN_SYMBOLS:
        module = import_module("stockagentcombined.agentsimulator")
        return getattr(module, name)
    if name == "run_simulation":
        module = import_module("stockagentcombined.simulation")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
