"""
Second-generation portfolio agent that fuses probabilistic forecasts,
LLM-derived views, and cost-aware optimisation.
"""

from .config import OptimizationConfig, PipelineConfig
from .forecasting import ForecastReturnSet, combine_forecast_sets, shrink_covariance
from .pipeline import AllocationPipeline, AllocationResult
from .views_schema import LLMViews, TickerView

__all__ = [
    "AllocationPipeline",
    "AllocationResult",
    "ForecastReturnSet",
    "LLMViews",
    "OptimizationConfig",
    "PipelineConfig",
    "TickerView",
    "combine_forecast_sets",
    "shrink_covariance",
]
