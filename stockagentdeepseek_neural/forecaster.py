"""Utilities for enriching DeepSeek prompts with neural forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import pandas as pd

from stockagent.agentsimulator.market_data import MarketDataBundle
from stockagentcombined.forecaster import CombinedForecast, CombinedForecastGenerator, ModelForecast


def _bundle_frame(symbol: str, bundle: MarketDataBundle) -> pd.DataFrame:
    frame = bundle.get_symbol_bars(symbol)
    if frame.empty:
        raise ValueError(f"No historical data available for symbol '{symbol}'.")
    df = frame.reset_index().rename(columns={"index": "timestamp"})
    if "timestamp" not in df.columns:
        raise ValueError("Expected resolved frame to contain a 'timestamp' column.")
    return df


@dataclass(frozen=True)
class ModelForecastSummary:
    model: str
    config_name: str
    average_price_mae: float
    forecasts: Mapping[str, float]


@dataclass(frozen=True)
class NeuralForecast:
    symbol: str
    combined: Mapping[str, float]
    best_model: Optional[str]
    selection_source: Optional[str]
    model_summaries: Mapping[str, ModelForecastSummary]


def _summarise_model_forecast(model_forecast: ModelForecast) -> ModelForecastSummary:
    return ModelForecastSummary(
        model=model_forecast.model,
        config_name=model_forecast.config_name,
        average_price_mae=model_forecast.average_price_mae,
        forecasts=model_forecast.forecasts,
    )


def build_neural_forecasts(
    *,
    symbols: Iterable[str],
    market_data: MarketDataBundle,
    prediction_length: int = 1,
    generator: Optional[CombinedForecastGenerator] = None,
) -> Dict[str, NeuralForecast]:
    """Generate combined neural forecasts for the supplied symbols."""
    generator = generator or CombinedForecastGenerator()
    historical_frames: MutableMapping[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            historical_frames[symbol] = _bundle_frame(symbol, market_data)
        except ValueError:
            continue

    if not historical_frames:
        raise ValueError("No historical frames could be extracted for the requested symbols.")

    combined_forecasts: Dict[str, CombinedForecast] = generator.generate(
        symbols=historical_frames.keys(),
        prediction_length=prediction_length,
        historical_data=historical_frames,
    )

    results: Dict[str, NeuralForecast] = {}
    for symbol, combined in combined_forecasts.items():
        summaries = {
            name: _summarise_model_forecast(model_forecast)
            for name, model_forecast in combined.model_forecasts.items()
        }
        results[symbol] = NeuralForecast(
            symbol=symbol,
            combined=combined.combined,
            best_model=combined.best_model,
            selection_source=combined.selection_source,
            model_summaries=summaries,
        )
    return results


__all__ = ["NeuralForecast", "ModelForecastSummary", "build_neural_forecasts"]
