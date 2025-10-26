from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from stockagentcombined.forecaster import CombinedForecast, CombinedForecastGenerator


@dataclass(frozen=True)
class SymbolForecast:
    symbol: str
    last_close: float
    predicted_close: float
    entry_price: float
    average_price_mae: float

    @property
    def predicted_return(self) -> float:
        if self.last_close <= 0:
            return 0.0
        return (self.predicted_close - self.last_close) / self.last_close

    @property
    def error_pct(self) -> float:
        if self.last_close <= 0:
            return 0.0
        return self.average_price_mae / self.last_close


def _weighted_mae(forecast: CombinedForecast) -> float:
    weights = forecast.weights or {}
    total = 0.0
    used = 0.0
    for name, model_forecast in forecast.model_forecasts.items():
        weight = weights.get(name, 0.0)
        if weight <= 0.0:
            continue
        total += weight * model_forecast.average_price_mae
        used += weight
    if used <= 0.0 and forecast.model_forecasts:
        total = sum(model.average_price_mae for model in forecast.model_forecasts.values()) / len(
            forecast.model_forecasts
        )
    return float(total)


class CombinedForecastAdapter:
    """
    Lightweight adapter that translates the Toto/Kronos combined forecasts into
    the simplified :class:`SymbolForecast` contract expected by the allocation
    pipeline.
    """

    def __init__(self, generator: CombinedForecastGenerator) -> None:
        self.generator = generator

    def forecast(
        self,
        symbol: str,
        history: pd.DataFrame,
    ) -> Optional[SymbolForecast]:
        if history.empty:
            return None
        try:
            payload = history.reset_index().rename(columns={"index": "timestamp"})
            if "timestamp" not in payload.columns:
                payload["timestamp"] = history.index
            forecast = self.generator.generate_for_symbol(
                symbol,
                prediction_length=1,
                historical_frame=payload,
            )
        except Exception as exc:
            logger.warning("Combined forecast failed for %s: %s", symbol, exc)
            return None

        last_row = history.iloc[-1]
        last_close = float(last_row.get("close", np.nan))
        if not np.isfinite(last_close) or last_close <= 0:
            return None

        predicted_close = float(forecast.combined.get("close", last_close))
        entry_price = float(forecast.combined.get("open", last_row.get("open", predicted_close)))
        mae = _weighted_mae(forecast)
        return SymbolForecast(
            symbol=symbol,
            last_close=last_close,
            predicted_close=predicted_close,
            entry_price=entry_price if np.isfinite(entry_price) else last_close,
            average_price_mae=mae,
        )
