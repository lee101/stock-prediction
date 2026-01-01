"""Chronos2 forecasting for long-term market simulation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import DataConfigLong, ForecastConfigLong
from .data import DailyDataLoader, is_crypto_symbol

logger = logging.getLogger(__name__)


@dataclass
class SymbolForecast:
    """Forecast for a single symbol."""

    symbol: str
    forecast_date: date  # Date being forecasted
    current_close: float  # Close price on day before forecast
    predicted_close: float  # Median predicted close
    predicted_high: float  # Median predicted high
    predicted_low: float  # Median predicted low
    predicted_close_p10: float  # 10th percentile predicted close
    predicted_close_p90: float  # 90th percentile predicted close

    @property
    def predicted_return(self) -> float:
        """Predicted percentage return."""
        if self.current_close <= 0:
            return 0.0
        return (self.predicted_close - self.current_close) / self.current_close

    @property
    def predicted_upside(self) -> float:
        """Predicted upside to p90."""
        if self.current_close <= 0:
            return 0.0
        return (self.predicted_close_p90 - self.current_close) / self.current_close

    @property
    def predicted_downside(self) -> float:
        """Predicted downside to p10."""
        if self.current_close <= 0:
            return 0.0
        return (self.predicted_close_p10 - self.current_close) / self.current_close

    @property
    def confidence_spread(self) -> float:
        """Spread between p10 and p90 as measure of uncertainty."""
        if self.current_close <= 0:
            return float("inf")
        return (self.predicted_close_p90 - self.predicted_close_p10) / self.current_close


@dataclass
class DailyForecasts:
    """All forecasts for a single trading day."""

    forecast_date: date
    forecasts: Dict[str, SymbolForecast]

    def get_ranked_symbols(
        self,
        metric: str = "predicted_return",
        ascending: bool = False,
    ) -> List[Tuple[str, float]]:
        """Get symbols ranked by a metric.

        Args:
            metric: Metric to rank by (predicted_return, predicted_upside, etc.)
            ascending: If True, sort ascending (lowest first)

        Returns:
            List of (symbol, metric_value) tuples, sorted
        """
        ranked = []
        for symbol, forecast in self.forecasts.items():
            value = getattr(forecast, metric, 0.0)
            if np.isfinite(value):
                ranked.append((symbol, value))

        ranked.sort(key=lambda x: x[1], reverse=not ascending)
        return ranked

    def get_top_n_symbols(
        self,
        n: int = 1,
        metric: str = "predicted_return",
        min_return: float = 0.0,
    ) -> List[str]:
        """Get top N symbols by predicted return.

        Args:
            n: Number of symbols to return
            metric: Metric to rank by
            min_return: Minimum predicted return to include

        Returns:
            List of top N symbols
        """
        ranked = self.get_ranked_symbols(metric=metric, ascending=False)

        # Filter by minimum return
        filtered = [(sym, val) for sym, val in ranked if val >= min_return]

        return [sym for sym, _ in filtered[:n]]


class Chronos2Forecaster:
    """Generates Chronos2 forecasts for all symbols."""

    def __init__(
        self,
        data_loader: DailyDataLoader,
        forecast_config: ForecastConfigLong,
    ) -> None:
        self.data_loader = data_loader
        self.config = forecast_config
        self._wrapper = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of Chronos2 wrapper."""
        if self._initialized:
            return

        try:
            from src.models.chronos2_wrapper import Chronos2OHLCWrapper

            self._wrapper = Chronos2OHLCWrapper.from_pretrained(
                model_id=self.config.model_id,
                device_map=self.config.device_map,
                default_context_length=self.config.context_length,
                quantile_levels=list(self.config.quantile_levels),
                default_batch_size=self.config.batch_size,
                preaugmentation_dirs=(
                    list(self.config.preaugmentation_dirs)
                    if self.config.use_preaugmentation
                    else None
                ),
            )
            self._initialized = True
            logger.info("Initialized Chronos2 forecaster with model %s", self.config.model_id)
        except Exception as e:
            logger.error("Failed to initialize Chronos2: %s", e)
            raise

    def forecast_symbol(
        self,
        symbol: str,
        target_date: date,
        use_multivariate: Optional[bool] = None,
    ) -> Optional[SymbolForecast]:
        """Generate forecast for a single symbol.

        Args:
            symbol: Trading symbol
            target_date: Date to forecast
            use_multivariate: Override multivariate setting

        Returns:
            SymbolForecast or None if forecast failed
        """
        self._ensure_initialized()

        # Get historical context
        context_df = self.data_loader.get_context_for_date(
            symbol,
            target_date,
            context_days=self.config.context_length,
        )

        if context_df.empty or len(context_df) < 10:
            logger.warning("Insufficient context for %s on %s", symbol, target_date)
            return None

        # Get current close (last available close before target_date)
        current_close = float(context_df.iloc[-1]["close"])

        # Determine whether to use multivariate
        should_multivariate = use_multivariate
        if should_multivariate is None:
            # Use multivariate for stocks (not crypto)
            should_multivariate = self.config.use_multivariate and not is_crypto_symbol(symbol)

        try:
            if should_multivariate and hasattr(self._wrapper, "predict_ohlc_multivariate"):
                batch = self._wrapper.predict_ohlc_multivariate(
                    context_df,
                    symbol=symbol,
                    prediction_length=self.config.prediction_length,
                    context_length=len(context_df),
                    quantile_levels=list(self.config.quantile_levels),
                )
            else:
                batch = self._wrapper.predict_ohlc(
                    context_df,
                    symbol=symbol,
                    prediction_length=self.config.prediction_length,
                    context_length=len(context_df),
                    quantile_levels=list(self.config.quantile_levels),
                )

            # Extract predictions from quantile frames
            q50 = batch.quantile_frames.get(0.5)
            q10 = batch.quantile_frames.get(0.1)
            q90 = batch.quantile_frames.get(0.9)

            if q50 is None or q50.empty:
                logger.warning("Empty predictions for %s on %s", symbol, target_date)
                return None

            # Get first row predictions (day 1)
            pred_close = float(q50.iloc[0].get("close", current_close))
            pred_high = float(q50.iloc[0].get("high", pred_close))
            pred_low = float(q50.iloc[0].get("low", pred_close))

            pred_close_p10 = float(q10.iloc[0].get("close", pred_close)) if q10 is not None else pred_close * 0.95
            pred_close_p90 = float(q90.iloc[0].get("close", pred_close)) if q90 is not None else pred_close * 1.05

            return SymbolForecast(
                symbol=symbol,
                forecast_date=target_date,
                current_close=current_close,
                predicted_close=pred_close,
                predicted_high=pred_high,
                predicted_low=pred_low,
                predicted_close_p10=pred_close_p10,
                predicted_close_p90=pred_close_p90,
            )

        except Exception as e:
            logger.error("Forecast failed for %s on %s: %s", symbol, target_date, e)
            return None

    def forecast_all_symbols(
        self,
        target_date: date,
        symbols: Optional[List[str]] = None,
    ) -> DailyForecasts:
        """Generate forecasts for all symbols on a target date.

        Args:
            target_date: Date to forecast
            symbols: Optional list of symbols (defaults to all tradable)

        Returns:
            DailyForecasts containing all symbol forecasts
        """
        self._ensure_initialized()

        if symbols is None:
            symbols = self.data_loader.get_tradable_symbols_on_date(target_date)

        forecasts = {}
        for symbol in symbols:
            forecast = self.forecast_symbol(symbol, target_date)
            if forecast is not None:
                forecasts[symbol] = forecast

        logger.info(
            "Generated %d forecasts for %s (out of %d symbols)",
            len(forecasts),
            target_date,
            len(symbols),
        )

        return DailyForecasts(
            forecast_date=target_date,
            forecasts=forecasts,
        )

    def compute_mae(
        self,
        forecasts: DailyForecasts,
        actual_prices: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """Compute Mean Absolute Error for forecasts.

        Args:
            forecasts: Forecasts to evaluate
            actual_prices: Dict of symbol -> {open, high, low, close}

        Returns:
            Dict with mae_close, mae_pct, etc.
        """
        errors = []
        pct_errors = []

        for symbol, forecast in forecasts.forecasts.items():
            if symbol not in actual_prices:
                continue

            actual = actual_prices[symbol]
            actual_close = actual.get("close")
            if actual_close is None:
                continue

            error = abs(forecast.predicted_close - actual_close)
            pct_error = error / actual_close if actual_close > 0 else 0.0

            errors.append(error)
            pct_errors.append(pct_error)

        if not errors:
            return {"mae_close": float("nan"), "mae_pct": float("nan")}

        return {
            "mae_close": float(np.mean(errors)),
            "mae_pct": float(np.mean(pct_errors)) * 100,  # As percentage
            "n_symbols": len(errors),
        }

    def unload(self) -> None:
        """Release GPU memory."""
        if self._wrapper is not None:
            self._wrapper.unload()
            self._wrapper = None
            self._initialized = False
            logger.info("Chronos2 forecaster unloaded")


def create_forecaster(
    data_config: DataConfigLong,
    forecast_config: ForecastConfigLong,
) -> Chronos2Forecaster:
    """Factory function to create a forecaster.

    Args:
        data_config: Data configuration
        forecast_config: Forecast configuration

    Returns:
        Initialized Chronos2Forecaster
    """
    data_loader = DailyDataLoader(data_config)
    data_loader.load_all_symbols()
    return Chronos2Forecaster(data_loader, forecast_config)
