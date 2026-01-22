"""Chronos2 forecasting for Solana tokens.

Generates price forecasts using Chronos2 time series model
from collected OHLC data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ForecastConfig, TokenConfig
from .data_collector import DataCollector, OHLCBar

logger = logging.getLogger(__name__)


@dataclass
class TokenForecast:
    """Forecast for a single token."""

    token_mint: str
    token_symbol: str
    forecast_time: datetime
    current_price: float

    # Predicted prices (for each step ahead)
    predicted_prices: List[float]  # Median predictions
    predicted_prices_p10: List[float]  # 10th percentile (pessimistic)
    predicted_prices_p90: List[float]  # 90th percentile (optimistic)

    # Summary metrics for first prediction
    @property
    def predicted_price(self) -> float:
        """Next period predicted price (median)."""
        return self.predicted_prices[0] if self.predicted_prices else self.current_price

    @property
    def predicted_return(self) -> float:
        """Predicted return for next period."""
        if self.current_price <= 0:
            return 0.0
        return (self.predicted_price - self.current_price) / self.current_price

    @property
    def predicted_upside(self) -> float:
        """Predicted upside (to p90)."""
        if self.current_price <= 0 or not self.predicted_prices_p90:
            return 0.0
        return (self.predicted_prices_p90[0] - self.current_price) / self.current_price

    @property
    def predicted_downside(self) -> float:
        """Predicted downside (to p10)."""
        if self.current_price <= 0 or not self.predicted_prices_p10:
            return 0.0
        return (self.predicted_prices_p10[0] - self.current_price) / self.current_price

    @property
    def confidence_spread(self) -> float:
        """Spread between p10 and p90 as uncertainty measure."""
        if self.current_price <= 0:
            return float("inf")
        if not self.predicted_prices_p10 or not self.predicted_prices_p90:
            return float("inf")
        return (
            self.predicted_prices_p90[0] - self.predicted_prices_p10[0]
        ) / self.current_price

    @property
    def signal_strength(self) -> float:
        """Signal strength: predicted_return / confidence_spread."""
        spread = self.confidence_spread
        if spread <= 0 or spread == float("inf"):
            return 0.0
        return self.predicted_return / spread

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "token_mint": self.token_mint,
            "token_symbol": self.token_symbol,
            "forecast_time": self.forecast_time.isoformat(),
            "current_price": self.current_price,
            "predicted_price": self.predicted_price,
            "predicted_return": self.predicted_return,
            "predicted_upside": self.predicted_upside,
            "predicted_downside": self.predicted_downside,
            "confidence_spread": self.confidence_spread,
            "signal_strength": self.signal_strength,
        }


@dataclass
class ForecastBatch:
    """Collection of forecasts for multiple tokens."""

    forecast_time: datetime
    forecasts: Dict[str, TokenForecast]  # token_mint -> forecast

    def get_ranked_tokens(
        self,
        metric: str = "predicted_return",
        min_value: float = 0.0,
        ascending: bool = False,
    ) -> List[Tuple[str, float]]:
        """Get tokens ranked by a metric.

        Args:
            metric: Metric to rank by (predicted_return, signal_strength, etc.)
            min_value: Minimum metric value to include
            ascending: If True, sort ascending

        Returns:
            List of (token_mint, metric_value) tuples, sorted
        """
        ranked = []
        for mint, forecast in self.forecasts.items():
            value = getattr(forecast, metric, 0.0)
            if np.isfinite(value) and value >= min_value:
                ranked.append((mint, value))

        ranked.sort(key=lambda x: x[1], reverse=not ascending)
        return ranked

    def get_best_token(
        self,
        metric: str = "predicted_return",
        min_return: float = 0.0,
    ) -> Optional[str]:
        """Get the best token by metric.

        Args:
            metric: Metric to use
            min_return: Minimum predicted return

        Returns:
            Token mint or None if no qualifying tokens
        """
        ranked = self.get_ranked_tokens(metric=metric, min_value=min_return)
        return ranked[0][0] if ranked else None


class TokenForecaster:
    """Generates forecasts for Solana tokens using Chronos2.

    Uses OHLC data from DataCollector to generate price forecasts
    with quantile estimates for uncertainty.
    """

    def __init__(
        self,
        data_collector: DataCollector,
        config: ForecastConfig,
    ) -> None:
        self.data_collector = data_collector
        self.config = config
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
            logger.info(f"Initialized Chronos2 forecaster with model {self.config.model_id}")

        except ImportError as e:
            logger.warning(f"Chronos2 wrapper not available: {e}")
            logger.info("Falling back to simple momentum-based forecasting")
            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize Chronos2: {e}")
            raise

    def _build_ohlc_frame(self, bars: List[OHLCBar]) -> pd.DataFrame:
        """Build a pandas OHLC dataframe from bars."""
        if not bars:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        sorted_bars = sorted(bars, key=lambda b: b.timestamp)
        df = pd.DataFrame(
            [
                {
                    "timestamp": b.timestamp,
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume,
                }
                for b in sorted_bars
            ]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def _simple_forecast(
        self,
        bars: List[OHLCBar],
        token: TokenConfig,
        prediction_length: int,
    ) -> Optional[TokenForecast]:
        """Simple momentum-based forecast when Chronos2 is unavailable.

        Uses exponential moving average and recent momentum.
        """
        if len(bars) < 10:
            return None

        closes = np.array([b.close for b in bars])
        current_price = closes[-1]

        # Simple momentum calculation
        returns = np.diff(closes) / closes[:-1]
        recent_returns = returns[-10:]  # Last 10 bars

        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns)

        # Project forward
        predicted_prices = []
        predicted_p10 = []
        predicted_p90 = []

        for i in range(1, prediction_length + 1):
            expected_return = mean_return * i
            predicted = current_price * (1 + expected_return)
            predicted_prices.append(predicted)

            # Uncertainty grows with horizon
            uncertainty = std_return * np.sqrt(i) * 1.65  # ~90% CI
            predicted_p10.append(predicted * (1 - uncertainty))
            predicted_p90.append(predicted * (1 + uncertainty))

        return TokenForecast(
            token_mint=token.mint,
            token_symbol=token.symbol,
            forecast_time=datetime.utcnow(),
            current_price=current_price,
            predicted_prices=predicted_prices,
            predicted_prices_p10=predicted_p10,
            predicted_prices_p90=predicted_p90,
        )

    def forecast_from_bars(
        self,
        token: TokenConfig,
        bars: List[OHLCBar],
        prediction_length: Optional[int] = None,
        context_length: Optional[int] = None,
    ) -> Optional[TokenForecast]:
        """Generate a forecast directly from an OHLC bar list.

        Args:
            token: Token configuration
            bars: OHLC bars in chronological order (or unsorted)
            prediction_length: Forecast horizon (defaults to config)
            context_length: Max context bars to use (defaults to config)

        Returns:
            TokenForecast or None if forecast failed
        """
        self._ensure_initialized()

        if prediction_length is None:
            prediction_length = self.config.prediction_length
        if context_length is None:
            context_length = self.config.context_length

        if len(bars) < 20:
            logger.warning(
                f"Insufficient data for {token.symbol}: {len(bars)} bars (need >= 20)"
            )
            return None

        context_bars = bars[-context_length:] if context_length else bars
        df = self._build_ohlc_frame(context_bars)
        if df.empty:
            return None

        current_price = float(df.iloc[-1]["close"])

        if self._wrapper is not None:
            try:
                batch = self._wrapper.predict_ohlc(
                    df,
                    symbol=token.symbol,
                    prediction_length=prediction_length,
                    context_length=len(df),
                    quantile_levels=list(self.config.quantile_levels),
                )

                q50 = batch.quantile_frames.get(0.5)
                q10 = batch.quantile_frames.get(0.1)
                q90 = batch.quantile_frames.get(0.9)

                if q50 is None or q50.empty:
                    logger.warning(f"Empty predictions for {token.symbol}")
                    return self._simple_forecast(context_bars, token, prediction_length)

                predicted_prices = q50["close"].tolist()
                predicted_p10 = (
                    q10["close"].tolist()
                    if q10 is not None
                    else [p * 0.95 for p in predicted_prices]
                )
                predicted_p90 = (
                    q90["close"].tolist()
                    if q90 is not None
                    else [p * 1.05 for p in predicted_prices]
                )

                return TokenForecast(
                    token_mint=token.mint,
                    token_symbol=token.symbol,
                    forecast_time=datetime.utcnow(),
                    current_price=current_price,
                    predicted_prices=predicted_prices,
                    predicted_prices_p10=predicted_p10,
                    predicted_prices_p90=predicted_p90,
                )

            except Exception as e:
                logger.error(f"Chronos2 forecast failed for {token.symbol}: {e}")
                return self._simple_forecast(context_bars, token, prediction_length)

        return self._simple_forecast(context_bars, token, prediction_length)

    def forecast_token(
        self,
        token: TokenConfig,
        context_bars: Optional[int] = None,
    ) -> Optional[TokenForecast]:
        """Generate forecast for a single token.

        Args:
            token: Token configuration
            context_bars: Number of context bars (defaults to config)

        Returns:
            TokenForecast or None if forecast failed
        """
        self._ensure_initialized()

        if context_bars is None:
            context_bars = self.config.context_length

        # Get OHLC bars
        bars = self.data_collector.get_context_bars(token.mint, n=context_bars)

        return self.forecast_from_bars(
            token=token,
            bars=bars,
            prediction_length=self.config.prediction_length,
            context_length=context_bars,
        )

    def forecast_all_tokens(
        self,
        tokens: Optional[List[TokenConfig]] = None,
    ) -> ForecastBatch:
        """Generate forecasts for all tokens.

        Args:
            tokens: List of tokens (defaults to tracked tokens)

        Returns:
            ForecastBatch with all forecasts
        """
        self._ensure_initialized()

        if tokens is None:
            tokens = self.data_collector.data_config.tracked_tokens

        forecasts = {}
        for token in tokens:
            forecast = self.forecast_token(token)
            if forecast is not None:
                forecasts[token.mint] = forecast

        logger.info(
            f"Generated {len(forecasts)} forecasts for {len(tokens)} tokens"
        )

        return ForecastBatch(
            forecast_time=datetime.utcnow(),
            forecasts=forecasts,
        )

    def evaluate_forecast(
        self,
        forecast: TokenForecast,
        actual_price: float,
    ) -> Dict[str, float]:
        """Evaluate a forecast against actual price.

        Args:
            forecast: Previous forecast
            actual_price: Actual price that occurred

        Returns:
            Dict with error metrics
        """
        predicted = forecast.predicted_price
        error = actual_price - predicted
        pct_error = error / forecast.current_price if forecast.current_price > 0 else 0

        # Direction accuracy
        predicted_direction = 1 if forecast.predicted_return > 0 else -1
        actual_direction = 1 if actual_price > forecast.current_price else -1
        direction_correct = predicted_direction == actual_direction

        # Was actual within prediction interval?
        in_interval = (
            forecast.predicted_prices_p10[0] <= actual_price <= forecast.predicted_prices_p90[0]
        )

        return {
            "error": error,
            "pct_error": pct_error * 100,
            "abs_pct_error": abs(pct_error) * 100,
            "direction_correct": direction_correct,
            "in_90_interval": in_interval,
        }

    def unload(self) -> None:
        """Release model resources."""
        if self._wrapper is not None:
            try:
                self._wrapper.unload()
            except Exception:
                pass
            self._wrapper = None
            self._initialized = False
            logger.info("Chronos2 forecaster unloaded")


def create_forecaster(
    data_collector: DataCollector,
    config: Optional[ForecastConfig] = None,
) -> TokenForecaster:
    """Factory function to create a forecaster.

    Args:
        data_collector: Data collector with OHLC data
        config: Forecast configuration (uses defaults if None)

    Returns:
        Initialized TokenForecaster
    """
    if config is None:
        config = ForecastConfig()

    return TokenForecaster(data_collector, config)
