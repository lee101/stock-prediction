"""
Forecast validation and correction for OHLC price predictions.

This module ensures that forecasted prices maintain logical ordering:
    low_price <= close_price <= high_price

It provides:
1. Validation functions to detect invalid forecasts
2. Correction functions to fix invalid forecasts
3. Retry logic for model predictions
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Callable, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class OHLCForecast:
    """OHLC forecast with validation."""
    open_price: float
    high_price: float
    low_price: float
    close_price: float

    def is_valid(self) -> bool:
        """
        Check if forecast maintains valid OHLC ordering.

        Valid OHLC requires:
        - low_price <= close_price <= high_price
        - low_price <= open_price <= high_price
        """
        close_valid = self.low_price <= self.close_price <= self.high_price
        open_valid = self.low_price <= self.open_price <= self.high_price
        return close_valid and open_valid

    def get_violations(self) -> list[str]:
        """Get list of validation violations."""
        violations = []

        if self.high_price < self.low_price:
            violations.append(f"inverted_highlow: high={self.high_price:.4f} < low={self.low_price:.4f}")

        if self.close_price > self.high_price:
            violations.append(f"close_exceeds_high: close={self.close_price:.4f} > high={self.high_price:.4f}")

        if self.close_price < self.low_price:
            violations.append(f"close_below_low: close={self.close_price:.4f} < low={self.low_price:.4f}")

        if self.open_price > self.high_price:
            violations.append(f"open_exceeds_high: open={self.open_price:.4f} > high={self.high_price:.4f}")

        if self.open_price < self.low_price:
            violations.append(f"open_below_low: open={self.open_price:.4f} < low={self.low_price:.4f}")

        return violations

    def correct(self) -> 'OHLCForecast':
        """
        Correct invalid forecasts to maintain OHLC ordering.

        Strategy (matching trade_stock_e2e.py:1487-1515):
        1. If high < low: set both to close
        2. If close > high: set high = close
        3. If close < low: set low = close
        4. If open > high: set high = open
        5. If open < low: set low = open

        Returns:
            Corrected OHLCForecast
        """
        corrected_open = self.open_price
        corrected_high = self.high_price
        corrected_low = self.low_price
        corrected_close = self.close_price

        # Fix inverted high/low - use close as reference
        if corrected_high < corrected_low:
            logger.warning(
                f"Correcting inverted high/low: high={corrected_high:.4f} < low={corrected_low:.4f}, "
                f"setting both to close={corrected_close:.4f}"
            )
            corrected_high = corrected_close
            corrected_low = corrected_close

        # Ensure close is within [low, high]
        if corrected_close > corrected_high:
            logger.warning(
                f"Correcting close exceeds high: close={corrected_close:.4f} > high={corrected_high:.4f}, "
                f"setting high={corrected_close:.4f}"
            )
            corrected_high = corrected_close

        if corrected_close < corrected_low:
            logger.warning(
                f"Correcting close below low: close={corrected_close:.4f} < low={corrected_low:.4f}, "
                f"setting low={corrected_close:.4f}"
            )
            corrected_low = corrected_close

        # Ensure open is within [low, high]
        if corrected_open > corrected_high:
            logger.warning(
                f"Correcting open exceeds high: open={corrected_open:.4f} > high={corrected_high:.4f}, "
                f"setting high={corrected_open:.4f}"
            )
            corrected_high = corrected_open

        if corrected_open < corrected_low:
            logger.warning(
                f"Correcting open below low: open={corrected_open:.4f} < low={corrected_low:.4f}, "
                f"setting low={corrected_open:.4f}"
            )
            corrected_low = corrected_open

        return OHLCForecast(
            open_price=corrected_open,
            high_price=corrected_high,
            low_price=corrected_low,
            close_price=corrected_close,
        )


def forecast_with_retry(
    forecast_fn: Callable[[], OHLCForecast],
    max_retries: int = 2,
    symbol: str = "UNKNOWN",
) -> Tuple[OHLCForecast, int]:
    """
    Attempt to get valid forecast with retries.

    Mirrors the retry logic from trade_stock_e2e.py:1436-1486.

    Args:
        forecast_fn: Function that returns an OHLCForecast
        max_retries: Maximum number of retries (default 2)
        symbol: Symbol name for logging

    Returns:
        Tuple of (forecast, retry_count)
        - If forecast is valid after retries: returns valid forecast
        - If all retries fail: returns corrected forecast
    """
    retry_count = 0

    while retry_count <= max_retries:
        if retry_count > 0:
            logger.info(f"{symbol}: Retrying forecast (attempt {retry_count + 1}/{max_retries + 1})")

        try:
            forecast = forecast_fn()
        except Exception as e:
            logger.warning(f"{symbol}: Forecast attempt {retry_count + 1} failed: {e}")
            retry_count += 1
            continue

        # Check if forecast is valid
        if forecast.is_valid():
            if retry_count > 0:
                logger.info(f"{symbol}: Forecast valid after {retry_count} retries")
            return forecast, retry_count

        # Log violations on first attempt
        if retry_count == 0:
            violations = forecast.get_violations()
            logger.warning(
                f"{symbol}: Invalid forecast detected: {', '.join(violations)}"
            )

        retry_count += 1

    # All retries exhausted - apply corrections
    logger.warning(
        f"{symbol}: All {max_retries} retries failed to produce valid forecast, applying corrections"
    )
    corrected_forecast = forecast.correct()

    return corrected_forecast, retry_count - 1


def validate_and_correct_forecast(
    open_price: float,
    high_price: float,
    low_price: float,
    close_price: float,
    symbol: str = "UNKNOWN",
) -> Tuple[float, float, float, float]:
    """
    Validate and correct a single OHLC forecast.

    Args:
        open_price: Predicted open price
        high_price: Predicted high price
        low_price: Predicted low price
        close_price: Predicted close price
        symbol: Symbol name for logging

    Returns:
        Tuple of (open, high, low, close) corrected prices
    """
    forecast = OHLCForecast(
        open_price=open_price,
        high_price=high_price,
        low_price=low_price,
        close_price=close_price,
    )

    if not forecast.is_valid():
        violations = forecast.get_violations()
        logger.warning(
            f"{symbol}: Invalid forecast detected: {', '.join(violations)}, applying corrections"
        )
        forecast = forecast.correct()

    return forecast.open_price, forecast.high_price, forecast.low_price, forecast.close_price
