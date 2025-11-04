#!/usr/bin/env python3
"""
Unit tests for forecast validation and correction.

Tests the src/forecast_validation.py module to ensure:
1. Valid forecasts are correctly identified
2. Invalid forecasts are detected
3. Corrections maintain OHLC ordering
4. Retry logic works as expected
"""

import pytest
from src.forecast_validation import (
    OHLCForecast,
    forecast_with_retry,
    validate_and_correct_forecast,
)


def test_valid_forecast():
    """Test that valid OHLC forecasts are identified correctly."""
    forecast = OHLCForecast(
        open_price=100.0,
        high_price=105.0,
        low_price=98.0,
        close_price=102.0,
    )

    assert forecast.is_valid(), "Forecast should be valid"
    assert len(forecast.get_violations()) == 0, "Should have no violations"


def test_inverted_high_low():
    """Test detection of inverted high/low prices."""
    forecast = OHLCForecast(
        open_price=100.0,
        high_price=98.0,  # High < Low (invalid)
        low_price=105.0,
        close_price=102.0,
    )

    assert not forecast.is_valid(), "Forecast should be invalid"
    violations = forecast.get_violations()
    assert any("inverted_highlow" in v for v in violations), "Should detect inverted high/low"


def test_close_exceeds_high():
    """Test detection of close exceeding high."""
    forecast = OHLCForecast(
        open_price=100.0,
        high_price=103.0,
        low_price=98.0,
        close_price=105.0,  # Close > High (invalid)
    )

    assert not forecast.is_valid(), "Forecast should be invalid"
    violations = forecast.get_violations()
    assert any("close_exceeds_high" in v for v in violations), "Should detect close > high"


def test_close_below_low():
    """Test detection of close below low."""
    forecast = OHLCForecast(
        open_price=100.0,
        high_price=105.0,
        low_price=98.0,
        close_price=95.0,  # Close < Low (invalid)
    )

    assert not forecast.is_valid(), "Forecast should be invalid"
    violations = forecast.get_violations()
    assert any("close_below_low" in v for v in violations), "Should detect close < low"


def test_open_exceeds_high():
    """Test detection of open exceeding high."""
    forecast = OHLCForecast(
        open_price=107.0,  # Open > High (invalid)
        high_price=105.0,
        low_price=98.0,
        close_price=102.0,
    )

    assert not forecast.is_valid(), "Forecast should be invalid"
    violations = forecast.get_violations()
    assert any("open_exceeds_high" in v for v in violations), "Should detect open > high"


def test_open_below_low():
    """Test detection of open below low."""
    forecast = OHLCForecast(
        open_price=95.0,  # Open < Low (invalid)
        high_price=105.0,
        low_price=98.0,
        close_price=102.0,
    )

    assert not forecast.is_valid(), "Forecast should be invalid"
    violations = forecast.get_violations()
    assert any("open_below_low" in v for v in violations), "Should detect open < low"


def test_correct_inverted_high_low():
    """Test correction of inverted high/low."""
    forecast = OHLCForecast(
        open_price=100.0,
        high_price=98.0,
        low_price=105.0,
        close_price=102.0,
    )

    corrected = forecast.correct()

    assert corrected.is_valid(), "Corrected forecast should be valid"
    # After correction: high=close, low=close, then open adjustment lowers low to open
    assert corrected.high_price == corrected.close_price, "High should be set to close"
    assert corrected.low_price <= corrected.open_price, "Low should accommodate open"
    assert corrected.low_price <= corrected.close_price <= corrected.high_price, "Should maintain OHLC order"


def test_correct_close_exceeds_high():
    """Test correction when close exceeds high."""
    forecast = OHLCForecast(
        open_price=100.0,
        high_price=103.0,
        low_price=98.0,
        close_price=105.0,
    )

    corrected = forecast.correct()

    assert corrected.is_valid(), "Corrected forecast should be valid"
    assert corrected.high_price == 105.0, "High should be adjusted to match close"
    assert corrected.close_price == 105.0, "Close should remain unchanged"


def test_correct_close_below_low():
    """Test correction when close is below low."""
    forecast = OHLCForecast(
        open_price=100.0,
        high_price=105.0,
        low_price=98.0,
        close_price=95.0,
    )

    corrected = forecast.correct()

    assert corrected.is_valid(), "Corrected forecast should be valid"
    assert corrected.low_price == 95.0, "Low should be adjusted to match close"
    assert corrected.close_price == 95.0, "Close should remain unchanged"


def test_correct_open_exceeds_high():
    """Test correction when open exceeds high."""
    forecast = OHLCForecast(
        open_price=107.0,
        high_price=105.0,
        low_price=98.0,
        close_price=102.0,
    )

    corrected = forecast.correct()

    assert corrected.is_valid(), "Corrected forecast should be valid"
    assert corrected.high_price == 107.0, "High should be adjusted to match open"
    assert corrected.open_price == 107.0, "Open should remain unchanged"


def test_correct_open_below_low():
    """Test correction when open is below low."""
    forecast = OHLCForecast(
        open_price=95.0,
        high_price=105.0,
        low_price=98.0,
        close_price=102.0,
    )

    corrected = forecast.correct()

    assert corrected.is_valid(), "Corrected forecast should be valid"
    assert corrected.low_price == 95.0, "Low should be adjusted to match open"
    assert corrected.open_price == 95.0, "Open should remain unchanged"


def test_forecast_with_retry_valid_first_attempt():
    """Test retry logic when forecast is valid on first attempt."""
    call_count = 0

    def valid_forecast_fn():
        nonlocal call_count
        call_count += 1
        return OHLCForecast(100.0, 105.0, 98.0, 102.0)

    forecast, retries = forecast_with_retry(valid_forecast_fn, max_retries=2)

    assert forecast.is_valid(), "Should return valid forecast"
    assert retries == 0, "Should not retry for valid forecast"
    assert call_count == 1, "Should only call forecast function once"


def test_forecast_with_retry_valid_after_one_retry():
    """Test retry logic when forecast becomes valid after one retry."""
    call_count = 0

    def eventually_valid_forecast_fn():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call: invalid
            return OHLCForecast(100.0, 98.0, 105.0, 102.0)
        else:
            # Second call: valid
            return OHLCForecast(100.0, 105.0, 98.0, 102.0)

    forecast, retries = forecast_with_retry(eventually_valid_forecast_fn, max_retries=2)

    assert forecast.is_valid(), "Should return valid forecast"
    assert retries == 1, "Should retry once"
    assert call_count == 2, "Should call forecast function twice"


def test_forecast_with_retry_all_retries_exhausted():
    """Test retry logic when all retries are exhausted."""
    call_count = 0

    def always_invalid_forecast_fn():
        nonlocal call_count
        call_count += 1
        # Always return invalid forecast
        return OHLCForecast(100.0, 98.0, 105.0, 102.0)

    forecast, retries = forecast_with_retry(always_invalid_forecast_fn, max_retries=2)

    assert forecast.is_valid(), "Should return corrected forecast"
    assert retries == 2, "Should exhaust all retries"
    assert call_count == 3, "Should call forecast function 3 times (initial + 2 retries)"


def test_forecast_with_retry_exception_handling():
    """Test retry logic handles exceptions."""
    call_count = 0

    def failing_forecast_fn():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise RuntimeError("Forecast failed")
        # Third attempt succeeds
        return OHLCForecast(100.0, 105.0, 98.0, 102.0)

    forecast, retries = forecast_with_retry(failing_forecast_fn, max_retries=2)

    assert forecast.is_valid(), "Should return valid forecast after exceptions"
    assert retries == 2, "Should count retries including exceptions"
    assert call_count == 3, "Should call forecast function 3 times"


def test_validate_and_correct_forecast_valid():
    """Test validation and correction for valid forecast."""
    o, h, l, c = validate_and_correct_forecast(
        open_price=100.0,
        high_price=105.0,
        low_price=98.0,
        close_price=102.0,
    )

    assert l <= c <= h, "Should maintain valid ordering"
    assert l <= o <= h, "Should maintain valid ordering for open"
    assert (o, h, l, c) == (100.0, 105.0, 98.0, 102.0), "Valid forecast should not be modified"


def test_validate_and_correct_forecast_invalid():
    """Test validation and correction for invalid forecast."""
    o, h, l, c = validate_and_correct_forecast(
        open_price=100.0,
        high_price=98.0,  # Inverted
        low_price=105.0,  # Inverted
        close_price=102.0,
    )

    assert l <= c <= h, "Should correct to valid ordering"
    assert l <= o <= h, "Should correct to valid ordering for open"


def test_edge_case_all_prices_equal():
    """Test edge case where all prices are equal."""
    forecast = OHLCForecast(100.0, 100.0, 100.0, 100.0)

    assert forecast.is_valid(), "All equal prices should be valid"
    assert len(forecast.get_violations()) == 0, "Should have no violations"


def test_edge_case_close_equals_high():
    """Test edge case where close equals high."""
    forecast = OHLCForecast(100.0, 105.0, 98.0, 105.0)

    assert forecast.is_valid(), "Close == High should be valid"


def test_edge_case_close_equals_low():
    """Test edge case where close equals low."""
    forecast = OHLCForecast(100.0, 105.0, 98.0, 98.0)

    assert forecast.is_valid(), "Close == Low should be valid"


def run_all_tests():
    """Run all tests and report results."""
    import sys

    print("=" * 60)
    print("Running Forecast Validation Tests")
    print("=" * 60)
    print()

    # Use pytest to run tests
    exit_code = pytest.main([__file__, "-v", "--tb=short"])

    sys.exit(exit_code)


if __name__ == "__main__":
    run_all_tests()
