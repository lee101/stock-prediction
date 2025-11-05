"""
Unit tests for pure functions extracted from backtest_test3_inline.py and trade_stock_e2e.py.
Tests validation, return calculations, signal calibration, and strategy logic.
"""

import pytest
import torch
import numpy as np

from src.backtest_pure_functions import (
    validate_forecast_order,
    compute_return_profile,
    calibrate_signal,
    simple_buy_sell_strategy,
    all_signals_strategy,
    buy_hold_strategy,
    calculate_position_notional_value,
)


class TestForecastValidation:
    """Tests for forecast order validation"""

    def test_valid_forecast_order(self):
        """low < high should be valid"""
        high_pred = torch.tensor([0.02, 0.03, 0.01])
        low_pred = torch.tensor([-0.01, 0.01, -0.02])
        valid = validate_forecast_order(high_pred, low_pred)
        assert torch.all(valid)

    def test_invalid_forecast_order(self):
        """high < low should be invalid"""
        high_pred = torch.tensor([0.01, 0.02])
        low_pred = torch.tensor([0.02, 0.03])
        valid = validate_forecast_order(high_pred, low_pred)
        assert not torch.any(valid)

    def test_mixed_validity(self):
        """Mix of valid and invalid forecasts"""
        high_pred = torch.tensor([0.02, 0.01, 0.03])
        low_pred = torch.tensor([-0.01, 0.02, 0.01])
        valid = validate_forecast_order(high_pred, low_pred)
        expected = torch.tensor([True, False, True])
        assert torch.equal(valid, expected)

    def test_equal_high_low(self):
        """Equal values should be invalid"""
        high_pred = torch.tensor([0.01, 0.01])
        low_pred = torch.tensor([0.01, 0.01])
        valid = validate_forecast_order(high_pred, low_pred)
        assert not torch.any(valid)


class TestReturnProfile:
    """Tests for return profile computation"""

    def test_compute_return_profile_positive(self):
        """Test with positive returns"""
        returns = np.array([0.01, 0.02, 0.015, 0.01])
        avg_daily, annualized = compute_return_profile(returns, trading_days_per_year=252)
        assert avg_daily == pytest.approx(0.01375)
        assert annualized == pytest.approx(0.01375 * 252)

    def test_compute_return_profile_negative(self):
        """Test with negative returns"""
        returns = np.array([-0.01, -0.02, -0.015])
        avg_daily, annualized = compute_return_profile(returns, trading_days_per_year=252)
        assert avg_daily < 0
        assert annualized < 0

    def test_compute_return_profile_empty(self):
        """Empty returns should return zeros"""
        returns = np.array([])
        avg_daily, annualized = compute_return_profile(returns, trading_days_per_year=252)
        assert avg_daily == 0.0
        assert annualized == 0.0

    def test_compute_return_profile_with_nan(self):
        """NaN values should be filtered"""
        returns = np.array([0.01, np.nan, 0.02, np.inf, 0.015])
        avg_daily, annualized = compute_return_profile(returns, trading_days_per_year=252)
        assert np.isfinite(avg_daily)
        assert np.isfinite(annualized)
        expected_avg = (0.01 + 0.02 + 0.015) / 3
        assert avg_daily == pytest.approx(expected_avg)

    def test_compute_return_profile_zero_trading_days(self):
        """Zero trading days should return zeros"""
        returns = np.array([0.01, 0.02])
        avg_daily, annualized = compute_return_profile(returns, trading_days_per_year=0)
        assert avg_daily == 0.0
        assert annualized == 0.0

    def test_compute_return_profile_crypto_365_days(self):
        """Crypto with 365 trading days"""
        returns = np.array([0.01, 0.01, 0.01])
        avg_daily, annualized = compute_return_profile(returns, trading_days_per_year=365)
        assert avg_daily == pytest.approx(0.01)
        assert annualized == pytest.approx(0.01 * 365)


class TestCalibrateSignal:
    """Tests for signal calibration"""

    def test_calibrate_signal_perfect_correlation(self):
        """Perfect linear correlation"""
        predictions = np.array([0.01, 0.02, 0.03, 0.04])
        actual_returns = np.array([0.01, 0.02, 0.03, 0.04])
        slope, intercept = calibrate_signal(predictions, actual_returns)

        assert slope == pytest.approx(1.0, abs=0.01)
        assert intercept == pytest.approx(0.0, abs=0.01)

    def test_calibrate_signal_scaled(self):
        """Predictions scaled by 2"""
        predictions = np.array([0.01, 0.02, 0.03])
        actual_returns = np.array([0.02, 0.04, 0.06])
        slope, intercept = calibrate_signal(predictions, actual_returns)

        assert slope == pytest.approx(2.0, abs=0.01)
        assert intercept == pytest.approx(0.0, abs=0.01)

    def test_calibrate_signal_with_offset(self):
        """Predictions with constant offset"""
        predictions = np.array([0.00, 0.01, 0.02])
        actual_returns = np.array([0.01, 0.02, 0.03])
        slope, intercept = calibrate_signal(predictions, actual_returns)

        assert slope == pytest.approx(1.0, abs=0.01)
        assert intercept == pytest.approx(0.01, abs=0.01)

    def test_calibrate_signal_insufficient_data(self):
        """Single data point should return defaults"""
        predictions = np.array([0.01])
        actual_returns = np.array([0.02])
        slope, intercept = calibrate_signal(predictions, actual_returns)

        assert slope == 1.0
        assert intercept == 0.0

    def test_calibrate_signal_mismatched_length(self):
        """Different lengths should use minimum"""
        predictions = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        actual_returns = np.array([0.01, 0.02, 0.03])
        slope, intercept = calibrate_signal(predictions, actual_returns)

        assert np.isfinite(slope)
        assert np.isfinite(intercept)


class TestSimpleBuySellStrategy:
    """Tests for simple buy/sell strategy"""

    def test_simple_buy_sell_crypto_long_only(self):
        """Crypto should only allow longs"""
        predictions = torch.tensor([0.01, 0.02, -0.01, -0.02, 0.0])
        positions = simple_buy_sell_strategy(predictions, is_crypto=True)

        expected = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
        assert torch.equal(positions, expected)

    def test_simple_buy_sell_stocks(self):
        """Stocks allow longs and shorts"""
        predictions = torch.tensor([0.01, 0.02, -0.01, -0.02, 0.0])
        positions = simple_buy_sell_strategy(predictions, is_crypto=False)

        expected = torch.tensor([1.0, 1.0, -1.0, -1.0, -1.0])
        assert torch.equal(positions, expected)

    def test_simple_buy_sell_zero_predictions_stock(self):
        """Zero predictions should be short for stocks"""
        predictions = torch.tensor([0.0, 0.0])
        positions = simple_buy_sell_strategy(predictions, is_crypto=False)

        expected = torch.tensor([-1.0, -1.0])
        assert torch.equal(positions, expected)


class TestAllSignalsStrategy:
    """Tests for all signals strategy"""

    def test_all_signals_all_bullish(self):
        """All positive signals"""
        close_pred = torch.tensor([0.01, 0.02])
        high_pred = torch.tensor([0.015, 0.025])
        low_pred = torch.tensor([0.005, 0.01])

        positions = all_signals_strategy(close_pred, high_pred, low_pred, is_crypto=False)
        expected = torch.tensor([1.0, 1.0])
        assert torch.equal(positions, expected)

    def test_all_signals_all_bearish(self):
        """All negative signals"""
        close_pred = torch.tensor([-0.01, -0.02])
        high_pred = torch.tensor([-0.005, -0.01])
        low_pred = torch.tensor([-0.015, -0.025])

        positions = all_signals_strategy(close_pred, high_pred, low_pred, is_crypto=False)
        expected = torch.tensor([-1.0, -1.0])
        assert torch.equal(positions, expected)

    def test_all_signals_mixed(self):
        """Mixed signals should hold (0)"""
        close_pred = torch.tensor([0.01, -0.01])
        high_pred = torch.tensor([0.015, 0.01])
        low_pred = torch.tensor([0.005, -0.02])

        positions = all_signals_strategy(close_pred, high_pred, low_pred, is_crypto=False)
        expected = torch.tensor([1.0, 0.0])
        assert torch.equal(positions, expected)

    def test_all_signals_crypto_no_shorts(self):
        """Crypto should not short"""
        close_pred = torch.tensor([-0.01, -0.02])
        high_pred = torch.tensor([-0.005, -0.01])
        low_pred = torch.tensor([-0.015, -0.025])

        positions = all_signals_strategy(close_pred, high_pred, low_pred, is_crypto=True)
        expected = torch.tensor([0.0, 0.0])
        assert torch.equal(positions, expected)


class TestBuyHoldStrategy:
    """Tests for buy and hold strategy"""

    def test_buy_hold_positive(self):
        """Positive predictions should buy"""
        predictions = torch.tensor([0.01, 0.02, 0.001])
        positions = buy_hold_strategy(predictions)

        expected = torch.tensor([1.0, 1.0, 1.0])
        assert torch.equal(positions, expected)

    def test_buy_hold_negative(self):
        """Negative predictions should hold (0)"""
        predictions = torch.tensor([-0.01, -0.02])
        positions = buy_hold_strategy(predictions)

        expected = torch.tensor([0.0, 0.0])
        assert torch.equal(positions, expected)

    def test_buy_hold_mixed(self):
        """Mixed predictions"""
        predictions = torch.tensor([0.01, -0.01, 0.0, 0.005])
        positions = buy_hold_strategy(predictions)

        expected = torch.tensor([1.0, 0.0, 0.0, 1.0])
        assert torch.equal(positions, expected)


class TestPositionCalculations:
    """Tests for position value calculations"""

    def test_position_notional_value_with_market_value(self):
        """Test notional value using market_value"""
        value = calculate_position_notional_value(
            market_value=-1500.0,
            qty=10.0,
            current_price=150.0
        )
        assert value == 1500.0

    def test_position_notional_value_with_qty_price(self):
        """Test notional value using qty * current_price"""
        value = calculate_position_notional_value(
            market_value=0.0,
            qty=10.0,
            current_price=150.0
        )
        assert value == 1500.0

    def test_position_notional_value_fallback_to_qty(self):
        """Test fallback to qty when no price available"""
        value = calculate_position_notional_value(
            market_value=0.0,
            qty=-25.0,
            current_price=0.0
        )
        assert value == 25.0

    def test_position_notional_value_with_nan(self):
        """Test handling of NaN values"""
        value = calculate_position_notional_value(
            market_value=np.nan,
            qty=10.0,
            current_price=50.0
        )
        assert value == 500.0

    def test_position_notional_value_negative_qty(self):
        """Test with negative qty (short position)"""
        value = calculate_position_notional_value(
            market_value=0.0,
            qty=-15.0,
            current_price=100.0
        )
        assert value == 1500.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
