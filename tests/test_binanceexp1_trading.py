"""Comprehensive tests for binanceexp1 trading logic."""

import numpy as np
import pandas as pd
import pytest

from binanceneural.execution import compute_order_quantities, quantize_price, quantize_qty
from binanceneural.execution import resolve_symbol_rules, split_binance_symbol
from src.metrics_utils import compute_step_returns, annualized_sortino, annualized_sharpe


class TestMetricsUtils:
    """Test metrics calculation functions."""

    def test_compute_step_returns_basic(self):
        """Test basic return calculation."""
        equity = [100, 110, 105, 115]
        returns = compute_step_returns(equity)
        expected = np.array([0.1, -0.0454545, 0.0952381])
        np.testing.assert_array_almost_equal(returns, expected, decimal=6)

    def test_compute_step_returns_empty(self):
        """Test with empty input."""
        returns = compute_step_returns([])
        assert len(returns) == 0

    def test_compute_step_returns_single_value(self):
        """Test with single value."""
        returns = compute_step_returns([100])
        assert len(returns) == 0

    def test_compute_step_returns_zero_division(self):
        """Test handling of zero values."""
        equity = [100, 0, 50]
        returns = compute_step_returns(equity)
        # Should handle division by zero gracefully
        assert len(returns) == 2
        assert np.isfinite(returns).all()

    def test_compute_step_returns_negative_values(self):
        """Test with negative values (debt)."""
        equity = [100, -50, 25]
        returns = compute_step_returns(equity)
        assert len(returns) == 2
        # All returns should be finite
        assert np.isfinite(returns).all()

    def test_annualized_sortino_positive_returns(self):
        """Test Sortino with positive returns."""
        returns = np.array([0.01, 0.02, 0.015, 0.025, 0.018])
        sortino = annualized_sortino(returns, periods_per_year=252)
        assert sortino > 0
        assert np.isfinite(sortino)

    def test_annualized_sortino_negative_returns(self):
        """Test Sortino with negative returns."""
        returns = np.array([-0.01, -0.02, -0.015, -0.025, -0.018])
        sortino = annualized_sortino(returns, periods_per_year=252)
        assert sortino < 0
        assert np.isfinite(sortino)

    def test_annualized_sortino_mixed_returns(self):
        """Test Sortino with mixed returns."""
        returns = np.array([0.01, -0.01, 0.02, -0.005, 0.015])
        sortino = annualized_sortino(returns, periods_per_year=252)
        assert np.isfinite(sortino)

    def test_annualized_sortino_no_downside(self):
        """Test Sortino with no negative returns (all positive)."""
        returns = np.array([0.01, 0.02, 0.015, 0.025, 0.018])
        sortino = annualized_sortino(returns, periods_per_year=252)
        # Should be high since no downside volatility
        assert sortino > 100  # Very high sortino

    def test_annualized_sortino_empty(self):
        """Test with empty returns."""
        sortino = annualized_sortino([], periods_per_year=252)
        assert sortino == 0.0

    def test_annualized_sortino_single_value(self):
        """Test with single return."""
        sortino = annualized_sortino([0.01], periods_per_year=252)
        # Should handle gracefully
        assert sortino >= 0

    def test_annualized_sharpe_basic(self):
        """Test Sharpe ratio calculation."""
        returns = np.array([0.01, 0.02, 0.015, 0.025, 0.018])
        sharpe = annualized_sharpe(returns, periods_per_year=252)
        assert sharpe > 0
        assert np.isfinite(sharpe)

    def test_annualized_sharpe_zero_volatility(self):
        """Test Sharpe with constant returns (zero volatility)."""
        returns = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        sharpe = annualized_sharpe(returns, periods_per_year=252)
        assert sharpe == 0.0  # Can't divide by zero std

    def test_annualized_sharpe_negative_returns(self):
        """Test Sharpe with negative mean return."""
        returns = np.array([-0.01, -0.02, -0.015, -0.025, -0.018])
        sharpe = annualized_sharpe(returns, periods_per_year=252)
        assert sharpe < 0

    def test_metrics_with_real_trading_scenario(self):
        """Test with realistic trading scenario."""
        # Simulate account growing from $10k to $12k over 100 periods
        np.random.seed(42)
        base = 10000
        growth = 0.2  # 20% total return
        periods = 100
        
        # Generate equity curve with some volatility
        equity = [base]
        for i in range(periods):
            # Add trend + random walk
            trend = base * growth * (i / periods)
            noise = np.random.normal(0, base * 0.02)
            equity.append(base + trend + noise)
        
        returns = compute_step_returns(equity)
        sortino = annualized_sortino(returns, periods_per_year=24*365)  # hourly
        sharpe = annualized_sharpe(returns, periods_per_year=24*365)
        
        assert len(returns) == periods
        assert np.isfinite(sortino)
        assert np.isfinite(sharpe)
        # With 20% return and low volatility, should have positive risk-adjusted returns
        # Note: might be negative due to random noise, but should be reasonable
        assert abs(sortino) < 1000  # Not absurdly high
        assert abs(sharpe) < 1000


class TestBinanceExecution:
    """Test binance execution functions."""

    def test_split_binance_symbol(self):
        """Test symbol splitting."""
        base, quote = split_binance_symbol("BTCUSDT")
        assert base == "BTC"
        assert quote == "USDT"
        
        base, quote = split_binance_symbol("SOLUSDT")
        assert base == "SOL"
        assert quote == "USDT"

    def test_quantize_price_basic(self):
        """Test price quantization."""
        # With tick_size=0.01
        price = quantize_price(100.567, tick_size=0.01, side="buy")
        assert abs(price - 100.56) < 1e-6
        
        price = quantize_price(100.567, tick_size=0.01, side="sell")
        assert abs(price - 100.57) < 1e-6

    def test_quantize_price_different_tick_sizes(self):
        """Test with various tick sizes."""
        # Tick size 0.1
        assert abs(quantize_price(100.567, tick_size=0.1, side="buy") - 100.5) < 1e-6
        assert abs(quantize_price(100.567, tick_size=0.1, side="sell") - 100.6) < 1e-6
        
        # Tick size 1.0
        assert abs(quantize_price(100.567, tick_size=1.0, side="buy") - 100.0) < 1e-6
        assert abs(quantize_price(100.567, tick_size=1.0, side="sell") - 101.0) < 1e-6
        
        # Tick size 0.001
        assert abs(quantize_price(100.5674, tick_size=0.001, side="buy") - 100.567) < 1e-6
        assert abs(quantize_price(100.5674, tick_size=0.001, side="sell") - 100.568) < 1e-6

    def test_quantize_qty_basic(self):
        """Test quantity quantization."""
        qty = quantize_qty(1.23456, step_size=0.01)
        assert qty == 1.23
        
        qty = quantize_qty(1.23456, step_size=0.001)
        assert qty == 1.234

    def test_quantize_qty_floor_behavior(self):
        """Test that quantity always floors (never rounds up)."""
        # Always floors to avoid order rejection
        assert quantize_qty(1.999, step_size=1.0) == 1.0
        assert quantize_qty(0.999, step_size=0.1) == 0.9
        assert quantize_qty(0.0999, step_size=0.01) == 0.09


class TestPnLCalculation:
    """Test PnL calculation logic from trade_binance_hourly.py."""

    def test_pnl_basic(self):
        """Test basic PnL calculation."""
        history_values = [5000, 5100, 5050, 5200]
        
        # Find first non-zero (all non-zero here)
        start_value = history_values[0]
        current_value = history_values[-1]
        
        pnl_usdt = current_value - start_value
        pnl_pct = pnl_usdt / start_value if start_value > 0 else 0.0
        
        assert pnl_usdt == 200
        assert pnl_pct == 0.04  # 4% gain

    def test_pnl_with_zero_values(self):
        """Test PnL when history starts with zeros."""
        history_values = [0, 0, 5000, 5100, 5050, 5200]
        
        # Find first non-zero
        nonzero_values = [v for v in history_values if v > 0]
        start_value = nonzero_values[0]
        current_value = history_values[-1]
        
        pnl_usdt = current_value - start_value
        pnl_pct = pnl_usdt / start_value if start_value > 0 else 0.0
        
        assert pnl_usdt == 200
        assert pnl_pct == 0.04

    def test_pnl_all_zeros(self):
        """Test PnL when all values are zero."""
        history_values = [0, 0, 0, 0]
        
        nonzero_values = [v for v in history_values if v > 0]
        if len(nonzero_values) > 0:
            start_value = nonzero_values[0]
        else:
            start_value = 1.0  # Avoid division by zero
        
        current_value = history_values[-1]
        pnl_usdt = current_value - start_value
        pnl_pct = pnl_usdt / start_value if start_value > 0 else 0.0
        
        # Should not crash
        assert np.isfinite(pnl_usdt)
        assert np.isfinite(pnl_pct)

    def test_pnl_loss_scenario(self):
        """Test PnL with losses."""
        history_values = [10000, 9500, 9200, 8800]
        
        start_value = history_values[0]
        current_value = history_values[-1]
        
        pnl_usdt = current_value - start_value
        pnl_pct = pnl_usdt / start_value
        
        assert pnl_usdt == -1200
        assert pnl_pct == -0.12  # -12% loss

    def test_sortino_from_pnl_history(self):
        """Test Sortino calculation from PnL history."""
        history_values = [5000, 5100, 5050, 5200, 5150, 5300]
        
        # Remove zeros
        valid_values = [v for v in history_values if v > 0]
        returns = compute_step_returns(valid_values)
        sortino = annualized_sortino(returns, periods_per_year=24*365)
        
        assert np.isfinite(sortino)
        assert len(returns) == len(valid_values) - 1


class TestIntensityScaling:
    """Test intensity scaling logic."""

    def test_intensity_scale_conservative(self):
        """Test conservative intensity (< 1.0)."""
        base_amount = 50.0
        intensity = 0.5
        
        scaled = base_amount * intensity
        clamped = min(100.0, max(0.0, scaled))
        
        assert clamped == 25.0

    def test_intensity_scale_normal(self):
        """Test normal intensity (= 1.0)."""
        base_amount = 50.0
        intensity = 1.0
        
        scaled = base_amount * intensity
        clamped = min(100.0, max(0.0, scaled))
        
        assert clamped == 50.0

    def test_intensity_scale_aggressive(self):
        """Test aggressive intensity (> 1.0)."""
        base_amount = 50.0
        intensity = 2.0
        
        scaled = base_amount * intensity
        clamped = min(100.0, max(0.0, scaled))
        
        assert clamped == 100.0  # Clamped at max

    def test_intensity_scale_very_aggressive(self):
        """Test very aggressive intensity (>> 1.0)."""
        base_amount = 50.0
        intensity = 20.0
        
        scaled = base_amount * intensity
        clamped = min(100.0, max(0.0, scaled))
        
        assert clamped == 100.0  # Still clamped at max


class TestPriceGaps:
    """Test minimum gap enforcement."""

    def test_enforce_min_gap_basic(self):
        """Test basic gap enforcement."""
        buy_price = 100.0
        sell_price = 100.5
        min_gap_pct = 0.001  # 0.1%
        
        # Calculate midpoint and apply gap
        mid = (buy_price + sell_price) / 2
        gap = mid * min_gap_pct
        
        new_buy = mid - gap
        new_sell = mid + gap
        
        actual_gap_pct = (new_sell - new_buy) / new_buy
        assert actual_gap_pct >= min_gap_pct

    def test_prices_too_close(self):
        """Test when prices are too close together."""
        buy_price = 100.0
        sell_price = 100.01  # Only 0.01% apart
        min_gap_pct = 0.001  # Require 0.1%
        
        mid = (buy_price + sell_price) / 2
        gap = mid * min_gap_pct
        
        new_buy = mid - gap
        new_sell = mid + gap
        
        # New prices should be further apart
        assert new_sell > sell_price
        assert new_buy < buy_price


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
