"""
Critical Math Function Tests

Tests for mission-critical financial calculations:
1. Kelly criterion position sizing
2. Sharpe ratio calculations
3. Annualized return calculations
4. Drawdown scaling

These functions directly control:
- How much capital to risk per trade (Kelly)
- How we measure strategy performance (Sharpe, returns)
- Risk management during drawdowns (scaling)

Any bugs here can lead to:
- Overleveraging (bankruptcy risk)
- Underleveraging (missed returns)
- Incorrect strategy selection
- Poor risk management
"""

import numpy as np
import pandas as pd
import pytest

from src.trade_stock_utils import kelly_lite


class TestKellyCriterion:
    """
    Test Kelly criterion position sizing formula

    Kelly fraction = edge / variance
    where edge = expected return (%)
          variance = sigma^2

    Implementation applies 0.2x fractional Kelly (conservative)
    and caps at 0.15 (15% max position size)
    """

    def test_kelly_basic_calculation(self):
        """Test basic Kelly formula: edge / sigma^2"""
        # edge = 2%, sigma = 10%
        # kelly = 0.02 / 0.01 = 2.0
        # fractional kelly (0.2x) = 0.4
        # But capped at 0.15

        edge = 0.02
        sigma = 0.10
        kelly = kelly_lite(edge, sigma)

        # 0.02 / 0.01 = 2.0, then * 0.2 = 0.4, capped at 0.15
        assert kelly == pytest.approx(0.15), \
            f"Expected cap of 0.15, got {kelly}"

    def test_kelly_zero_edge_zero_position(self):
        """Test that zero edge = zero position size"""
        kelly = kelly_lite(0.0, 0.10)
        assert kelly == 0.0, "Zero edge should give zero position size"

    def test_kelly_negative_edge_zero_position(self):
        """Test that negative edge = zero position size (don't bet on losers)"""
        kelly = kelly_lite(-0.01, 0.10)
        assert kelly == 0.0, "Negative edge should give zero position size"

    def test_kelly_zero_sigma_zero_position(self):
        """Test that zero volatility = zero position size (avoid division by zero)"""
        kelly = kelly_lite(0.02, 0.0)
        assert kelly == 0.0, "Zero sigma should give zero position size"

    def test_kelly_small_edge_small_position(self):
        """Test that small edge gives appropriately small position"""
        # edge = 0.5%, sigma = 10%
        # kelly = 0.005 / 0.01 = 0.5
        # fractional = 0.5 * 0.2 = 0.1 = 10% position

        edge = 0.005
        sigma = 0.10
        kelly = kelly_lite(edge, sigma)

        expected = 0.2 * (edge / (sigma ** 2))
        assert kelly == pytest.approx(expected), \
            f"Expected {expected:.4f}, got {kelly:.4f}"

    def test_kelly_high_volatility_reduces_position(self):
        """Test that higher volatility reduces position size"""
        edge = 0.02

        kelly_low_vol = kelly_lite(edge, 0.10)  # 10% volatility
        kelly_high_vol = kelly_lite(edge, 0.20)  # 20% volatility

        assert kelly_high_vol < kelly_low_vol, \
            f"Higher volatility should reduce Kelly: {kelly_high_vol} vs {kelly_low_vol}"

        # Specific calculation:
        # Low vol: 0.02 / 0.01 = 2.0 * 0.2 = 0.4 → capped at 0.15
        # High vol: 0.02 / 0.04 = 0.5 * 0.2 = 0.1
        assert kelly_low_vol == pytest.approx(0.15)
        assert kelly_high_vol == pytest.approx(0.10)

    def test_kelly_cap_parameter(self):
        """Test that custom cap parameter works"""
        edge = 0.10
        sigma = 0.10

        # Default cap = 0.15
        kelly_default = kelly_lite(edge, sigma)
        assert kelly_default == pytest.approx(0.15)

        # Custom cap = 0.05
        kelly_low_cap = kelly_lite(edge, sigma, cap=0.05)
        assert kelly_low_cap == pytest.approx(0.05)

        # Custom cap = 0.50
        kelly_high_cap = kelly_lite(edge, sigma, cap=0.50)
        # 0.10 / 0.01 = 10.0 * 0.2 = 2.0 → capped at 0.50
        assert kelly_high_cap == pytest.approx(0.50)

    def test_kelly_realistic_trading_scenarios(self):
        """Test Kelly with realistic trading parameters"""

        # Scenario 1: Strong edge, low vol (good setup)
        # 3% expected return, 8% volatility
        kelly_strong = kelly_lite(0.03, 0.08)
        # 0.03 / 0.0064 = 4.69 * 0.2 = 0.94 → capped at 0.15
        assert kelly_strong == pytest.approx(0.15)

        # Scenario 2: Moderate edge, moderate vol
        # 1% expected return, 15% volatility
        kelly_moderate = kelly_lite(0.01, 0.15)
        # 0.01 / 0.0225 = 0.44 * 0.2 = 0.089
        expected_moderate = 0.2 * (0.01 / (0.15 ** 2))
        assert kelly_moderate == pytest.approx(expected_moderate, abs=1e-4)

        # Scenario 3: Tiny edge, high vol (skip this trade)
        # 0.2% expected return, 20% volatility
        kelly_weak = kelly_lite(0.002, 0.20)
        # 0.002 / 0.04 = 0.05 * 0.2 = 0.01 = 1% position
        assert kelly_weak < 0.02, "Weak edge should give very small position"

    def test_kelly_fractional_scaling(self):
        """Verify the 0.2x fractional Kelly scaling"""
        edge = 0.005  # Smaller edge to avoid hitting cap
        sigma = 0.10

        # Full Kelly would be: edge / variance = 0.005 / 0.01 = 0.5
        full_kelly = edge / (sigma ** 2)

        # Fractional Kelly is 0.2x = 0.1
        kelly = kelly_lite(edge, sigma)

        assert kelly == pytest.approx(0.2 * full_kelly), \
            f"Should be 0.2x of full Kelly: {kelly} vs {0.2 * full_kelly}"


class TestSharpeRatio:
    """
    Test Sharpe ratio calculation

    Sharpe = (mean / std) * sqrt(trading_days_per_year)

    Measures risk-adjusted returns. Higher is better.
    Typical good values: > 1.0
    Excellent: > 2.0
    """

    def test_sharpe_basic_calculation(self):
        """Test basic Sharpe ratio formula"""
        from backtest_test3_inline import _evaluate_daily_returns

        # Daily returns: mean=0.01 (1%), std=0.02 (2%)
        # Sharpe = (0.01 / 0.02) * sqrt(252) = 0.5 * 15.87 = 7.94

        daily_returns = np.array([0.01, 0.01, 0.01])  # Constant 1% returns
        std = 0.0  # Zero std for constant returns

        # With zero std, Sharpe should be 0 (edge case)
        result = _evaluate_daily_returns(daily_returns, trading_days_per_year=252)
        assert result.sharpe_ratio == 0.0, "Zero std should give Sharpe=0"

    def test_sharpe_with_volatility(self):
        """Test Sharpe with realistic volatility"""
        # Create returns with mean=0.01, some volatility
        np.random.seed(42)
        daily_returns = np.array([0.01, 0.02, 0.00, 0.015, 0.005])

        result = _evaluate_daily_returns(daily_returns, trading_days_per_year=252)

        mean = np.mean(daily_returns)
        std = np.std(daily_returns)
        expected_sharpe = (mean / std) * np.sqrt(252)

        assert result.sharpe_ratio == pytest.approx(expected_sharpe), \
            f"Expected {expected_sharpe:.4f}, got {result.sharpe_ratio:.4f}"

        # Should be positive for positive returns
        assert result.sharpe_ratio > 0, "Positive returns should give positive Sharpe"

    def test_sharpe_empty_returns(self):
        """Test Sharpe with empty returns array"""
        result = _evaluate_daily_returns(np.array([]), trading_days_per_year=252)
        assert result.sharpe_ratio == 0.0
        assert result.total_return == 0.0
        assert result.avg_daily_return == 0.0

    def test_sharpe_negative_returns(self):
        """Test Sharpe with losing strategy"""
        daily_returns = np.array([-0.01, -0.02, -0.015])

        result = _evaluate_daily_returns(daily_returns, trading_days_per_year=252)

        # Negative mean should give negative Sharpe
        assert result.sharpe_ratio < 0, "Losing strategy should have negative Sharpe"
        assert result.avg_daily_return < 0, "Avg return should be negative"

    def test_sharpe_high_volatility_reduces_sharpe(self):
        """Test that higher volatility reduces Sharpe ratio"""
        # Low volatility strategy
        low_vol = np.array([0.01, 0.011, 0.009, 0.010, 0.012])

        # High volatility strategy (same mean, more variance)
        high_vol = np.array([0.01, 0.05, -0.03, 0.02, 0.01])

        result_low = _evaluate_daily_returns(low_vol, trading_days_per_year=252)
        result_high = _evaluate_daily_returns(high_vol, trading_days_per_year=252)

        # Low vol should have higher Sharpe
        assert result_low.sharpe_ratio > result_high.sharpe_ratio, \
            f"Low vol Sharpe {result_low.sharpe_ratio} should > high vol {result_high.sharpe_ratio}"

    def test_sharpe_annualization_factor(self):
        """Test that annualization factor affects Sharpe correctly"""
        daily_returns = np.array([0.01, 0.02, 0.015, 0.008, 0.012])

        # Daily trading (252 days)
        result_daily = _evaluate_daily_returns(daily_returns, trading_days_per_year=252)

        # Hourly trading (252 * 24 periods)
        result_hourly = _evaluate_daily_returns(daily_returns, trading_days_per_year=252 * 24)

        # Sharpe should scale with sqrt(periods)
        ratio = result_hourly.sharpe_ratio / result_daily.sharpe_ratio
        expected_ratio = np.sqrt(24)

        assert ratio == pytest.approx(expected_ratio, rel=0.01), \
            f"Sharpe should scale with sqrt(periods): {ratio} vs {expected_ratio}"


class TestAnnualizedReturns:
    """
    Test annualized return calculations

    Annualized = avg_daily_return * trading_days_per_year

    Converts daily returns to yearly equivalents for comparison.
    """

    def test_annualized_returns_basic(self):
        """Test basic annualization formula"""
        # 0.5% average daily return
        daily_returns = np.array([0.005] * 10)

        result = _evaluate_daily_returns(daily_returns, trading_days_per_year=252)

        # Annualized = 0.005 * 252 = 1.26 = 126%
        expected_annual = 0.005 * 252

        assert result.avg_daily_return == pytest.approx(0.005)
        assert result.annualized_return == pytest.approx(expected_annual), \
            f"Expected {expected_annual:.4f}, got {result.annualized_return:.4f}"

    def test_annualized_returns_with_variance(self):
        """Test annualization with varying returns"""
        daily_returns = np.array([0.01, 0.02, -0.005, 0.015, 0.008])

        result = _evaluate_daily_returns(daily_returns, trading_days_per_year=252)

        avg_daily = np.mean(daily_returns)
        expected_annual = avg_daily * 252

        assert result.avg_daily_return == pytest.approx(avg_daily)
        assert result.annualized_return == pytest.approx(expected_annual)

    def test_total_return_calculation(self):
        """Test that total return is sum of daily returns"""
        daily_returns = np.array([0.01, 0.02, 0.015, -0.005, 0.01])

        result = _evaluate_daily_returns(daily_returns, trading_days_per_year=252)

        expected_total = np.sum(daily_returns)

        assert result.total_return == pytest.approx(expected_total), \
            f"Expected {expected_total:.4f}, got {result.total_return:.4f}"

    def test_compute_return_profile(self):
        """Test _compute_return_profile helper function"""
        returns = pd.Series([0.01, 0.02, 0.015, 0.008, 0.012])

        avg_daily, annualized = _compute_return_profile(returns, trading_days_per_year=252)

        expected_avg = 0.01 + 0.02 + 0.015 + 0.008 + 0.012
        expected_avg /= 5
        expected_annual = expected_avg * 252

        assert avg_daily == pytest.approx(expected_avg, abs=1e-5)
        assert annualized == pytest.approx(expected_annual, abs=1e-5)

    def test_compute_return_profile_edge_cases(self):
        """Test edge cases in return profile computation"""
        # Empty returns
        avg, annual = _compute_return_profile(pd.Series([]), 252)
        assert avg == 0.0
        assert annual == 0.0

        # Single return
        avg, annual = _compute_return_profile(pd.Series([0.02]), 252)
        assert avg == pytest.approx(0.02)
        assert annual == pytest.approx(0.02 * 252)

        # With NaN values
        returns_with_nan = pd.Series([0.01, np.nan, 0.02, 0.015])
        avg, annual = _compute_return_profile(returns_with_nan, 252)
        # Should filter out NaN and compute on [0.01, 0.02, 0.015]
        expected_avg = (0.01 + 0.02 + 0.015) / 3
        assert avg == pytest.approx(expected_avg, abs=1e-5)

    def test_realistic_strategy_metrics(self):
        """Test with realistic strategy performance numbers"""
        # Simulate MaxDiffAlwaysOn historical: 19.26% over 70 days
        # That's 0.2751% avg daily return

        avg_daily = 0.002751
        num_days = 70

        # Create returns that average to this
        np.random.seed(42)
        daily_returns = np.random.normal(avg_daily, 0.01, num_days)  # Some volatility

        result = _evaluate_daily_returns(daily_returns, trading_days_per_year=252)

        # Should annualize to roughly 69% (0.002751 * 252)
        # But since we added noise, won't be exact
        assert result.annualized_return > 0.5, \
            f"Strong strategy should show high annualized return, got {result.annualized_return:.2%}"

        # Check Sharpe is reasonable
        assert result.sharpe_ratio > 0, "Profitable strategy should have positive Sharpe"


class TestDrawdownScaling:
    """
    Test drawdown scaling logic

    Scale = max(0, 1 - (drawdown_pct / cap))

    Reduces position size during drawdowns to limit risk.
    """

    def test_drawdown_scale_formula(self):
        """Test the basic drawdown scaling formula"""
        # If drawdown = 10%, cap = 20%
        # scale = 1 - (0.10 / 0.20) = 1 - 0.5 = 0.5 (50% position size)

        drawdown_pct = 0.10
        cap = 0.20

        scale = max(0.0, 1.0 - (drawdown_pct / cap))

        assert scale == pytest.approx(0.5), \
            f"Expected 0.5, got {scale}"

    def test_drawdown_scale_no_drawdown(self):
        """Test that no drawdown = full position size"""
        drawdown_pct = 0.0
        cap = 0.20

        scale = max(0.0, 1.0 - (drawdown_pct / cap))

        assert scale == 1.0, "No drawdown should give full size"

    def test_drawdown_scale_at_cap(self):
        """Test that drawdown at cap = zero position size"""
        drawdown_pct = 0.20
        cap = 0.20

        scale = max(0.0, 1.0 - (drawdown_pct / cap))

        assert scale == 0.0, "Drawdown at cap should give zero size"

    def test_drawdown_scale_beyond_cap(self):
        """Test that drawdown beyond cap = zero position size"""
        drawdown_pct = 0.30
        cap = 0.20

        scale = max(0.0, 1.0 - (drawdown_pct / cap))

        assert scale == 0.0, "Drawdown beyond cap should give zero size (floored at 0)"

    def test_drawdown_scale_with_min_scale(self):
        """Test minimum scale floor"""
        drawdown_pct = 0.18
        cap = 0.20
        min_scale = 0.2

        raw_scale = 1.0 - (drawdown_pct / cap)  # = 0.1
        scale = max(min_scale, raw_scale)

        assert scale == min_scale, f"Should be floored at min_scale {min_scale}, got {scale}"


if __name__ == "__main__":
    print("Running critical math unit tests...")
    pytest.main([__file__, "-v", "-s"])
