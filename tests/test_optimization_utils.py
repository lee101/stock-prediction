import numpy as np
import pytest
import torch
from loss_utils import (
    calculate_trading_profit_torch_with_entry_buysell,
)
from src.optimization_utils import (
    optimize_always_on_multipliers,
    optimize_entry_exit_multipliers,
    optimize_entry_exit_multipliers_with_callback,
    optimize_single_parameter,
)


def test_optimize_entry_exit_multipliers_basic():
    """Test basic optimization finds reasonable multipliers"""
    np.random.seed(42)
    torch.manual_seed(42)

    n = 30
    close_actual = torch.randn(n) * 0.02
    high_actual = close_actual + torch.abs(torch.randn(n)) * 0.01
    low_actual = close_actual - torch.abs(torch.randn(n)) * 0.01

    high_pred = torch.randn(n) * 0.01 + 0.005
    low_pred = torch.randn(n) * 0.01 - 0.005
    positions = torch.ones(n)  # all long

    h_mult, l_mult, profit = optimize_entry_exit_multipliers(
        close_actual,
        positions,
        high_actual,
        high_pred,
        low_actual,
        low_pred,
        maxiter=20,
        popsize=8,
    )

    # Check multipliers are within bounds
    assert -0.03 <= h_mult <= 0.03
    assert -0.03 <= l_mult <= 0.03

    # Check profit is reasonable (not NaN/inf)
    assert np.isfinite(profit)

    # Verify profit calculation is correct
    verify_profit = calculate_trading_profit_torch_with_entry_buysell(
        None,
        None,
        close_actual,
        positions,
        high_actual,
        high_pred + h_mult,
        low_actual,
        low_pred + l_mult,
    ).item()

    assert abs(profit - verify_profit) < 1e-5


def test_optimize_finds_better_than_zero():
    """Optimization should find better multipliers than [0, 0]"""
    np.random.seed(123)
    torch.manual_seed(123)

    n = 50
    # Create data where high exit at +2% and low entry at -1% is optimal
    close_actual = torch.randn(n) * 0.01
    high_actual = close_actual + 0.02 + torch.randn(n) * 0.005
    low_actual = close_actual - 0.01 + torch.randn(n) * 0.005

    high_pred = torch.zeros(n)  # predict no movement
    low_pred = torch.zeros(n)
    positions = torch.ones(n)

    # Baseline profit with no multipliers
    baseline_profit = calculate_trading_profit_torch_with_entry_buysell(
        None,
        None,
        close_actual,
        positions,
        high_actual,
        high_pred,
        low_actual,
        low_pred,
    ).item()

    h_mult, l_mult, opt_profit = optimize_entry_exit_multipliers(
        close_actual,
        positions,
        high_actual,
        high_pred,
        low_actual,
        low_pred,
        maxiter=30,
        popsize=10,
    )

    # Optimized should be better than baseline
    assert opt_profit > baseline_profit, f"Optimized {opt_profit} should beat baseline {baseline_profit}"


def test_optimize_with_shorts():
    """Test optimization works correctly with short positions"""
    np.random.seed(456)
    torch.manual_seed(456)

    n = 40
    close_actual = torch.randn(n) * 0.02
    high_actual = close_actual + torch.abs(torch.randn(n)) * 0.01
    low_actual = close_actual - torch.abs(torch.randn(n)) * 0.01

    high_pred = torch.randn(n) * 0.01
    low_pred = torch.randn(n) * 0.01
    positions = -torch.ones(n)  # all short

    h_mult, l_mult, profit = optimize_entry_exit_multipliers(
        close_actual,
        positions,
        high_actual,
        high_pred,
        low_actual,
        low_pred,
        maxiter=20,
        popsize=8,
    )

    assert np.isfinite(profit)
    assert -0.03 <= h_mult <= 0.03
    assert -0.03 <= l_mult <= 0.03


def test_optimize_entry_exit_multipliers_with_callback():
    """Test callback-based optimizer"""
    np.random.seed(789)

    # Simple quadratic function: profit = -(h-0.01)^2 - (l+0.02)^2 + 1
    # Optimal at h=0.01, l=-0.02
    def profit_calc(h_mult, l_mult):
        return -((h_mult - 0.01) ** 2) - (l_mult + 0.02) ** 2 + 1.0

    h_mult, l_mult, profit = optimize_entry_exit_multipliers_with_callback(
        profit_calc,
        maxiter=30,
        popsize=10,
    )

    # Should find near-optimal solution
    assert abs(h_mult - 0.01) < 0.005, f"Expected h≈0.01, got {h_mult}"
    assert abs(l_mult - (-0.02)) < 0.005, f"Expected l≈-0.02, got {l_mult}"
    assert profit > 0.99, f"Expected profit≈1.0, got {profit}"


def test_optimize_single_parameter():
    """Test single parameter optimization"""

    # Quadratic: profit = -(x-0.015)^2 + 1
    # Optimal at x=0.015
    def profit_calc(x):
        return -((x - 0.015) ** 2) + 1.0

    param, profit = optimize_single_parameter(
        profit_calc,
        bounds=(-0.05, 0.05),
        maxiter=20,
        popsize=8,
    )

    assert abs(param - 0.015) < 0.002, f"Expected param≈0.015, got {param}"
    assert profit > 0.99, f"Expected profit≈1.0, got {profit}"


def test_optimization_reproducibility():
    """Test that same seed produces same results"""
    np.random.seed(42)
    torch.manual_seed(42)

    n = 30
    close_actual = torch.randn(n) * 0.02
    high_actual = close_actual + 0.01
    low_actual = close_actual - 0.01
    high_pred = torch.randn(n) * 0.01
    low_pred = torch.randn(n) * 0.01
    positions = torch.ones(n)

    # Run twice with same seed
    h1, l1, p1 = optimize_entry_exit_multipliers(
        close_actual,
        positions,
        high_actual,
        high_pred,
        low_actual,
        low_pred,
        seed=42,
        maxiter=20,
        popsize=8,
    )

    h2, l2, p2 = optimize_entry_exit_multipliers(
        close_actual,
        positions,
        high_actual,
        high_pred,
        low_actual,
        low_pred,
        seed=42,
        maxiter=20,
        popsize=8,
    )

    assert h1 == h2
    assert l1 == l2
    assert p1 == p2


def test_optimization_bounds_respected():
    """Test that optimizer respects bounds"""

    def profit_calc(h_mult, l_mult):
        # Function that would want to go outside bounds
        return h_mult * 10 + l_mult * 10

    h_mult, l_mult, profit = optimize_entry_exit_multipliers_with_callback(
        profit_calc,
        bounds=[(-0.03, 0.03), (-0.03, 0.03)],
        maxiter=20,
        popsize=8,
    )

    # Should hit upper bounds
    assert -0.03 <= h_mult <= 0.03
    assert -0.03 <= l_mult <= 0.03
    # Should be at or very close to upper bounds
    assert h_mult > 0.025
    assert l_mult > 0.025


def test_optimize_always_on_crypto():
    """Test AlwaysOn optimizer for crypto (buy only)"""
    np.random.seed(100)
    torch.manual_seed(100)

    n = 40
    close_actual = torch.randn(n) * 0.02
    high_actual = close_actual + torch.abs(torch.randn(n)) * 0.01
    low_actual = close_actual - torch.abs(torch.randn(n)) * 0.01
    high_pred = torch.randn(n) * 0.01
    low_pred = torch.randn(n) * 0.01

    buy_indicator = torch.ones(n)
    sell_indicator = torch.zeros(n)

    h_mult, l_mult, profit = optimize_always_on_multipliers(
        close_actual,
        buy_indicator,
        sell_indicator,
        high_actual,
        high_pred,
        low_actual,
        low_pred,
        is_crypto=True,
        maxiter=20,
        popsize=8,
        workers=1,
    )

    assert -0.03 <= h_mult <= 0.03
    assert -0.03 <= l_mult <= 0.03
    assert np.isfinite(profit)


def test_optimize_always_on_stocks():
    """Test AlwaysOn optimizer for stocks (buy + sell)"""
    np.random.seed(200)
    torch.manual_seed(200)

    n = 40
    close_actual = torch.randn(n) * 0.02
    high_actual = close_actual + torch.abs(torch.randn(n)) * 0.01
    low_actual = close_actual - torch.abs(torch.randn(n)) * 0.01
    high_pred = torch.randn(n) * 0.01
    low_pred = torch.randn(n) * 0.01

    buy_indicator = torch.ones(n)
    sell_indicator = -torch.ones(n)

    h_mult, l_mult, profit = optimize_always_on_multipliers(
        close_actual,
        buy_indicator,
        sell_indicator,
        high_actual,
        high_pred,
        low_actual,
        low_pred,
        is_crypto=False,
        maxiter=20,
        popsize=8,
        workers=1,
    )

    assert -0.03 <= h_mult <= 0.03
    assert -0.03 <= l_mult <= 0.03
    assert np.isfinite(profit)


def test_optimize_always_on_with_custom_fee():
    """Test AlwaysOn with custom trading fee"""
    np.random.seed(300)
    torch.manual_seed(300)

    n = 30
    close_actual = torch.randn(n) * 0.02
    high_actual = close_actual + 0.02
    low_actual = close_actual - 0.01
    high_pred = torch.zeros(n)
    low_pred = torch.zeros(n)

    buy_indicator = torch.ones(n)
    sell_indicator = torch.zeros(n)

    # Optimize with crypto fee (0.15%)
    h1, l1, p1 = optimize_always_on_multipliers(
        close_actual,
        buy_indicator,
        sell_indicator,
        high_actual,
        high_pred,
        low_actual,
        low_pred,
        is_crypto=True,
        trading_fee=0.0015,
        maxiter=15,
        popsize=6,
        workers=1,
    )

    # Optimize with equity fee (0.05%)
    h2, l2, p2 = optimize_always_on_multipliers(
        close_actual,
        buy_indicator,
        sell_indicator,
        high_actual,
        high_pred,
        low_actual,
        low_pred,
        is_crypto=True,
        trading_fee=0.0005,
        maxiter=15,
        popsize=6,
        workers=1,
    )

    # Lower fee should give higher profit
    assert p2 > p1


def test_optimize_always_on_parallel():
    """Test AlwaysOn with parallel workers"""
    np.random.seed(400)
    torch.manual_seed(400)

    n = 30
    close_actual = torch.randn(n) * 0.02
    high_actual = close_actual + torch.abs(torch.randn(n)) * 0.01
    low_actual = close_actual - torch.abs(torch.randn(n)) * 0.01
    high_pred = torch.randn(n) * 0.01
    low_pred = torch.randn(n) * 0.01

    buy_indicator = torch.ones(n)
    sell_indicator = torch.zeros(n)

    h_mult, l_mult, profit = optimize_always_on_multipliers(
        close_actual,
        buy_indicator,
        sell_indicator,
        high_actual,
        high_pred,
        low_actual,
        low_pred,
        is_crypto=True,
        maxiter=10,
        popsize=5,
        workers=-1,  # parallel
    )

    assert -0.03 <= h_mult <= 0.03
    assert -0.03 <= l_mult <= 0.03
    assert np.isfinite(profit)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
