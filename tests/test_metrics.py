"""Integration tests verifying Python Sortino matches C env formula.

The C environment (pufferlib_market/src/trading_env.c) computes Sortino as:
    downside_dev = sqrt(sum_neg_sq / ret_count)   # RMS of negative returns
    sortino = mean_ret / downside_dev * sqrt(periods_per_year)

These tests verify that src/metrics_utils.annualized_sortino uses the same
formula so that Python evaluation metrics are comparable to C training metrics.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics_utils import annualized_sharpe, annualized_sortino, compute_step_returns


# ---------------------------------------------------------------------------
# Helper: C-style Sortino (manual implementation matching trading_env.c)
# ---------------------------------------------------------------------------

def compute_sortino_c_style(returns: np.ndarray, periods_per_year: float) -> float:
    """Replicate the exact C env Sortino formula in Python for cross-checking.

    C code (trading_env.c):
        float mean_ret = ag->sum_ret / ag->ret_count;
        float downside_dev = sqrtf(ag->sum_neg_sq / ag->ret_count);
        sortino = mean_ret / downside_dev * sqrtf(ppy);
    """
    arr = np.asarray(returns, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0
    mean_ret = float(np.mean(arr))
    neg = arr[arr < 0.0]
    if neg.size == 0:
        return 0.0
    # C uses sum_neg_sq / ret_count where ret_count counts ALL steps (not just neg)
    # but sum_neg_sq only accumulates for negative returns.
    # That is equivalent to: sqrt( mean_of_neg_squares_over_ALL_steps )
    sum_neg_sq = float(np.sum(neg ** 2))
    ret_count = arr.size
    downside_dev = float(np.sqrt(sum_neg_sq / ret_count))
    if downside_dev <= 1e-8:
        return 0.0
    ppy = float(periods_per_year) if periods_per_year > 0 else 365.0
    return float(mean_ret / downside_dev * np.sqrt(ppy))


def compute_sortino_python(returns: np.ndarray, periods_per_year: float) -> float:
    """Thin wrapper around the canonical Python implementation."""
    return annualized_sortino(returns, periods_per_year=periods_per_year)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_sortino_consistency_daily():
    """Python Sortino matches C env formula within 1% for daily returns."""
    returns = np.array([0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.008])
    py_sortino = compute_sortino_python(returns, periods_per_year=365)
    c_sortino = compute_sortino_c_style(returns, periods_per_year=365)

    assert abs(py_sortino - c_sortino) / max(abs(py_sortino), 1e-6) < 0.01, (
        f"Sortino mismatch: Python={py_sortino:.4f}, C-style={c_sortino:.4f}"
    )


def test_sortino_consistency_hourly():
    """Python Sortino matches C env formula within 1% for hourly returns."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0001, 0.005, size=200)
    ppy = 8760.0
    py_sortino = compute_sortino_python(returns, periods_per_year=ppy)
    c_sortino = compute_sortino_c_style(returns, periods_per_year=ppy)

    assert abs(py_sortino - c_sortino) / max(abs(py_sortino), 1e-6) < 0.01, (
        f"Sortino mismatch (hourly): Python={py_sortino:.4f}, C-style={c_sortino:.4f}"
    )


def test_sortino_c_formula_not_ddof1():
    """Verify annualized_sortino uses C env formula, not sample std (ddof=1).

    C formula: downside_dev = sqrt(sum(neg**2) / total_n)
    Sample std: downside_dev = sqrt(sum((x-mean(x))**2) / (n_neg - 1))
    These differ both in the divisor and in subtracting the mean.
    """
    returns = np.array([0.01, -0.02, 0.03, -0.01, 0.005, -0.015])
    neg = returns[returns < 0]

    # C env formula: sum_neg_sq / ret_count (total n, not neg-only n)
    c_downside = float(np.sqrt(np.sum(neg ** 2) / returns.size))
    ddof1_downside = float(np.std(neg, ddof=1))

    # The two formulas must give different numbers so the test is meaningful.
    assert not np.isclose(c_downside, ddof1_downside), (
        "Test is degenerate: C-style and ddof=1 std happen to be equal for this input"
    )

    mean_ret = float(np.mean(returns))
    ppy = 365.0
    expected_c = mean_ret / c_downside * np.sqrt(ppy)

    result = annualized_sortino(returns, periods_per_year=ppy)
    assert np.isclose(result, expected_c), (
        f"Expected C-style Sortino={expected_c:.4f}, got {result:.4f}"
    )


def test_sortino_zero_on_empty():
    assert annualized_sortino([], periods_per_year=365) == 0.0


def test_sortino_zero_on_single_return():
    assert annualized_sortino([0.01], periods_per_year=365) == 0.0


def test_sortino_all_positive_falls_back_to_sharpe():
    """All-positive returns: Sortino should fall back to Sharpe (no downside)."""
    returns = [0.01, 0.02, 0.03, 0.015]
    sharpe = annualized_sharpe(returns, periods_per_year=365)
    sortino = annualized_sortino(returns, periods_per_year=365)
    assert np.isclose(sortino, sharpe), (
        f"Expected Sortino={sharpe:.4f} (==Sharpe) when all returns positive, got {sortino:.4f}"
    )


def test_sortino_annualisation_factor_365_not_252():
    """Crypto daily: annualisation must use 365, not 252."""
    returns = np.array([0.01, -0.02, 0.015, -0.005, 0.02, -0.01])
    neg = returns[returns < 0]
    mean_ret = float(np.mean(returns))
    # C env formula: sqrt(sum_neg_sq / total_n)
    c_dd = float(np.sqrt(np.sum(neg ** 2) / returns.size))

    expected_365 = mean_ret / c_dd * np.sqrt(365.0)
    expected_252 = mean_ret / c_dd * np.sqrt(252.0)

    result = annualized_sortino(returns, periods_per_year=365)
    assert np.isclose(result, expected_365), (
        f"Expected 365-annualised Sortino={expected_365:.4f}, got {result:.4f}"
    )
    assert not np.isclose(result, expected_252), (
        "Sortino with periods_per_year=365 should NOT equal 252-annualised value"
    )


def test_sortino_annualisation_factor_8760_hourly():
    """Crypto hourly: annualisation must use 8760."""
    returns = np.array([0.001, -0.002, 0.0015, -0.0005, 0.002, -0.001])
    neg = returns[returns < 0]
    mean_ret = float(np.mean(returns))
    # C env formula: sqrt(sum_neg_sq / total_n)
    c_dd = float(np.sqrt(np.sum(neg ** 2) / returns.size))
    expected_8760 = mean_ret / c_dd * np.sqrt(8760.0)

    result = annualized_sortino(returns, periods_per_year=8760)
    assert np.isclose(result, expected_8760), (
        f"Expected 8760-annualised Sortino={expected_8760:.4f}, got {result:.4f}"
    )


def test_sharpe_default_periods_365():
    """Default periods_per_year for annualized_sharpe is 365 (crypto-safe)."""
    import inspect
    sig = inspect.signature(annualized_sharpe)
    default = sig.parameters["periods_per_year"].default
    assert default == 365.0, (
        f"annualized_sharpe default periods_per_year should be 365.0, got {default}"
    )


def test_sortino_default_periods_365():
    """Default periods_per_year for annualized_sortino is 365 (crypto-safe)."""
    import inspect
    sig = inspect.signature(annualized_sortino)
    default = sig.parameters["periods_per_year"].default
    assert default == 365.0, (
        f"annualized_sortino default periods_per_year should be 365.0, got {default}"
    )


def test_sortino_nonfinite_values_ignored():
    """NaN/inf in returns are silently dropped before computation."""
    returns_clean = np.array([0.01, -0.02, 0.015, -0.005])
    returns_noisy = np.array([0.01, np.nan, -0.02, np.inf, 0.015, -0.005])
    s_clean = annualized_sortino(returns_clean, periods_per_year=365)
    s_noisy = annualized_sortino(returns_noisy, periods_per_year=365)
    assert np.isclose(s_clean, s_noisy), (
        f"NaN/inf should be ignored: clean={s_clean:.4f}, noisy={s_noisy:.4f}"
    )


def test_compute_step_returns_basic():
    """Equity curve [100, 110, 99] → returns [0.10, -0.10]."""
    returns = compute_step_returns([100.0, 110.0, 99.0])
    expected = np.array([0.10, (99.0 - 110.0) / 110.0])
    assert np.allclose(returns, expected)


def test_c_style_helper_matches_formula():
    """Sanity-check the C-style helper itself against hand-calculated values."""
    # returns = [0.01, -0.02, 0.03]
    # neg = [-0.02]
    # sum_neg_sq = 0.0004, ret_count = 3
    # downside_dev = sqrt(0.0004 / 3) = sqrt(0.000133..) = 0.01155..
    # mean_ret = (0.01 - 0.02 + 0.03) / 3 = 0.02/3 = 0.006667
    # sortino = 0.006667 / 0.01155 * sqrt(365) = 11.028..
    returns = np.array([0.01, -0.02, 0.03])
    result = compute_sortino_c_style(returns, periods_per_year=365.0)
    expected = (0.02 / 3) / np.sqrt(0.0004 / 3) * np.sqrt(365.0)
    assert np.isclose(result, expected, rtol=1e-5), (
        f"C-style helper={result:.6f}, expected={expected:.6f}"
    )
