"""Tests for pufferlib_market.evaluate_sliding.

No live GPU or real checkpoint required — uses synthetic MktdData and a
random/fixed policy function.
"""

from __future__ import annotations

import numpy as np
import pytest

from pufferlib_market.evaluate_sliding import (
    WindowResult,
    aggregate_sliding_results,
    compute_calmar,
    sliding_window_eval,
)
from pufferlib_market.hourly_replay import MktdData


# ---------------------------------------------------------------------------
# Helpers to build synthetic MktdData
# ---------------------------------------------------------------------------

def _make_data(num_timesteps: int, num_symbols: int = 1) -> MktdData:
    """Build a minimal valid MKTD v2 dataset with flat prices."""
    features = np.zeros((num_timesteps, num_symbols, 16), dtype=np.float32)
    prices = np.ones((num_timesteps, num_symbols, 5), dtype=np.float32)
    # O=H=L=C=1.0, V=1.0 — no price movement, zero return
    tradable = np.ones((num_timesteps, num_symbols), dtype=np.uint8)
    return MktdData(
        version=2,
        symbols=[f"SYM{i}" for i in range(num_symbols)],
        features=features,
        prices=prices,
        tradable=tradable,
    )


def _make_trending_data(num_timesteps: int, drift: float = 0.001) -> MktdData:
    """Single symbol with a constant upward drift each bar."""
    prices_arr = np.ones((num_timesteps, 1, 5), dtype=np.float32)
    for t in range(num_timesteps):
        p = 1.0 * (1.0 + drift) ** t
        prices_arr[t, 0, :] = p  # O=H=L=C=V=p
    features = np.zeros((num_timesteps, 1, 16), dtype=np.float32)
    tradable = np.ones((num_timesteps, 1), dtype=np.uint8)
    return MktdData(
        version=2,
        symbols=["AAA"],
        features=features,
        prices=prices_arr,
        tradable=tradable,
    )


# ---------------------------------------------------------------------------
# Tests: compute_calmar
# ---------------------------------------------------------------------------

def test_calmar_basic():
    c = compute_calmar(0.2, 0.1)
    assert c == pytest.approx(2.0)


def test_calmar_zero_drawdown_positive_return():
    c = compute_calmar(0.15, 0.0)
    assert c == float("inf")


def test_calmar_zero_drawdown_zero_return():
    c = compute_calmar(0.0, 0.0)
    assert c == 0.0


def test_calmar_negative_return():
    c = compute_calmar(-0.1, 0.2)
    assert c == pytest.approx(-0.5)


def test_calmar_uses_abs_drawdown():
    # max_drawdown is conventionally positive; function uses abs() defensively
    c1 = compute_calmar(0.2, 0.1)
    c2 = compute_calmar(0.2, -0.1)
    assert c1 == pytest.approx(c2)


# ---------------------------------------------------------------------------
# Tests: window count
# ---------------------------------------------------------------------------

def test_window_count_exact():
    """1000 bars, episode_len=100, stride=50 -> floor((1000-100)/50) = 18 windows."""
    data = _make_data(1000)
    flat_policy = lambda obs: 0  # always flat
    results = sliding_window_eval(
        flat_policy,
        data,
        episode_len=100,
        stride=50,
        fee_rate=0.0,
        enable_drawdown_profit_early_exit=False,
    )
    # starts = range(0, 1000-100, 50) = range(0, 900, 50) -> 18 values
    assert len(results) == 18


def test_window_count_no_overlap():
    """stride == episode_len gives non-overlapping windows."""
    data = _make_data(500)
    flat_policy = lambda obs: 0
    results = sliding_window_eval(
        flat_policy,
        data,
        episode_len=100,
        stride=100,
        fee_rate=0.0,
        enable_drawdown_profit_early_exit=False,
    )
    # starts = range(0, 500-100, 100) = [0, 100, 200, 300] -> 4 windows
    assert len(results) == 4


def test_window_indices_are_sequential():
    data = _make_data(300)
    flat_policy = lambda obs: 0
    results = sliding_window_eval(
        flat_policy,
        data,
        episode_len=50,
        stride=50,
        fee_rate=0.0,
        enable_drawdown_profit_early_exit=False,
    )
    for i, r in enumerate(results):
        assert r.window_idx == i
        assert r.start_offset == i * 50


# ---------------------------------------------------------------------------
# Tests: flat market (zero return policy)
# ---------------------------------------------------------------------------

def test_flat_market_zero_return():
    """With flat prices and always-flat policy, total return should be ~0."""
    data = _make_data(200)
    flat_policy = lambda obs: 0  # action 0 = flat
    results = sliding_window_eval(
        flat_policy,
        data,
        episode_len=50,
        stride=50,
        fee_rate=0.0,
        enable_drawdown_profit_early_exit=False,
    )
    for r in results:
        assert r.total_return == pytest.approx(0.0, abs=1e-6)
        assert r.max_drawdown == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests: trending market (long policy should profit)
# ---------------------------------------------------------------------------

def test_long_policy_profits_in_uptrend():
    """With a constant upward drift and always-long policy, returns should be positive."""
    data = _make_trending_data(300, drift=0.005)
    # action 1 = long symbol 0
    long_policy = lambda obs: 1
    results = sliding_window_eval(
        long_policy,
        data,
        episode_len=50,
        stride=50,
        fee_rate=0.0,
        enable_drawdown_profit_early_exit=False,
    )
    for r in results:
        assert r.total_return > 0.0, f"Expected positive return, got {r.total_return}"


# ---------------------------------------------------------------------------
# Tests: WindowResult fields are correct types
# ---------------------------------------------------------------------------

def test_window_result_types():
    data = _make_data(200)
    flat_policy = lambda obs: 0
    results = sliding_window_eval(
        flat_policy,
        data,
        episode_len=50,
        stride=100,
        fee_rate=0.0,
        enable_drawdown_profit_early_exit=False,
    )
    assert len(results) > 0
    r = results[0]
    assert isinstance(r.window_idx, int)
    assert isinstance(r.start_offset, int)
    assert isinstance(r.total_return, float)
    assert isinstance(r.max_drawdown, float)
    assert isinstance(r.sortino, float)
    assert isinstance(r.num_trades, int)
    assert isinstance(r.win_rate, float)
    assert isinstance(r.stopped_early, bool)


# ---------------------------------------------------------------------------
# Tests: aggregate_sliding_results
# ---------------------------------------------------------------------------

def test_aggregate_basic():
    results = [
        WindowResult(0, 0,   0.10, 0.05, 1.0, 5, 0.6, False),
        WindowResult(1, 50, -0.05, 0.08, 0.5, 3, 0.4, False),
        WindowResult(2, 100, 0.20, 0.03, 2.0, 7, 0.7, False),
    ]
    stats = aggregate_sliding_results(results, episode_len=50, periods_per_year=8760.0)
    assert stats["num_windows"] == 3
    assert stats["mean_return"] == pytest.approx(0.25 / 3, abs=1e-9)
    assert stats["pct_profitable"] == pytest.approx(2.0 / 3, abs=1e-9)
    assert stats["worst_max_drawdown"] == pytest.approx(0.08)
    # Calmar should be finite (annualized_return / worst_max_drawdown)
    assert np.isfinite(stats["calmar"])
    assert isinstance(stats["annualized_return"], float)


def test_aggregate_calmar_uses_worst_drawdown():
    results = [
        WindowResult(0, 0,  0.10, 0.02, 1.0, 5, 0.6, False),
        WindowResult(1, 50, 0.10, 0.50, 1.0, 5, 0.6, False),  # worst DD
    ]
    stats = aggregate_sliding_results(results, episode_len=50, periods_per_year=8760.0)
    # calmar = annualized_return / 0.50
    assert stats["worst_max_drawdown"] == pytest.approx(0.50)
    expected_calmar = compute_calmar(stats["annualized_return"], 0.50)
    assert stats["calmar"] == pytest.approx(expected_calmar)


def test_aggregate_empty_returns_empty():
    stats = aggregate_sliding_results([], episode_len=50, periods_per_year=8760.0)
    assert stats == {}


# ---------------------------------------------------------------------------
# Tests: edge cases / error handling
# ---------------------------------------------------------------------------

def test_episode_len_too_large_raises():
    data = _make_data(50)
    flat_policy = lambda obs: 0
    with pytest.raises(ValueError, match="episode_len"):
        sliding_window_eval(flat_policy, data, episode_len=100, stride=10, fee_rate=0.0)


def test_invalid_stride_raises():
    data = _make_data(200)
    flat_policy = lambda obs: 0
    with pytest.raises(ValueError, match="stride"):
        sliding_window_eval(flat_policy, data, episode_len=50, stride=0, fee_rate=0.0)


def test_no_windows_fit_raises():
    """When episode_len+1 >= T and stride has no room, error is raised."""
    data = _make_data(101)
    flat_policy = lambda obs: 0
    # T=101, episode_len=101: episode_len+1=102 > T=101 -> ValueError about episode_len
    with pytest.raises(ValueError):
        sliding_window_eval(flat_policy, data, episode_len=101, stride=1, fee_rate=0.0)


# ---------------------------------------------------------------------------
# Tests: multi-symbol data
# ---------------------------------------------------------------------------

def test_multi_symbol_data():
    data = _make_data(200, num_symbols=3)
    flat_policy = lambda obs: 0
    results = sliding_window_eval(
        flat_policy,
        data,
        episode_len=50,
        stride=50,
        fee_rate=0.0,
        enable_drawdown_profit_early_exit=False,
    )
    assert len(results) == 3  # range(0, 150, 50)
    for r in results:
        assert r.total_return == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests: fee reduces return
# ---------------------------------------------------------------------------

def test_fee_reduces_return():
    """Long policy with fees should earn less than without fees."""
    data = _make_trending_data(200, drift=0.002)
    long_policy = lambda obs: 1
    results_no_fee = sliding_window_eval(
        long_policy, data, episode_len=50, stride=50, fee_rate=0.0,
        enable_drawdown_profit_early_exit=False,
    )
    results_with_fee = sliding_window_eval(
        long_policy, data, episode_len=50, stride=50, fee_rate=0.01,
        enable_drawdown_profit_early_exit=False,
    )
    mean_no_fee = np.mean([r.total_return for r in results_no_fee])
    mean_with_fee = np.mean([r.total_return for r in results_with_fee])
    assert mean_no_fee > mean_with_fee
