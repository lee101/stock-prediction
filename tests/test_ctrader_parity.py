"""Parity test: ctrader libmarket_sim.so weight simulator vs pure Python reference.

This validates the continuous target-weight path that ctrader exposes via
``simulate_target_weights``. It is the *overlapping subset* shared between the
C production sim and a Python reference -- ctrader does not expose discrete
binary-fill semantics (those live in pufferlib_market's binding C env, which has
its own validators). See PARITY_SESSION_2026_04_07.md Phase 3 for scope notes.

Run: ``source .venv/bin/activate && pytest tests/test_ctrader_parity.py -v``
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from ctrader.market_sim_ffi import (
    WeightSimConfig,
    simulate_target_weights,
)


def _py_simulate_target_weights(
    close: np.ndarray,
    weights: np.ndarray,
    *,
    initial_cash: float,
    fee_rate: float,
    borrow_rate_per_period: float = 0.0,
    max_gross_leverage: float = 1.0,
    can_short: bool = False,
) -> tuple[float, float, float, np.ndarray]:
    """Pure-Python mirror of ctrader/market_sim.c::simulate_target_weights.

    Returns (final_equity, total_fees, total_turnover, equity_curve).
    Test inputs are crafted so weight clamping is the identity, so we omit it
    here -- the C clamp is exercised separately by ctrader's own unit tests.
    """
    n_bars, n_symbols = close.shape
    equity = float(initial_cash)
    eq_curve = np.zeros(n_bars, dtype=np.float64)
    eq_curve[0] = equity
    prev_w = np.zeros(n_symbols, dtype=np.float64)
    total_fees = 0.0
    total_turnover = 0.0

    for t in range(n_bars - 1):
        next_w = weights[t].astype(np.float64)
        # sanity: tests construct inputs that don't need clamping
        if not can_short:
            assert (next_w >= -1e-12).all(), "test bug: negative weight without can_short"
        gross_long = float(next_w[next_w > 0].sum())
        gross_short = float(-next_w[next_w < 0].sum())
        assert gross_long + gross_short <= max_gross_leverage + 1e-9

        turnover = float(np.abs(next_w - prev_w).sum())
        cash_w = 1.0 - float(next_w.sum())
        growth = cash_w
        for s in range(n_symbols):
            p0 = close[t, s]
            p1 = close[t + 1, s]
            rel = (p1 / p0) if (p0 > 0.0 and p1 > 0.0) else 1.0
            growth += next_w[s] * rel

        fees = equity * turnover * fee_rate
        borrow_exposure = max(gross_long - 1.0, 0.0) + gross_short
        borrow_cost = equity * borrow_exposure * borrow_rate_per_period
        equity_after = max(equity - fees - borrow_cost, 0.0)
        equity = equity_after * growth
        if not math.isfinite(equity) or equity < 0:
            equity = 0.0
        eq_curve[t + 1] = equity
        total_fees += fees
        total_turnover += turnover
        prev_w = next_w

    return float(eq_curve[-1]), float(total_fees), float(total_turnover), eq_curve


def _cfg(**over) -> WeightSimConfig:
    base = dict(
        initial_cash=10_000.0,
        max_gross_leverage=1.0,
        fee_rate=0.0,
        borrow_rate_per_period=0.0,
        periods_per_year=8760.0,
        can_short=0,
    )
    base.update(over)
    return WeightSimConfig(**base)


def test_zero_weights_no_pnl_no_fees():
    n_bars, n_sym = 50, 3
    rng = np.random.default_rng(0)
    close = 100.0 + rng.standard_normal((n_bars, n_sym)).cumsum(axis=0)
    weights = np.zeros((n_bars, n_sym), dtype=np.float64)

    summary, eq = simulate_target_weights(close, weights, _cfg(fee_rate=0.001))
    py_eq_final, py_fees, py_turnover, _ = _py_simulate_target_weights(
        close, weights, initial_cash=10_000.0, fee_rate=0.001
    )

    assert summary.final_equity == pytest.approx(10_000.0, rel=0, abs=1e-9)
    assert summary.total_fees == pytest.approx(0.0, abs=1e-12)
    assert summary.total_turnover == pytest.approx(0.0, abs=1e-12)
    assert summary.final_equity == pytest.approx(py_eq_final, rel=1e-12, abs=1e-9)
    assert summary.total_fees == pytest.approx(py_fees, abs=1e-12)
    assert summary.total_turnover == pytest.approx(py_turnover, abs=1e-12)


def test_single_symbol_full_alloc_deterministic_path():
    # Two-bar segments with known returns, full allocation to symbol 0.
    close = np.array(
        [[100.0], [110.0], [99.0], [121.0], [121.0]], dtype=np.float64
    )
    weights = np.ones((5, 1), dtype=np.float64)  # always 100% in symbol 0
    cfg = _cfg(fee_rate=0.0010)  # 10 bps

    summary, eq = simulate_target_weights(close, weights, cfg)
    py_eq_final, py_fees, py_turnover, py_curve = _py_simulate_target_weights(
        close, weights, initial_cash=10_000.0, fee_rate=0.0010
    )

    # Manual ground truth at bar 1 only:
    #   t=0 -> next_w=[1.0], prev=[0.0], turnover=1.0
    #   fees = 10000 * 1.0 * 0.001 = 10
    #   equity_after = 9990
    #   growth = (1-1) + 1.0 * (110/100) = 1.10
    #   equity_1 = 9990 * 1.10 = 10989
    assert eq[1] == pytest.approx(10989.0, rel=1e-12)
    # Subsequent bars: turnover=0, no fees, equity scales by close ratio.
    assert summary.final_equity == pytest.approx(py_eq_final, rel=1e-12)
    assert summary.total_fees == pytest.approx(py_fees, rel=1e-12)
    assert summary.total_turnover == pytest.approx(py_turnover, rel=1e-12)
    assert summary.total_fees == pytest.approx(10.0, rel=1e-12)
    assert summary.total_turnover == pytest.approx(1.0, rel=1e-12)
    assert np.allclose(eq, py_curve, rtol=1e-12, atol=1e-9)


def test_two_symbol_random_walk_with_fees_matches_python_reference():
    rng = np.random.default_rng(42)
    n_bars, n_sym = 200, 2
    rets = rng.normal(0.0, 0.01, size=(n_bars, n_sym))
    close = 100.0 * np.exp(rets.cumsum(axis=0))
    # Smooth target weights summing to <=1, no shorts.
    raw = rng.uniform(0.0, 0.5, size=(n_bars, n_sym))
    raw = np.minimum(raw, 0.45)  # ensure gross <= 0.9 < 1.0
    weights = raw

    cfg = _cfg(fee_rate=0.0010, periods_per_year=365.0)
    summary, eq = simulate_target_weights(close, weights, cfg)
    py_final, py_fees, py_turnover, py_curve = _py_simulate_target_weights(
        close, weights, initial_cash=10_000.0, fee_rate=0.0010
    )

    assert summary.final_equity == pytest.approx(py_final, rel=1e-10, abs=1e-6)
    assert summary.total_fees == pytest.approx(py_fees, rel=1e-10, abs=1e-9)
    assert summary.total_turnover == pytest.approx(py_turnover, rel=1e-10, abs=1e-12)
    assert np.allclose(eq, py_curve, rtol=1e-10, atol=1e-6)
