"""Tests for the Reg-T overnight gross-leverage cap.

The cap mirrors xgbnew/live_trader._eod_deleverage_tick which forces
account gross exposure ≤ equity * eod_max_gross_leverage before close.
In daily-bar mode each bar IS one overnight, so the simulator simply
clips effective max_leverage to min(max_leverage, cap) for sizing.

Test plan covers the contract documented in
project_eod_deleverage_audit_2026_04_29:
    1. cap=None matches legacy behavior (identity).
    2. cap > max_leverage is a no-op (still uses max_leverage).
    3. cap < max_leverage produces results identical to running with
       max_leverage = cap (the convergence test).
    4. cap accepts int, float; rejects 0/negative/NaN.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from pufferlib_market.hourly_replay import MktdData, simulate_daily_policy


def _flat_uptrend_data(closes: list[float]) -> MktdData:
    closes_np = np.asarray(closes, dtype=np.float32)
    T = len(closes_np)
    features = np.zeros((T, 1, 16), dtype=np.float32)
    prices = np.zeros((T, 1, 5), dtype=np.float32)
    prices[:, 0, 0] = closes_np
    prices[:, 0, 1] = closes_np
    prices[:, 0, 2] = closes_np
    prices[:, 0, 3] = closes_np
    tradable = np.ones((T, 1), dtype=np.uint8)
    return MktdData(
        version=2,
        symbols=["AAA"],
        features=features,
        prices=prices,
        tradable=tradable,
    )


def test_overnight_cap_disabled_matches_legacy_identity():
    """cap=None preserves legacy uncapped behavior bit-for-bit."""
    data = _flat_uptrend_data([100.0, 110.0, 121.0])

    legacy = simulate_daily_policy(
        data, lambda obs: 1, max_steps=2,
        fee_rate=0.0, max_leverage=2.0, periods_per_year=365.0,
    )
    capped_none = simulate_daily_policy(
        data, lambda obs: 1, max_steps=2,
        fee_rate=0.0, max_leverage=2.0, periods_per_year=365.0,
        overnight_max_gross_leverage=None,
    )

    assert capped_none.total_return == pytest.approx(legacy.total_return, abs=1e-12)
    assert capped_none.num_trades == legacy.num_trades
    assert capped_none.max_drawdown == pytest.approx(legacy.max_drawdown, abs=1e-12)


def test_overnight_cap_above_max_leverage_is_noop():
    """cap=10 with max_leverage=1.5 → identical to no cap."""
    data = _flat_uptrend_data([100.0, 110.0, 121.0])

    legacy = simulate_daily_policy(
        data, lambda obs: 1, max_steps=2,
        fee_rate=0.0, max_leverage=1.5, periods_per_year=365.0,
    )
    capped_above = simulate_daily_policy(
        data, lambda obs: 1, max_steps=2,
        fee_rate=0.0, max_leverage=1.5, periods_per_year=365.0,
        overnight_max_gross_leverage=10.0,
    )
    assert capped_above.total_return == pytest.approx(legacy.total_return, abs=1e-12)
    assert capped_above.num_trades == legacy.num_trades


def test_overnight_cap_below_max_leverage_matches_running_at_cap():
    """The headline correctness fix: cap=2.0 with max_leverage=3.0 must produce
    the SAME equity curve as running with max_leverage=2.0 directly.

    This is the core invariant for the D_s16 deploy decision: lev=2.5 and
    lev=3.0 candidates capped at 2.0 must converge to lev=2.0 numbers."""
    data = _flat_uptrend_data([100.0, 102.0, 104.04, 106.12, 108.24])

    baseline_lev2 = simulate_daily_policy(
        data, lambda obs: 1, max_steps=4,
        fee_rate=0.0, max_leverage=2.0, periods_per_year=365.0,
    )
    capped_lev3_to_2 = simulate_daily_policy(
        data, lambda obs: 1, max_steps=4,
        fee_rate=0.0, max_leverage=3.0, periods_per_year=365.0,
        overnight_max_gross_leverage=2.0,
    )

    assert capped_lev3_to_2.total_return == pytest.approx(
        baseline_lev2.total_return, abs=1e-9,
    )
    assert capped_lev3_to_2.max_drawdown == pytest.approx(
        baseline_lev2.max_drawdown, abs=1e-9,
    )
    assert capped_lev3_to_2.num_trades == baseline_lev2.num_trades

    # And explicitly ensure cap actually changed something vs uncapped lev=3.
    uncapped_lev3 = simulate_daily_policy(
        data, lambda obs: 1, max_steps=4,
        fee_rate=0.0, max_leverage=3.0, periods_per_year=365.0,
    )
    assert uncapped_lev3.total_return > baseline_lev2.total_return + 1e-6


def test_overnight_cap_at_lev_15_with_cap_2_no_effect():
    """User-stated expectation #2: at lev=1.5 with cap=2.0, no effect."""
    data = _flat_uptrend_data([100.0, 102.0, 104.04, 106.12])

    uncapped = simulate_daily_policy(
        data, lambda obs: 1, max_steps=3,
        fee_rate=0.0, max_leverage=1.5, periods_per_year=365.0,
    )
    capped = simulate_daily_policy(
        data, lambda obs: 1, max_steps=3,
        fee_rate=0.0, max_leverage=1.5, periods_per_year=365.0,
        overnight_max_gross_leverage=2.0,
    )
    assert capped.total_return == pytest.approx(uncapped.total_return, abs=1e-12)


def test_overnight_cap_rejects_zero_or_negative():
    data = _flat_uptrend_data([100.0, 101.0, 102.0])

    with pytest.raises(ValueError):
        simulate_daily_policy(
            data, lambda obs: 1, max_steps=2,
            fee_rate=0.0, max_leverage=1.0, periods_per_year=365.0,
            overnight_max_gross_leverage=0.0,
        )

    with pytest.raises(ValueError):
        simulate_daily_policy(
            data, lambda obs: 1, max_steps=2,
            fee_rate=0.0, max_leverage=1.0, periods_per_year=365.0,
            overnight_max_gross_leverage=-1.0,
        )


def test_overnight_cap_rejects_nonfinite():
    data = _flat_uptrend_data([100.0, 101.0, 102.0])
    with pytest.raises(ValueError):
        simulate_daily_policy(
            data, lambda obs: 1, max_steps=2,
            fee_rate=0.0, max_leverage=1.0, periods_per_year=365.0,
            overnight_max_gross_leverage=float("nan"),
        )
    with pytest.raises(ValueError):
        simulate_daily_policy(
            data, lambda obs: 1, max_steps=2,
            fee_rate=0.0, max_leverage=1.0, periods_per_year=365.0,
            overnight_max_gross_leverage=float("inf"),
        )


def test_overnight_cap_with_drawdown_path_matches_lev_at_cap():
    """Stress the convergence test on a noisy path with drawdown.

    With the cap engaged the curve must still match running at the cap directly,
    even when max_drawdown is non-zero."""
    # Up, down, down, up, up
    closes = [100.0, 105.0, 98.0, 95.0, 99.0, 103.0]
    data = _flat_uptrend_data(closes)

    baseline = simulate_daily_policy(
        data, lambda obs: 1, max_steps=5,
        fee_rate=0.0005, slippage_bps=2.0,
        max_leverage=2.0, periods_per_year=365.0,
        fill_buffer_bps=0.0,
    )
    capped = simulate_daily_policy(
        data, lambda obs: 1, max_steps=5,
        fee_rate=0.0005, slippage_bps=2.0,
        max_leverage=3.0, periods_per_year=365.0,
        fill_buffer_bps=0.0,
        overnight_max_gross_leverage=2.0,
    )

    assert capped.total_return == pytest.approx(baseline.total_return, abs=1e-9)
    assert capped.max_drawdown == pytest.approx(baseline.max_drawdown, abs=1e-9)
    assert capped.sortino == pytest.approx(baseline.sortino, abs=1e-9)


def test_overnight_cap_int_input_accepted():
    """Cap should accept int as well as float."""
    data = _flat_uptrend_data([100.0, 101.0, 102.01])
    out = simulate_daily_policy(
        data, lambda obs: 1, max_steps=2,
        fee_rate=0.0, max_leverage=3.0, periods_per_year=365.0,
        overnight_max_gross_leverage=2,
    )
    assert math.isfinite(out.total_return)
