"""Time-Under-Water + Ulcer-Index tests for ``xgbnew.backtest``.

Validates the equity-curve pain metrics on hand-rolled ``DayResult`` lists
so the math is exercised independently of the full simulate() path.

Monotone-up  → TuW = 0, Ulcer = 0.
Flat-then-dip → known closed-form TuW and Ulcer.
V-shape      → dip days are under water, recovery day is not.
Matches the max_dd denominator convention (depth / running-max).
"""
from __future__ import annotations

from datetime import date, timedelta

import math

import pytest

from xgbnew.backtest import BacktestConfig, DayResult, _compute_result


def _mk_day_results(equities: list[float], *, initial: float = 10_000.0) -> list[DayResult]:
    """Build a DayResult list whose equity path is ``[initial, *equities]``.

    daily_return_pct is set from the close-to-close on the equity curve so
    downstream metrics (Sharpe/Sortino) remain consistent, but TuW + Ulcer
    derive only from ``equity_end`` + ``config.initial_cash`` so the path
    is what's under test.
    """
    drs: list[DayResult] = []
    prev = initial
    d = date(2025, 1, 2)
    for i, eq in enumerate(equities):
        daily_ret_pct = 100.0 * (eq - prev) / prev
        drs.append(DayResult(
            day=d + timedelta(days=i),
            equity_start=prev,
            equity_end=eq,
            daily_return_pct=daily_ret_pct,
            trades=[],
            n_candidates=0,
        ))
        prev = eq
    return drs


def _cfg(initial: float = 10_000.0) -> BacktestConfig:
    return BacktestConfig(initial_cash=initial)


def test_monotone_up_tuw_and_ulcer_zero():
    drs = _mk_day_results([10_100, 10_200, 10_300, 10_400, 10_500])
    r = _compute_result(drs, _cfg())
    assert r.time_under_water_pct == pytest.approx(0.0, abs=1e-12)
    assert r.ulcer_index == pytest.approx(0.0, abs=1e-12)
    assert r.max_drawdown_pct == pytest.approx(0.0, abs=1e-12)


def test_flat_with_single_dip_known_values():
    # Equity: [10000, 10000, 9000, 10000, 10000]
    # Running max: [10000, 10000, 10000, 10000, 10000]
    # Dd frac:     [0,       0,    0.1,  0,     0]
    # 6 samples total (initial + 5); 1 sample strictly underwater → 1/6.
    drs = _mk_day_results([10_000, 9_000, 10_000, 10_000])
    r = _compute_result(drs, _cfg())
    # Sample count = 1 + len(drs) = 5.
    # drawdowns: [0, 0, 0.1, 0, 0]  → 1/5 samples underwater.
    assert r.time_under_water_pct == pytest.approx(20.0, abs=1e-9)
    # Ulcer = sqrt(mean(dd^2)) * 100 = sqrt(0.01/5)*100 = sqrt(0.002)*100
    expected_ulcer = math.sqrt(0.002) * 100.0
    assert r.ulcer_index == pytest.approx(expected_ulcer, abs=1e-9)
    assert r.max_drawdown_pct == pytest.approx(10.0, abs=1e-9)


def test_v_shape_dip_then_recover():
    # Equity: [10000, 9500, 9000, 9500, 10000]
    # Running max:  [10000, 10000, 10000, 10000, 10000]
    # Dd frac:      [0, 0.05, 0.10, 0.05, 0]
    drs = _mk_day_results([9_500, 9_000, 9_500, 10_000])
    r = _compute_result(drs, _cfg())
    # 3 of 5 samples strictly underwater (drs days 1, 2, 3).
    assert r.time_under_water_pct == pytest.approx(60.0, abs=1e-9)
    expected_ulcer = math.sqrt((0.05**2 + 0.10**2 + 0.05**2) / 5) * 100.0
    assert r.ulcer_index == pytest.approx(expected_ulcer, abs=1e-9)
    assert r.max_drawdown_pct == pytest.approx(10.0, abs=1e-9)


def test_new_high_resets_underwater():
    # Make a new peak mid-run: dd should reset to 0 from that point on.
    # Equity: [10000, 11000, 10500, 12000]
    #   dd:   [0,      0,    ~4.54%, 0]  → 1 of 4 samples underwater.
    drs = _mk_day_results([11_000, 10_500, 12_000])
    r = _compute_result(drs, _cfg())
    assert r.time_under_water_pct == pytest.approx(25.0, abs=1e-9)
    dd = 500.0 / 11_000.0   # 4.545...%
    expected_ulcer = math.sqrt((dd * dd) / 4) * 100.0
    assert r.ulcer_index == pytest.approx(expected_ulcer, abs=1e-9)


def test_long_drawdown_high_tuw_same_max_dd():
    # Same max drawdown depth (10%) but sustained over many days → Ulcer
    # and TuW both dominate while max_dd alone misses the duration.
    #
    # 10 days each at equity 9000 after a peak of 10000.
    equities = [10_000] + [9_000] * 10
    drs = _mk_day_results(equities)
    r = _compute_result(drs, _cfg())
    # Sample count = 1 + 11 = 12; first 2 at peak (initial + day 0 both 10k),
    # next 10 at 9k → 10/12 underwater.
    assert r.time_under_water_pct == pytest.approx(10.0 / 12.0 * 100.0, abs=1e-9)
    # Ulcer = sqrt(10 * 0.1^2 / 12)*100 = sqrt(0.1/12)*100
    expected_ulcer = math.sqrt(0.1 / 12.0) * 100.0
    assert r.ulcer_index == pytest.approx(expected_ulcer, abs=1e-9)
    assert r.max_drawdown_pct == pytest.approx(10.0, abs=1e-9)


def test_empty_day_results_zero_pain():
    r = _compute_result([], _cfg())
    assert r.time_under_water_pct == 0.0
    assert r.ulcer_index == 0.0


def test_ulcer_penalizes_long_over_short_same_max_dd():
    """Key property: Ulcer distinguishes a 1-day −10% vs 10-day −10% even
    though max-DD is identical. TuW captures duration; Ulcer integrates
    depth^2 × duration."""
    short_dip = _compute_result(
        _mk_day_results([10_000, 9_000, 10_000, 10_000]),
        _cfg(),
    )
    long_dip = _compute_result(
        _mk_day_results([10_000] + [9_000] * 10 + [10_000]),
        _cfg(),
    )
    # Same max drawdown depth.
    assert short_dip.max_drawdown_pct == pytest.approx(10.0, abs=1e-9)
    assert long_dip.max_drawdown_pct == pytest.approx(10.0, abs=1e-9)
    # But Ulcer and TuW are both strictly larger for the long dip.
    assert long_dip.ulcer_index > short_dip.ulcer_index
    assert long_dip.time_under_water_pct > short_dip.time_under_water_pct
