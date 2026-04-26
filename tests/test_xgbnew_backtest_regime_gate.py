"""Regime gate + vol-target sizing on xgbnew.backtest.

Focused on the two new BacktestConfig knobs:
  * ``regime_gate_window`` — skip days when SPY < SPY_ma(window).
  * ``vol_target_ann``     — scale daily exposure by min(1, target/realised).

We call the pure helpers ``_build_regime_flags`` / ``_build_vol_scale``
directly because they're the load-bearing contract; ``simulate()``
integrates them into the per-day loop.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

import xgbnew.backtest as backtest
from xgbnew.backtest import (
    BacktestConfig,
    _build_regime_flags,
    _build_vol_scale,
    simulate,
)
from xgbnew.features import cross_sectional_regime_keep_by_date


def _spy_updown(n: int = 120, seed: int = 7) -> pd.Series:
    """Half uptrend, half downtrend SPY series so MA50 is unambiguous."""
    rng = np.random.default_rng(seed)
    up = 100.0 * np.exp(np.cumsum(rng.normal(0.002, 0.005, n // 2)))
    down = up[-1] * np.exp(np.cumsum(rng.normal(-0.003, 0.005, n - n // 2)))
    prices = np.concatenate([up, down])
    start = date(2024, 1, 2)
    dates = [start + timedelta(days=int(i)) for i in range(n)]
    return pd.Series(prices, index=pd.Index(dates, name="date"))


def test_regime_gate_disabled_returns_all_open():
    sp = _spy_updown()
    dates = pd.Index(sp.index)
    closed = _build_regime_flags(sp, dates, window=0)
    assert not closed.any()


def test_regime_gate_ma50_closes_more_on_downtrend_tail():
    sp = _spy_updown(n=200)
    dates = pd.Index(sp.index)
    closed = _build_regime_flags(sp, dates, window=50)
    # First half is uptrend → expect mostly open. Last 30 days are well into the
    # downtrend → expect mostly closed.
    early_open = (~closed.iloc[50:100]).sum()
    late_closed = closed.iloc[-30:].sum()
    assert early_open >= 40, f"uptrend should keep gate open: {early_open}/50"
    assert late_closed >= 20, f"downtrend should close gate: {late_closed}/30"


def test_regime_gate_missing_dates_default_open():
    sp = _spy_updown()
    # Holiday-like missing date, SPY has no row for it.
    extra_dates = pd.Index(list(sp.index) + [date(2099, 1, 1)])
    closed = _build_regime_flags(sp, extra_dates, window=50)
    assert closed.loc[date(2099, 1, 1)] is np.False_ or bool(closed.loc[date(2099, 1, 1)]) is False


def test_regime_gate_available_at_open_uses_prior_close():
    """A same-day SPY breakdown must not close the gate before the open."""
    start = date(2024, 1, 2)
    dates = pd.Index([start + timedelta(days=int(i)) for i in range(4)])
    sp = pd.Series([100.0, 100.0, 100.0, 50.0], index=dates)

    as_of_close = _build_regime_flags(
        sp,
        dates,
        window=2,
        available_at_open=False,
    )
    available_at_open = _build_regime_flags(
        sp,
        dates,
        window=2,
        available_at_open=True,
    )

    assert bool(as_of_close.iloc[-1]) is True
    assert bool(available_at_open.iloc[-1]) is False


def test_vol_scale_disabled_returns_one():
    sp = _spy_updown()
    dates = pd.Index(sp.index)
    scale = _build_vol_scale(sp, dates, target_ann=0.0)
    assert (scale == 1.0).all()


def test_vol_scale_clips_to_one_in_low_vol():
    """If realised vol is much lower than target, scale should pin at 1.0."""
    # Synthetic near-zero-vol SPY series.
    n = 80
    start = date(2024, 1, 2)
    dates = pd.Index([start + timedelta(days=int(i)) for i in range(n)])
    sp = pd.Series(100.0 + np.linspace(0.0, 0.001, n), index=dates)
    scale = _build_vol_scale(sp, dates, target_ann=0.15)
    # After the first 20 days have vol data, scale should be exactly 1.0 (clipped).
    assert (scale.iloc[21:] == 1.0).all()


def test_vol_scale_down_scales_in_high_vol():
    """Target 15% ann vs ~50% realised ann → scale around 0.3."""
    rng = np.random.default_rng(0)
    n = 80
    start = date(2024, 1, 2)
    dates = pd.Index([start + timedelta(days=int(i)) for i in range(n)])
    daily_sigma = 0.50 / np.sqrt(252)
    log_ret = rng.normal(0.0, daily_sigma, n)
    prices = 100.0 * np.exp(np.cumsum(log_ret))
    sp = pd.Series(prices, index=dates)
    scale = _build_vol_scale(sp, dates, target_ann=0.15, lookback_days=20)
    tail = scale.iloc[25:]
    # Everything should be strictly < 1 (we're well above target) and roughly near 0.3.
    assert (tail < 1.0).all()
    assert 0.15 < float(tail.median()) < 0.55


def test_vol_scale_available_at_open_uses_prior_close():
    """A same-day SPY shock must not alter exposure already set at the open."""
    n = 30
    start = date(2024, 1, 2)
    dates = pd.Index([start + timedelta(days=int(i)) for i in range(n)])
    prices = np.full(n, 100.0)
    prices[-1] = 50.0
    sp = pd.Series(prices, index=dates)

    as_of_close = _build_vol_scale(
        sp,
        dates,
        target_ann=0.10,
        lookback_days=2,
        available_at_open=False,
    )
    available_at_open = _build_vol_scale(
        sp,
        dates,
        target_ann=0.10,
        lookback_days=2,
        available_at_open=True,
    )

    assert as_of_close.iloc[-1] < 1.0
    assert available_at_open.iloc[-1] == 1.0


def test_regime_gate_tolerates_duplicate_spy_dates():
    """Regression: real SPY CSVs carry multiple bars per date. The helper
    must not raise on a duplicate-indexed input — it should collapse to
    per-date last close before the rolling MA."""
    sp = _spy_updown(n=200)
    # Duplicate every label (simulating hourly bars collapsed to `dt.date`).
    dup = pd.concat([sp, sp])
    dates = pd.Index(sp.index)
    closed = _build_regime_flags(dup, dates, window=50)
    assert len(closed) == len(dates)
    assert closed.iloc[-30:].sum() >= 20


def test_vol_scale_tolerates_duplicate_spy_dates():
    """Same duplicate-label regression for the vol-scale helper."""
    rng = np.random.default_rng(0)
    n = 80
    start = date(2024, 1, 2)
    dates = pd.Index([start + timedelta(days=int(i)) for i in range(n)])
    daily_sigma = 0.50 / np.sqrt(252)
    log_ret = rng.normal(0.0, daily_sigma, n)
    prices = 100.0 * np.exp(np.cumsum(log_ret))
    sp = pd.Series(prices, index=dates)
    dup = pd.concat([sp, sp])
    scale = _build_vol_scale(dup, dates, target_ann=0.15, lookback_days=20)
    assert len(scale) == len(dates)
    tail = scale.iloc[25:]
    assert (tail < 1.0).all()


def test_backtest_config_knob_defaults():
    cfg = BacktestConfig()
    assert cfg.regime_gate_window == 0
    assert cfg.vol_target_ann == 0.0


def test_backtest_config_knobs_settable():
    cfg = BacktestConfig(regime_gate_window=50, vol_target_ann=0.15)
    assert cfg.regime_gate_window == 50
    assert cfg.vol_target_ann == 0.15


def test_simulate_vol_target_uses_open_available_spy_scale(monkeypatch):
    d1 = date(2025, 1, 2)
    d2 = date(2025, 1, 3)
    df = pd.DataFrame([
        {
            "date": d1,
            "symbol": "A",
            "actual_open": 100.0,
            "actual_close": 102.0,
            "spread_bps": 2.0,
            "dolvol_20d_log": 20.0,
        },
        {
            "date": d2,
            "symbol": "A",
            "actual_open": 100.0,
            "actual_close": 102.0,
            "spread_bps": 2.0,
            "dolvol_20d_log": 20.0,
        },
    ])
    seen: dict[str, bool] = {}

    def fake_build_vol_scale(_spy_close, all_dates, **kwargs):
        seen["available_at_open"] = bool(kwargs.get("available_at_open"))
        return pd.Series(1.0, index=all_dates)

    monkeypatch.setattr(backtest, "_build_vol_scale", fake_build_vol_scale)
    backtest.simulate(
        df,
        model=None,
        config=BacktestConfig(
            top_n=1,
            min_dollar_vol=0.0,
            max_spread_bps=10.0,
            vol_target_ann=0.10,
        ),
        precomputed_scores=pd.Series([0.9, 0.9], index=df.index),
        spy_close_by_date=pd.Series([100.0, 101.0], index=pd.Index([d1, d2])),
    )

    assert seen == {"available_at_open": True}


def test_simulate_regime_gate_uses_open_available_spy_flags(monkeypatch):
    d1 = date(2025, 1, 2)
    d2 = date(2025, 1, 3)
    df = pd.DataFrame([
        {
            "date": d1,
            "symbol": "A",
            "actual_open": 100.0,
            "actual_close": 102.0,
            "spread_bps": 2.0,
            "dolvol_20d_log": 20.0,
        },
        {
            "date": d2,
            "symbol": "A",
            "actual_open": 100.0,
            "actual_close": 102.0,
            "spread_bps": 2.0,
            "dolvol_20d_log": 20.0,
        },
    ])
    seen: dict[str, bool] = {}

    def fake_build_regime_flags(_spy_close, all_dates, **kwargs):
        seen["available_at_open"] = bool(kwargs.get("available_at_open"))
        return pd.Series(False, index=all_dates)

    monkeypatch.setattr(backtest, "_build_regime_flags", fake_build_regime_flags)
    backtest.simulate(
        df,
        model=None,
        config=BacktestConfig(
            top_n=1,
            min_dollar_vol=0.0,
            max_spread_bps=10.0,
            regime_gate_window=2,
        ),
        precomputed_scores=pd.Series([0.9, 0.9], index=df.index),
        spy_close_by_date=pd.Series([100.0, 101.0], index=pd.Index([d1, d2])),
    )

    assert seen == {"available_at_open": True}


def test_cross_sectional_regime_gate_drops_wide_dispersion_days():
    d1 = date(2025, 1, 2)
    d2 = date(2025, 1, 3)
    df = pd.DataFrame([
        {"date": d1, "symbol": "A", "actual_open": 100.0, "actual_close": 101.0,
         "spread_bps": 2.0, "dolvol_20d_log": 20.0, "ret_5d": -0.01},
        {"date": d1, "symbol": "B", "actual_open": 100.0, "actual_close": 101.0,
         "spread_bps": 2.0, "dolvol_20d_log": 20.0, "ret_5d": 0.01},
        {"date": d2, "symbol": "A", "actual_open": 100.0, "actual_close": 101.0,
         "spread_bps": 2.0, "dolvol_20d_log": 20.0, "ret_5d": -0.30},
        {"date": d2, "symbol": "B", "actual_open": 100.0, "actual_close": 101.0,
         "spread_bps": 2.0, "dolvol_20d_log": 20.0, "ret_5d": 0.30},
    ])
    keep_by_date = cross_sectional_regime_keep_by_date(
        df,
        regime_cs_iqr_max=0.05,
    )
    assert keep_by_date.to_dict() == {d1: True, d2: False}

    result = simulate(
        df,
        model=None,
        config=BacktestConfig(
            top_n=1,
            min_dollar_vol=0.0,
            max_spread_bps=10.0,
            regime_cs_iqr_max=0.05,
        ),
        precomputed_scores=pd.Series([0.9, 0.8, 0.9, 0.8], index=df.index),
    )

    assert [day.day for day in result.day_results] == [d1]
