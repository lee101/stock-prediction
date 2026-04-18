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

from xgbnew.backtest import (
    BacktestConfig,
    _build_regime_flags,
    _build_vol_scale,
)


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


def test_backtest_config_knob_defaults():
    cfg = BacktestConfig()
    assert cfg.regime_gate_window == 0
    assert cfg.vol_target_ann == 0.0


def test_backtest_config_knobs_settable():
    cfg = BacktestConfig(regime_gate_window=50, vol_target_ann=0.15)
    assert cfg.regime_gate_window == 50
    assert cfg.vol_target_ann == 0.15
