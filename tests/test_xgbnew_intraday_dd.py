"""Tests for xgbnew.backtest intraday excursion proxy.

The backtest carries same-day ``actual_high`` / ``actual_low`` through the
dataset so we can report the worst unrealized drawdown observed within
each traded day (the distance from the entry fill to the bar low, times
leverage). This is the "realized vs unrealized DD" metric the user asked
for — pure OHLC based, no hourly data required.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xgbnew.backtest import _intraday_excursion_pct  # noqa: E402


def _row(hi, lo):
    return pd.Series({"actual_high": hi, "actual_low": lo})


def test_long_drawdown_pure():
    # Entry at 100, bar low 95 → 5% DD at 1x.
    dd, up = _intraday_excursion_pct(_row(101.0, 95.0), entry_ref=100.0, leverage=1.0)
    assert dd == 5.0
    assert up == 1.0


def test_leverage_scales_both_legs():
    dd, up = _intraday_excursion_pct(_row(102.0, 98.0), entry_ref=100.0, leverage=2.0)
    assert dd == 4.0  # 2% * 2x
    assert up == 4.0  # 2% * 2x


def test_missing_high_low_returns_zero():
    dd, up = _intraday_excursion_pct(_row(0.0, 0.0), entry_ref=100.0, leverage=1.0)
    assert dd == 0.0
    assert up == 0.0


def test_bar_above_entry_no_dd():
    # Gap up — bar's low is above entry, DD should clamp to 0.
    dd, up = _intraday_excursion_pct(_row(110.0, 102.0), entry_ref=100.0, leverage=1.0)
    assert dd == 0.0
    assert up == 10.0


def test_bar_below_entry_no_runup():
    # Gap down — bar's high is below entry, runup clamps to 0.
    dd, up = _intraday_excursion_pct(_row(99.0, 90.0), entry_ref=100.0, leverage=1.0)
    assert dd == 10.0
    assert up == 0.0


def test_zero_entry_ref_returns_zero():
    dd, up = _intraday_excursion_pct(_row(100.0, 95.0), entry_ref=0.0, leverage=1.0)
    assert dd == 0.0
    assert up == 0.0


def test_nan_high_low_returns_zero():
    dd, up = _intraday_excursion_pct(
        _row(float("nan"), float("nan")), entry_ref=100.0, leverage=1.0
    )
    assert dd == 0.0
    assert up == 0.0


def test_missing_columns_returns_zero():
    # Row without actual_high / actual_low at all (legacy dataset path).
    dd, up = _intraday_excursion_pct(pd.Series({}), entry_ref=100.0, leverage=1.0)
    assert dd == 0.0
    assert up == 0.0


def test_large_crash_scaled_by_leverage():
    dd, up = _intraday_excursion_pct(_row(101.0, 80.0), entry_ref=100.0, leverage=3.0)
    assert dd == pytest.approx(60.0)   # 20% * 3x
    assert up == pytest.approx(3.0)    # 1%  * 3x
