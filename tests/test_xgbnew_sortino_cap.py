"""Tests for xgbnew.backtest._sortino_semi.

Exercises the semi-deviation based Sortino that replaces the old
``down_r.std()`` implementation, which blew up to ~6e8 when the
window had ≤1 losing day (len(down_r)-1 → 0 → 1e-9 floor in the
denominator).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xgbnew.backtest import _SORTINO_CAP, _sortino_semi  # noqa: E402


def test_no_losses_returns_cap_for_positive_mean():
    rets = np.array([0.01, 0.005, 0.02, 0.0, 0.015])
    s = _sortino_semi(rets, mean_r=float(rets.mean()), ann_factor=np.sqrt(252))
    assert s == _SORTINO_CAP, f"no-loss + positive mean → cap, got {s}"


def test_all_zeros_returns_zero():
    rets = np.zeros(10)
    s = _sortino_semi(rets, mean_r=0.0, ann_factor=np.sqrt(252))
    assert s == 0.0


def test_single_losing_day_stays_bounded():
    """The bug: one losing day → down_r has len 1 → std floor 1e-9 → sortino ~6e8."""
    rets = np.array([0.01] * 50 + [-0.0001])
    s = _sortino_semi(rets, mean_r=float(rets.mean()), ann_factor=np.sqrt(252))
    # With the semi-dev fix this stays well below the cap.
    assert abs(s) <= _SORTINO_CAP + 1e-6, f"bounded at cap, got {s}"
    # And it should still be a reasonably large positive (mostly positive returns).
    assert s > 5.0, f"expected clearly-positive sortino, got {s}"


def test_mixed_returns_reasonable():
    rng = np.random.default_rng(42)
    rets = rng.normal(loc=0.002, scale=0.01, size=252)
    s = _sortino_semi(rets, mean_r=float(rets.mean()), ann_factor=np.sqrt(252))
    assert abs(s) <= _SORTINO_CAP
    # Mildly positive mean, mixed tails → modest positive Sortino
    assert 0.5 < s < 15.0, f"expected moderate Sortino, got {s}"


def test_negative_mean_large_losses():
    rets = np.array([-0.02, -0.015, 0.005, -0.03, 0.01, -0.025])
    s = _sortino_semi(rets, mean_r=float(rets.mean()), ann_factor=np.sqrt(252))
    assert s < 0, f"negative mean + real downside → negative Sortino, got {s}"
    assert abs(s) <= _SORTINO_CAP


def test_short_series_returns_zero():
    # n < 2 → undefined semi-deviation → 0.0 by contract
    assert _sortino_semi(np.array([0.01]), 0.01, np.sqrt(252)) == 0.0
    assert _sortino_semi(np.array([]), 0.0, np.sqrt(252)) == 0.0
