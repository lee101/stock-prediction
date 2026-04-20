"""Tests for xgbnew.features.add_cross_sectional_ranks.

Cross-sectional ranks are per-day, computed across all symbols in the
supplied DataFrame. This module verifies:
  1. Ranks are computed per-day, not globally.
  2. Hand-crafted panel → predictable rank ordering (ret_1d highest → 1.0).
  3. NaN source values → filled with fill_value (0.5 by default).
  4. Single-symbol days don't crash (edge case — guard behaviour).
  5. Missing required 'date' column raises ValueError with clear message.
  6. Missing source column raises ValueError.
  7. Output doesn't mutate the input DataFrame in place.
  8. Default source→rank mapping adds exactly the DAILY_RANK_FEATURE_COLS.
  9. Ranks are in (0, 1] (or exactly fill_value where source was NaN).
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xgbnew.features import (  # noqa: E402
    DAILY_RANK_FEATURE_COLS,
    add_cross_sectional_ranks,
)


def _panel(n_days: int = 3, n_syms: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        day = date(2025, 1, 2 + d)
        for k in range(n_syms):
            rows.append({
                "date": day,
                "symbol": f"SYM{k}",
                "ret_1d":         float(rng.normal()),
                "ret_5d":         float(rng.normal()),
                "vol_20d":        float(abs(rng.normal())),
                "dolvol_20d_log": float(10.0 + rng.normal()),
                "rsi_14":         float(50.0 + 10.0 * rng.normal()),
            })
    return pd.DataFrame(rows)


def test_ranks_are_per_day_not_global():
    df = _panel(n_days=2, n_syms=3)
    # Make day 1 ret_1d all HIGHER than any day 0 ret_1d so a global rank
    # would put day 1 rows uniformly above day 0.
    df.loc[df["date"] == date(2025, 1, 2), "ret_1d"] = [-3.0, -2.0, -1.0]
    df.loc[df["date"] == date(2025, 1, 3), "ret_1d"] = [10.0, 20.0, 30.0]
    out = add_cross_sectional_ranks(df)
    # Per-day ranks: the *within-day* ordering for day 0 must match day 1.
    day0 = out[out["date"] == date(2025, 1, 2)]["rank_ret_1d"].values
    day1 = out[out["date"] == date(2025, 1, 3)]["rank_ret_1d"].values
    np.testing.assert_allclose(day0, day1, rtol=0, atol=1e-9,
        err_msg=f"per-day ranks differ: day0={day0} day1={day1}")
    # Both days should span [1/3, 1.0] (3 symbols, average method): 1/3, 2/3, 3/3.
    np.testing.assert_allclose(sorted(day0), [1/3, 2/3, 1.0], atol=1e-9)


def test_hand_crafted_ordering():
    """Largest source value in a day → rank 1.0; smallest → 1/N."""
    df = pd.DataFrame([
        {"date": date(2025, 1, 2), "symbol": "A", "ret_1d": 0.01,
         "ret_5d": 0.0, "vol_20d": 0.1, "dolvol_20d_log": 5.0, "rsi_14": 50},
        {"date": date(2025, 1, 2), "symbol": "B", "ret_1d": 0.05,
         "ret_5d": 0.0, "vol_20d": 0.1, "dolvol_20d_log": 5.0, "rsi_14": 50},
        {"date": date(2025, 1, 2), "symbol": "C", "ret_1d": -0.02,
         "ret_5d": 0.0, "vol_20d": 0.1, "dolvol_20d_log": 5.0, "rsi_14": 50},
    ])
    out = add_cross_sectional_ranks(df)
    got = dict(zip(out["symbol"], out["rank_ret_1d"]))
    assert got["B"] == pytest.approx(1.0), f"B largest → 1.0, got {got}"
    assert got["A"] == pytest.approx(2/3), f"A middle → 2/3, got {got}"
    assert got["C"] == pytest.approx(1/3), f"C smallest → 1/3, got {got}"


def test_nan_source_filled_with_fill_value():
    df = pd.DataFrame([
        {"date": date(2025, 1, 2), "symbol": "A", "ret_1d": np.nan,
         "ret_5d": 0.0, "vol_20d": 0.1, "dolvol_20d_log": 5.0, "rsi_14": 50},
        {"date": date(2025, 1, 2), "symbol": "B", "ret_1d": 0.05,
         "ret_5d": 0.0, "vol_20d": 0.1, "dolvol_20d_log": 5.0, "rsi_14": 50},
    ])
    out = add_cross_sectional_ranks(df, fill_value=0.5)
    # A's ret_1d was NaN → rank NaN → filled with 0.5.
    # B is the only non-NaN → rank = 1.0.
    by_sym = dict(zip(out["symbol"], out["rank_ret_1d"]))
    assert by_sym["A"] == pytest.approx(0.5)
    assert by_sym["B"] == pytest.approx(1.0)


def test_single_symbol_day_no_crash():
    """If a day only has one symbol, rank = 1.0 (pandas default)."""
    df = pd.DataFrame([
        {"date": date(2025, 1, 2), "symbol": "A", "ret_1d": 0.01,
         "ret_5d": 0.0, "vol_20d": 0.1, "dolvol_20d_log": 5.0, "rsi_14": 50},
    ])
    out = add_cross_sectional_ranks(df)
    assert out["rank_ret_1d"].iloc[0] == pytest.approx(1.0)


def test_missing_date_column_raises():
    df = pd.DataFrame([{"symbol": "A", "ret_1d": 0.01, "ret_5d": 0.0,
                         "vol_20d": 0.1, "dolvol_20d_log": 5.0, "rsi_14": 50}])
    with pytest.raises(ValueError, match="requires a 'date' column"):
        add_cross_sectional_ranks(df)


def test_missing_source_column_raises():
    df = pd.DataFrame([
        {"date": date(2025, 1, 2), "symbol": "A",
         "ret_5d": 0.0, "vol_20d": 0.1, "dolvol_20d_log": 5.0, "rsi_14": 50},
        # ret_1d missing entirely
    ])
    with pytest.raises(ValueError, match="source column 'ret_1d' missing"):
        add_cross_sectional_ranks(df)


def test_does_not_mutate_input():
    df = _panel(n_days=2, n_syms=3)
    before_cols = set(df.columns)
    before_shape = df.shape
    _ = add_cross_sectional_ranks(df)
    # Original df must still have only the original columns.
    assert set(df.columns) == before_cols
    assert df.shape == before_shape


def test_default_mapping_produces_daily_rank_feature_cols():
    df = _panel(n_days=2, n_syms=3)
    out = add_cross_sectional_ranks(df)
    for col in DAILY_RANK_FEATURE_COLS:
        assert col in out.columns, f"{col} missing from output"
    # No extra rank cols beyond the registered set
    extra = [c for c in out.columns if c.startswith("rank_")]
    assert set(extra) == set(DAILY_RANK_FEATURE_COLS), f"extra rank cols: {extra}"


def test_ranks_in_unit_interval():
    df = _panel(n_days=3, n_syms=10, seed=7)
    # Scatter some NaNs.
    df.loc[df.index[::5], "ret_1d"] = np.nan
    out = add_cross_sectional_ranks(df, fill_value=0.5)
    for col in DAILY_RANK_FEATURE_COLS:
        vals = out[col].values
        assert np.all(np.isfinite(vals)), f"{col} has non-finite values"
        assert np.all(vals > 0.0) and np.all(vals <= 1.0), (
            f"{col} outside (0, 1]: min={vals.min()} max={vals.max()}"
        )
