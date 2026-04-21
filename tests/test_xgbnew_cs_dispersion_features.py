"""Tests for add_cross_sectional_dispersion feature builder.

Adds day-level ``cs_iqr_ret5`` and ``cs_skew_ret5`` features (broadcast
same value to every row on the same date). Leak-free: ret_5d is lag-1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xgbnew.features import (
    DAILY_DISPERSION_FEATURE_COLS,
    add_cross_sectional_dispersion,
)


def _panel(rows: list[tuple]) -> pd.DataFrame:
    """Build a minimal dataframe for the function under test."""
    return pd.DataFrame(rows, columns=["date", "symbol", "ret_5d"])


def test_dispersion_adds_both_feature_cols():
    df = _panel([
        ("2025-01-06", "A", -0.01),
        ("2025-01-06", "B", +0.01),
        ("2025-01-06", "C", +0.02),
        ("2025-01-06", "D", -0.02),
    ])
    out = add_cross_sectional_dispersion(df)
    for col in DAILY_DISPERSION_FEATURE_COLS:
        assert col in out.columns


def test_dispersion_iqr_matches_manual_calc():
    df = _panel([
        ("2025-01-06", "A", -0.02),
        ("2025-01-06", "B", -0.01),
        ("2025-01-06", "C", +0.01),
        ("2025-01-06", "D", +0.02),
    ])
    out = add_cross_sectional_dispersion(df)
    # Q75 = 0.0125, Q25 = -0.0125 → IQR = 0.025
    expected_iqr = 0.025
    for v in out["cs_iqr_ret5"].tolist():
        assert v == pytest.approx(expected_iqr, abs=1e-5)


def test_dispersion_is_per_day_broadcast():
    """Two days with different dispersion → every row in a day gets
    the same value; values differ across days."""
    df = _panel([
        ("2025-01-06", "A", -0.02),
        ("2025-01-06", "B", -0.01),
        ("2025-01-06", "C", +0.01),
        ("2025-01-06", "D", +0.02),
        ("2025-01-07", "A", -0.10),
        ("2025-01-07", "B", -0.05),
        ("2025-01-07", "C", +0.05),
        ("2025-01-07", "D", +0.10),
    ])
    out = add_cross_sectional_dispersion(df)
    day0 = out[out["date"] == "2025-01-06"]
    day1 = out[out["date"] == "2025-01-07"]
    # Intra-day uniformity
    assert day0["cs_iqr_ret5"].nunique() == 1
    assert day1["cs_iqr_ret5"].nunique() == 1
    # Inter-day differences: day1 has ~5× wider IQR than day0
    assert day1["cs_iqr_ret5"].iloc[0] > 4 * day0["cs_iqr_ret5"].iloc[0]


def test_dispersion_skew_sign_flipped_for_mirrored_days():
    """Positive-skew day (right-tail) vs negative-skew day."""
    df = _panel([
        ("2025-01-06", "A", -0.01),
        ("2025-01-06", "B", -0.01),
        ("2025-01-06", "C", -0.01),
        ("2025-01-06", "D", +0.20),   # positive skew
        ("2025-01-07", "A", +0.01),
        ("2025-01-07", "B", +0.01),
        ("2025-01-07", "C", +0.01),
        ("2025-01-07", "D", -0.20),   # negative skew
    ])
    out = add_cross_sectional_dispersion(df)
    day0 = out[out["date"] == "2025-01-06"]["cs_skew_ret5"].iloc[0]
    day1 = out[out["date"] == "2025-01-07"]["cs_skew_ret5"].iloc[0]
    assert day0 > 0.0
    assert day1 < 0.0
    assert day0 == pytest.approx(-day1, abs=1e-4)   # mirror symmetry


def test_dispersion_single_symbol_day_does_not_crash():
    df = _panel([("2025-01-06", "A", +0.01)])
    out = add_cross_sectional_dispersion(df)
    assert "cs_iqr_ret5" in out.columns
    assert "cs_skew_ret5" in out.columns
    # Skew of a single value is NaN → filled with default 0.0
    assert float(out["cs_skew_ret5"].iloc[0]) == pytest.approx(0.0, abs=1e-5)


def test_dispersion_missing_date_raises():
    df = pd.DataFrame({"symbol": ["A"], "ret_5d": [0.01]})
    with pytest.raises(ValueError, match="date"):
        add_cross_sectional_dispersion(df)


def test_dispersion_missing_ret5d_raises():
    df = pd.DataFrame({"date": ["2025-01-06"], "symbol": ["A"]})
    with pytest.raises(ValueError, match="ret_5d"):
        add_cross_sectional_dispersion(df)


def test_dispersion_fill_value_for_nan_days():
    """If a day's ret_5d is entirely NaN, dispersion falls back to fill_value."""
    df = pd.DataFrame({
        "date":   ["2025-01-06", "2025-01-06", "2025-01-07", "2025-01-07"],
        "symbol": ["A", "B", "A", "B"],
        "ret_5d": [np.nan, np.nan, -0.01, +0.01],
    })
    out = add_cross_sectional_dispersion(df, fill_value=0.0)
    nan_day = out[out["date"] == "2025-01-06"]
    normal_day = out[out["date"] == "2025-01-07"]
    assert float(nan_day["cs_iqr_ret5"].iloc[0]) == pytest.approx(0.0, abs=1e-5)
    assert float(normal_day["cs_iqr_ret5"].iloc[0]) > 0.0


def test_dispersion_preserves_existing_cols():
    df = _panel([
        ("2025-01-06", "A", -0.01),
        ("2025-01-06", "B", +0.01),
    ])
    df["extra_col"] = [1.0, 2.0]
    out = add_cross_sectional_dispersion(df)
    assert "extra_col" in out.columns
    assert out["extra_col"].tolist() == [1.0, 2.0]


def test_dispersion_output_dtypes_are_float32():
    df = _panel([
        ("2025-01-06", "A", -0.01),
        ("2025-01-06", "B", +0.01),
    ])
    out = add_cross_sectional_dispersion(df)
    assert out["cs_iqr_ret5"].dtype == np.float32
    assert out["cs_skew_ret5"].dtype == np.float32
