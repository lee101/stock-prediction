"""Tests for xgbnew.symbol_kfold.

Covers the bucketing logic (every symbol in exactly one bucket, buckets
are roughly equal-sized, liquidity mode respects the anchor-day ordering)
and the model-path resolver (globs + comma-lists collapse to unique).
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

from xgbnew.symbol_kfold import _bucket_symbols, _resolve_model_paths  # noqa: E402


def _mock_oos_df(syms, liquidities=None, vols=None):
    rows = []
    for i, s in enumerate(syms):
        rows.append({
            "date": date(2025, 1, 2),
            "symbol": s,
            "dolvol_20d_log": float(liquidities[i] if liquidities else i),
            "vol_20d":        float(vols[i]        if vols        else i),
        })
    return pd.DataFrame(rows)


def test_alpha_buckets_cover_all_symbols_no_overlap():
    syms = [f"SYM{i:03d}" for i in range(20)]
    df = _mock_oos_df(syms)
    out = _bucket_symbols(syms, df, mode="alpha", n_buckets=4)
    collected = []
    for lst in out.values():
        collected.extend(lst)
    assert sorted(collected) == sorted(syms), "every symbol exactly once"
    # Each bucket is 4 or 5 symbols for 20/4.
    sizes = sorted(len(v) for v in out.values())
    assert sizes == [5, 5, 5, 5]


def test_liquidity_buckets_ordered_by_dolvol():
    syms = ["A", "B", "C", "D", "E", "F", "G", "H"]
    liq = [5.0, 1.0, 8.0, 3.0, 7.0, 2.0, 4.0, 6.0]
    df = _mock_oos_df(syms, liquidities=liq)
    out = _bucket_symbols(syms, df, mode="liquidity", n_buckets=4)
    # Bucket 0 = lowest liquidity (sorted asc). We check ordering inside
    # the bucket list is asc by sorted-liquidity order.
    b0 = out[0]; b3 = out[3]
    # Lowest two liquidities are B(1.0), F(2.0) — bucket 0 should hold these.
    assert set(b0) == {"B", "F"}
    # Highest two are C(8.0), E(7.0).
    assert set(b3) == {"C", "E"}


def test_n_buckets_lt_2_raises():
    syms = ["A", "B"]
    df = _mock_oos_df(syms)
    with pytest.raises(ValueError, match="n_buckets"):
        _bucket_symbols(syms, df, mode="alpha", n_buckets=1)


def test_not_enough_symbols_for_bucket_count_raises():
    syms = ["A", "B"]
    df = _mock_oos_df(syms)
    with pytest.raises(ValueError, match="can't make"):
        _bucket_symbols(syms, df, mode="alpha", n_buckets=4)


def test_liquidity_needs_dolvol_col():
    syms = ["A", "B", "C", "D"]
    df = pd.DataFrame({
        "date": [date(2025, 1, 2)] * 4,
        "symbol": syms,
    })
    with pytest.raises(ValueError, match="dolvol_20d_log"):
        _bucket_symbols(syms, df, mode="liquidity", n_buckets=2)


def test_volatility_needs_vol20d_col():
    syms = ["A", "B", "C", "D"]
    df = pd.DataFrame({
        "date": [date(2025, 1, 2)] * 4,
        "symbol": syms,
    })
    with pytest.raises(ValueError, match="vol_20d"):
        _bucket_symbols(syms, df, mode="volatility", n_buckets=2)


def test_resolve_model_paths_glob(tmp_path):
    # Create some dummy pkl files.
    (tmp_path / "a.pkl").write_text("x")
    (tmp_path / "b.pkl").write_text("x")
    (tmp_path / "c.pkl").write_text("x")
    pattern = str(tmp_path / "*.pkl")
    resolved = _resolve_model_paths(pattern)
    assert len(resolved) == 3
    assert all(p.name.endswith(".pkl") for p in resolved)


def test_resolve_model_paths_comma_and_dedup(tmp_path):
    (tmp_path / "a.pkl").write_text("x")
    (tmp_path / "b.pkl").write_text("x")
    # Comma list with duplicate — dedup by resolved path.
    spec = f"{tmp_path}/a.pkl,{tmp_path}/b.pkl,{tmp_path}/a.pkl"
    resolved = _resolve_model_paths(spec)
    assert len(resolved) == 2


def test_unknown_bucket_mode_raises():
    syms = ["A", "B", "C", "D"]
    df = _mock_oos_df(syms)
    with pytest.raises(ValueError, match="unknown bucket mode"):
        _bucket_symbols(syms, df, mode="nonsense", n_buckets=2)
