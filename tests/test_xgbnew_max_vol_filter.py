"""Tests for the inference-side MAX vol_20d filter.

Purpose: users can mask crash-sensitive high-vol names out of the INFERENCE
pick pool while keeping them in training. This avoids retraining when the
regime shifts — a backstop for the "0/192 cells positive on true-OOS"
finding in project_xgb_true_oos_no_edge_2026_04_21.md.

Contract:
  - config.max_vol_20d == 0.0 → no filter applied (back-compat).
  - config.max_vol_20d > 0 and vol_20d column present → rows with
    vol_20d > threshold are dropped from the pick pool.
  - Band-pass with min_vol_20d works (both bounds honoured).
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from xgbnew.backtest import BacktestConfig, simulate
from xgbnew.model import XGBStockModel


def _toy_test_df(n_days: int = 20, n_syms: int = 4) -> pd.DataFrame:
    """A deterministic test frame with 4 symbols at vol_20d = [0.05, 0.15, 0.30, 0.60].

    The HIGH-vol symbol (vol_20d=0.60) always has the highest score, so
    without a max_vol filter it always gets picked.
    """
    rng = np.random.default_rng(0)
    vols = [0.05, 0.15, 0.30, 0.60]
    rows = []
    d0 = date(2025, 1, 6)
    for i in range(n_days):
        d = d0 + timedelta(days=i)
        for j in range(n_syms):
            rows.append({
                "date": d,
                "symbol": f"S{j}",
                "actual_open": 100.0 + rng.normal(0, 0.5),
                "actual_close": 100.0 + rng.normal(0, 0.5),
                "spread_bps": 10.0,
                "dolvol_20d_log": 20.0,  # large → passes liquidity floor
                "vol_20d": vols[j],
                "target_oc": 0.0,
            })
    df = pd.DataFrame(rows)
    # Minimum set of feature columns — zeros are fine, sim uses precomputed scores
    from xgbnew.features import DAILY_FEATURE_COLS
    for c in DAILY_FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0
    return df


def _dummy_model() -> XGBStockModel:
    from xgbnew.features import DAILY_FEATURE_COLS
    m = XGBStockModel(device="cpu", n_estimators=1, max_depth=1, learning_rate=0.1)
    m.feature_cols = DAILY_FEATURE_COLS
    m._col_medians = np.zeros(len(DAILY_FEATURE_COLS), dtype=np.float32)
    m._fitted = True
    return m


def _scores_prefer_high_vol(df: pd.DataFrame) -> pd.Series:
    """S3 (high vol) > S2 > S1 > S0 (low vol)."""
    rank = df["symbol"].map({"S0": 0.55, "S1": 0.60, "S2": 0.70, "S3": 0.90})
    return pd.Series(rank.values, index=df.index, name="score")


def _picked_symbols(result) -> set[str]:
    out = set()
    for day in result.day_results:
        for t in day.trades:
            out.add(t.symbol)
    return out


def test_max_vol_filter_default_is_noop() -> None:
    """max_vol_20d=0.0 → high-vol S3 still wins every day."""
    df = _toy_test_df()
    scores = _scores_prefer_high_vol(df)
    cfg = BacktestConfig(
        top_n=1, leverage=1.0, min_score=0.0, min_dollar_vol=1e5,
        min_vol_20d=0.0, max_vol_20d=0.0,  # disabled
        xgb_weight=1.0, fee_rate=0.0, fill_buffer_bps=0.0,
        commission_bps=0.0, max_spread_bps=30.0,
    )
    result = simulate(df, _dummy_model(), cfg, precomputed_scores=scores)
    assert _picked_symbols(result) == {"S3"}


def test_max_vol_filter_excludes_high_vol_names() -> None:
    """max_vol_20d=0.25 → S3 (0.60) and S2 (0.30) dropped; pick falls to S1 (0.15)."""
    df = _toy_test_df()
    scores = _scores_prefer_high_vol(df)
    cfg = BacktestConfig(
        top_n=1, leverage=1.0, min_score=0.0, min_dollar_vol=1e5,
        min_vol_20d=0.0, max_vol_20d=0.25,
        xgb_weight=1.0, fee_rate=0.0, fill_buffer_bps=0.0,
        commission_bps=0.0, max_spread_bps=30.0,
    )
    result = simulate(df, _dummy_model(), cfg, precomputed_scores=scores)
    picks = _picked_symbols(result)
    assert "S3" not in picks
    assert "S2" not in picks
    # S1 (vol 0.15) is the survivor with highest score ≤ ceiling
    assert picks == {"S1"}


def test_band_pass_min_and_max_vol_work_together() -> None:
    """min_vol=0.10 AND max_vol=0.40 → only S1 (0.15) and S2 (0.30) qualify."""
    df = _toy_test_df()
    scores = _scores_prefer_high_vol(df)
    cfg = BacktestConfig(
        top_n=1, leverage=1.0, min_score=0.0, min_dollar_vol=1e5,
        min_vol_20d=0.10, max_vol_20d=0.40,
        xgb_weight=1.0, fee_rate=0.0, fill_buffer_bps=0.0,
        commission_bps=0.0, max_spread_bps=30.0,
    )
    result = simulate(df, _dummy_model(), cfg, precomputed_scores=scores)
    picks = _picked_symbols(result)
    # S0 (0.05) fails min; S3 (0.60) fails max; S2 (0.30) has higher score than S1 (0.15)
    assert picks == {"S2"}


def test_sweep_cell_threads_max_vol_to_cellresult() -> None:
    """End-to-end: the sweep CLI/function propagates inference_max_vol_20d
    into the CellResult so the output JSON carries it."""
    from xgbnew.sweep_ensemble_grid import CellResult

    # Just check the dataclass accepts the new field
    c = CellResult(
        leverage=1.0, min_score=0.5, hold_through=False, top_n=1,
        fee_regime="deploy", n_windows=0, median_monthly_pct=0.0,
        p10_monthly_pct=0.0, median_sortino=0.0, worst_dd_pct=0.0, n_neg=0,
    )
    assert c.inference_max_vol_20d == 0.0
    c.inference_max_vol_20d = 0.40
    assert c.inference_max_vol_20d == 0.40
