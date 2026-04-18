"""Tests for the walk-forward OOS harness in ``xgbnew.eval_walk_forward``.

These tests are fast: they build tiny synthetic OHLCV data for a handful
of symbols and run the walk-forward loop with a deliberately-small XGB
model. No production data is touched.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xgbnew.backtest import BacktestConfig
from xgbnew.eval_walk_forward import (
    FoldResult,
    WalkForwardResult,
    _train_slice,
    build_refit_schedule,
    run_walk_forward_core,
)
from xgbnew.features import DAILY_FEATURE_COLS, build_features_for_symbol


# ─── build_refit_schedule ────────────────────────────────────────────────────

def test_schedule_single_fold_when_stride_exceeds_span():
    s = build_refit_schedule(date(2024, 1, 1), date(2024, 1, 10), refit_stride_days=100)
    assert s == [(date(2024, 1, 1), date(2024, 1, 10))]


def test_schedule_multiple_folds_cover_span_contiguously():
    s = build_refit_schedule(date(2024, 1, 1), date(2024, 1, 31), refit_stride_days=10)
    # Expect 4 folds: 1-10, 11-20, 21-30, 31-31
    assert len(s) == 4
    assert s[0] == (date(2024, 1, 1), date(2024, 1, 10))
    assert s[-1] == (date(2024, 1, 31), date(2024, 1, 31))
    for i in range(len(s) - 1):
        assert s[i + 1][0] == s[i][1] + timedelta(days=1)


def test_schedule_rejects_bad_input():
    with pytest.raises(ValueError):
        build_refit_schedule(date(2024, 2, 1), date(2024, 1, 1), 10)
    with pytest.raises(ValueError):
        build_refit_schedule(date(2024, 1, 1), date(2024, 1, 10), 0)


# ─── _train_slice no-lookahead property ──────────────────────────────────────

def test_train_slice_excludes_fold_start_date():
    df = pd.DataFrame({
        "date": [date(2024, 1, d) for d in range(1, 21)],
        "x": np.arange(20),
    })
    sl = _train_slice(df, fold_start=date(2024, 1, 10), train_window_days=None)
    assert sl["date"].max() == date(2024, 1, 9)
    # 2024-01-10 must NOT be in the training slice.
    assert date(2024, 1, 10) not in set(sl["date"])


def test_train_window_days_bounds_training_slice():
    df = pd.DataFrame({
        "date": [date(2024, 1, d) for d in range(1, 21)],
        "x": np.arange(20),
    })
    sl = _train_slice(df, fold_start=date(2024, 1, 15), train_window_days=5)
    # Expect dates 10..14 (5 rows).
    assert sl["date"].min() == date(2024, 1, 10)
    assert sl["date"].max() == date(2024, 1, 14)
    assert len(sl) == 5


# ─── End-to-end walk-forward on synthetic data ───────────────────────────────

def _build_synthetic_feat_df(n_symbols: int = 6, n_days: int = 900, seed: int = 0) -> pd.DataFrame:
    """Build a mini feature DataFrame across a handful of symbols.

    Prices are a random walk with a slight drift so there's *some* signal.
    """
    rng = np.random.default_rng(seed)
    parts = []
    start = pd.Timestamp("2021-01-04", tz="UTC")
    for s_idx in range(n_symbols):
        prices = 100.0 + np.cumsum(rng.standard_normal(n_days) * 0.8)
        prices = np.clip(prices, 1.0, None)
        ts = pd.date_range(start, periods=n_days, freq="B", tz="UTC")
        df = pd.DataFrame({
            "timestamp": ts,
            "open": prices * (1.0 + rng.normal(0, 0.002, n_days)),
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": rng.uniform(1e6, 1e7, n_days),
        })
        feat = build_features_for_symbol(df, symbol=f"SYM{s_idx}")
        parts.append(feat)
    out = pd.concat(parts, ignore_index=True)
    out = out.dropna(subset=DAILY_FEATURE_COLS[:5])
    return out


def _tiny_xgb_params() -> dict:
    # Small model → fast training for unit tests.
    return dict(n_estimators=10, max_depth=3, learning_rate=0.2, random_state=7)


def test_walk_forward_produces_folds_and_no_lookahead():
    feat_df = _build_synthetic_feat_df(n_symbols=6, n_days=600)
    oos_start = date(2022, 10, 1)
    oos_end = date(2023, 3, 31)
    schedule = build_refit_schedule(oos_start, oos_end, refit_stride_days=30)

    bt_cfg = BacktestConfig(
        top_n=1,
        leverage=1.0,
        xgb_weight=1.0,
        fee_rate=0.0,
        fill_buffer_bps=0.0,
        min_dollar_vol=0.0,
        max_spread_bps=10_000.0,
    )
    res = run_walk_forward_core(
        feat_df,
        schedule,
        xgb_params=_tiny_xgb_params(),
        backtest_cfg=bt_cfg,
        min_train_rows=200,
    )
    assert isinstance(res, WalkForwardResult)
    # Expect at least a few folds actually trained.
    trained_folds = [fr for fr in res.folds if fr.oos_days_traded > 0]
    assert len(trained_folds) >= 3
    # No fold should have a training-date equal to or after its own fold_start.
    for fr in trained_folds:
        assert fr.train_date_max is not None
        assert fr.train_date_max < fr.fold_start
    # Daily return dates should be sorted and cover at least one day per trained fold.
    assert res.daily_returns.index.is_monotonic_increasing
    assert len(res.daily_returns) >= len(trained_folds)


def test_walk_forward_skips_folds_with_insufficient_training_data():
    feat_df = _build_synthetic_feat_df(n_symbols=3, n_days=400)
    # Very early OOS start → first folds will have < min_train_rows.
    oos_start = date(2021, 2, 1)
    oos_end = date(2022, 6, 30)
    schedule = build_refit_schedule(oos_start, oos_end, refit_stride_days=60)

    bt_cfg = BacktestConfig(
        top_n=1, leverage=1.0, xgb_weight=1.0, fee_rate=0.0, fill_buffer_bps=0.0,
        min_dollar_vol=0.0, max_spread_bps=10_000.0,
    )
    res = run_walk_forward_core(
        feat_df,
        schedule,
        xgb_params=_tiny_xgb_params(),
        backtest_cfg=bt_cfg,
        min_train_rows=5_000,  # impossibly high → skips every fold
    )
    trained = [fr for fr in res.folds if fr.oos_days_traded > 0]
    assert trained == []
    assert res.median_fold_monthly_pct == 0.0
    assert res.n_folds == 0


def test_walk_forward_sliding_window_limits_training_set_size():
    feat_df = _build_synthetic_feat_df(n_symbols=6, n_days=900)
    oos_start = date(2023, 6, 1)
    oos_end = date(2023, 9, 30)
    schedule = build_refit_schedule(oos_start, oos_end, refit_stride_days=30)

    bt_cfg = BacktestConfig(
        top_n=1, leverage=1.0, xgb_weight=1.0, fee_rate=0.0, fill_buffer_bps=0.0,
        min_dollar_vol=0.0, max_spread_bps=10_000.0,
    )
    res = run_walk_forward_core(
        feat_df,
        schedule,
        xgb_params=_tiny_xgb_params(),
        backtest_cfg=bt_cfg,
        min_train_rows=100,
        train_window_days=180,  # only ~180 calendar days of training
    )
    trained = [fr for fr in res.folds if fr.oos_days_traded > 0]
    assert trained, "expected at least one trained fold"
    for fr in trained:
        assert fr.train_date_min is not None
        span = (fr.train_date_max - fr.train_date_min).days
        assert span <= 180
