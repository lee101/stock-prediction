"""Tests for BacktestConfig.regime_cs_iqr_max / regime_cs_skew_min.

Gates entire DAYS based on cross-sectional regime signals computed
from the universe's ret_5d distribution that day. Leak-free since
ret_5d is lag-1 (close[t-1]/close[t-6]).
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from xgbnew.backtest import BacktestConfig, simulate
from xgbnew.features import DAILY_FEATURE_COLS
from xgbnew.model import XGBStockModel


def _dummy_model() -> XGBStockModel:
    m = XGBStockModel(device="cpu", n_estimators=1, max_depth=1, learning_rate=0.1)
    m.feature_cols = DAILY_FEATURE_COLS
    m._col_medians = np.zeros(len(DAILY_FEATURE_COLS), dtype=np.float32)
    m._fitted = True
    return m


def _build_regime_panel(n_days: int = 10) -> pd.DataFrame:
    """10 trading days × 4 symbols.

    Days alternate between:
      TIGHT-IQR days  (0, 2, 4, ...):  ret_5d ∈ [-0.02, -0.01, +0.01, +0.02]
        → IQR ≈ 0.02, skew ~ 0.0
      WIDE-IQR days   (1, 3, 5, ...):  ret_5d ∈ [-0.10, -0.05, +0.05, +0.10]
        → IQR ≈ 0.15, skew ~ 0.0

    target_oc is POSITIVE on TIGHT days (+1%) and NEGATIVE on WIDE (-1%),
    so gating wide days out should flip total PnL sign.
    """
    d0 = date(2025, 1, 6)
    rows = []
    for i in range(n_days):
        d = d0 + timedelta(days=i)
        wide = (i % 2) == 1
        if wide:
            ret5s = [-0.10, -0.05, 0.05, 0.10]
            target = -0.01
        else:
            ret5s = [-0.02, -0.01, 0.01, 0.02]
            target = +0.01
        for sym, r5 in zip(["A", "B", "C", "D"], ret5s):
            row = {
                "date": d,
                "symbol": sym,
                "actual_open": 100.0,
                "actual_close": 101.0 if target > 0 else 99.0,
                "actual_high": 102.0,
                "actual_low":   98.0,
                "spread_bps": 5.0,
                "dolvol_20d_log": 20.0,
                "vol_20d": 0.20,
                "ret_5d":  r5,
                "ret_20d": r5 * 2,
                "target_oc": target,
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    for col in DAILY_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    return df


def _scores_all_equal(df: pd.DataFrame) -> pd.Series:
    return pd.Series(0.99, index=df.index, name="score")


def test_regime_iqr_disabled_defaults_keeps_all_days():
    df = _build_regime_panel()
    cfg = BacktestConfig(
        top_n=1, leverage=1.0, min_score=0.0, min_dollar_vol=1e5,
        xgb_weight=1.0, fee_rate=0.0, fill_buffer_bps=0.0, commission_bps=0.0,
    )
    res = simulate(df, _dummy_model(), cfg, precomputed_scores=_scores_all_equal(df))
    # All 10 days simulated (no gate).
    assert len(res.day_results) == 10


def test_regime_iqr_max_gates_wide_days():
    """With threshold ≈ tight-day IQR (~0.03), only TIGHT days survive."""
    df = _build_regime_panel()
    cfg = BacktestConfig(
        top_n=1, leverage=1.0, min_score=0.0, min_dollar_vol=1e5,
        xgb_weight=1.0, fee_rate=0.0, fill_buffer_bps=0.0, commission_bps=0.0,
        regime_cs_iqr_max=0.05,  # tight ~0.02, wide ~0.15 → 0.05 keeps tight only
    )
    res = simulate(df, _dummy_model(), cfg, precomputed_scores=_scores_all_equal(df))
    # 5 tight days (i % 2 == 0) should survive.
    assert len(res.day_results) == 5
    # All kept days are winners (+1% target) → total_return strictly positive.
    assert res.total_return_pct > 0.0


def test_regime_iqr_max_gate_flips_pnl_sign():
    """Without gate: 5 winning + 5 losing days → PnL ≈ 0 or negative.
    With gate: only winning days → PnL strongly positive.
    """
    df = _build_regime_panel()
    cfg_no_gate = BacktestConfig(
        top_n=1, leverage=1.0, min_score=0.0, min_dollar_vol=1e5,
        xgb_weight=1.0, fee_rate=0.0, fill_buffer_bps=0.0, commission_bps=0.0,
    )
    cfg_gate = BacktestConfig(
        top_n=1, leverage=1.0, min_score=0.0, min_dollar_vol=1e5,
        xgb_weight=1.0, fee_rate=0.0, fill_buffer_bps=0.0, commission_bps=0.0,
        regime_cs_iqr_max=0.05,
    )
    scores = _scores_all_equal(df)
    r_none = simulate(df, _dummy_model(), cfg_no_gate, precomputed_scores=scores)
    r_gate = simulate(df, _dummy_model(), cfg_gate, precomputed_scores=scores)
    # Gated version must strictly outperform (winning-only days).
    assert r_gate.total_return_pct > r_none.total_return_pct


def test_regime_iqr_threshold_very_loose_keeps_all():
    df = _build_regime_panel()
    cfg = BacktestConfig(
        top_n=1, leverage=1.0, min_score=0.0, min_dollar_vol=1e5,
        xgb_weight=1.0, fee_rate=0.0, fill_buffer_bps=0.0, commission_bps=0.0,
        regime_cs_iqr_max=10.0,  # far above max IQR (~0.15) → keeps all
    )
    res = simulate(df, _dummy_model(), cfg, precomputed_scores=_scores_all_equal(df))
    assert len(res.day_results) == 10


def test_regime_iqr_threshold_very_tight_keeps_none():
    df = _build_regime_panel()
    cfg = BacktestConfig(
        top_n=1, leverage=1.0, min_score=0.0, min_dollar_vol=1e5,
        xgb_weight=1.0, fee_rate=0.0, fill_buffer_bps=0.0, commission_bps=0.0,
        regime_cs_iqr_max=0.001,  # below all IQRs → no days kept
    )
    res = simulate(df, _dummy_model(), cfg, precomputed_scores=_scores_all_equal(df))
    assert len(res.day_results) == 0
    assert res.total_return_pct == 0.0


def _build_skew_panel(n_days: int = 6) -> pd.DataFrame:
    """Alternate days with positive vs negative skew in ret_5d.

    Pos-skew day: values [-0.01, -0.01, -0.01, +0.20] → right-tail heavy, skew > 0
    Neg-skew day: values [+0.01, +0.01, +0.01, -0.20] → left-tail heavy,  skew < 0

    Target_oc +1% on pos-skew days, -1% on neg-skew days.
    """
    d0 = date(2025, 1, 6)
    rows = []
    for i in range(n_days):
        d = d0 + timedelta(days=i)
        if (i % 2) == 0:   # positive skew
            ret5s = [-0.01, -0.01, -0.01, +0.20]
            target = +0.01
        else:              # negative skew
            ret5s = [+0.01, +0.01, +0.01, -0.20]
            target = -0.01
        for sym, r5 in zip(["A", "B", "C", "D"], ret5s):
            row = {
                "date": d, "symbol": sym,
                "actual_open": 100.0,
                "actual_close": 101.0 if target > 0 else 99.0,
                "actual_high": 102.0, "actual_low": 98.0,
                "spread_bps": 5.0, "dolvol_20d_log": 20.0,
                "vol_20d": 0.20,
                "ret_5d": r5, "ret_20d": 0.0,
                "target_oc": target,
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    for col in DAILY_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    return df


def test_regime_skew_min_gates_negative_skew_days():
    df = _build_skew_panel()
    cfg = BacktestConfig(
        top_n=1, leverage=1.0, min_score=0.0, min_dollar_vol=1e5,
        xgb_weight=1.0, fee_rate=0.0, fill_buffer_bps=0.0, commission_bps=0.0,
        regime_cs_skew_min=0.0,  # keep only days with positive skew
    )
    res = simulate(df, _dummy_model(), cfg, precomputed_scores=_scores_all_equal(df))
    # Only 3 pos-skew days survive (all winners).
    assert len(res.day_results) == 3
    assert res.total_return_pct > 0.0


def test_regime_combined_iqr_and_skew():
    """With BOTH gates active: gate_iqr keeps some, gate_skew keeps intersection."""
    df = _build_skew_panel()  # IQR on these days: pos ~0.16, neg ~0.16 (similar)
    cfg = BacktestConfig(
        top_n=1, leverage=1.0, min_score=0.0, min_dollar_vol=1e5,
        xgb_weight=1.0, fee_rate=0.0, fill_buffer_bps=0.0, commission_bps=0.0,
        regime_cs_iqr_max=10.0,  # effectively off
        regime_cs_skew_min=0.0,
    )
    res = simulate(df, _dummy_model(), cfg, precomputed_scores=_scores_all_equal(df))
    assert len(res.day_results) == 3
    assert res.total_return_pct > 0.0
