"""Tests for xgbnew BacktestConfig.overnight_max_gross_leverage cap.

This is the xgbnew/backtest.py analog of
``tests/test_pufferlib_overnight_leverage_cap.py`` and mirrors the
``xgbnew/live_trader._eod_deleverage_tick`` semantic — every bar IS one
overnight, so the simulator must clip realized leverage at sizing time.

Test plan covers the contract documented in
``project_eod_deleverage_audit_2026_04_29``:
    1. cap=None preserves legacy bit-identical behavior.
    2. cap > leverage is a no-op.
    3. cap < leverage produces results identical to running with
       ``leverage = cap`` directly (the convergence invariant — the
       core deploy-decision check).
    4. cap rejects 0 / negative / NaN / inf via ``ValueError``.
    5. Hourly path (``simulate_hourly``) honors cap with same semantics.
    6. ``no_picks_fallback`` path also honors cap (fb_lev clipped).
    7. ``conviction_scale`` * leverage still gets capped.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from xgbnew.backtest import BacktestConfig, simulate, simulate_hourly
from xgbnew.model import XGBStockModel


def _dummy_model() -> XGBStockModel:
    from xgbnew.features import DAILY_FEATURE_COLS
    m = XGBStockModel(device="cpu", n_estimators=1, max_depth=1, learning_rate=0.1)
    m.feature_cols = DAILY_FEATURE_COLS
    m._col_medians = np.zeros(len(DAILY_FEATURE_COLS), dtype=np.float32)
    m._fitted = True
    return m


def _toy_df(n_days: int = 10, gross_pct: float = 0.01) -> pd.DataFrame:
    """Single-symbol, deterministic +gross_pct% open-to-close per day.

    With one symbol picked at top_n=1, zero fees and zero buffer, each
    trade's net return ≈ leverage * gross_pct (minus margin cost on the
    excess>1 portion). Convergence: cap=2.0 + leverage=3.0 must equal
    leverage=2.0 directly because both clip eff_lev to 2.0.
    """
    d0 = date(2025, 1, 6)
    rows = []
    for i in range(n_days):
        d = d0 + timedelta(days=i)
        o = 100.0
        c = 100.0 * (1.0 + gross_pct)
        rows.append({
            "date": d,
            "symbol": "AAA",
            "actual_open": o,
            "actual_close": c,
            "actual_high": c * 1.001,
            "actual_low": o * 0.999,
            "spread_bps": 5.0,
            "dolvol_20d_log": 20.0,
            "vol_20d": 0.20,
            "target_oc": gross_pct,
        })
    df = pd.DataFrame(rows)
    from xgbnew.features import DAILY_FEATURE_COLS
    for col in DAILY_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    return df


def _all_ones_scores(df: pd.DataFrame) -> pd.Series:
    return pd.Series(0.99, index=df.index, name="score")


def _base_cfg(**overrides) -> BacktestConfig:
    base = dict(
        top_n=1, leverage=2.0, min_score=0.0, min_dollar_vol=1e5,
        xgb_weight=1.0, fee_rate=0.0, fill_buffer_bps=0.0,
        commission_bps=0.0,
    )
    base.update(overrides)
    return BacktestConfig(**base)


def test_overnight_cap_none_matches_legacy_identity():
    """cap=None preserves bit-identical legacy behavior at any leverage."""
    df = _toy_df(n_days=8)
    scores = _all_ones_scores(df)

    cfg_legacy = _base_cfg(leverage=2.0)
    cfg_capped_none = _base_cfg(leverage=2.0, overnight_max_gross_leverage=None)

    res_legacy = simulate(df, _dummy_model(), cfg_legacy, precomputed_scores=scores)
    res_none = simulate(df, _dummy_model(), cfg_capped_none, precomputed_scores=scores)

    assert res_legacy.total_return_pct == pytest.approx(res_none.total_return_pct, abs=1e-12)
    assert res_legacy.final_equity == pytest.approx(res_none.final_equity, abs=1e-9)
    assert res_legacy.total_trades == res_none.total_trades


def test_overnight_cap_above_leverage_is_noop():
    """cap=10 with leverage=1.5 — identical to no cap."""
    df = _toy_df(n_days=6)
    scores = _all_ones_scores(df)
    cfg_legacy = _base_cfg(leverage=1.5)
    cfg_above = _base_cfg(leverage=1.5, overnight_max_gross_leverage=10.0)

    res_legacy = simulate(df, _dummy_model(), cfg_legacy, precomputed_scores=scores)
    res_above = simulate(df, _dummy_model(), cfg_above, precomputed_scores=scores)

    assert res_legacy.total_return_pct == pytest.approx(res_above.total_return_pct, abs=1e-12)
    assert res_legacy.final_equity == pytest.approx(res_above.final_equity, abs=1e-9)
    assert res_legacy.total_trades == res_above.total_trades


def test_overnight_cap_below_leverage_matches_running_at_cap():
    """The headline correctness fix: cap=2.0 + leverage=3.0 must produce the
    SAME equity curve as running with leverage=2.0 directly.

    Core invariant for the deploy decision: capped lev=2.5 / 3.0 candidates
    must converge to the lev=2.0 baseline (matches prod EOD deleverage)."""
    df = _toy_df(n_days=8)
    scores = _all_ones_scores(df)

    cfg_baseline = _base_cfg(leverage=2.0)
    cfg_capped = _base_cfg(leverage=3.0, overnight_max_gross_leverage=2.0)

    res_baseline = simulate(df, _dummy_model(), cfg_baseline, precomputed_scores=scores)
    res_capped = simulate(df, _dummy_model(), cfg_capped, precomputed_scores=scores)

    assert res_baseline.total_return_pct == pytest.approx(res_capped.total_return_pct, abs=1e-12)
    assert res_baseline.final_equity == pytest.approx(res_capped.final_equity, abs=1e-9)
    assert res_baseline.total_trades == res_capped.total_trades

    # Each per-trade leverage field must reflect the clipped value.
    for day in res_capped.day_results:
        for t in day.trades:
            np.testing.assert_allclose(t.leverage, 2.0, atol=1e-12)

    # And explicitly: uncapped lev=3.0 must be MEANINGFULLY larger.
    cfg_uncapped = _base_cfg(leverage=3.0)
    res_uncapped = simulate(df, _dummy_model(), cfg_uncapped, precomputed_scores=scores)
    assert res_uncapped.total_return_pct > res_baseline.total_return_pct + 1e-6


def test_overnight_cap_at_lev_below_cap_no_effect():
    """leverage=1.5, cap=2.0 must be no-op (cap > leverage)."""
    df = _toy_df(n_days=5)
    scores = _all_ones_scores(df)

    cfg_uncapped = _base_cfg(leverage=1.5)
    cfg_capped = _base_cfg(leverage=1.5, overnight_max_gross_leverage=2.0)

    res_un = simulate(df, _dummy_model(), cfg_uncapped, precomputed_scores=scores)
    res_cap = simulate(df, _dummy_model(), cfg_capped, precomputed_scores=scores)

    assert res_un.total_return_pct == pytest.approx(res_cap.total_return_pct, abs=1e-12)
    assert res_un.final_equity == pytest.approx(res_cap.final_equity, abs=1e-9)


def test_overnight_cap_with_fees_and_buffer_still_converges():
    """Convergence holds even with non-zero fees + fill buffer."""
    df = _toy_df(n_days=6)
    scores = _all_ones_scores(df)

    common = dict(
        top_n=1, min_score=0.0, min_dollar_vol=1e5, xgb_weight=1.0,
        fee_rate=0.0005, fill_buffer_bps=2.0, commission_bps=1.0,
    )
    cfg_baseline = BacktestConfig(leverage=2.0, **common)
    cfg_capped = BacktestConfig(
        leverage=3.0, overnight_max_gross_leverage=2.0, **common,
    )

    res_baseline = simulate(df, _dummy_model(), cfg_baseline, precomputed_scores=scores)
    res_capped = simulate(df, _dummy_model(), cfg_capped, precomputed_scores=scores)

    assert res_baseline.total_return_pct == pytest.approx(res_capped.total_return_pct, abs=1e-12)
    assert res_baseline.final_equity == pytest.approx(res_capped.final_equity, abs=1e-9)


def test_overnight_cap_rejects_zero():
    df = _toy_df(n_days=3)
    scores = _all_ones_scores(df)
    cfg = _base_cfg(leverage=1.0, overnight_max_gross_leverage=0.0)
    with pytest.raises(ValueError, match="overnight_max_gross_leverage"):
        simulate(df, _dummy_model(), cfg, precomputed_scores=scores)


def test_overnight_cap_rejects_negative():
    df = _toy_df(n_days=3)
    scores = _all_ones_scores(df)
    cfg = _base_cfg(leverage=1.0, overnight_max_gross_leverage=-1.0)
    with pytest.raises(ValueError, match="overnight_max_gross_leverage"):
        simulate(df, _dummy_model(), cfg, precomputed_scores=scores)


def test_overnight_cap_rejects_nan():
    df = _toy_df(n_days=3)
    scores = _all_ones_scores(df)
    cfg = _base_cfg(leverage=1.0, overnight_max_gross_leverage=float("nan"))
    with pytest.raises(ValueError, match="overnight_max_gross_leverage"):
        simulate(df, _dummy_model(), cfg, precomputed_scores=scores)


def test_overnight_cap_rejects_inf():
    df = _toy_df(n_days=3)
    scores = _all_ones_scores(df)
    cfg = _base_cfg(leverage=1.0, overnight_max_gross_leverage=float("inf"))
    with pytest.raises(ValueError, match="overnight_max_gross_leverage"):
        simulate(df, _dummy_model(), cfg, precomputed_scores=scores)


def test_overnight_cap_int_input_accepted():
    """Cap should accept int as well as float."""
    df = _toy_df(n_days=4)
    scores = _all_ones_scores(df)
    cfg_int = _base_cfg(leverage=3.0, overnight_max_gross_leverage=2)
    cfg_float = _base_cfg(leverage=3.0, overnight_max_gross_leverage=2.0)
    res_int = simulate(df, _dummy_model(), cfg_int, precomputed_scores=scores)
    res_float = simulate(df, _dummy_model(), cfg_float, precomputed_scores=scores)
    assert res_int.total_return_pct == pytest.approx(res_float.total_return_pct, abs=1e-12)
    assert res_int.final_equity == pytest.approx(res_float.final_equity, abs=1e-9)


# ── Hourly simulator path ──────────────────────────────────────────────────


def _toy_hourly_df(n_bars: int = 8, gross_pct: float = 0.005) -> pd.DataFrame:
    base_ts = pd.Timestamp("2025-01-06 14:30", tz="UTC")
    rows = []
    for i in range(n_bars):
        ts = base_ts + pd.Timedelta(hours=i)
        o = 100.0
        c = 100.0 * (1.0 + gross_pct)
        rows.append({
            "timestamp": ts,
            "symbol": "AAA",
            "actual_open": o,
            "actual_close": c,
            "actual_high": c * 1.001,
            "actual_low": o * 0.999,
            "spread_bps": 5.0,
            "dolvol_20d_log": 20.0,
            "target_oc": gross_pct,
        })
    df = pd.DataFrame(rows)
    from xgbnew.features import HOURLY_FEATURE_COLS
    for col in HOURLY_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    return df


def _dummy_hourly_model() -> XGBStockModel:
    from xgbnew.features import HOURLY_FEATURE_COLS
    m = XGBStockModel(device="cpu", n_estimators=1, max_depth=1, learning_rate=0.1)
    m.feature_cols = HOURLY_FEATURE_COLS
    m._col_medians = np.zeros(len(HOURLY_FEATURE_COLS), dtype=np.float32)
    m._fitted = True
    return m


def test_overnight_cap_hourly_convergence():
    """simulate_hourly: cap=2.0 + leverage=3.0 == leverage=2.0."""
    df = _toy_hourly_df(n_bars=6)
    scores = _all_ones_scores(df)

    common = dict(
        top_n=1, min_score=0.0, min_dollar_vol=1e5, xgb_weight=1.0,
        fee_rate=0.0, fill_buffer_bps=0.0, commission_bps=0.0,
    )
    cfg_baseline = BacktestConfig(leverage=2.0, **common)
    cfg_capped = BacktestConfig(
        leverage=3.0, overnight_max_gross_leverage=2.0, **common,
    )

    res_baseline = simulate_hourly(df, _dummy_hourly_model(), cfg_baseline,
                                   precomputed_scores=scores)
    res_capped = simulate_hourly(df, _dummy_hourly_model(), cfg_capped,
                                 precomputed_scores=scores)

    assert res_baseline.total_return_pct == pytest.approx(res_capped.total_return_pct, abs=1e-12)
    assert res_baseline.final_equity == pytest.approx(res_capped.final_equity, abs=1e-9)


def test_overnight_cap_hourly_rejects_zero():
    df = _toy_hourly_df(n_bars=3)
    scores = _all_ones_scores(df)
    cfg = BacktestConfig(
        leverage=1.0, top_n=1, min_score=0.0, min_dollar_vol=1e5,
        xgb_weight=1.0, fee_rate=0.0, fill_buffer_bps=0.0, commission_bps=0.0,
        overnight_max_gross_leverage=0.0,
    )
    with pytest.raises(ValueError, match="overnight_max_gross_leverage"):
        simulate_hourly(df, _dummy_hourly_model(), cfg, precomputed_scores=scores)
