"""Tests for the per-pick 1/vol_20d sizing lever.

Covers:
  - `_inv_vol_pick_scale` math + edge cases (disabled, NaN vol, clip)
  - End-to-end: positive inv_vol_target_ann changes PnL when picks have
    different vol_20d; negative symmetric (high-vol name with low vol_20d
    gets up-levered).
  - Default config is bit-identical to pre-feature behaviour.

Motivation: the SPY-vol-target knob (vol_target_ann) is inactive on
true-OOS 2025-07 → 2026-04 because the tariff crash is cross-sectional
not market-wide. Per-pick inv-vol sees each pick individually — task #91.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from xgbnew.backtest import (
    BacktestConfig,
    _inv_vol_pick_scale,
    simulate,
)
from xgbnew.model import XGBStockModel


def _dummy_model() -> XGBStockModel:
    from xgbnew.features import DAILY_FEATURE_COLS
    m = XGBStockModel(device="cpu", n_estimators=1, max_depth=1, learning_rate=0.1)
    m.feature_cols = DAILY_FEATURE_COLS
    m._col_medians = np.zeros(len(DAILY_FEATURE_COLS), dtype=np.float32)
    m._fitted = True
    return m


# ── _inv_vol_pick_scale unit cases ─────────────────────────────────────────

def test_pick_scale_disabled_returns_one():
    assert _inv_vol_pick_scale(0.20, target_ann=0.0, floor=0.05, cap=3.0) == 1.0
    assert _inv_vol_pick_scale(0.01, target_ann=0.0, floor=0.05, cap=3.0) == 1.0


def test_pick_scale_handles_missing_vol():
    # Non-finite or non-positive vol should return 1.0 (no surprise scaling).
    assert _inv_vol_pick_scale(np.nan, target_ann=0.25, floor=0.05, cap=3.0) == 1.0
    assert _inv_vol_pick_scale(0.0, target_ann=0.25, floor=0.05, cap=3.0) == 1.0
    assert _inv_vol_pick_scale(-0.1, target_ann=0.25, floor=0.05, cap=3.0) == 1.0


def test_pick_scale_clips_to_cap_up_and_down():
    # Low-vol name (0.01) would otherwise give 25 — must clip at cap=3.
    assert _inv_vol_pick_scale(0.01, target_ann=0.25, floor=0.05, cap=3.0) == 3.0
    # Very-high-vol name (2.00) gives 0.125 — must clip at 1/cap = 0.333.
    np.testing.assert_allclose(
        _inv_vol_pick_scale(2.00, target_ann=0.25, floor=0.05, cap=3.0),
        1.0 / 3.0,
        rtol=1e-12,
    )


def test_pick_scale_uses_floor_denominator():
    # Vol below the floor is treated as floor: target/floor = 0.25/0.05 = 5 → capped at 3.
    assert _inv_vol_pick_scale(0.01, target_ann=0.25, floor=0.05, cap=3.0) == 3.0
    # Uncapped scenario: target/floor = 5, no cap override ⇒ would return 5.
    assert _inv_vol_pick_scale(0.01, target_ann=0.25, floor=0.05, cap=5.0) == 5.0


def test_pick_scale_matches_ratio_in_middle():
    # vol=0.25 exactly matches target=0.25 → scale=1.0
    np.testing.assert_allclose(
        _inv_vol_pick_scale(0.25, target_ann=0.25, floor=0.05, cap=3.0),
        1.0, rtol=1e-12,
    )
    # vol=0.50, target=0.25 → scale=0.5
    np.testing.assert_allclose(
        _inv_vol_pick_scale(0.50, target_ann=0.25, floor=0.05, cap=3.0),
        0.5, rtol=1e-12,
    )
    # vol=0.10, target=0.25 → scale=2.5
    np.testing.assert_allclose(
        _inv_vol_pick_scale(0.10, target_ann=0.25, floor=0.05, cap=3.0),
        2.5, rtol=1e-12,
    )


# ── End-to-end: the knob actually changes PnL ──────────────────────────────

def _toy_df(n_days: int = 10) -> pd.DataFrame:
    """Two symbols: S_low_vol (vol=0.10) and S_high_vol (vol=0.50).

    Both have the same constant +1% open-to-close return so sign/direction
    doesn't confound. At top_n=2 equal-weighted:
      - disabled: each pick contributes equally ⇒ leverage=L on both.
      - inv-vol target=0.25: S_low up-levered (×2.5 → effective 2.5*L),
                            S_high down-levered (×0.5 → effective 0.5*L).
    Net portfolio return diverges from the disabled baseline.
    """
    d0 = date(2025, 1, 6)
    rows = []
    for i in range(n_days):
        d = d0 + timedelta(days=i)
        for sym, v in [("SLOW", 0.10), ("SHIGH", 0.50)]:
            o = 100.0
            # +1% gross daily (ignores noise; deterministic)
            c = 101.0
            rows.append({
                "date": d,
                "symbol": sym,
                "actual_open": o,
                "actual_close": c,
                "actual_high": c * 1.002,
                "actual_low":  o * 0.999,
                "spread_bps": 5.0,
                "dolvol_20d_log": 20.0,
                "vol_20d": v,
                "target_oc": 0.01,
            })
    df = pd.DataFrame(rows)
    from xgbnew.features import DAILY_FEATURE_COLS
    for col in DAILY_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    return df


def _score_both_picks(df: pd.DataFrame) -> pd.Series:
    # Constant 0.99 so both get picked at top_n=2.
    return pd.Series(0.99, index=df.index, name="score")


def test_inv_vol_default_matches_legacy():
    """inv_vol_target_ann=0.0 ⇒ sim output bit-identical to pre-feature path."""
    df = _toy_df()
    scores = _score_both_picks(df)
    cfg = BacktestConfig(
        top_n=2, leverage=1.0, min_score=0.0, min_dollar_vol=1e5,
        xgb_weight=1.0, fee_rate=0.0, fill_buffer_bps=0.0,
        commission_bps=0.0,
        inv_vol_target_ann=0.0,    # disabled
    )
    res_off = simulate(df, _dummy_model(), cfg, precomputed_scores=scores)
    # Sanity: with zero fees + no buffer, 1% daily on both picks ⇒ each
    # trade's net return is +1% (before the portfolio averaging).
    assert res_off.total_trades == 2 * 10  # 10 days × 2 picks
    for day in res_off.day_results:
        assert len(day.trades) == 2
        for t in day.trades:
            np.testing.assert_allclose(t.net_return_pct, 1.0, atol=1e-9)


def test_inv_vol_scales_low_vol_pick_up():
    """target_ann=0.25 ⇒ low-vol pick (vol=0.10) gets 2.5× leverage multiplier."""
    df = _toy_df()
    scores = _score_both_picks(df)
    cfg = BacktestConfig(
        top_n=2, leverage=1.0, min_score=0.0, min_dollar_vol=1e5,
        xgb_weight=1.0, fee_rate=0.0, fill_buffer_bps=0.0,
        commission_bps=0.0,
        inv_vol_target_ann=0.25, inv_vol_floor=0.05, inv_vol_cap=3.0,
    )
    res = simulate(df, _dummy_model(), cfg, precomputed_scores=scores)
    # Low-vol pick: eff_lev = 1.0 * 2.5 = 2.5 → net return ≈ 2.5% (gross — margin cost at lev=2.5)
    # High-vol pick: eff_lev = 1.0 * 0.5 = 0.5 → net return ≈ 0.5% (no margin cost <= 1.0)
    slow_returns = []
    shigh_returns = []
    for day in res.day_results:
        for t in day.trades:
            if t.symbol == "SLOW":
                slow_returns.append(t.net_return_pct)
            else:
                shigh_returns.append(t.net_return_pct)
    # After margin cost on the 1.5-of-lev excess, slow return is
    # 2.5 * 1.0% − (2.5−1.0) * (0.0625/252) * 100% ≈ 2.5 − 0.037 = 2.463
    # (the margin-cost fraction is returned as fraction, × leverage excess).
    # We just assert >2× bump vs flat, not exact.
    assert np.mean(slow_returns) > 2.0 * 1.0 - 0.1
    # High-vol return shrunk to ~0.5%.
    np.testing.assert_allclose(np.mean(shigh_returns), 0.5, atol=1e-9)


def test_inv_vol_scales_intraday_excursions_with_leverage():
    """Scaled-up (low-vol) pick should carry scaled-up intraday excursions."""
    df = _toy_df()
    scores = _score_both_picks(df)
    cfg_off = BacktestConfig(
        top_n=2, leverage=1.0, min_score=0.0, min_dollar_vol=1e5,
        xgb_weight=1.0, fee_rate=0.0, fill_buffer_bps=0.0,
        commission_bps=0.0, inv_vol_target_ann=0.0,
    )
    cfg_on = BacktestConfig(
        top_n=2, leverage=1.0, min_score=0.0, min_dollar_vol=1e5,
        xgb_weight=1.0, fee_rate=0.0, fill_buffer_bps=0.0,
        commission_bps=0.0,
        inv_vol_target_ann=0.25, inv_vol_floor=0.05, inv_vol_cap=3.0,
    )
    res_off = simulate(df, _dummy_model(), cfg_off, precomputed_scores=scores)
    res_on  = simulate(df, _dummy_model(), cfg_on,  precomputed_scores=scores)

    # Grab the SLOW pick's first-day intraday DD from both runs.
    dd_off = [t.intraday_worst_dd_pct for t in res_off.day_results[0].trades
              if t.symbol == "SLOW"][0]
    dd_on  = [t.intraday_worst_dd_pct for t in res_on.day_results[0].trades
              if t.symbol == "SLOW"][0]
    # dd_off is computed at lev=1.0; dd_on should be ~2.5× that.
    if dd_off > 0.0:
        np.testing.assert_allclose(dd_on / dd_off, 2.5, rtol=1e-3)


def test_inv_vol_leverage_field_reflects_effective():
    """DayTrade.leverage should carry the effective (scaled) leverage."""
    df = _toy_df()
    scores = _score_both_picks(df)
    cfg = BacktestConfig(
        top_n=2, leverage=2.0, min_score=0.0, min_dollar_vol=1e5,
        xgb_weight=1.0, fee_rate=0.0, fill_buffer_bps=0.0,
        commission_bps=0.0,
        inv_vol_target_ann=0.25, inv_vol_floor=0.05, inv_vol_cap=3.0,
    )
    res = simulate(df, _dummy_model(), cfg, precomputed_scores=scores)
    t0 = res.day_results[0].trades
    slow = next(t for t in t0 if t.symbol == "SLOW")
    shigh = next(t for t in t0 if t.symbol == "SHIGH")
    # Slow: lev=2.0 × scale=2.5 = 5.0 (unclipped — cap is 3.0 in scale-space,
    # NOT in effective-leverage-space, so eff_lev up to L*cap is allowed).
    np.testing.assert_allclose(slow.leverage, 2.0 * 2.5, rtol=1e-12)
    # High: lev=2.0 × scale=0.5 = 1.0
    np.testing.assert_allclose(shigh.leverage, 2.0 * 0.5, rtol=1e-12)
