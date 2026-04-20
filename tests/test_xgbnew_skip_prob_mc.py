"""Missed-order Monte Carlo tests for xgbnew/backtest.simulate.

Validates:
  * skip_prob=0.0  → bit-for-bit identity with the pre-MC path.
  * skip_prob=1.0  → zero churn-day trades (only hold-through carries
                     survive, and they only survive once seeded).
  * monotonic-ish: mean survival fraction ≈ (1 - skip_prob) for many
                   independent draws.
  * reproducibility: same (skip_prob, skip_seed) → identical equity
                     curve; different seeds → generally different.
  * hold-through continuations are NOT skipped (no order fires).
"""
from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from xgbnew.backtest import BacktestConfig, simulate
from xgbnew.features import DAILY_FEATURE_COLS
from xgbnew.model import XGBStockModel


def _dummy_model() -> XGBStockModel:
    m = XGBStockModel(device="cpu", n_estimators=1, max_depth=1,
                      learning_rate=0.1)
    m.feature_cols = DAILY_FEATURE_COLS
    m._col_medians = np.zeros(len(DAILY_FEATURE_COLS), dtype=np.float32)
    m._fitted = True
    return m


def _make_df(n_days: int = 20, n_syms: int = 3,
             daily_drift_pct: float = 0.3,
             seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic daily OHLC with positive-drift picks.

    Each day has n_syms candidates; the top-scored symbol rotates day
    over day so picks are never identical two days in a row (prevents
    accidental hold-through).
    Returns (df, scores) — scores aligned to df.index.
    """
    rng = np.random.default_rng(seed)
    syms = [f"SYM{i}" for i in range(n_syms)]
    rows = []
    scores = []
    d0 = date(2025, 1, 2)
    for d_idx in range(n_days):
        day = d0 + timedelta(days=d_idx)
        # Rotate top-score across symbols to avoid hold-through by default.
        top_idx = d_idx % n_syms
        for s_idx, sym in enumerate(syms):
            o = 100.0 + rng.normal(0, 0.5)
            c = o * (1.0 + daily_drift_pct / 100.0 + rng.normal(0, 0.001))
            score = 0.9 if s_idx == top_idx else 0.3 + 0.01 * rng.random()
            row = {
                "date": day, "symbol": sym,
                "actual_open": float(o), "actual_close": float(c),
                "actual_high": float(max(o, c) * 1.002),
                "actual_low": float(min(o, c) * 0.998),
                "spread_bps": 10.0,
                "dolvol_20d_log": np.log1p(1e9),
                "vol_20d": 0.25,
                "target_oc": (c - o) / o,
            }
            for fc in DAILY_FEATURE_COLS:
                row.setdefault(fc, 0.0)
            rows.append(row)
            scores.append(score)
    df = pd.DataFrame(rows)
    scores_s = pd.Series(scores, index=df.index, name="_score")
    return df, scores_s


def _cfg(**overrides) -> BacktestConfig:
    base = BacktestConfig(
        top_n=1, leverage=1.0, xgb_weight=1.0,
        fee_rate=0.0, fill_buffer_bps=0.0, commission_bps=0.0,
        min_dollar_vol=1e6, min_vol_20d=0.0, max_spread_bps=50.0,
        hold_through=False, min_score=0.0,
    )
    return replace(base, **overrides)


def test_skip_prob_zero_is_identity():
    df, scores = _make_df(n_days=15, seed=1)
    model = _dummy_model()
    cfg_base = _cfg()
    cfg_mc0 = _cfg(skip_prob=0.0, skip_seed=123)
    r_base = simulate(df, model, cfg_base, precomputed_scores=scores)
    r_mc0 = simulate(df, model, cfg_mc0, precomputed_scores=scores)
    # Identity — disable should not alter any scalar.
    assert r_base.final_equity == pytest.approx(r_mc0.final_equity, rel=1e-12)
    assert r_base.total_return_pct == pytest.approx(
        r_mc0.total_return_pct, rel=1e-12)
    assert r_base.total_trades == r_mc0.total_trades


def test_skip_prob_one_zeros_all_trades():
    df, scores = _make_df(n_days=15, seed=2)
    model = _dummy_model()
    r = simulate(df, model, _cfg(skip_prob=1.0, skip_seed=0),
                 precomputed_scores=scores)
    # Every pick skipped → zero realised trades across all days.
    assert r.total_trades == 0
    assert r.final_equity == pytest.approx(10_000.0, rel=1e-9)


def test_skip_prob_reproducibility_same_seed():
    df, scores = _make_df(n_days=30, seed=3)
    model = _dummy_model()
    r1 = simulate(df, model, _cfg(skip_prob=0.3, skip_seed=77),
                  precomputed_scores=scores)
    r2 = simulate(df, model, _cfg(skip_prob=0.3, skip_seed=77),
                  precomputed_scores=scores)
    assert r1.final_equity == pytest.approx(r2.final_equity, rel=1e-12)
    assert r1.total_trades == r2.total_trades


def test_skip_prob_different_seeds_differ():
    df, scores = _make_df(n_days=30, seed=4)
    model = _dummy_model()
    r1 = simulate(df, model, _cfg(skip_prob=0.5, skip_seed=1),
                  precomputed_scores=scores)
    r2 = simulate(df, model, _cfg(skip_prob=0.5, skip_seed=2),
                  precomputed_scores=scores)
    # Very unlikely to match exactly across seeds on 30 days × p=0.5.
    assert r1.total_trades != r2.total_trades or \
           r1.final_equity != r2.final_equity


def test_skip_prob_survival_rate_matches_expectation():
    df, scores = _make_df(n_days=200, seed=5)
    model = _dummy_model()
    # Baseline: all picks survive.
    r_base = simulate(df, model, _cfg(), precomputed_scores=scores)
    base_trades = r_base.total_trades

    p = 0.25
    # Average over many MC seeds — survival rate should track (1-p)
    # within a reasonable statistical tolerance.
    observed = []
    for seed in range(40):
        r = simulate(df, model, _cfg(skip_prob=p, skip_seed=seed),
                     precomputed_scores=scores)
        observed.append(r.total_trades / max(base_trades, 1))
    mean_survival = float(np.mean(observed))
    assert abs(mean_survival - (1.0 - p)) < 0.03, (
        f"mean survival {mean_survival:.3f} vs expected {1-p:.3f} "
        f"(tol 0.03)")


def test_skip_prob_clamped_to_valid_range():
    df, scores = _make_df(n_days=10, seed=6)
    model = _dummy_model()
    # -0.5 is clamped to 0.0 — must equal the non-MC run.
    r_low = simulate(df, model, _cfg(skip_prob=-0.5, skip_seed=0),
                     precomputed_scores=scores)
    r_base = simulate(df, model, _cfg(), precomputed_scores=scores)
    assert r_low.total_trades == r_base.total_trades

    # 1.5 is clamped to 1.0 — must zero trades.
    r_hi = simulate(df, model, _cfg(skip_prob=1.5, skip_seed=0),
                    precomputed_scores=scores)
    assert r_hi.total_trades == 0


def _hold_through_df():
    """Build a df where the same top-symbol wins every day, so with
    hold_through=True every day after the first is a continuation."""
    rng = np.random.default_rng(10)
    syms = ["SYM0", "SYM1"]
    rows = []
    scores = []
    d0 = date(2025, 1, 2)
    for d_idx in range(20):
        day = d0 + timedelta(days=d_idx)
        for s_idx, sym in enumerate(syms):
            o = 100.0 + rng.normal(0, 0.5)
            c = o * (1.003 + rng.normal(0, 0.001))
            row = {
                "date": day, "symbol": sym,
                "actual_open": float(o), "actual_close": float(c),
                "actual_high": float(max(o, c) * 1.002),
                "actual_low": float(min(o, c) * 0.998),
                "spread_bps": 10.0, "dolvol_20d_log": np.log1p(1e9),
                "vol_20d": 0.25, "target_oc": (c - o) / o,
            }
            for fc in DAILY_FEATURE_COLS:
                row.setdefault(fc, 0.0)
            rows.append(row)
            # SYM0 always wins → same pick every day.
            scores.append(0.95 if s_idx == 0 else 0.3)
    df = pd.DataFrame(rows)
    return df, pd.Series(scores, index=df.index, name="_score")


def test_hold_through_continuations_not_skipped():
    """Under hold_through, after day 1 every subsequent day is a pure
    continuation with no new order — skip_prob cannot drop them."""
    df, scores = _hold_through_df()
    model = _dummy_model()
    cfg_base = _cfg(hold_through=True)
    cfg_mc   = _cfg(hold_through=True, skip_prob=1.0, skip_seed=0)
    r_base = simulate(df, model, cfg_base, precomputed_scores=scores)
    r_mc   = simulate(df, model, cfg_mc,   precomputed_scores=scores)
    # Day 1 is a churn day — at skip_prob=1.0 the opening trade is
    # killed, so the hold-through chain never starts. That's 0 trades.
    # Baseline (no skip) has 20 trades (1 per day).
    assert r_base.total_trades == 20
    assert r_mc.total_trades == 0


def test_hold_through_day1_survives_skip_resets_state():
    """If day-1 survives skip_prob, subsequent continuations MUST fire
    regardless of later RNG draws (continuations don't consume rng)."""
    df, scores = _hold_through_df()
    model = _dummy_model()
    # Find a seed where day 1 survives skip_prob=0.5.
    for seed in range(50):
        cfg_mc = _cfg(hold_through=True, skip_prob=0.5, skip_seed=seed)
        r = simulate(df, model, cfg_mc, precomputed_scores=scores)
        if r.total_trades >= 1:
            # Day 1 fires as a churn trade; days 2..20 should be
            # continuations (19 additional trades). Total = 20.
            assert r.total_trades == 20, (
                f"seed={seed} day1 survived but continuations didn't "
                f"fire (got {r.total_trades})")
            return
    pytest.fail("no seed survived day-1 skip — expected ~25 of 50")
