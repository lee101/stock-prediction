"""No-picks fallback tests for xgbnew.backtest.

The fallback fires when the day's pick pool is empty (because no symbol's
score clears ``min_score`` or every candidate was filtered out by the
dolvol/vol/spread/rank filters or missed-order MC). Instead of sitting in
cash, we look up the configured fallback symbol (SPY/QQQ) in the
pre-filter test_df and trade it at a reduced allocation.

Also covers conviction-scaled allocation, which sizes today's exposure by
the top-of-pool score.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from xgbnew.backtest import BacktestConfig, simulate


def _build_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    # huge dolvol so filters never drop rows
    df["dolvol_20d_log"] = np.log1p(1e9)
    return df


def _row(day: date, sym: str, o: float, c: float, score: float, spread: float = 10.0) -> dict:
    return {
        "date": day,
        "symbol": sym,
        "actual_open": o,
        "actual_close": c,
        "spread_bps": spread,
        "_score_stub": score,
    }


BASE = dict(
    top_n=1,
    fill_buffer_bps=5.0,
    commission_bps=0.0,
    fee_rate=0.001,  # 10 bps per side
    min_dollar_vol=0.0,
    max_spread_bps=100.0,
    xgb_weight=1.0,
    leverage=1.0,
)


def test_fallback_defaults_off():
    cfg = BacktestConfig()
    assert cfg.no_picks_fallback_symbol == ""
    assert cfg.no_picks_fallback_alloc_scale == 0.5
    assert cfg.conviction_scaled_alloc is False


def test_no_picks_fallback_fires_when_no_candidate_clears_min_score():
    """Two days: day 1 AAA wins (score=0.9); day 2 nobody clears ms=0.85.
    With fallback=SPY alloc=0.5, day 2 buys SPY half-leverage."""
    d1, d2 = date(2024, 1, 2), date(2024, 1, 3)
    rows = [
        _row(d1, "AAA", o=100.0, c=105.0, score=0.9),
        _row(d1, "SPY", o=400.0, c=401.0, score=0.1),
        _row(d2, "AAA", o=110.0, c=108.0, score=0.2),  # below ms
        _row(d2, "BBB", o=100.0, c=99.0, score=0.5),   # below ms
        _row(d2, "SPY", o=402.0, c=406.0, score=0.1),  # fallback target
    ]
    df = _build_df(rows)
    scores = df["_score_stub"]

    cfg_cash = BacktestConfig(**BASE, min_score=0.85)
    cfg_fb = BacktestConfig(
        **BASE, min_score=0.85,
        no_picks_fallback_symbol="SPY", no_picks_fallback_alloc_scale=0.5,
    )

    r_cash = simulate(df, model=None, config=cfg_cash, precomputed_scores=scores)  # type: ignore[arg-type]
    r_fb = simulate(df, model=None, config=cfg_fb, precomputed_scores=scores)     # type: ignore[arg-type]

    # Cash variant: only day 1 trades.
    assert len(r_cash.day_results) == 1
    assert r_cash.day_results[0].trades[0].symbol == "AAA"

    # Fallback variant: day 1 (AAA) AND day 2 (SPY).
    assert len(r_fb.day_results) == 2
    day1_fb = r_fb.day_results[0]
    day2_fb = r_fb.day_results[1]
    assert day1_fb.trades[0].symbol == "AAA"
    assert day2_fb.trades[0].symbol == "SPY"
    # SPY went 402 -> 406 = +1% at 0.5x leverage ≈ +0.5% gross, less fees.
    assert day2_fb.trades[0].leverage == 0.5
    assert day2_fb.trades[0].gross_return_pct > 0.5  # before cost
    # Net pct should be between 0.2% and 0.5% (0.5x gross minus 2*10bps fee × 0.5L)
    assert 0.2 < day2_fb.daily_return_pct < 0.6


def test_fallback_respects_alloc_scale_zero_to_one():
    """alloc_scale=0 → no fallback trade, acts like cash.
    alloc_scale=1 → full leverage applied to SPY."""
    d1 = date(2024, 1, 2)
    rows = [
        _row(d1, "AAA", o=100.0, c=105.0, score=0.2),  # below ms
        _row(d1, "SPY", o=400.0, c=410.0, score=0.0),
    ]
    df = _build_df(rows)
    scores = df["_score_stub"]

    cfg_none = BacktestConfig(
        **BASE, min_score=0.85,
        no_picks_fallback_symbol="SPY", no_picks_fallback_alloc_scale=0.0,
    )
    cfg_full = BacktestConfig(
        **BASE, min_score=0.85,
        no_picks_fallback_symbol="SPY", no_picks_fallback_alloc_scale=1.0,
    )

    r_none = simulate(df, model=None, config=cfg_none, precomputed_scores=scores)  # type: ignore[arg-type]
    r_full = simulate(df, model=None, config=cfg_full, precomputed_scores=scores)  # type: ignore[arg-type]

    assert len(r_none.day_results) == 0
    assert len(r_full.day_results) == 1
    assert r_full.day_results[0].trades[0].symbol == "SPY"
    assert r_full.day_results[0].trades[0].leverage == 1.0


def test_fallback_handles_missing_fallback_symbol_for_day_gracefully():
    """If the fallback symbol has no row on day 2 (data gap), we stay in cash."""
    d1, d2 = date(2024, 1, 2), date(2024, 1, 3)
    rows = [
        _row(d1, "AAA", o=100.0, c=105.0, score=0.9),
        _row(d1, "SPY", o=400.0, c=401.0, score=0.1),
        _row(d2, "AAA", o=110.0, c=108.0, score=0.2),  # below ms, no SPY row for d2
    ]
    df = _build_df(rows)
    scores = df["_score_stub"]

    cfg = BacktestConfig(
        **BASE, min_score=0.85,
        no_picks_fallback_symbol="SPY", no_picks_fallback_alloc_scale=0.5,
    )
    r = simulate(df, model=None, config=cfg, precomputed_scores=scores)  # type: ignore[arg-type]
    # Only day 1 traded — d2 SPY missing.
    assert len(r.day_results) == 1


def test_fallback_preserves_pick_days_unchanged():
    """On days where a real pick fires, fallback does NOT fire — the PnL
    should exactly match a no-fallback run."""
    d1 = date(2024, 1, 2)
    rows = [
        _row(d1, "AAA", o=100.0, c=105.0, score=0.9),  # clears ms
        _row(d1, "SPY", o=400.0, c=401.0, score=0.1),
    ]
    df = _build_df(rows)
    scores = df["_score_stub"]

    cfg_cash = BacktestConfig(**BASE, min_score=0.85)
    cfg_fb = BacktestConfig(
        **BASE, min_score=0.85,
        no_picks_fallback_symbol="SPY", no_picks_fallback_alloc_scale=0.5,
    )

    r_cash = simulate(df, model=None, config=cfg_cash, precomputed_scores=scores)  # type: ignore[arg-type]
    r_fb = simulate(df, model=None, config=cfg_fb, precomputed_scores=scores)     # type: ignore[arg-type]

    # Identical — fallback was never used.
    assert len(r_cash.day_results) == 1 == len(r_fb.day_results)
    np.testing.assert_allclose(
        r_cash.day_results[0].daily_return_pct,
        r_fb.day_results[0].daily_return_pct,
    )


def test_fallback_survives_when_row_filter_drops_fallback_from_pick_pool():
    """Fallback lookup is on the PRE-FILTER test_df snapshot, so even if
    aggressive max_ret_20d_rank_pct or vol filters would drop SPY from
    the pick pool, the fallback still fires on no-pick days (AAA remains
    scoreable but below ms; SPY removed from pool via spread filter but
    still recoverable via fallback_lookup)."""
    d1 = date(2024, 1, 2)
    rows = [
        _row(d1, "AAA", o=100.0, c=105.0, score=0.2, spread=5.0),  # below ms
        # SPY has a super-wide spread so max_spread_bps filter drops it
        # from the pick pool — but fallback_lookup was taken BEFORE the
        # filter so it can still fire.
        _row(d1, "SPY", o=400.0, c=405.0, score=0.0, spread=500.0),
    ]
    df = _build_df(rows)
    scores = df["_score_stub"]
    cfg = BacktestConfig(
        top_n=1, fill_buffer_bps=5.0, commission_bps=0.0, fee_rate=0.001,
        min_dollar_vol=0.0, max_spread_bps=30.0,  # drops SPY from pool
        xgb_weight=1.0, leverage=1.0, min_score=0.85,
        no_picks_fallback_symbol="SPY", no_picks_fallback_alloc_scale=1.0,
    )
    r = simulate(df, model=None, config=cfg, precomputed_scores=scores)  # type: ignore[arg-type]
    assert len(r.day_results) == 1
    assert r.day_results[0].trades[0].symbol == "SPY"


def test_fallback_fires_when_all_rows_are_filtered_out_for_day():
    """Fallback must still fire if the filtered pick pool has no rows at all.

    This covers the case where every row on the day is removed by live-replicable
    filters before the simulator groups by date.
    """
    d1 = date(2024, 1, 2)
    rows = [
        _row(d1, "AAA", o=100.0, c=105.0, score=0.2, spread=500.0),
        _row(d1, "SPY", o=400.0, c=405.0, score=0.0, spread=500.0),
    ]
    df = _build_df(rows)
    scores = df["_score_stub"]
    base_no_spread = {k: v for k, v in BASE.items() if k != "max_spread_bps"}
    cfg = BacktestConfig(
        **base_no_spread, min_score=0.85, max_spread_bps=30.0,
        no_picks_fallback_symbol="SPY", no_picks_fallback_alloc_scale=0.5,
    )
    r = simulate(df, model=None, config=cfg, precomputed_scores=scores)  # type: ignore[arg-type]
    assert len(r.day_results) == 1
    assert r.day_results[0].n_candidates == 0
    assert r.day_results[0].trades[0].symbol == "SPY"
    assert r.day_results[0].trades[0].leverage == 0.5


def test_fallback_does_not_carry_via_hold_through():
    """A SPY fallback on day 1 must NOT cause day 2 to claim a hold_through
    continuation — day 2 should fire a fresh trade (or fallback)."""
    d1, d2 = date(2024, 1, 2), date(2024, 1, 3)
    rows = [
        _row(d1, "AAA", o=100.0, c=105.0, score=0.2),  # below ms
        _row(d1, "SPY", o=400.0, c=404.0, score=0.0),
        _row(d2, "AAA", o=106.0, c=110.0, score=0.9),  # clears ms
        _row(d2, "SPY", o=404.0, c=408.0, score=0.0),
    ]
    df = _build_df(rows)
    scores = df["_score_stub"]

    cfg = BacktestConfig(
        **BASE, min_score=0.85, hold_through=True,
        no_picks_fallback_symbol="SPY", no_picks_fallback_alloc_scale=0.5,
    )
    r = simulate(df, model=None, config=cfg, precomputed_scores=scores)  # type: ignore[arg-type]
    assert len(r.day_results) == 2
    # Day 2 must be AAA, not a SPY continuation.
    assert r.day_results[0].trades[0].symbol == "SPY"
    assert r.day_results[1].trades[0].symbol == "AAA"
    # Day 2 AAA should pay full round-trip fees (churn day, not continuation).
    assert r.day_results[1].trades[0].fee_rate > 0.0


def test_hold_through_no_pick_keeps_previous_position_marked_to_market():
    """Live hold-through does not liquidate when no new pick clears min_score.

    The simulator must account for the stale hold close-to-close instead of
    skipping the day as cash.
    """
    d1, d2 = date(2024, 1, 2), date(2024, 1, 3)
    rows = [
        _row(d1, "AAA", o=100.0, c=105.0, score=0.90),
        _row(d1, "SPY", o=400.0, c=401.0, score=0.10),
        _row(d2, "AAA", o=105.0, c=110.0, score=0.20),  # below ms, still held live
        _row(d2, "SPY", o=401.0, c=402.0, score=0.10),
    ]
    df = _build_df(rows)
    scores = df["_score_stub"]

    cfg = BacktestConfig(**BASE, min_score=0.85, hold_through=True)
    r = simulate(df, model=None, config=cfg, precomputed_scores=scores)  # type: ignore[arg-type]

    assert len(r.day_results) == 2
    day2_trade = r.day_results[1].trades[0]
    assert day2_trade.symbol == "AAA"
    assert day2_trade.entry_fill_price == pytest.approx(105.0)
    assert day2_trade.exit_fill_price == pytest.approx(110.0)
    assert day2_trade.fee_rate == 0.0
    assert day2_trade.fill_buffer_bps == 0.0
    assert day2_trade.net_return_pct == pytest.approx((110.0 - 105.0) / 105.0 * 100.0)


def test_hold_through_state_uses_executed_trade_set_after_missed_order_filter():
    """Continuation state must reflect filled trades, not pre-skip candidates."""
    d1, d2 = date(2024, 1, 2), date(2024, 1, 3)
    rows = [
        _row(d1, "AAA", o=100.0, c=105.0, score=0.95),
        _row(d1, "BBB", o=100.0, c=102.0, score=0.90),
        _row(d2, "AAA", o=105.0, c=108.0, score=0.10),
        _row(d2, "BBB", o=102.0, c=104.0, score=0.10),
    ]
    df = _build_df(rows)
    scores = df["_score_stub"]

    base_top2 = {k: v for k, v in BASE.items() if k != "top_n"}
    cfg = BacktestConfig(
        **base_top2,
        top_n=2,
        min_score=0.85,
        hold_through=True,
        skip_prob=0.5,
        skip_seed=0,
    )
    r = simulate(df, model=None, config=cfg, precomputed_scores=scores)  # type: ignore[arg-type]

    assert len(r.day_results) == 2
    day1_symbols = {trade.symbol for trade in r.day_results[0].trades}
    day2_symbols = {trade.symbol for trade in r.day_results[1].trades}
    assert day1_symbols == {"AAA"}
    assert day2_symbols == {"AAA"}
    assert r.day_results[1].trades[0].fee_rate == 0.0
    assert r.day_results[1].trades[0].fill_buffer_bps == 0.0


def test_conviction_scaled_alloc_scales_leverage():
    """top_score below alloc_low → 0 alloc (skip day); above alloc_high →
    full; linear between. Zero-alloc days are dropped (no pick trade) so
    the no-picks fallback path owns them."""
    d1, d2, d3 = date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)
    # score 0.55 = lo → scale=0 (DROPPED — no trade)
    # score 0.70 = (.70-.55)/(.85-.55) = 0.5 scale → half lev = 1.0
    # score 0.90 = above hi → full lev = 2.0
    rows = [
        _row(d1, "AAA", o=100.0, c=110.0, score=0.55),
        _row(d2, "AAA", o=100.0, c=110.0, score=0.70),
        _row(d3, "AAA", o=100.0, c=110.0, score=0.90),
    ]
    df = _build_df(rows)
    scores = df["_score_stub"]

    base_no_lev = {k: v for k, v in BASE.items() if k != "leverage"}
    cfg = BacktestConfig(
        **base_no_lev, min_score=0.0, leverage=2.0,
        conviction_scaled_alloc=True,
        conviction_alloc_low=0.55, conviction_alloc_high=0.85,
    )
    r = simulate(df, model=None, config=cfg, precomputed_scores=scores)  # type: ignore[arg-type]
    # Day 1 dropped (zero conviction), days 2 and 3 traded.
    assert len(r.day_results) == 2
    levs = [t.leverage for dr in r.day_results for t in dr.trades]
    np.testing.assert_allclose(levs[0], 1.0, rtol=1e-9)
    assert levs[1] == 2.0


def test_conviction_scale_triggers_fallback_when_low_score():
    """When the top score drops the alloc to 0 under conviction scaling,
    the no-picks fallback should fire at its configured alloc."""
    d1 = date(2024, 1, 2)
    rows = [
        _row(d1, "AAA", o=100.0, c=110.0, score=0.40),  # below conviction lo=0.55
        _row(d1, "SPY", o=400.0, c=408.0, score=0.0),
    ]
    df = _build_df(rows)
    scores = df["_score_stub"]
    base_no_lev = {k: v for k, v in BASE.items() if k != "leverage"}
    cfg = BacktestConfig(
        **base_no_lev, min_score=0.0, leverage=2.0,
        conviction_scaled_alloc=True,
        conviction_alloc_low=0.55, conviction_alloc_high=0.85,
        no_picks_fallback_symbol="SPY", no_picks_fallback_alloc_scale=0.5,
    )
    r = simulate(df, model=None, config=cfg, precomputed_scores=scores)  # type: ignore[arg-type]
    assert len(r.day_results) == 1
    # AAA trade was sized to eff_lev=0 — but we still expect fallback to
    # ALSO fire. Trades list should include both or prefer fallback.
    syms = [t.symbol for t in r.day_results[0].trades]
    assert "SPY" in syms
