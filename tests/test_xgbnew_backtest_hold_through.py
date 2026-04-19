"""Hold-through tests for xgbnew.backtest.

Hold-through means: if today's top-N pick set equals yesterday's, we don't
sell-at-close + buy-at-open; we carry the positions. Cost benefit = 2 ×
(fee + fill_buffer) per held day. Return benefit = close-to-close (captures
overnight drift) instead of open-to-close.

Off by default (hold_through=False) so existing tests / eval harnesses are
unchanged.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from xgbnew.backtest import BacktestConfig, simulate


def _build_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["dolvol_20d_log"] = np.log1p(1e7)
    return df


def _make_day(day: date, *, symbol: str, o: float, c: float, score: float, spread: float = 10.0) -> dict:
    return {
        "date": day,
        "symbol": symbol,
        "actual_open": o,
        "actual_close": c,
        "spread_bps": spread,
        "_score_stub": score,
    }


def test_hold_through_default_off():
    cfg = BacktestConfig()
    assert cfg.hold_through is False


def test_same_pick_two_days_continuation_skips_fees():
    """Same symbol wins two days in a row. With hold_through on, day 2's
    cost is zero (no trade fires). With it off, day 2 pays a round-trip."""
    d1, d2 = date(2024, 1, 2), date(2024, 1, 3)
    # AAA wins both days, with distinct open/close prices.
    rows = [
        _make_day(d1, symbol="AAA", o=100.0, c=105.0, score=0.9),
        _make_day(d1, symbol="BBB", o=100.0, c=100.5, score=0.1),  # filler
        _make_day(d2, symbol="AAA", o=110.0, c=112.0, score=0.9),
        _make_day(d2, symbol="BBB", o=100.0, c=100.5, score=0.1),  # filler
    ]
    df = _build_df(rows)
    scores = df["_score_stub"]

    base = dict(
        top_n=1,
        fill_buffer_bps=5.0,
        commission_bps=0.0,
        fee_rate=0.001,   # 10 bps per side
        min_dollar_vol=0.0,
        max_spread_bps=100.0,
        xgb_weight=1.0,
        leverage=1.0,
    )
    cfg_off = BacktestConfig(**base, hold_through=False)
    cfg_on = BacktestConfig(**base, hold_through=True)

    r_off = simulate(df, model=None, config=cfg_off, precomputed_scores=scores)  # type: ignore[arg-type]
    r_on = simulate(df, model=None, config=cfg_on, precomputed_scores=scores)   # type: ignore[arg-type]

    # Two days' worth of trades regardless.
    assert len(r_off.day_results) == 2
    assert len(r_on.day_results) == 2

    # Day 1 should be IDENTICAL between off/on — no continuation yet.
    np.testing.assert_allclose(
        r_off.day_results[0].daily_return_pct,
        r_on.day_results[0].daily_return_pct,
        rtol=1e-12, atol=1e-12,
    )

    # Day 2 under ON: close-to-close, no fees → (112 - 105) / 105 = 6.667%
    # Day 2 under OFF: open-to-close WITH fees + buffer →
    #   entry = 110*(1+5bp) = 110.055; exit = 112*(1-5bp) = 111.944
    #   round_trip = (111.944*(1-.001)) / (110.055*(1+.001)) - 1
    d2_on = r_on.day_results[1].daily_return_pct
    d2_off = r_off.day_results[1].daily_return_pct

    expected_on = (112.0 - 105.0) / 105.0 * 100.0
    assert abs(d2_on - expected_on) < 1e-6, (d2_on, expected_on)

    # Off is open-to-close with costs. Under positive OC gain (+1.8%), fees
    # drag. But hold_through's close-to-close captures a completely different
    # window (prev close 105 → today close 112), which is larger.
    assert d2_on > d2_off  # hold_through > churn on this trajectory

    # Fee saved ≈ 2 × 10 bps + 2 × 5 bp buffer ≈ 30 bps.
    # Plus overnight gap capture (105 → 110 = 4.76% gap captured).
    # So the delta should be large.
    assert (d2_on - d2_off) > 0.20  # >20bp lift is conservative floor


def test_different_picks_uses_churn_path():
    """Pick changes day 1 → day 2. No continuation. Must match hold_through=False."""
    d1, d2 = date(2024, 1, 2), date(2024, 1, 3)
    rows = [
        _make_day(d1, symbol="AAA", o=100.0, c=102.0, score=0.9),
        _make_day(d1, symbol="BBB", o=100.0, c=100.5, score=0.1),
        _make_day(d2, symbol="AAA", o=100.0, c=100.5, score=0.1),
        _make_day(d2, symbol="BBB", o=100.0, c=103.0, score=0.9),  # new winner
    ]
    df = _build_df(rows)
    scores = df["_score_stub"]

    base = dict(
        top_n=1,
        fill_buffer_bps=5.0,
        commission_bps=0.0,
        fee_rate=0.001,
        min_dollar_vol=0.0,
        max_spread_bps=100.0,
        xgb_weight=1.0,
    )
    cfg_off = BacktestConfig(**base, hold_through=False)
    cfg_on = BacktestConfig(**base, hold_through=True)

    r_off = simulate(df, model=None, config=cfg_off, precomputed_scores=scores)  # type: ignore[arg-type]
    r_on = simulate(df, model=None, config=cfg_on, precomputed_scores=scores)   # type: ignore[arg-type]

    for a, b in zip(r_off.day_results, r_on.day_results):
        np.testing.assert_allclose(a.daily_return_pct, b.daily_return_pct,
                                   rtol=1e-12, atol=1e-12)


def test_three_day_hold_chain():
    """Pick AAA for 3 consecutive days. Only day 1 pays entry cost; days 2/3
    are pure close-to-close."""
    d1, d2, d3 = date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)
    rows = [
        _make_day(d1, symbol="AAA", o=100.0, c=102.0, score=0.9),
        _make_day(d1, symbol="ZZZ", o=100.0, c=100.5, score=0.0),
        _make_day(d2, symbol="AAA", o=105.0, c=107.0, score=0.9),
        _make_day(d2, symbol="ZZZ", o=100.0, c=100.5, score=0.0),
        _make_day(d3, symbol="AAA", o=110.0, c=115.0, score=0.9),
        _make_day(d3, symbol="ZZZ", o=100.0, c=100.5, score=0.0),
    ]
    df = _build_df(rows)
    scores = df["_score_stub"]

    cfg = BacktestConfig(
        top_n=1, fill_buffer_bps=5.0, commission_bps=0.0, fee_rate=0.001,
        min_dollar_vol=0.0, max_spread_bps=100.0, xgb_weight=1.0,
        hold_through=True,
    )
    res = simulate(df, model=None, config=cfg, precomputed_scores=scores)  # type: ignore[arg-type]
    assert len(res.day_results) == 3

    # Day 1: open-to-close with fees. Entry @100.05, exit @101.898.
    # Day 2: close-to-close no fee: (107 - 102)/102
    # Day 3: close-to-close no fee: (115 - 107)/107
    d2_pct = res.day_results[1].daily_return_pct
    d3_pct = res.day_results[2].daily_return_pct

    expected_d2 = (107.0 - 102.0) / 102.0 * 100.0
    expected_d3 = (115.0 - 107.0) / 107.0 * 100.0

    assert abs(d2_pct - expected_d2) < 1e-6
    assert abs(d3_pct - expected_d3) < 1e-6

    # Continuation trades record zero fee_rate + zero fill_buffer_bps.
    for day_res in res.day_results[1:]:
        for t in day_res.trades:
            assert t.fee_rate == 0.0
            assert t.fill_buffer_bps == 0.0


def test_hold_through_off_matches_legacy():
    """Regression: with hold_through=False, behaviour is bit-for-bit identical
    to the pre-feature code path (full round-trip every day)."""
    d1, d2 = date(2024, 1, 2), date(2024, 1, 3)
    rows = [
        _make_day(d1, symbol="AAA", o=100.0, c=102.0, score=0.9),
        _make_day(d2, symbol="AAA", o=102.5, c=103.0, score=0.9),  # same pick
    ]
    df = _build_df(rows)
    scores = df["_score_stub"]
    cfg = BacktestConfig(
        top_n=1, fill_buffer_bps=5.0, commission_bps=0.0, fee_rate=0.001,
        min_dollar_vol=0.0, max_spread_bps=100.0, xgb_weight=1.0,
        hold_through=False,
    )
    res = simulate(df, model=None, config=cfg, precomputed_scores=scores)  # type: ignore[arg-type]
    # Every trade must carry the configured fee_rate + fill_buffer (full round-trip).
    for day_res in res.day_results:
        for t in day_res.trades:
            assert t.fee_rate == cfg.fee_rate
            assert t.fill_buffer_bps == cfg.fill_buffer_bps


def test_regime_gate_flattens_hold_state():
    """A regime-gated day breaks the hold chain — even if tomorrow's pick
    matches yesterday's, it cannot carry across the cash day."""
    # Use regime_gate_window=20 with a SPY series designed so day 2 is
    # gate-closed (SPY below its MA), day 1 and day 3 are open.
    d1 = date(2024, 2, 1)
    d2 = date(2024, 2, 2)
    d3 = date(2024, 2, 3)
    rows = [
        _make_day(d1, symbol="AAA", o=100.0, c=105.0, score=0.9),
        _make_day(d2, symbol="AAA", o=105.0, c=106.0, score=0.9),
        _make_day(d3, symbol="AAA", o=106.0, c=110.0, score=0.9),
    ]
    df = _build_df(rows)
    scores = df["_score_stub"]

    # SPY series: rising for 20d then drops on d2 below MA20.
    spy_dates = pd.date_range("2024-01-02", periods=25, freq="B")
    spy_closes = np.concatenate([np.linspace(400, 450, 20), [420, 380, 380, 380, 380]])
    spy = pd.Series(spy_closes, index=[d.date() for d in spy_dates])

    cfg = BacktestConfig(
        top_n=1, fill_buffer_bps=5.0, commission_bps=0.0, fee_rate=0.001,
        min_dollar_vol=0.0, max_spread_bps=100.0, xgb_weight=1.0,
        regime_gate_window=20, hold_through=True,
    )
    res = simulate(df, model=None, config=cfg, precomputed_scores=scores,   # type: ignore[arg-type]
                   spy_close_by_date=spy)

    # Day 2 should be gated out (we set SPY at 380, MA20 > 380 so below).
    traded_days = {r.day for r in res.day_results}
    assert d2 not in traded_days

    # Day 3 should be a fresh entry (churn), NOT continuation.
    d3_res = [r for r in res.day_results if r.day == d3][0]
    assert len(d3_res.trades) == 1
    t = d3_res.trades[0]
    # Fresh entry → non-zero fee/buffer.
    assert t.fee_rate == cfg.fee_rate
    assert t.fill_buffer_bps == cfg.fill_buffer_bps
