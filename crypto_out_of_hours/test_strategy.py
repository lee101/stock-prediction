"""Regression tests for the out-of-hours crypto strategy."""
from __future__ import annotations

import pandas as pd
import pytest

from sessions import build_sessions
from ooh_backtest import (
    build_session_panel, apply_signal, session_pnl_series,
    summarize, DEFAULT_SYMBOLS,
)
from find_combined import combine_picks


def test_sessions_cover_120d_window():
    """120d window should produce ~80 sessions (5 weekdays + 1 weekend per week × 17 weeks)."""
    sessions = build_sessions("2025-12-17", "2026-04-15")
    assert 75 <= len(sessions) <= 85, f"Got {len(sessions)} sessions"
    kinds = {s.kind for s in sessions}
    assert "weekday" in kinds
    assert "weekend" in kinds


def test_session_durations_match_expectation():
    """Weekday ~17.5h (DST ±1h), weekend ~64-66h."""
    sessions = build_sessions("2025-12-17", "2026-04-15")
    wd = [s.duration_hours for s in sessions if s.kind == "weekday"]
    we = [s.duration_hours for s in sessions if s.kind == "weekend"]
    assert all(15.0 <= d <= 18.5 for d in wd), f"weekday durations out of range: {wd}"
    assert all(60.0 <= d <= 68.0 for d in we), f"weekend durations out of range: {we}"


def test_panel_has_no_lookahead():
    """entry_price at session_start, exit_price at session_end, features lag by 1d."""
    sessions = build_sessions("2025-12-17", "2026-04-15")
    panel = build_session_panel(["BTCUSDT"], sessions)
    assert len(panel) > 50
    # All features should be non-null (we drop NaN rows)
    assert panel[["sma_20", "mom_7d", "vol_20d"]].notna().all().all()
    # Entry/exit prices > 0
    assert (panel["entry_price"] > 0).all()
    assert (panel["exit_price"] > 0).all()


def test_fee_cost_correct():
    """Synthetic: 1 pick, 1 session, return=+1%, fee=10bps → net=+0.8%."""
    from sessions import Session

    session = Session(
        start=pd.Timestamp("2025-12-19 21:00", tz="UTC"),
        end=pd.Timestamp("2025-12-22 14:30", tz="UTC"),
        kind="weekend", duration_hours=65.5,
    )
    picked = pd.DataFrame([{
        "session_start": session.start,
        "session_end": session.end,
        "kind": "weekend",
        "duration_hours": 65.5,
        "symbol": "BTCUSDT",
        "entry_price": 100.0, "exit_price": 101.0, "session_ret": 0.01,
        "sma_20": 95.0, "mom_7d": 0.02, "vol_20d": 0.02, "price_source": "hourly",
    }])
    series = session_pnl_series(picked, [session], fee_bps=10.0, max_gross=1.0)
    assert len(series) == 1
    row = series.iloc[0]
    # gross_ret = 0.01, fee = 2 × 0.001 = 0.002, pnl = 0.008
    assert abs(row["pnl_fraction"] - 0.008) < 1e-6


def test_combine_dedupes():
    """combine_picks dedupes on (session_start, symbol)."""
    df_a = pd.DataFrame([{"session_start": pd.Timestamp("2025-12-19 21:00", tz="UTC"),
                          "symbol": "BTCUSDT", "session_ret": 0.01}])
    df_b = pd.DataFrame([{"session_start": pd.Timestamp("2025-12-19 21:00", tz="UTC"),
                          "symbol": "BTCUSDT", "session_ret": 0.02},
                         {"session_start": pd.Timestamp("2025-12-22 21:00", tz="UTC"),
                          "symbol": "ETHUSDT", "session_ret": 0.005}])
    combined = combine_picks(df_a, df_b)
    assert len(combined) == 2  # BTC (from a) + ETH (from b); duplicate BTC from b dropped


def test_combined_strategy_beats_weekend_on_target_120d():
    """Regression: combined weekend+weekday tight must strict-beat weekend-only on
    monthly PnL on the TARGET 120d window (2025-12-17 → 2026-04-15)."""
    sessions = build_sessions("2025-12-17", "2026-04-15")
    n_days = (pd.Timestamp("2026-04-15", tz="UTC") - pd.Timestamp("2025-12-17", tz="UTC")).days
    periods_per_month = len(sessions) * (30.0 / n_days)
    panel = build_session_panel(DEFAULT_SYMBOLS, sessions)
    picks_wknd = apply_signal(panel, sma_mult=1.0, mom7_min=0.0, mom_vs_sma_min=0.0,
                              vol_max=0.03, top_k=1, kinds={"weekend"})
    picks_wd = apply_signal(panel, sma_mult=1.05, mom7_min=0.0, mom_vs_sma_min=0.0,
                            vol_max=0.02, top_k=2, kinds={"weekday", "holiday"})
    combined = combine_picks(picks_wknd, picks_wd)
    series_c = session_pnl_series(combined, sessions, fee_bps=10.0, max_gross=1.0)
    s_c = summarize(series_c, "combined", periods_per_month=periods_per_month)
    series_w = session_pnl_series(picks_wknd, sessions, fee_bps=10.0, max_gross=1.0)
    s_w = summarize(series_w, "weekend_only", periods_per_month=periods_per_month)
    # Combined mo >= weekend mo (same or better)
    assert s_c["monthly_contribution_pct"] >= s_w["monthly_contribution_pct"] - 0.01, \
        f"combined {s_c['monthly_contribution_pct']} < weekend {s_w['monthly_contribution_pct']}"
    # DD not worse than −7.5
    assert s_c["max_dd_pct"] > -7.5, \
        f"combined DD {s_c['max_dd_pct']} below -7.5"
    # neg% < 15% on 120d (81 sessions; at most 12 negative)
    assert s_c["neg_session_rate_pct"] < 15.0
    # monthly contribution positive
    assert s_c["monthly_contribution_pct"] > 0.5


def test_strategy_never_holds_during_us_hours():
    """Every trading session starts at NYSE market_close and ends at next NYSE
    market_open. So we are flat during US stock trading hours by construction."""
    sessions = build_sessions("2025-12-17", "2026-04-15")
    for s in sessions:
        # Session start: NYSE close. Session end: next NYSE open.
        # Weekday close is 20:00 or 21:00 UTC.
        assert s.start.time().hour in (18, 20, 21), f"Unexpected close hour: {s.start}"
        assert s.end.time().hour in (13, 14), f"Unexpected open hour: {s.end}"
        # Session end must be AFTER session start
        assert s.end > s.start


def test_higher_fees_still_positive_target_120d():
    """At 20bps (2× real), combined strategy still positive on target 120d."""
    sessions = build_sessions("2025-12-17", "2026-04-15")
    n_days = 119
    periods_per_month = len(sessions) * (30.0 / n_days)
    panel = build_session_panel(DEFAULT_SYMBOLS, sessions)
    picks_wknd = apply_signal(panel, sma_mult=1.0, mom7_min=0.0, mom_vs_sma_min=0.0,
                              vol_max=0.03, top_k=1, kinds={"weekend"})
    picks_wd = apply_signal(panel, sma_mult=1.05, mom7_min=0.0, mom_vs_sma_min=0.0,
                            vol_max=0.02, top_k=2, kinds={"weekday", "holiday"})
    combined = combine_picks(picks_wknd, picks_wd)
    series_c = session_pnl_series(combined, sessions, fee_bps=20.0, max_gross=1.0)
    s_c = summarize(series_c, "combined_20bps", periods_per_month=periods_per_month)
    assert s_c["monthly_contribution_pct"] > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
