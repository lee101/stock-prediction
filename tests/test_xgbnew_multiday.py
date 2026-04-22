"""Smoke tests for xgbnew_multiday prototype."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xgbnew_multiday.dataset import _fwd_columns_for_symbol, HORIZONS
from xgbnew_multiday.backtest import (
    MultiDayConfig,
    _compute_trade_return,
    _symbol_abs_ret_prior,
    build_daily_candidate_table,
    simulate,
    summarize,
    Trade,
)


def _make_ohlc(n_days: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-02", periods=n_days, freq="B", tz="UTC")
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n_days)))
    df = pd.DataFrame({
        "timestamp": ts,
        "open": prices * (1.0 + rng.normal(0, 0.002, n_days)),
        "high": prices * 1.01,
        "low":  prices * 0.99,
        "close": prices,
        "volume": rng.uniform(1e6, 5e6, n_days),
        "symbol": "TEST",
    })
    df["date"] = df["timestamp"].dt.date
    return df


def test_fwd_columns_valid_mask_trails():
    """Last N-1 rows of a symbol should have valid_fwd_{N}d == 0."""
    df = _make_ohlc(30)
    fwd = _fwd_columns_for_symbol(df, horizons=(1, 3, 10))
    assert fwd["valid_fwd_1d"].iloc[-1] == 1   # can always compute 1-day fwd
    assert fwd["valid_fwd_3d"].iloc[-1] == 0   # need close[t+2] which is missing
    assert fwd["valid_fwd_3d"].iloc[-3] == 1
    assert fwd["valid_fwd_10d"].iloc[-9] == 0
    assert fwd["valid_fwd_10d"].iloc[-10] == 1


def test_fwd_columns_match_close_over_open():
    df = _make_ohlc(20, seed=1)
    fwd = _fwd_columns_for_symbol(df, horizons=(1, 5))
    # Row 0: target_fwd_1d = close[0] / open[0] - 1
    expected = df["close"].iloc[0] / df["open"].iloc[0] - 1.0
    assert np.isclose(fwd["target_fwd_1d"].iloc[0], expected, atol=1e-8)
    # Row 2: target_fwd_5d = close[6] / open[2] - 1
    expected_5 = df["close"].iloc[6] / df["open"].iloc[2] - 1.0
    assert np.isclose(fwd["target_fwd_5d"].iloc[2], expected_5, atol=1e-8)


def test_compute_trade_return_fees_and_buffer():
    entry = pd.Series({"actual_open": 100.0, "actual_close": 101.0, "symbol": "X"})
    exit_ = pd.Series({"actual_open": 102.0, "actual_close": 103.0, "symbol": "X"})
    cfg = MultiDayConfig(
        leverage=1.0, fee_bps_per_side=10.0, fill_buffer_bps=5.0,
    )
    gross, net = _compute_trade_return(entry, exit_, horizon=3, cfg=cfg)
    # gross = close/open - 1 = 103/100 - 1 = 0.03
    assert np.isclose(gross, 0.03, atol=1e-6)
    # Net includes 2x10bps fee + 2x5bps buffer drag + margin cost (lev=1, 0 cost)
    # fill_open = 100 * (1 + 5bps) = 100.05
    # fill_close = 103 * (1 - 5bps) = 102.9485
    # raw_net = 102.9485/100.05 - 1 = 0.02897... - 2*10bps = 0.02697
    expected_raw = 103.0 * (1 - 5e-4) / (100.0 * (1 + 5e-4)) - 1.0
    expected_net = expected_raw - 2 * 10e-4
    assert np.isclose(net, expected_net, atol=1e-6)


def test_compute_trade_return_lev2_adds_margin_cost():
    entry = pd.Series({"actual_open": 100.0, "actual_close": 101.0, "symbol": "X"})
    exit_ = pd.Series({"actual_open": 102.0, "actual_close": 103.0, "symbol": "X"})
    cfg1 = MultiDayConfig(leverage=1.0)
    cfg2 = MultiDayConfig(leverage=2.0)
    _, net1 = _compute_trade_return(entry, exit_, horizon=10, cfg=cfg1)
    _, net2 = _compute_trade_return(entry, exit_, horizon=10, cfg=cfg2)
    # lev=2 should approximately double raw return and then subtract margin cost
    # margin_cost ~ (2-1)*0.0625*9/365 ~ 0.00154
    raw = 103.0 * (1 - 5e-4) / (100.0 * (1 + 5e-4)) - 1.0
    expected_net2 = 2.0 * raw - 2 * 10e-4 - 0.0625 * 9 / 365.0
    assert np.isclose(net2, expected_net2, atol=1e-6)


def test_symbol_abs_ret_prior_nonneg():
    # Fake train_df with 2 symbols, 30 rows each
    rows = []
    for sym, seed in [("A", 0), ("B", 1)]:
        df = _make_ohlc(30, seed=seed)
        fwd = _fwd_columns_for_symbol(df, horizons=(1, 3))
        df = pd.concat([df.reset_index(drop=True), fwd], axis=1)
        df["symbol"] = sym
        rows.append(df)
    train = pd.concat(rows, ignore_index=True)
    prior = _symbol_abs_ret_prior(train, (1, 3))
    assert "abs_prior_1d" in prior.columns
    assert "abs_prior_3d" in prior.columns
    assert (prior["abs_prior_1d"] >= 0).all()
    # 3d should generally have larger mean |return| than 1d
    assert prior["abs_prior_3d"].mean() >= prior["abs_prior_1d"].mean()


def test_build_candidate_table_picks_best_horizon():
    # Build a tiny test_df with known valid masks and known priors
    test_df = pd.DataFrame({
        "symbol": ["A", "A", "B"],
        "date": [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 1)],
        "actual_open": [100.0, 101.0, 50.0],
        "actual_close": [102.0, 100.0, 51.0],
        "valid_fwd_1d": [1, 1, 1],
        "valid_fwd_3d": [1, 0, 1],
    })
    probs = {1: np.array([0.55, 0.70, 0.60]), 3: np.array([0.80, 0.80, 0.51])}
    prior = pd.DataFrame({
        "symbol": ["A", "B"],
        "abs_prior_1d": [0.01, 0.005],
        "abs_prior_3d": [0.05, 0.01],
    }).set_index("symbol")
    cfg = MultiDayConfig(horizons=(1, 3), min_prob=0.5, min_expected_ret=0.0)
    cand = build_daily_candidate_table(test_df, probs, prior, cfg)
    # Row 0 (A, 2025-01-01): 1d exp = (0.55-0.5)*2*0.01=0.001; 3d exp = (0.8-0.5)*2*0.05=0.03
    # -> picks 3d
    assert cand.iloc[0]["horizon_pick"] == 3
    # Row 1 (A, 2025-01-02): 3d invalid -> must pick 1d
    assert cand.iloc[1]["horizon_pick"] == 1
    # Row 2 (B): 1d exp = (0.6-0.5)*2*0.005=0.001; 3d exp = (0.51-0.5)*2*0.01=0.0002
    # -> picks 1d
    assert cand.iloc[2]["horizon_pick"] == 1


def test_simulate_single_slot_non_overlapping():
    # Build a synthetic 5-day test_df + candidate with a single horizon-3 trade
    dates = [date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 6), date(2025, 1, 7), date(2025, 1, 8)]
    test_df = pd.DataFrame({
        "symbol": ["A"] * 5,
        "date": dates,
        "actual_open":  [100, 101, 102, 103, 104],
        "actual_close": [100.5, 101.5, 103, 104, 105],
    }).astype({"actual_open": "float64", "actual_close": "float64"})

    cand = pd.DataFrame({
        "date": dates,
        "symbol": ["A"] * 5,
        "horizon_pick": [3, 3, 3, 3, 3],
        "prob_pick": [0.9, 0.9, 0.9, 0.9, 0.9],
        "expected_ret_pick": [0.05, 0.05, 0.05, 0.05, 0.05],
        "score": [0.05, 0.05, 0.05, 0.05, 0.05],
        "actual_open": test_df["actual_open"].values,
    })
    cfg = MultiDayConfig(
        horizons=(3,), leverage=1.0, fee_bps_per_side=0.0,
        fill_buffer_bps=0.0, decision_lag=1, min_prob=0.5, min_expected_ret=0.0,
    )
    res = simulate(test_df, cand, cfg, initial_cash=10_000.0)
    # With K=1 slots, dl=1: trade 1 opens day-0 (2025-01-02), exits day-2
    # (2025-01-06). On same day 2025-01-06 the slot is freed and a new
    # trade opens at day-2's open (102) with exit at day-4 (2025-01-08).
    # Entry at day-3 (2025-01-07) would exit at day-5 (out of bounds).
    # Day-4 entry also out of bounds. Expect exactly 2 trades.
    assert len(res.trades) == 2
    t1 = res.trades[0]
    assert t1.horizon == 3
    assert t1.entry_date == date(2025, 1, 2)
    assert t1.exit_date == date(2025, 1, 6)
    assert np.isclose(t1.gross_ret, 0.03, atol=1e-6)  # 103/100 - 1
    t2 = res.trades[1]
    assert t2.entry_date == date(2025, 1, 6)
    assert t2.exit_date == date(2025, 1, 8)
    # gross = close[4]/open[2] - 1 = 105/102 - 1 = 0.02941176
    assert np.isclose(t2.gross_ret, 105.0 / 102.0 - 1.0, atol=1e-6)


def test_simulate_horizon_1_same_day_close():
    """Horizon=1 trade should open and close within the same day (baseline)."""
    dates = [date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 6)]
    test_df = pd.DataFrame({
        "symbol": ["A"] * 3,
        "date": dates,
        "actual_open":  [100.0, 101.0, 102.0],
        "actual_close": [101.0, 102.0, 103.0],
    })
    cand = pd.DataFrame({
        "date": dates,
        "symbol": ["A"] * 3,
        "horizon_pick": [1, 1, 1],
        "prob_pick": [0.9, 0.9, 0.9],
        "expected_ret_pick": [0.01, 0.01, 0.01],
        "score": [0.01, 0.01, 0.01],
        "actual_open": test_df["actual_open"].values,
    })
    cfg = MultiDayConfig(
        horizons=(1,), leverage=1.0, fee_bps_per_side=0.0,
        fill_buffer_bps=0.0, decision_lag=1, min_prob=0.5, min_expected_ret=0.0,
        top_n_slots=1,
    )
    res = simulate(test_df, cand, cfg, initial_cash=10_000.0)
    # dl=1: 3 trades, one per day, all horizon=1 (same-day close)
    assert len(res.trades) == 3
    assert all(t.horizon == 1 for t in res.trades)
    assert all(t.entry_date == t.exit_date for t in res.trades)
    # Trade on day 0: close[0]/open[0] - 1 = 101/100 - 1 = 0.01
    assert np.isclose(res.trades[0].gross_ret, 0.01, atol=1e-6)


def test_simulate_3slot_holds_three_concurrent():
    """K=3 slots should allow 3 different symbols to be held concurrently."""
    dates = [date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 6), date(2025, 1, 7), date(2025, 1, 8)]
    rows = []
    for sym, base in [("A", 100.0), ("B", 50.0), ("C", 25.0)]:
        for i, d in enumerate(dates):
            rows.append({
                "symbol": sym, "date": d,
                "actual_open":  base * (1.0 + 0.01 * i),
                "actual_close": base * (1.0 + 0.01 * i + 0.005),
            })
    test_df = pd.DataFrame(rows)
    cand_rows = []
    for sym, exp in [("A", 0.03), ("B", 0.025), ("C", 0.02)]:
        for d in dates:
            cand_rows.append({
                "date": d, "symbol": sym, "horizon_pick": 3,
                "prob_pick": 0.9, "expected_ret_pick": exp, "score": exp,
                "actual_open": 0.0,
            })
    cand = pd.DataFrame(cand_rows)
    cfg = MultiDayConfig(
        horizons=(3,), leverage=1.0, fee_bps_per_side=0.0,
        fill_buffer_bps=0.0, decision_lag=1, min_prob=0.5, min_expected_ret=0.0,
        top_n_slots=3,
    )
    res = simulate(test_df, cand, cfg, initial_cash=10_000.0)
    # On day 0 (2025-01-02), three slots open for A, B, C with horizon=3
    # -> exit day 2 (2025-01-06). On day 2 after closing, 3 slots re-open
    # -> exit day 4 (2025-01-08). Expect 6 trades (3 symbols × 2 batches).
    assert len(res.trades) == 6
    by_entry = {}
    for t in res.trades:
        by_entry.setdefault(t.entry_date, set()).add(t.symbol)
    assert by_entry[date(2025, 1, 2)] == {"A", "B", "C"}
    assert by_entry[date(2025, 1, 6)] == {"A", "B", "C"}


def test_summarize_empty():
    summary = summarize([], pd.Series(dtype=float), 10_000.0)
    assert summary["n_trades"] == 0
    assert summary["median_monthly_pnl_pct"] == 0.0


def test_summarize_known_trades():
    dates = pd.date_range("2025-01-02", periods=50, freq="B")
    equity = pd.Series(
        np.linspace(10_000.0, 11_000.0, 50),
        index=dates.date,
    )
    trades = [
        Trade(dates[0].date(), dates[2].date(), "A", 3, 0.8, 0.02, 0.021, 0.019, 3),
        Trade(dates[3].date(), dates[7].date(), "B", 5, 0.7, 0.01, 0.008, 0.005, 5),
    ]
    summary = summarize(trades, equity, 10_000.0)
    assert summary["n_trades"] == 2
    assert summary["avg_hold_days"] == 4.0
    assert summary["total_return_pct"] > 0.0
    # neg_window_frac should be 0 since equity is monotonically increasing
    assert summary["neg_window_frac"] == 0.0
