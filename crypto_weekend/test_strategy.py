"""Sanity tests for crypto_weekend strategy."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from backtest import (
    build_weekend_panel, apply_signal, weekend_pnl_series,
    add_holdout_rows, summarize,
)
from backtest_tight import tight_filter


def test_panel_has_fri_sun_pairs():
    panel = build_weekend_panel(["BTCUSDT"])
    assert len(panel) > 300
    # All fri_date rows should have sun_close
    assert panel["sun_close"].notna().all()
    # weekend_ret should be reasonable
    assert panel["weekend_ret"].abs().max() < 1.0  # no 100%+ moves


def test_fee_cost_applied_correctly():
    # Construct synthetic: 1 pick, 1 weekend, return=+1%, fee=10bps
    df = pd.DataFrame([{
        "symbol": "X", "fri_date": pd.Timestamp("2020-01-03", tz="UTC"),
        "fri_close": 100.0, "sun_close": 101.0,
        "sma_20": 95.0, "mom_7d": 0.02, "vol_20d": 0.02,
        "weekend_ret": 0.01,
    }])
    weekly = weekend_pnl_series(df, fee_bps=10.0, max_gross=1.0)
    assert len(weekly) == 1
    # gross_ret = 0.01 * 1.0 = 0.01
    # fee_cost = 2 * 0.001 * 1.0 = 0.002
    # pnl = 0.008
    assert abs(weekly.iloc[0]["pnl_fraction"] - 0.008) < 1e-6


def test_tight_filter_passes_conservative_bar_oos():
    """Regression: the chosen candidate must clear a conservative OOS bar.

    BTC+ETH+SOL, sma_mult=1.0, mom_vs_sma=5%, vol_max=3%, gross=1.0.
    """
    panel = build_weekend_panel(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    p = tight_filter(panel, sma_mult=1.0, mom7_min=0.0, mom30_min=0.05,
                     vol_max=0.03)
    weekly_raw = weekend_pnl_series(p, fee_bps=10.0, max_gross=1.0)
    all_fridays = sorted(panel["fri_date"].unique())
    weekly = add_holdout_rows(weekly_raw, all_fridays)
    oos = weekly[weekly["fri_date"] > pd.Timestamp("2022-06-30", tz="UTC")]
    s = summarize(oos, "oos")
    # Mean positive
    assert s["mean_weekly_pnl_pct"] > 0.10
    # Worst drawdown bounded
    assert s["max_dd_pct"] > -10.0
    # Negative-weekend rate below 15%
    assert s["neg_weekend_rate_pct"] < 15.0
    # Sortino positive
    assert s["sortino_weekly"] > 0.05


def test_tight_filter_is_subset_of_panel():
    panel = build_weekend_panel(["BTCUSDT", "ETHUSDT"])
    p = tight_filter(panel, sma_mult=1.0, mom7_min=0.0, mom30_min=0.05,
                     vol_max=0.03)
    assert len(p) <= len(panel)
    # All filtered rows have fri_close > sma_20
    assert (p["fri_close"] > p["sma_20"]).all()
    # Vol constraint
    assert (p["vol_20d"] <= 0.03).all()


def test_higher_fees_still_positive():
    """At 20bps fees (2× real), mean weekly PnL still positive."""
    panel = build_weekend_panel(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    p = tight_filter(panel, sma_mult=1.0, mom7_min=0.0, mom30_min=0.05,
                     vol_max=0.03)
    weekly_raw = weekend_pnl_series(p, fee_bps=20.0, max_gross=1.0)
    all_fridays = sorted(panel["fri_date"].unique())
    weekly = add_holdout_rows(weekly_raw, all_fridays)
    oos = weekly[weekly["fri_date"] > pd.Timestamp("2022-06-30", tz="UTC")]
    s = summarize(oos, "oos")
    assert s["mean_weekly_pnl_pct"] > 0.0
