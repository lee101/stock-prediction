from __future__ import annotations

import pandas as pd

from alpacaconstrainedexp.marketsimulator.selector import SelectionConfig, run_best_trade_simulation


def _make_frames(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.date_range("2025-11-01", periods=3, freq="h", tz="UTC")
    bars = pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": symbol,
            "open": [100.0, 100.0, 100.0],
            "high": [103.0, 103.0, 103.0],
            "low": [97.0, 97.0, 97.0],
            "close": [100.0, 100.0, 100.0],
            "predicted_high_p50_h1": [105.0, 105.0, 105.0],
            "predicted_low_p50_h1": [95.0, 95.0, 95.0],
            "predicted_close_p50_h1": [98.0, 98.0, 98.0],
        }
    )
    actions = pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": symbol,
            "buy_price": [98.0, 98.0, 98.0],
            "sell_price": [102.0, 102.0, 102.0],
            "buy_amount": [1.0, 1.0, 1.0],
            "sell_amount": [1.0, 1.0, 1.0],
            "trade_amount": [1.0, 1.0, 1.0],
        }
    )
    return bars, actions


def test_short_only_symbol_opens_and_closes():
    bars, actions = _make_frames("YELP")
    cfg = SelectionConfig(
        initial_cash=10_000.0,
        min_edge=0.0,
        risk_weight=0.0,
        edge_mode="high_low",
        enforce_market_hours=False,
        close_at_eod=False,
        long_symbols=["NVDA"],
        short_symbols=["YELP"],
        fee_by_symbol={"YELP": 0.0},
    )
    result = run_best_trade_simulation(bars, actions, cfg, horizon=1)
    sides = [t.side for t in result.trades]
    assert "sell_short" in sides
    assert "buy_to_cover" in sides


def test_long_constraint_blocks_non_longable():
    bars_nvda, actions_nvda = _make_frames("NVDA")
    bars_yelp, actions_yelp = _make_frames("YELP")
    actions_yelp["sell_amount"] = 0.0
    bars = pd.concat([bars_nvda, bars_yelp], ignore_index=True)
    actions = pd.concat([actions_nvda, actions_yelp], ignore_index=True)
    cfg = SelectionConfig(
        initial_cash=10_000.0,
        min_edge=0.0,
        risk_weight=0.0,
        edge_mode="high_low",
        enforce_market_hours=False,
        close_at_eod=False,
        long_symbols=["NVDA"],
        short_symbols=["YELP"],
        fee_by_symbol={"NVDA": 0.0, "YELP": 0.0},
    )
    result = run_best_trade_simulation(bars, actions, cfg, horizon=1)
    traded_symbols = {t.symbol for t in result.trades}
    assert "NVDA" in traded_symbols
    # Longs should not open for YELP.
    assert all(t.symbol == "NVDA" for t in result.trades)
