from __future__ import annotations

import pandas as pd

from newnanoalpacahourlyexp.marketsimulator.hourly_trader import (
    HourlyTraderMarketSimulator,
    HourlyTraderSimulationConfig,
)


def test_hourly_trader_simulator_decision_lag_one_bar_places_then_fills_next_bar():
    ts0 = pd.Timestamp("2026-01-01T00:00:00Z")
    ts1 = ts0 + pd.Timedelta(hours=1)
    ts2 = ts1 + pd.Timedelta(hours=1)

    bars = pd.DataFrame(
        [
            {"timestamp": ts0, "symbol": "BTCUSD", "high": 10.5, "low": 9.5, "close": 10.0},
            {"timestamp": ts1, "symbol": "BTCUSD", "high": 10.5, "low": 9.5, "close": 10.0},
            {"timestamp": ts2, "symbol": "BTCUSD", "high": 11.0, "low": 9.5, "close": 10.0},
        ]
    )

    actions = pd.DataFrame(
        [
            # Place buy at ts0 -> eligible at ts1, fills at 10.
            {"timestamp": ts0, "symbol": "BTCUSD", "buy_price": 10.0, "sell_price": 11.0, "buy_amount": 100.0, "sell_amount": 0.0},
            # Place sell at ts1 -> eligible at ts2, fills at 11.
            {"timestamp": ts1, "symbol": "BTCUSD", "buy_price": 10.0, "sell_price": 11.0, "buy_amount": 0.0, "sell_amount": 100.0},
            # Last row won't fill within window, but keeps columns present.
            {"timestamp": ts2, "symbol": "BTCUSD", "buy_price": 10.0, "sell_price": 11.0, "buy_amount": 0.0, "sell_amount": 0.0},
        ]
    )

    sim = HourlyTraderMarketSimulator(
        HourlyTraderSimulationConfig(
            initial_cash=1000.0,
            allocation_usd=1000.0,
            allocation_pct=None,
            intensity_scale=1.0,
            price_offset_pct=0.0,
            min_gap_pct=0.001,
            decision_lag_bars=1,
            fee_by_symbol={"BTCUSD": 0.0},
            enforce_market_hours=True,
        )
    )
    result = sim.run(bars, actions)

    assert result.final_positions == {}
    assert result.final_reserved_cash == 0.0
    assert result.final_cash == 1100.0
    assert [f.timestamp for f in result.fills] == [ts1, ts2]
    assert [f.side for f in result.fills] == ["buy", "sell"]
    assert result.metrics["total_return"] == 0.1


def test_hourly_trader_simulator_portfolio_allocation_splits_cash_across_symbols():
    ts0 = pd.Timestamp("2026-01-01T00:00:00Z")
    bars = pd.DataFrame(
        [
            {"timestamp": ts0, "symbol": "BTCUSD", "high": 10.0, "low": 10.0, "close": 10.0},
            {"timestamp": ts0, "symbol": "SOLUSD", "high": 10.0, "low": 10.0, "close": 10.0},
        ]
    )
    actions = pd.DataFrame(
        [
            {"timestamp": ts0, "symbol": "BTCUSD", "buy_price": 10.0, "sell_price": 11.0, "buy_amount": 100.0, "sell_amount": 0.0},
            {"timestamp": ts0, "symbol": "SOLUSD", "buy_price": 10.0, "sell_price": 11.0, "buy_amount": 100.0, "sell_amount": 0.0},
        ]
    )

    sim = HourlyTraderMarketSimulator(
        HourlyTraderSimulationConfig(
            initial_cash=1000.0,
            allocation_usd=None,
            allocation_pct=1.0,
            allocation_mode="portfolio",
            decision_lag_bars=1,
            fee_by_symbol={"BTCUSD": 0.0, "SOLUSD": 0.0},
        )
    )
    result = sim.run(bars, actions)

    # At ts0, both buys are placed and reserve the full $1000 split evenly.
    assert result.per_hour.iloc[0]["reserved_cash"] == 1000.0
    assert result.per_hour.iloc[0]["available_cash"] == 0.0
    assert result.per_hour.iloc[0]["open_orders"] == 2.0


def test_hourly_trader_simulator_default_live_mode_no_same_side_add_and_full_exit():
    ts0 = pd.Timestamp("2026-01-01T00:00:00Z")
    ts1 = ts0 + pd.Timedelta(hours=1)
    ts2 = ts1 + pd.Timedelta(hours=1)

    bars = pd.DataFrame(
        [
            {"timestamp": ts0, "symbol": "ETHUSD", "high": 10.5, "low": 9.5, "close": 10.0},
            {"timestamp": ts1, "symbol": "ETHUSD", "high": 10.5, "low": 9.5, "close": 10.0},
            {"timestamp": ts2, "symbol": "ETHUSD", "high": 11.5, "low": 9.5, "close": 10.0},
        ]
    )

    actions = pd.DataFrame(
        [
            {"timestamp": ts0, "symbol": "ETHUSD", "buy_price": 10.0, "sell_price": 11.0, "buy_amount": 100.0, "sell_amount": 0.0},
            {"timestamp": ts1, "symbol": "ETHUSD", "buy_price": 10.0, "sell_price": 11.0, "buy_amount": 100.0, "sell_amount": 10.0},
            {"timestamp": ts2, "symbol": "ETHUSD", "buy_price": 10.0, "sell_price": 11.0, "buy_amount": 0.0, "sell_amount": 0.0},
        ]
    )

    sim = HourlyTraderMarketSimulator(
        HourlyTraderSimulationConfig(
            initial_cash=1000.0,
            allocation_usd=1000.0,
            allocation_pct=None,
            decision_lag_bars=1,
            fee_by_symbol={"ETHUSD": 0.0},
        )
    )
    result = sim.run(bars, actions)

    # Default config should not pyramid while already long and should quote full exit size.
    assert [f.side for f in result.fills] == ["buy", "sell"]
    assert result.final_positions == {}
    assert result.final_cash == 1100.0


def test_hourly_trader_simulator_fill_buffer_bps_requires_trade_through_limit():
    ts0 = pd.Timestamp("2026-01-01T00:00:00Z")
    ts1 = ts0 + pd.Timedelta(hours=1)
    ts2 = ts1 + pd.Timedelta(hours=1)

    bars = pd.DataFrame(
        [
            {"timestamp": ts0, "symbol": "ETHUSD", "high": 100.1, "low": 99.9, "close": 100.0},
            # Buy at 100.0 with 5 bps buffer requires low <= 99.95 -> fills here (99.94).
            {"timestamp": ts1, "symbol": "ETHUSD", "high": 100.05, "low": 99.94, "close": 100.0},
            # Sell at 100.0 with 5 bps buffer requires high >= 100.05 -> does not fill here (100.04).
            {"timestamp": ts2, "symbol": "ETHUSD", "high": 100.04, "low": 99.9, "close": 100.0},
        ]
    )

    actions = pd.DataFrame(
        [
            {"timestamp": ts0, "symbol": "ETHUSD", "buy_price": 100.0, "sell_price": 100.0, "buy_amount": 100.0, "sell_amount": 0.0},
            {"timestamp": ts1, "symbol": "ETHUSD", "buy_price": 100.0, "sell_price": 100.0, "buy_amount": 0.0, "sell_amount": 100.0},
            {"timestamp": ts2, "symbol": "ETHUSD", "buy_price": 100.0, "sell_price": 100.0, "buy_amount": 0.0, "sell_amount": 0.0},
        ]
    )

    sim = HourlyTraderMarketSimulator(
        HourlyTraderSimulationConfig(
            initial_cash=1000.0,
            allocation_usd=1000.0,
            allocation_pct=None,
            decision_lag_bars=1,
            fill_buffer_bps=5.0,
            fee_by_symbol={"ETHUSD": 0.0},
        )
    )
    result = sim.run(bars, actions)
    assert [f.side for f in result.fills] == ["buy"]
    assert "ETHUSD" in result.final_positions
    assert result.final_positions["ETHUSD"] > 0.0
