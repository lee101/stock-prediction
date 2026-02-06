from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from newnanoalpacahourlyexp.marketsimulator.simulator import (
    AlpacaMarketSimulator,
    SimulationConfig,
    _infer_periods_per_year,
)


def _bars(symbol: str, timestamps: list[datetime], *, prices: list[float]) -> pd.DataFrame:
    rows = []
    for ts, price in zip(timestamps, prices):
        rows.append(
            {
                "timestamp": ts,
                "symbol": symbol,
                "open": price,
                "high": price * 1.01,
                "low": price * 0.99,
                "close": price,
                "volume": 1000.0,
            }
        )
    return pd.DataFrame(rows)


def _actions(symbol: str, timestamps: list[datetime], *, buy_price: float, sell_price: float) -> pd.DataFrame:
    rows = []
    for ts in timestamps:
        rows.append(
            {
                "timestamp": ts,
                "symbol": symbol,
                "buy_price": buy_price,
                "sell_price": sell_price,
                "buy_amount": 1.0,
                "sell_amount": 1.0,
            }
        )
    return pd.DataFrame(rows)


def test_infer_periods_per_year_stock_estimate() -> None:
    timestamps = [
        datetime(2025, 1, 2, 19, 30, tzinfo=timezone.utc),
        datetime(2025, 1, 2, 20, 30, tzinfo=timezone.utc),
        datetime(2025, 1, 3, 19, 30, tzinfo=timezone.utc),
        datetime(2025, 1, 3, 20, 30, tzinfo=timezone.utc),
    ]
    periods = _infer_periods_per_year(pd.Series(timestamps), "stock")
    assert periods == 252 * 2


def test_market_hours_block_trades_outside_session() -> None:
    symbol = "AAPL"
    ts = [datetime(2025, 1, 2, 22, 0, tzinfo=timezone.utc)]  # 17:00 ET, after close
    bars = _bars(symbol, ts, prices=[100.0])
    actions = _actions(symbol, ts, buy_price=100.0, sell_price=101.0)

    sim = AlpacaMarketSimulator(
        SimulationConfig(
            initial_cash=1000.0,
            enforce_market_hours=True,
            close_at_eod=True,
            fee_by_symbol={symbol: 0.0},
        )
    )
    result = sim.run(bars, actions).per_symbol[symbol]
    assert result.final_inventory == 0.0
    assert len(result.trades) == 0


def test_eod_close_forces_flat_position() -> None:
    symbol = "AAPL"
    ts = [
        datetime(2025, 1, 2, 19, 30, tzinfo=timezone.utc),  # 14:30 ET
        datetime(2025, 1, 2, 20, 30, tzinfo=timezone.utc),  # 15:30 ET (EOD bar)
    ]
    bars = _bars(symbol, ts, prices=[100.0, 101.0])
    actions = _actions(symbol, ts, buy_price=100.0, sell_price=105.0)

    sim = AlpacaMarketSimulator(
        SimulationConfig(
            initial_cash=1000.0,
            enforce_market_hours=True,
            close_at_eod=True,
            fee_by_symbol={symbol: 0.0},
        )
    )
    result = sim.run(bars, actions).per_symbol[symbol]
    assert result.final_inventory == 0.0
    reasons = {trade.reason for trade in result.trades if trade.side == "sell"}
    assert "eod" in reasons


def test_simulator_allows_short_when_enabled_for_short_only_stock() -> None:
    symbol = "EBAY"
    ts = [
        datetime(2025, 1, 2, 19, 30, tzinfo=timezone.utc),
        datetime(2025, 1, 2, 20, 30, tzinfo=timezone.utc),
    ]
    bars = _bars(symbol, ts, prices=[100.0, 95.0])
    # Make the first bar short-entry (sell fill), second bar cover (buy fill).
    bars.loc[0, "high"] = 106.0
    bars.loc[0, "low"] = 99.0
    bars.loc[1, "high"] = 97.0
    bars.loc[1, "low"] = 90.0

    actions = _actions(symbol, ts, buy_price=92.0, sell_price=104.0)

    sim = AlpacaMarketSimulator(
        SimulationConfig(
            initial_cash=1000.0,
            enforce_market_hours=False,
            close_at_eod=False,
            allow_short=True,
            fee_by_symbol={symbol: 0.0},
        )
    )
    result = sim.run(bars, actions).per_symbol[symbol]
    assert result.final_inventory == 0.0
    assert result.final_cash > 1000.0
    assert [t.side for t in result.trades] == ["sell", "buy"]
