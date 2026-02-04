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
