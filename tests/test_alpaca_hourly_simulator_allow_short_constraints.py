from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from newnanoalpacahourlyexp.marketsimulator.simulator import AlpacaMarketSimulator, SimulationConfig


def _bars(symbol: str, timestamps: list[datetime], *, ohlc: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    rows = []
    for ts, (o, h, l, c) in zip(timestamps, ohlc):
        rows.append(
            {
                "timestamp": ts,
                "symbol": symbol,
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": 1000.0,
            }
        )
    return pd.DataFrame(rows)


def _actions(
    symbol: str,
    timestamps: list[datetime],
    *,
    buy_price: float,
    sell_price: float,
    buy_amounts: list[float],
    sell_amounts: list[float],
) -> pd.DataFrame:
    if len(timestamps) != len(buy_amounts) or len(timestamps) != len(sell_amounts):
        raise ValueError("timestamps, buy_amounts, sell_amounts must have the same length")
    rows = []
    for ts, b, s in zip(timestamps, buy_amounts, sell_amounts):
        rows.append(
            {
                "timestamp": ts,
                "symbol": symbol,
                "buy_price": float(buy_price),
                "sell_price": float(sell_price),
                "buy_amount": float(b),
                "sell_amount": float(s),
            }
        )
    return pd.DataFrame(rows)


def test_allow_short_flat_enters_only_one_side_when_both_fill() -> None:
    symbol = "JPM"  # not in default long-only/short-only groups => can_long and can_short.
    ts = [
        datetime(2025, 1, 2, 19, 30, tzinfo=timezone.utc),
        datetime(2025, 1, 2, 20, 30, tzinfo=timezone.utc),
    ]
    bars = _bars(symbol, ts, ohlc=[(100.0, 110.0, 90.0, 100.0), (100.0, 110.0, 90.0, 100.0)])
    actions = _actions(
        symbol,
        ts,
        buy_price=95.0,
        sell_price=105.0,
        buy_amounts=[100.0, 0.0],
        sell_amounts=[50.0, 0.0],
    )

    sim = AlpacaMarketSimulator(
        SimulationConfig(
            initial_cash=10_000.0,
            enforce_market_hours=False,
            close_at_eod=False,
            allow_short=True,
            fee_by_symbol={symbol: 0.0},
        )
    )
    result = sim.run(bars, actions).per_symbol[symbol]

    # When flat and both buy/sell could fill, we should enter at most one side (no same-bar flip).
    assert [t.side for t in result.trades] == ["buy"]
    assert result.final_inventory > 0.0


def test_allow_short_long_state_cannot_flip_to_short() -> None:
    symbol = "JPM"
    ts = [
        datetime(2025, 1, 2, 19, 30, tzinfo=timezone.utc),
        datetime(2025, 1, 2, 20, 30, tzinfo=timezone.utc),
    ]
    # Bar 0: buy can fill, sell cannot. Bar 1: sell can fill, buy cannot.
    bars = _bars(symbol, ts, ohlc=[(100.0, 94.0, 90.0, 92.0), (105.0, 110.0, 100.0, 105.0)])
    actions = _actions(
        symbol,
        ts,
        buy_price=95.0,
        sell_price=105.0,
        buy_amounts=[10.0, 0.0],
        sell_amounts=[0.0, 100.0],
    )

    sim = AlpacaMarketSimulator(
        SimulationConfig(
            initial_cash=10_000.0,
            enforce_market_hours=False,
            close_at_eod=False,
            allow_short=True,
            fee_by_symbol={symbol: 0.0},
        )
    )
    result = sim.run(bars, actions).per_symbol[symbol]

    # Sell while long should only exit the long; it must not open a short entry.
    assert [t.side for t in result.trades] == ["buy", "sell"]
    assert abs(result.final_inventory) < 1e-9


def test_allow_short_short_state_cannot_flip_to_long() -> None:
    symbol = "JPM"
    ts = [
        datetime(2025, 1, 2, 19, 30, tzinfo=timezone.utc),
        datetime(2025, 1, 2, 20, 30, tzinfo=timezone.utc),
    ]
    # Bar 0: sell can fill, buy cannot. Bar 1: buy can fill, sell cannot.
    bars = _bars(symbol, ts, ohlc=[(100.0, 110.0, 100.0, 105.0), (95.0, 100.0, 90.0, 92.0)])
    actions = _actions(
        symbol,
        ts,
        buy_price=95.0,
        sell_price=105.0,
        buy_amounts=[0.0, 100.0],
        sell_amounts=[10.0, 0.0],
    )

    sim = AlpacaMarketSimulator(
        SimulationConfig(
            initial_cash=10_000.0,
            enforce_market_hours=False,
            close_at_eod=False,
            allow_short=True,
            fee_by_symbol={symbol: 0.0},
        )
    )
    result = sim.run(bars, actions).per_symbol[symbol]

    # Buy while short should only cover; it must not open a long entry.
    assert [t.side for t in result.trades] == ["sell", "buy"]
    assert abs(result.final_inventory) < 1e-9

