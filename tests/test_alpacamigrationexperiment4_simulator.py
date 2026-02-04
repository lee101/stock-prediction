from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from alpacamigrationexperiment4.marketsimulator.simulator import (
    AlpacaMarketSimulator,
    SimulationConfig,
)


def _bars(symbol: str, ts: datetime, *, open_price: float, high: float, low: float, close: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": ts,
                "symbol": symbol,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1000.0,
            }
        ]
    )


def _actions(symbol: str, ts: datetime, *, buy_price: float, sell_price: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": ts,
                "symbol": symbol,
                "buy_price": buy_price,
                "sell_price": sell_price,
                "buy_amount": 1.0,
                "sell_amount": 1.0,
            }
        ]
    )


def test_intrabar_round_trips_increase_cash() -> None:
    symbol = "BTCUSD"
    ts = datetime(2025, 1, 2, 12, 0, tzinfo=timezone.utc)
    bars = _bars(symbol, ts, open_price=10.0, high=12.0, low=9.0, close=10.5)
    actions = _actions(symbol, ts, buy_price=10.0, sell_price=11.0)

    sim = AlpacaMarketSimulator(
        SimulationConfig(
            initial_cash=100.0,
            fee_by_symbol={symbol: 0.0},
            allow_intrabar_round_trips=True,
            max_round_trips_per_bar=3,
        )
    )
    result = sim.run(bars, actions).per_symbol[symbol]
    assert result.final_inventory == 0.0
    assert result.final_cash == pytest.approx(130.0)
    assert result.per_hour["cycle_count"].iloc[0] == 3.0
    assert result.per_hour["cycle_qty"].iloc[0] == 10.0


def test_intrabar_round_trips_disabled_by_default() -> None:
    symbol = "BTCUSD"
    ts = datetime(2025, 1, 2, 12, 0, tzinfo=timezone.utc)
    bars = _bars(symbol, ts, open_price=10.0, high=12.0, low=9.0, close=10.5)
    actions = _actions(symbol, ts, buy_price=10.0, sell_price=11.0)

    sim = AlpacaMarketSimulator(
        SimulationConfig(
            initial_cash=100.0,
            fee_by_symbol={symbol: 0.0},
        )
    )
    result = sim.run(bars, actions).per_symbol[symbol]
    assert result.per_hour["cycle_count"].iloc[0] == 0.0
