from datetime import datetime

import pandas as pd
import pytest

from marketsimulator.state import (
    PriceSeries,
    SimulatedClock,
    SimulatedPosition,
    SimulationState,
)


def _price_series(symbol: str, prices: list[float]) -> PriceSeries:
    frame = pd.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1, 9, 30, 0) for _ in prices],
            "Close": prices,
        }
    )
    # Start the cursor at the last price so mark-to-market uses the provided value.
    return PriceSeries(symbol=symbol, frame=frame, cursor=len(prices) - 1)


def test_equity_marks_to_market_for_long_position() -> None:
    clock = SimulatedClock(datetime(2024, 1, 1, 9, 30))
    position = SimulatedPosition(
        symbol="AAPL",
        qty=1,
        side="buy",
        avg_entry_price=100.0,
        current_price=110.0,
    )
    series = _price_series("AAPL", [100.0, 110.0])
    state = SimulationState(
        clock=clock,
        prices={"AAPL": series},
        cash=900.0,
        positions={"AAPL": position},
    )

    state._recalculate_equity()

    assert state.equity == pytest.approx(900.0 + 110.0)
    assert state.buying_power == pytest.approx(900.0 * 2)


def test_equity_marks_to_market_for_short_position() -> None:
    clock = SimulatedClock(datetime(2024, 1, 1, 9, 30))
    position = SimulatedPosition(
        symbol="AAPL",
        qty=1,
        side="sell",
        avg_entry_price=100.0,
        current_price=90.0,
    )
    series = _price_series("AAPL", [100.0, 90.0])
    state = SimulationState(
        clock=clock,
        prices={"AAPL": series},
        cash=1100.0,
        positions={"AAPL": position},
    )

    state._recalculate_equity()

    assert state.equity == pytest.approx(1100.0 - 90.0)
    assert state.buying_power == pytest.approx(1100.0 * 2)
