from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.leverage_settings import (
    LeverageSettings,
    get_leverage_settings,
    reset_leverage_settings,
    set_leverage_settings,
)

from marketsimulator.state import (
    PriceSeries,
    SimulatedClock,
    SimulatedPosition,
    SimulationState,
)


@pytest.fixture(autouse=True)
def leverage_settings_override():
    settings = LeverageSettings(annual_cost=0.0675, trading_days_per_year=252, max_gross_leverage=1.5)
    set_leverage_settings(settings)
    yield
    reset_leverage_settings()

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

    expected_equity = 900.0 + 110.0
    expected_gross = 110.0
    expected_buying_power = max(0.0, 1.5 * expected_equity - expected_gross)

    assert state.equity == pytest.approx(expected_equity)
    assert state.buying_power == pytest.approx(expected_buying_power)


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

    expected_equity = 1100.0 - 90.0
    expected_gross = 90.0
    expected_buying_power = max(0.0, 1.5 * expected_equity - expected_gross)

    assert state.equity == pytest.approx(expected_equity)
    assert state.buying_power == pytest.approx(expected_buying_power)


def test_financing_cost_accrues_on_leveraged_position() -> None:
    start_time = datetime(2024, 1, 1, 9, 30)
    clock = SimulatedClock(start_time)
    dates = [start_time, start_time + timedelta(days=1)]
    frame = pd.DataFrame({"timestamp": dates, "Close": [100.0, 102.0]})
    series = PriceSeries(symbol="AAPL", frame=frame, cursor=0)
    state = SimulationState(clock=clock, prices={"AAPL": series}, cash=100_000.0)

    state.ensure_position("AAPL", qty=1200, side="buy", price=100.0)
    gross_before = state.gross_exposure
    equity_before = max(state.equity, 0.0)
    settings = get_leverage_settings()
    daily_rate = settings.annual_cost / settings.trading_days_per_year

    previous_cash = state.cash
    previous_time = state.clock.current
    state.advance_time(1)
    delta_seconds = (state.clock.current - previous_time).total_seconds()

    expected_borrow = max(0.0, gross_before - equity_before)
    expected_cost = expected_borrow * daily_rate * (delta_seconds / 86400.0)

    cost_charged = previous_cash - state.cash
    assert cost_charged == pytest.approx(expected_cost, rel=1e-6, abs=1e-6)
    assert state.financing_cost_paid == pytest.approx(expected_cost, rel=1e-6, abs=1e-6)
