from __future__ import annotations

from datetime import datetime, timedelta

from bagsfm.config import SimulationConfig, TokenConfig
from bagsfm.data_collector import OHLCBar
from bagsfm.simulator import MarketSimulator, forecast_threshold_strategy


def _make_bars(count: int, start: datetime, step_minutes: int = 10) -> list[OHLCBar]:
    bars = []
    price = 1e-7
    for i in range(count):
        timestamp = start + timedelta(minutes=step_minutes * i)
        price *= 1.002
        bars.append(
            OHLCBar(
                timestamp=timestamp,
                token_mint="TEST",
                token_symbol="TST",
                open=price,
                high=price * 1.001,
                low=price * 0.999,
                close=price,
                volume=0.0,
                num_ticks=1,
            )
        )
    return bars


def test_run_walk_forward_backtest_basic():
    bars = {"TEST": _make_bars(12, datetime(2026, 1, 1))}
    tokens = {"TEST": TokenConfig(symbol="TST", mint="TEST", decimals=9)}
    sim_config = SimulationConfig(initial_sol=1.0, max_position_pct=0.5)

    simulator = MarketSimulator(sim_config)

    def flip_strategy(state, prices, forecasts):
        actions = {}
        for mint in prices:
            actions[mint] = "buy" if mint not in state.positions else "sell"
        return actions

    result = simulator.run_walk_forward_backtest(
        bars=bars,
        tokens=tokens,
        strategy_fn=flip_strategy,
        forecaster=None,
    )

    assert len(result.equity_curve) == len(bars["TEST"])
    assert result.total_trades > 0


def test_forecast_threshold_strategy_actions():
    strategy = forecast_threshold_strategy(min_return=0.01, max_drawdown_return=-0.01)

    class _DummyForecast:
        def __init__(self, value):
            self.predicted_return = value

    dummy_state = type("State", (), {"positions": {"A": object()}})()
    actions = strategy(
        dummy_state,
        prices={"A": 1.0, "B": 1.0},
        forecasts={"A": _DummyForecast(-0.02), "B": _DummyForecast(0.02)},
    )

    assert actions["A"] == "sell"
    assert actions["B"] == "buy"


def test_simulator_respects_max_position_sol():
    bars = {"TEST": _make_bars(3, datetime(2026, 1, 1))}
    tokens = {"TEST": TokenConfig(symbol="TST", mint="TEST", decimals=9)}
    sim_config = SimulationConfig(initial_sol=2.0, max_position_pct=1.0, max_position_sol=0.1)

    simulator = MarketSimulator(sim_config)
    simulator.reset()

    trade = simulator.open_position(
        token=tokens["TEST"],
        sol_amount=1.0,
        price_sol=bars["TEST"][0].close,
        timestamp=bars["TEST"][0].timestamp,
    )

    assert trade is not None
    assert trade.notional_sol <= sim_config.max_position_sol + 1e-9
