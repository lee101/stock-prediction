import pandas as pd

from hourlycryptomarketsimulator import HourlyCryptoMarketSimulator, SimulationConfig


def test_simulator_generates_trades():
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="H", tz="UTC"),
            "high": [10.5, 10.8, 11.0],
            "low": [9.5, 10.0, 10.2],
            "close": [10.2, 10.6, 10.9],
        }
    )
    actions = pd.DataFrame(
        {
            "timestamp": bars["timestamp"],
            "buy_price": [10.0, 10.4, 10.6],
            "sell_price": [10.6, 10.9, 11.2],
            "trade_amount": [1.0, 0.5, 0.3],
        }
    )
    sim = HourlyCryptoMarketSimulator(SimulationConfig(initial_cash=5_000.0))
    result = sim.run(bars, actions)
    assert result.trades, "Expected at least one trade"
    assert result.metrics["total_return"] > -1.0
