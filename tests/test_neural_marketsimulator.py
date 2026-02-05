import pandas as pd
import pytest

from binanceneural.marketsimulator import BinanceMarketSimulator, SimulationConfig


def _make_bars(symbol: str, closes: list[float]) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=len(closes), freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": symbol,
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
        }
    )
    return frame


def _make_actions(symbol: str, buy_price: float, sell_price: float) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": symbol,
            "buy_price": [buy_price, buy_price],
            "sell_price": [sell_price, sell_price],
            "buy_amount": [100.0, 0.0],
            "sell_amount": [0.0, 100.0],
        }
    )


def test_single_symbol_simulation_positive_return():
    bars = _make_bars("BTCUSD", [10.0, 12.0])
    actions = _make_actions("BTCUSD", buy_price=10.0, sell_price=12.0)
    sim = BinanceMarketSimulator(SimulationConfig(initial_cash=100.0, maker_fee=0.0))
    result = sim.run(bars, actions)

    assert "BTCUSD" in result.per_symbol
    metrics = result.per_symbol["BTCUSD"].metrics
    assert metrics["total_return"] == pytest.approx(0.2)
    assert result.metrics["total_return"] == pytest.approx(0.2)


def test_multi_symbol_combined_equity():
    bars_a = _make_bars("ETHUSD", [10.0, 12.0])
    bars_b = _make_bars("LINKUSD", [10.0, 9.0])
    actions_a = _make_actions("ETHUSD", buy_price=10.0, sell_price=12.0)
    actions_b = _make_actions("LINKUSD", buy_price=10.0, sell_price=9.0)

    bars = pd.concat([bars_a, bars_b], ignore_index=True)
    actions = pd.concat([actions_a, actions_b], ignore_index=True)

    sim = BinanceMarketSimulator(SimulationConfig(initial_cash=100.0, maker_fee=0.0))
    result = sim.run(bars, actions)

    assert set(result.per_symbol.keys()) == {"ETHUSD", "LINKUSD"}
    assert result.metrics["total_return"] == pytest.approx(0.05)
