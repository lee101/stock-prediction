from __future__ import annotations

import pandas as pd

import backtest_hybrid_rl_gemini as hybrid
from unified_orchestrator.rl_gemini_bridge import RLSignal


def _make_daily_bars(symbol: str, periods: int = 90) -> pd.DataFrame:
    index = pd.date_range("2025-10-01 00:00:00+00:00", periods=periods, freq="D", tz="UTC")
    closes = [100.0 + i for i in range(periods)]
    return pd.DataFrame(
        {
            "timestamp": index,
            "symbol": symbol,
            "open": [price - 1.0 for price in closes],
            "high": [price + 2.0 for price in closes],
            "low": [price - 2.0 for price in closes],
            "close": closes,
            "volume": [10_000 + i for i in range(periods)],
        }
    )


def test_resolve_backtest_symbols_auto_uses_trained_universe() -> None:
    execution_symbols, load_symbols = hybrid.resolve_backtest_symbols(
        ["auto"],
        ["BTCUSD", "ETHUSD", "SOLUSD"],
    )

    assert execution_symbols == ["BTCUSD", "ETHUSD", "SOLUSD"]
    assert load_symbols == ["BTCUSD", "ETHUSD", "SOLUSD"]


def test_resolve_backtest_symbols_keeps_requested_subset_and_loads_union() -> None:
    execution_symbols, load_symbols = hybrid.resolve_backtest_symbols(
        ["ETHUSD", "BTCUSD"],
        ["BTCUSD", "ETHUSD", "SOLUSD"],
    )

    assert execution_symbols == ["ETHUSD", "BTCUSD"]
    assert load_symbols == ["BTCUSD", "ETHUSD", "SOLUSD"]


def test_get_rl_signals_uses_trained_symbol_order(monkeypatch) -> None:
    trained_symbols = ["BTCUSD", "ETHUSD", "SOLUSD"]
    bars_dict = {symbol: _make_daily_bars(symbol) for symbol in trained_symbols}
    date = bars_dict["BTCUSD"]["timestamp"].iloc[-1]

    monkeypatch.setattr(
        hybrid,
        "_load_bridge_and_trained_symbols",
        lambda checkpoint_path, hidden_size=1024: (object(), trained_symbols),
    )

    def _fake_signal_map(history_frames, bridge):
        assert set(history_frames) == set(trained_symbols)
        return {
            "BTCUSD": RLSignal(0, "BTCUSD", "long", 0.91, 1.2, 0.5),
            "ETHUSD": RLSignal(1, "ETHUSD", "flat", 0.12, 0.0, 0.0),
        }

    monkeypatch.setattr(hybrid, "_build_crypto_rl_signal_map", _fake_signal_map)

    signals = hybrid.get_rl_signals("checkpoint.pt", bars_dict, date)

    assert [signal.symbol for signal in signals] == trained_symbols
    assert [signal.rl_direction for signal in signals] == ["long", "flat", "flat"]
    assert signals[0].rl_confidence == 0.91
    assert signals[2].rl_confidence == 0.0


def test_resolve_effective_end_date_clamps_to_available_data() -> None:
    bars_dict = {"BTCUSD": _make_daily_bars("BTCUSD", periods=10)}
    requested_end, effective_end = hybrid.resolve_effective_end_date(
        "2026-03-19",
        bars_dict,
        ["BTCUSD"],
    )

    assert str(requested_end.date()) == "2026-03-19"
    assert effective_end == bars_dict["BTCUSD"]["timestamp"].max()
