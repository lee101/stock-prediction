import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binance_worksteal.gemini_intraday_eval import _build_universe_summary, _cache_path, _sanitize_model_name
from binance_worksteal.strategy import Position


def _make_bars(prices: list[float], symbol: str) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2026-01-01", tz="UTC")
    for idx, close in enumerate(prices):
        rows.append(
            {
                "timestamp": start + pd.Timedelta(days=idx),
                "open": float(close),
                "high": float(close),
                "low": float(close),
                "close": float(close),
                "volume": 1000.0,
                "symbol": symbol,
            }
        )
    return pd.DataFrame(rows)


def test_cache_path_separates_models_and_prompts():
    path_25 = _cache_path(model="gemini-2.5-flash", prompt="prompt-a")
    path_31 = _cache_path(model="gemini-3.1-flash-lite-preview", prompt="prompt-a")
    path_other_prompt = _cache_path(model="gemini-2.5-flash", prompt="prompt-b")

    assert path_25 != path_31
    assert path_25 != path_other_prompt
    assert _sanitize_model_name("gemini-3.1/flash lite") == "gemini-3.1_flash_lite"


def test_build_universe_summary_marks_held_symbols():
    history = {
        "BTCUSD": _make_bars([100.0, 101.0], "BTCUSD"),
        "ETHUSD": _make_bars([200.0, 198.0], "ETHUSD"),
    }
    current_bars = {sym: frame.iloc[-1] for sym, frame in history.items()}
    positions = {
        "ETHUSD": Position(
            symbol="ETHUSD",
            direction="long",
            entry_price=198.0,
            entry_date=pd.Timestamp("2026-01-02", tz="UTC"),
            quantity=1.0,
            cost_basis=198.0,
            peak_price=198.0,
            target_exit_price=204.0,
            stop_price=182.0,
        )
    }

    summary = _build_universe_summary(current_bars=current_bars, history=history, positions=positions)

    assert "BTCUSD" in summary
    assert "ETHUSD" in summary
    assert "HELD" in summary
