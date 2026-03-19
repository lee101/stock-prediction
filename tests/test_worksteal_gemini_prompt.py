import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binance_worksteal.gemini_overlay import build_daily_prompt


def _bars(symbol: str) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=20, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": [100.0 + i for i in range(20)],
            "high": [101.0 + i for i in range(20)],
            "low": [99.0 + i for i in range(20)],
            "close": [100.0 + i for i in range(20)],
            "volume": [1000.0] * 20,
            "symbol": [symbol] * 20,
        }
    )


def test_build_daily_prompt_describes_btc_fdusd_execution():
    prompt = build_daily_prompt(
        symbol="BTCUSD",
        bars=_bars("BTCUSD"),
        current_price=119.0,
        rule_signal={"buy_target": 118.5},
        fee_bps=0,
        entry_proximity_bps=25.0,
    )

    assert "BTCFDUSD" in prompt
    assert "within 25 bps" in prompt
    assert "FDUSD pairs" in prompt


def test_build_daily_prompt_describes_alt_usdt_execution():
    prompt = build_daily_prompt(
        symbol="SOLUSD",
        bars=_bars("SOLUSD"),
        current_price=119.0,
        rule_signal={"buy_target": 118.5},
        fee_bps=10,
        entry_proximity_bps=25.0,
    )

    assert "SOLUSDT" in prompt
    assert "USDT pairs" in prompt
