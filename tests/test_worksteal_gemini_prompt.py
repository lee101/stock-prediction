import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binance_worksteal.gemini_overlay import (
    build_daily_prompt, DailyTradePlan, load_forecast_daily,
    _execution_pair_info, list_forecast_coverage, backtest_gemini_decisions,
)


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
        symbol="AVAXUSD",
        bars=_bars("AVAXUSD"),
        current_price=119.0,
        rule_signal={"buy_target": 118.5},
        fee_bps=10,
        entry_proximity_bps=25.0,
    )
    assert "AVAXUSDT" in prompt
    assert "USDT pairs" in prompt


def test_build_daily_prompt_no_proximity():
    prompt = build_daily_prompt(
        symbol="DOGEUSD",
        bars=_bars("DOGEUSD"),
        current_price=0.35,
        rule_signal={"buy_target": 0.30},
        fee_bps=10,
    )
    assert "DOGEUSDT" in prompt
    assert "within" not in prompt
    assert "USDT pairs" in prompt


def test_execution_pair_info_fdusd():
    pair, note = _execution_pair_info("BTCUSD")
    assert pair == "BTCFDUSD"
    assert "0% maker fee" in note


def test_execution_pair_info_usdt():
    pair, note = _execution_pair_info("AVAXUSD")
    assert pair == "AVAXUSDT"
    assert "10bps" in note


def test_daily_trade_plan_defaults():
    plan = DailyTradePlan(action="hold")
    assert plan.buy_price == 0.0
    assert plan.confidence == 0.0
    assert plan.reasoning == ""


def test_load_forecast_daily_nonexistent(tmp_path):
    result = load_forecast_daily("FAKEUSD", cache_root=tmp_path)
    assert result is None


def test_list_forecast_coverage_empty(tmp_path):
    cov = list_forecast_coverage(["BTCUSD", "FAKEUSD"], cache_root=tmp_path)
    assert "BTCUSD" in cov["missing"]
    assert "FAKEUSD" in cov["missing"]
    assert len(cov["covered"]) == 0


def test_backtest_gemini_decisions_empty_cache(tmp_path):
    results = backtest_gemini_decisions(
        bars=pd.DataFrame(),
        candidates=[{"symbol": "BTCUSD", "date": "2026-03-01", "current_price": 50000.0, "rule_signal": {}}],
        use_cache=True,
        cache_dir=tmp_path / "bt_cache",
    )
    assert len(results) == 1
    assert results[0]["plan"] is None


def test_backtest_gemini_decisions_uses_cache(tmp_path):
    import json
    cache_dir = tmp_path / "bt_cache"
    cache_dir.mkdir()
    (cache_dir / "BTCUSD_2026-03-01.json").write_text(json.dumps({
        "action": "buy", "buy_price": 49000.0, "sell_price": 52000.0,
        "stop_price": 47000.0, "confidence": 0.8, "reasoning": "test",
    }))
    results = backtest_gemini_decisions(
        bars=pd.DataFrame(),
        candidates=[{"symbol": "BTCUSD", "date": "2026-03-01", "current_price": 50000.0, "rule_signal": {}}],
        use_cache=True,
        cache_dir=cache_dir,
    )
    assert len(results) == 1
    assert results[0]["action"] == "buy"
    assert results[0]["buy_price"] == 49000.0
    assert results[0]["confidence"] == 0.8


def test_prompt_contains_forecast_section():
    bars = _bars("ETHUSD")
    fc = {"predicted_close_p50": 122.0, "predicted_high_p50": 125.0}
    prompt = build_daily_prompt(
        symbol="ETHUSD", bars=bars, current_price=119.0,
        rule_signal={}, forecast_24h=fc,
    )
    assert "CHRONOS2 24h FORECAST" in prompt
    assert "122.00" in prompt


def test_prompt_contains_position_info():
    bars = _bars("BTCUSD")
    pos_info = {
        "quantity": 0.5, "entry_price": 100.0,
        "peak_price": 120.0, "target_sell": 115.0, "stop_price": 90.0,
    }
    prompt = build_daily_prompt(
        symbol="BTCUSD", bars=bars, current_price=119.0,
        rule_signal={}, position_info=pos_info,
    )
    assert "CURRENT POSITION" in prompt
    assert "0.500000" in prompt
