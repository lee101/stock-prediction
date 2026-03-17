from __future__ import annotations

import pandas as pd

from llm_hourly_trader.gemini_wrapper import TradePlan
from scripts import analyze_actionable_reprompts as module


def test_analyze_window_counts_actionable_daily_plans(monkeypatch) -> None:
    index = pd.date_range("2026-03-14 00:00:00+00:00", periods=48, freq="h", tz="UTC")
    bars = pd.DataFrame(
        {
            "timestamp": index,
            "symbol": ["BTCUSD"] * len(index),
            "open": [100.0] * len(index),
            "high": [101.0] * len(index),
            "low": [99.0] * len(index),
            "close": [100.0 + i * 0.1 for i in range(len(index))],
            "volume": [1_000.0] * len(index),
        }
    )
    plans = iter(
        [
            TradePlan("long", 99.5, 101.0, 0.7, "buy"),
            TradePlan("hold", 0.0, 0.0, 0.3, "flat"),
        ]
    )

    monkeypatch.setattr(module.bh, "load_bars", lambda symbol: bars.copy())
    monkeypatch.setattr(module.bh, "load_forecasts", lambda symbol, horizon: pd.DataFrame())
    monkeypatch.setattr(module.bh, "build_prompt", lambda **kwargs: "prompt")
    monkeypatch.setattr(module, "call_llm", lambda *args, **kwargs: next(plans))

    result = module.analyze_window(
        ["BTCUSD"],
        model="gemini-3.1-flash-lite-preview",
        start_ts="2026-03-14T00:00:00Z",
        end_ts="2026-03-15T23:00:00Z",
        reprompt_policy="entry_only",
        review_max_confidence=0.6,
    )

    assert result["total"]["decisions"] == 2
    assert result["total"]["actionable"] == 1
    assert result["total"]["flat_holds"] == 1
    assert result["total"]["entry_only_reviews"] == 1
    assert result["total"]["would_review"] == 0
    assert result["per_symbol"]["BTCUSD"]["actionable_pct"] == 0.5
    assert result["per_symbol"]["BTCUSD"]["entry_only_pct"] == 0.5
    assert result["per_symbol"]["BTCUSD"]["would_review_pct"] == 0.0
    assert result["per_symbol"]["BTCUSD"]["categories"]["entry_with_exit"] == 1
    assert result["per_symbol"]["BTCUSD"]["categories"]["flat_hold"] == 1
