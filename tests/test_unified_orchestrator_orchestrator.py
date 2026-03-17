from types import SimpleNamespace

import pandas as pd

from llm_hourly_trader.gemini_wrapper import TradePlan
from unified_orchestrator.orchestrator import (
    _has_actionable_crypto_position,
    get_crypto_signals,
    _with_fallback_crypto_exit_target,
)
from unified_orchestrator.state import Position


def test_has_actionable_crypto_position_ignores_dust() -> None:
    dust = Position(
        symbol="BTCUSD",
        qty=1e-8,
        avg_price=71_000.0,
        current_price=72_000.0,
        broker="alpaca",
    )
    assert _has_actionable_crypto_position(dust) is False

    material = Position(
        symbol="BTCUSD",
        qty=0.001,
        avg_price=71_000.0,
        current_price=72_000.0,
        broker="alpaca",
    )
    assert _has_actionable_crypto_position(material) is True


def test_with_fallback_crypto_exit_target_sets_sell_for_held_position() -> None:
    held = Position(
        symbol="ETHUSD",
        qty=0.2,
        avg_price=2_100.0,
        current_price=2_150.0,
        broker="alpaca",
    )
    plan = TradePlan(
        direction="hold",
        buy_price=0.0,
        sell_price=0.0,
        confidence=0.0,
        reasoning="gemini_hold",
        allocation_pct=0.0,
    )

    updated = _with_fallback_crypto_exit_target("ETHUSD", plan, held, current_price=2_150.0)

    assert updated.direction == "hold"
    assert updated.sell_price > 2_150.0
    assert updated.confidence >= 0.05
    assert "fallback_exit_target" in updated.reasoning


def test_with_fallback_crypto_exit_target_preserves_valid_exit() -> None:
    held = Position(
        symbol="BTCUSD",
        qty=0.01,
        avg_price=71_000.0,
        current_price=72_000.0,
        broker="alpaca",
    )
    plan = TradePlan(
        direction="hold",
        buy_price=0.0,
        sell_price=72_800.0,
        confidence=0.42,
        reasoning="existing_target",
        allocation_pct=0.0,
    )

    updated = _with_fallback_crypto_exit_target("BTCUSD", plan, held, current_price=72_000.0)

    assert updated is plan


def test_get_crypto_signals_forwards_reprompt_passes(monkeypatch) -> None:
    index = pd.date_range("2026-03-15 00:00:00+00:00", periods=30, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": index,
            "symbol": ["BTCUSD"] * len(index),
            "open": [100.0 + i for i in range(len(index))],
            "high": [100.5 + i for i in range(len(index))],
            "low": [99.5 + i for i in range(len(index))],
            "close": [100.2 + i for i in range(len(index))],
            "volume": [1_000.0] * len(index),
        }
    )
    snapshot = SimpleNamespace(alpaca_positions={})
    seen: list[tuple[int, str | None, str | None, float | None]] = []

    monkeypatch.setattr(
        "unified_orchestrator.orchestrator.get_rl_bridge",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "unified_orchestrator.orchestrator._fetch_crypto_history_frames",
        lambda data_client, symbols, now: {"BTCUSD": frame},
    )
    monkeypatch.setattr(
        "unified_orchestrator.orchestrator._choose_forecast_cache_root",
        lambda symbols, candidates: None,
    )
    monkeypatch.setattr(
        "unified_orchestrator.orchestrator._load_forecast_frames",
        lambda symbols, root: ({}, {}),
    )
    monkeypatch.setattr(
        "unified_orchestrator.orchestrator.build_unified_prompt",
        lambda **kwargs: "prompt",
    )

    def _fake_call_llm(
        prompt: str,
        *,
        model: str,
        thinking_level: str | None = None,
        review_thinking_level: str | None = None,
        reprompt_passes: int = 1,
        reprompt_policy: str = "always",
        review_max_confidence: float | None = None,
        review_model: str | None = None,
    ) -> TradePlan:
        seen.append((reprompt_passes, review_model, review_thinking_level, review_max_confidence))
        assert reprompt_policy == "actionable"
        return TradePlan("hold", 0.0, 0.0, 0.4, "ok", 0.0)

    monkeypatch.setattr("unified_orchestrator.orchestrator.call_llm", _fake_call_llm)

    signals = get_crypto_signals(
        ["BTCUSD"],
        snapshot,
        reprompt_passes=2,
        reprompt_policy="actionable",
        review_max_confidence=0.6,
        review_model="gemini-2.5-flash",
        review_thinking_level="LOW",
    )

    assert seen == [(2, "gemini-2.5-flash", "LOW", 0.6)]
    assert signals["BTCUSD"].direction == "hold"
