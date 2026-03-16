from __future__ import annotations

from unified_orchestrator import orchestrator
from unified_orchestrator.state import UnifiedPortfolioSnapshot


def test_run_cycle_forwards_llm_runtime_settings(monkeypatch) -> None:
    snapshot = UnifiedPortfolioSnapshot(
        alpaca_cash=10_000.0,
        regime="STOCK_HOURS",
    )
    captured: dict[str, tuple[str, str, bool, tuple[str, ...]]] = {}

    monkeypatch.setattr(orchestrator, "build_snapshot", lambda now=None: snapshot)
    monkeypatch.setattr(orchestrator, "save_snapshot", lambda snapshot: None)
    monkeypatch.setattr(orchestrator, "read_pending_fills", lambda since_minutes=65: [])

    def _fake_get_crypto_signals(symbols, snapshot, model, thinking_level, reasoning_effort, dry_run):
        captured["crypto"] = (thinking_level, reasoning_effort, dry_run, tuple(symbols))
        return {}

    def _fake_get_stock_signals(symbols, snapshot, model, thinking_level, reasoning_effort, dry_run):
        captured["stock"] = (thinking_level, reasoning_effort, dry_run, tuple(symbols))
        return {}

    monkeypatch.setattr(orchestrator, "get_crypto_signals", _fake_get_crypto_signals)
    monkeypatch.setattr(orchestrator, "get_stock_signals", _fake_get_stock_signals)

    orchestrator.run_cycle(
        crypto_symbols=["BTCUSD"],
        stock_symbols=["NVDA"],
        model="gemini-3.1-flash-lite-preview",
        thinking_level="HIGH",
        reasoning_effort="high",
        dry_run=True,
    )

    assert captured["crypto"] == ("HIGH", "high", True, ("BTCUSD",))
    assert captured["stock"] == ("HIGH", "high", True, ("NVDA",))
