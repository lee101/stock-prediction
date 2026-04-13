from __future__ import annotations

import sys

import pytest

from unified_orchestrator import orchestrator
from unified_orchestrator.state import UnifiedPortfolioSnapshot


def test_run_cycle_forwards_llm_runtime_settings(monkeypatch) -> None:
    snapshot = UnifiedPortfolioSnapshot(
        alpaca_cash=10_000.0,
        regime="STOCK_HOURS",
    )
    captured: dict[str, tuple] = {}

    monkeypatch.setattr(orchestrator, "build_snapshot", lambda now=None: snapshot)
    monkeypatch.setattr(orchestrator, "save_snapshot", lambda snapshot: None)
    monkeypatch.setattr(orchestrator, "read_pending_fills", lambda since_minutes=65: [])

    def _fake_get_crypto_signals(
        symbols, snapshot, model, thinking_level,
        review_thinking_level, reprompt_passes, reprompt_policy,
        review_max_confidence, review_model, dry_run,
    ):
        captured["crypto"] = (thinking_level, dry_run, tuple(symbols))
        return {}

    def _fake_get_stock_signals(
        symbols, snapshot, model, thinking_level,
        review_thinking_level, reprompt_passes, reprompt_policy,
        review_max_confidence, review_model, dry_run,
    ):
        captured["stock"] = (thinking_level, dry_run, tuple(symbols))
        return {}

    monkeypatch.setattr(orchestrator, "get_crypto_signals", _fake_get_crypto_signals)
    monkeypatch.setattr(orchestrator, "get_stock_signals", _fake_get_stock_signals)

    orchestrator.run_cycle(
        crypto_symbols=["BTCUSD"],
        stock_symbols=["NVDA"],
        model="gemini-3.1-flash-lite-preview",
        thinking_level="HIGH",
        dry_run=True,
    )

    assert captured["crypto"] == ("HIGH", True, ("BTCUSD",))
    assert captured["stock"] == ("HIGH", True, ("NVDA",))


def test_build_arg_parser_defaults_to_dry_run_and_exposes_fetch_workers(monkeypatch) -> None:
    monkeypatch.setenv("ORCH_BAR_FETCH_WORKERS", "7")

    parser = orchestrator._build_arg_parser()
    args = parser.parse_args([])

    assert args.live is False
    assert args.bar_fetch_workers == 7


def test_build_arg_parser_allows_live_mode_and_custom_fetch_workers() -> None:
    parser = orchestrator._build_arg_parser()
    args = parser.parse_args(["--live", "--bar-fetch-workers", "9"])

    assert args.live is True
    assert args.bar_fetch_workers == 9


def test_startup_config_summary_includes_effective_runtime_settings(monkeypatch) -> None:
    monkeypatch.setenv("STATE_DIR", "/tmp/runtime-state-dir")
    parser = orchestrator._build_arg_parser()
    args = parser.parse_args(["--crypto-symbols", "BTCUSD", "ETHUSD", "--stock-symbols", "NVDA"])

    summary = orchestrator._startup_config_summary(args, dry_run=True)

    assert summary["mode"] == "DRY_RUN"
    assert summary["crypto_symbols"] == ["BTCUSD", "ETHUSD"]
    assert summary["stock_symbols"] == ["NVDA"]
    assert summary["bar_fetch_workers"] >= 1
    assert summary["state_dir"] == "/tmp/runtime-state-dir"
    assert summary["cycle_event_log"] == "/tmp/runtime-state-dir/orchestrator_cycle_events.jsonl"


def test_main_respects_custom_lock_name_in_live_mode(monkeypatch) -> None:
    """--lock-name is respected so multiple orchestrators can coexist on disjoint symbols."""
    lock_calls: list[tuple[str, str]] = []

    class _StopMain(RuntimeError):
        pass

    monkeypatch.setattr(sys, "argv", ["orchestrator.py", "--live", "--lock-name", "llm_stock_writer", "--once"])
    monkeypatch.setattr(orchestrator, "require_explicit_live_trading_enable", lambda _service: None)
    monkeypatch.setattr(
        orchestrator,
        "acquire_alpaca_account_lock",
        lambda owner, *, account_name: lock_calls.append((owner, account_name)) or type("Lock", (), {"path": "/tmp/live.lock"})(),
    )
    monkeypatch.setattr(orchestrator, "save_snapshot", lambda snapshot: None)
    monkeypatch.setattr(orchestrator.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        orchestrator,
        "_run_cycle_with_runtime_logging",
        lambda **kwargs: (_ for _ in ()).throw(_StopMain()),
    )

    with pytest.raises(_StopMain):
        orchestrator.main()

    # The custom lock name is used as-is (allows coexistence with daily-rl-trader
    # which holds alpaca_live_writer on a disjoint symbol set).
    assert lock_calls == [("unified-orchestrator", "llm_stock_writer")]
