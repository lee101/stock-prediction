from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

import trade_daily_stock_prod as daily_stock
import cancel_multi_orders as cancel_guard
import unified_hourly_experiment.trade_unified_hourly_meta as meta_stock
import unified_orchestrator.orchestrator as orchestrator
from src import alpaca_account_lock as account_lock


def test_lock_path_for_account_uses_state_dir(tmp_path: Path) -> None:
    path = account_lock.lock_path_for_account("alpaca_live_writer", state_dir=tmp_path)
    assert path == tmp_path / "account_locks" / "alpaca_live_writer.lock"


def test_acquire_alpaca_account_lock_writes_metadata(tmp_path: Path) -> None:
    lock = account_lock.acquire_alpaca_account_lock(
        "unit-test-service",
        account_name="alpaca_live_writer",
        state_dir=tmp_path,
    )
    try:
        payload = json.loads(lock.path.read_text())
        assert payload["service_name"] == "unit-test-service"
        assert payload["account_name"] == "alpaca_live_writer"
        assert payload["pid"] > 0
    finally:
        lock.release()


def test_acquire_alpaca_account_lock_reacquires_after_release(tmp_path: Path) -> None:
    first = account_lock.acquire_alpaca_account_lock(
        "first-service",
        account_name="alpaca_live_writer",
        state_dir=tmp_path,
    )
    first.release()

    second = account_lock.acquire_alpaca_account_lock(
        "second-service",
        account_name="alpaca_live_writer",
        state_dir=tmp_path,
    )
    try:
        payload = json.loads(second.path.read_text())
        assert payload["service_name"] == "second-service"
    finally:
        second.release()


def test_acquire_alpaca_account_lock_reports_current_holder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = account_lock.acquire_alpaca_account_lock(
        "first-service",
        account_name="alpaca_live_writer",
        state_dir=tmp_path,
    )
    original_flock = account_lock.fcntl.flock

    def _fake_flock(fd: int, operation: int):
        raise BlockingIOError("locked")

    monkeypatch.setattr(account_lock.fcntl, "flock", _fake_flock)
    try:
        with pytest.raises(RuntimeError, match="holder_service=first-service"):
            account_lock.acquire_alpaca_account_lock(
                "second-service",
                account_name="alpaca_live_writer",
                state_dir=tmp_path,
            )
    finally:
        monkeypatch.setattr(account_lock.fcntl, "flock", original_flock)
        first.release()


def test_require_explicit_live_trading_enable_blocks_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALLOW_ALPACA_LIVE_TRADING", raising=False)
    with pytest.raises(RuntimeError, match="ALLOW_ALPACA_LIVE_TRADING=1 is required"):
        account_lock.require_explicit_live_trading_enable("unit-test-service")


def test_require_explicit_live_trading_enable_allows_truthy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOW_ALPACA_LIVE_TRADING", "1")
    account_lock.require_explicit_live_trading_enable("unit-test-service")


def test_live_entrypoints_reference_account_lock() -> None:
    assert "acquire_alpaca_account_lock" in inspect.getsource(daily_stock.main)
    assert "acquire_alpaca_account_lock" in inspect.getsource(cancel_guard.main)
    assert "acquire_alpaca_account_lock" in inspect.getsource(orchestrator.main)
    assert "acquire_alpaca_account_lock" in inspect.getsource(meta_stock.main)
    assert "require_explicit_live_trading_enable" in inspect.getsource(daily_stock.main)
    assert "require_explicit_live_trading_enable" in inspect.getsource(cancel_guard.main)
    assert "require_explicit_live_trading_enable" in inspect.getsource(orchestrator.main)
    assert "require_explicit_live_trading_enable" in inspect.getsource(meta_stock.main)


def test_agents_md_documents_single_live_writer_rule() -> None:
    agents_path = Path(__file__).resolve().parent.parent / "AGENTS.md"
    assert agents_path.exists()
    text = agents_path.read_text()
    assert "Exactly one scheduled live writer process may run against a given Alpaca account." in text
    assert "Automatic live exits must not realize a loss unless they are an explicit force-exit path." in text
    assert "ALLOW_ALPACA_LIVE_TRADING=1" in text
