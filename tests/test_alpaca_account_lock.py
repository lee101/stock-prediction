from __future__ import annotations

import ast
import json
import time
from pathlib import Path
from threading import Barrier, Event, Thread

import pytest

import trade_daily_stock_prod as daily_stock
import cancel_multi_orders as cancel_guard
import unified_hourly_experiment.trade_unified_hourly_meta as meta_stock
import unified_orchestrator.orchestrator as orchestrator
from src import alpaca_account_lock as account_lock


def _module_function_source(module: object, function_name: str) -> str:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise AssertionError(f"Module {module!r} does not expose a source file")
    path = Path(module_file)
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    lines = source.splitlines()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise AssertionError(f"Function {function_name!r} not found in {path}")


def test_lock_path_for_account_uses_state_dir(tmp_path: Path) -> None:
    path = account_lock.lock_path_for_account("alpaca_live_writer", state_dir=tmp_path)
    assert path == tmp_path / "account_locks" / "alpaca_live_writer.lock"


def test_normalize_alpaca_account_name_normalizes_spaces_and_case() -> None:
    assert account_lock.normalize_alpaca_account_name(" Alpaca Live Writer ") == "alpaca_live_writer"


@pytest.mark.parametrize(
    "raw_name",
    [
        "",
        "   ",
        "alpaca/live",
        r"alpaca\\live",
        "alpaca..live",
        "alpaca!live",
        "x" * 65,
    ],
)
def test_normalize_alpaca_account_name_rejects_unsupported_values(raw_name: str) -> None:
    with pytest.raises(ValueError):
        account_lock.normalize_alpaca_account_name(raw_name)


def test_lock_path_for_account_normalizes_spaces(tmp_path: Path) -> None:
    path = account_lock.lock_path_for_account(" Alpaca Live Writer ", state_dir=tmp_path)
    assert path == tmp_path / "account_locks" / "alpaca_live_writer.lock"


def test_lock_path_for_account_rejects_path_like_account_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported Alpaca account name"):
        account_lock.lock_path_for_account("../alpaca_live_writer", state_dir=tmp_path)

    with pytest.raises(ValueError, match="Unsupported Alpaca account name"):
        account_lock.lock_path_for_account(r"..\\alpaca_live_writer", state_dir=tmp_path)


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
    first_key = str(first.path.resolve())
    first.release()
    assert first_key not in account_lock._HELD_LOCKS

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


def test_acquire_alpaca_account_lock_is_idempotent_for_same_service(tmp_path: Path) -> None:
    first = account_lock.acquire_alpaca_account_lock(
        "same-service",
        account_name="alpaca_live_writer",
        state_dir=tmp_path,
    )
    try:
        second = account_lock.acquire_alpaca_account_lock(
            "same-service",
            account_name="alpaca_live_writer",
            state_dir=tmp_path,
        )
        assert second is first
    finally:
        first.release()


def test_acquire_alpaca_account_lock_is_thread_safe_for_same_service(tmp_path: Path) -> None:
    barrier = Barrier(3)
    results: list[account_lock.AlpacaAccountLock] = []

    def _worker() -> None:
        barrier.wait()
        lock = account_lock.acquire_alpaca_account_lock(
            "same-service",
            account_name="alpaca_live_writer",
            state_dir=tmp_path,
        )
        results.append(lock)
        barrier.wait()

    threads = [Thread(target=_worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert len(results) == 2
    assert results[0] is results[1]
    results[0].release()


def test_acquire_alpaca_account_lock_rejects_different_in_process_service(tmp_path: Path) -> None:
    first = account_lock.acquire_alpaca_account_lock(
        "first-service",
        account_name="alpaca_live_writer",
        state_dir=tmp_path,
    )
    try:
        with pytest.raises(RuntimeError, match="held in-process"):
            account_lock.acquire_alpaca_account_lock(
                "second-service",
                account_name="alpaca_live_writer",
                state_dir=tmp_path,
            )
    finally:
        first.release()


def test_acquire_alpaca_account_lock_rejects_conflicting_service_across_threads(tmp_path: Path) -> None:
    start_barrier = Barrier(3)
    release_event = Event()
    results: list[tuple[str, object]] = []

    def _worker(service_name: str) -> None:
        start_barrier.wait()
        try:
            lock = account_lock.acquire_alpaca_account_lock(
                service_name,
                account_name="alpaca_live_writer",
                state_dir=tmp_path,
            )
        except Exception as exc:  # pragma: no cover - asserted below
            results.append(("error", exc))
            return
        results.append(("lock", lock))
        release_event.wait(timeout=5.0)
        lock.release()

    threads = [
        Thread(target=_worker, args=("first-service",)),
        Thread(target=_worker, args=("second-service",)),
    ]
    for thread in threads:
        thread.start()
    start_barrier.wait()
    deadline = time.time() + 5.0
    while len(results) < 2 and time.time() < deadline:
        time.sleep(0.01)
    release_event.set()
    for thread in threads:
        thread.join(timeout=5.0)

    locks = [payload for kind, payload in results if kind == "lock"]
    errors = [payload for kind, payload in results if kind == "error"]
    assert len(locks) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "held in-process" in str(errors[0])


def test_acquire_alpaca_account_lock_reports_current_holder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = account_lock.acquire_alpaca_account_lock(
        "first-service",
        account_name="alpaca_live_writer",
        state_dir=tmp_path,
    )
    original_flock = account_lock.fcntl.flock

    def _fake_flock(fd: int, operation: int):
        raise BlockingIOError("locked")

    key = str(first.path.resolve())
    held = account_lock._HELD_LOCKS.pop(key)
    assert held is first
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
    daily_stock_main = _module_function_source(daily_stock, "main")
    cancel_guard_main = _module_function_source(cancel_guard, "main")
    orchestrator_main = _module_function_source(orchestrator, "main")
    meta_stock_main = _module_function_source(meta_stock, "main")

    assert "acquire_alpaca_account_lock" in daily_stock_main
    assert "acquire_alpaca_account_lock" in cancel_guard_main
    assert "acquire_alpaca_account_lock" in orchestrator_main
    assert "acquire_alpaca_account_lock" in meta_stock_main
    assert "require_explicit_live_trading_enable" in daily_stock_main
    assert "require_explicit_live_trading_enable" in cancel_guard_main
    assert "require_explicit_live_trading_enable" in orchestrator_main
    assert "require_explicit_live_trading_enable" in meta_stock_main


def test_agents_md_documents_single_live_writer_rule() -> None:
    agents_path = Path(__file__).resolve().parent.parent / "AGENTS.md"
    assert agents_path.exists()
    text = agents_path.read_text()
    assert "Exactly one scheduled live writer process may run against a given Alpaca account." in text
    assert "Automatic live exits must not realize a loss unless they are an explicit force-exit path." in text
    assert "ALLOW_ALPACA_LIVE_TRADING=1" in text
