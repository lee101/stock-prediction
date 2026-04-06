from __future__ import annotations

import threading

from src.background_refresher import BackgroundRefreshRegistry


class _FakeEngine:
    def __init__(self) -> None:
        self.first_refresh = threading.Event()
        self.calls = 0

    def refresh_prices(self) -> None:
        self.calls += 1
        self.first_refresh.set()


def test_background_refresh_registry_reference_counts_shared_engine() -> None:
    registry = BackgroundRefreshRegistry[_FakeEngine](
        thread_name_prefix="test-refresh",
        min_poll_seconds=1,
    )
    engine = _FakeEngine()

    thread_one = registry.ensure(
        engine,
        refresh_fn=lambda current: current.refresh_prices(),
        default_poll_seconds=1,
    )
    thread_two = registry.ensure(
        engine,
        refresh_fn=lambda current: current.refresh_prices(),
        default_poll_seconds=1,
    )

    assert thread_one is thread_two
    assert engine.first_refresh.wait(timeout=1.0)

    registry.stop(engine, timeout=1.0)
    assert thread_one.is_alive()

    registry.stop(engine, timeout=1.0)
    assert not thread_one.is_alive()


def test_background_refresh_registry_removes_dead_handle_after_stop() -> None:
    registry = BackgroundRefreshRegistry[_FakeEngine](
        thread_name_prefix="test-refresh",
        min_poll_seconds=1,
    )
    engine = _FakeEngine()

    thread = registry.ensure(
        engine,
        refresh_fn=lambda current: current.refresh_prices(),
        default_poll_seconds=1,
    )
    assert engine.first_refresh.wait(timeout=1.0)

    registry.stop(engine, timeout=1.0)

    assert not thread.is_alive()
    with registry.lock:
        assert engine not in registry.refreshers


def test_background_refresh_registry_honors_restart_request_while_stop_unwinds() -> None:
    registry = BackgroundRefreshRegistry[_FakeEngine](
        thread_name_prefix="test-refresh",
        min_poll_seconds=1,
    )
    engine = _FakeEngine()
    release_refresh = threading.Event()

    def refresh_once_then_wait(current: _FakeEngine) -> None:
        current.calls += 1
        current.first_refresh.set()
        if current.calls == 1:
            release_refresh.wait(timeout=1.0)

    thread = registry.ensure(
        engine,
        refresh_fn=refresh_once_then_wait,
        default_poll_seconds=1,
    )
    assert engine.first_refresh.wait(timeout=1.0)

    registry.stop(engine, timeout=0.01)
    assert thread.is_alive()

    with registry.lock:
        handle = registry.refreshers[engine]
        release_refresh.set()
        handle.restart_requested = True
        handle.owners = 1

    deadline = threading.Event()
    deadline.wait(timeout=0.2)
    assert thread.is_alive()
    assert engine.calls >= 2

    registry.stop(engine, timeout=1.0)
    assert not thread.is_alive()
