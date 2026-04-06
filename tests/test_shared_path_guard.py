from __future__ import annotations

import gc
import threading
from pathlib import Path

from src.shared_path_guard import ReaderWriterGuard, SharedPathGuardRegistry


def test_shared_path_guard_registry_reuses_guards_per_path(tmp_path: Path) -> None:
    registry = SharedPathGuardRegistry()

    first = registry.get(tmp_path / "a.lock")
    second = registry.get(tmp_path / "a.lock")
    third = registry.get(tmp_path / "b.lock")

    assert first is second
    assert first is not third


def test_shared_path_guard_registry_reuses_guards_for_equivalent_paths(tmp_path: Path) -> None:
    registry = SharedPathGuardRegistry()

    path = tmp_path / "nested" / "a.lock"
    path.parent.mkdir(parents=True, exist_ok=True)

    canonical = registry.get(path)
    alias = registry.get(path.parent / ".." / "nested" / "a.lock")

    assert canonical is alias


def test_shared_path_guard_registry_drops_unreferenced_guards(tmp_path: Path) -> None:
    registry = SharedPathGuardRegistry()

    guard = registry.get(tmp_path / "a.lock")
    assert len(registry._guards) == 1

    del guard
    gc.collect()

    assert len(registry._guards) == 0


def test_reader_writer_guard_blocks_new_readers_when_writer_is_waiting() -> None:
    guard = ReaderWriterGuard()
    events: list[str] = []
    writer_waiting = threading.Event()
    release_reader = threading.Event()
    writer_done = threading.Event()

    def first_reader() -> None:
        guard.acquire_read()
        try:
            events.append("reader-1-acquired")
            writer_waiting.wait(timeout=1)
            release_reader.wait(timeout=1)
        finally:
            guard.release_read()
            events.append("reader-1-released")

    def writer() -> None:
        writer_waiting.set()
        guard.acquire_write()
        try:
            events.append("writer-acquired")
        finally:
            guard.release_write()
            events.append("writer-released")
            writer_done.set()

    def second_reader() -> None:
        writer_waiting.wait(timeout=1)
        guard.acquire_read()
        try:
            events.append("reader-2-acquired")
        finally:
            guard.release_read()
            events.append("reader-2-released")

    t1 = threading.Thread(target=first_reader)
    tw = threading.Thread(target=writer)
    t2 = threading.Thread(target=second_reader)

    t1.start()
    writer_waiting.wait(timeout=1)
    tw.start()
    t2.start()

    writer_waiting.wait(timeout=1)
    release_reader.set()

    t1.join(timeout=1)
    tw.join(timeout=1)
    t2.join(timeout=1)

    assert writer_done.is_set()
    assert events.index("writer-acquired") < events.index("reader-2-acquired")
