from __future__ import annotations

from pathlib import Path
import threading
import weakref


class ReaderWriterGuard:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    def acquire_read(self) -> None:
        with self._condition:
            while self._writer or self._waiting_writers > 0:
                self._condition.wait()
            self._readers += 1

    def release_read(self) -> None:
        with self._condition:
            self._readers -= 1
            if self._readers == 0:
                self._condition.notify_all()

    def acquire_write(self) -> None:
        with self._condition:
            self._waiting_writers += 1
            try:
                while self._writer or self._readers > 0:
                    self._condition.wait()
                self._writer = True
            finally:
                self._waiting_writers -= 1

    def release_write(self) -> None:
        with self._condition:
            self._writer = False
            self._condition.notify_all()


class SharedPathGuardRegistry:
    def __init__(self) -> None:
        self._guards: weakref.WeakValueDictionary[str, ReaderWriterGuard] = weakref.WeakValueDictionary()
        self._lock = threading.Lock()

    @staticmethod
    def _normalize_key(path: Path) -> str:
        # Normalize equivalent path spellings so callers sharing the same file
        # through different relative/absolute forms still synchronize.
        return str(path.expanduser().resolve(strict=False))

    def get(self, path: Path) -> ReaderWriterGuard:
        key = self._normalize_key(path)
        with self._lock:
            guard = self._guards.get(key)
            if guard is None:
                guard = ReaderWriterGuard()
                self._guards[key] = guard
            return guard


_shared_registry = SharedPathGuardRegistry()


def shared_path_guard(path: Path) -> ReaderWriterGuard:
    return _shared_registry.get(path)
