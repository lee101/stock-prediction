"""Thin in-process read/write lock keyed by file path.

Provides cooperative locking so that concurrent threads within the same
process do not clobber each other when reading/writing the same file.
This is *not* a cross-process file lock; for that use ``fcntl`` or a
dedicated library.
"""

from __future__ import annotations

import threading
from pathlib import Path


class _PathGuard:
    """A simple read/write lock for a single path."""

    def __init__(self) -> None:
        self._lock = threading.RLock()

    # -- public API expected by trade_daily_stock_prod --

    def acquire_read(self) -> None:
        self._lock.acquire()

    def release_read(self) -> None:
        self._lock.release()

    def acquire_write(self) -> None:
        self._lock.acquire()

    def release_write(self) -> None:
        self._lock.release()


_guards: dict[str, _PathGuard] = {}
_guards_lock = threading.Lock()


def shared_path_guard(path: Path | str) -> _PathGuard:
    """Return a shared guard instance for *path*.

    All callers that pass the same resolved path get the same guard, so
    concurrent ``load_state`` / ``save_state`` calls serialise correctly.
    """
    key = str(Path(path).resolve())
    with _guards_lock:
        if key not in _guards:
            _guards[key] = _PathGuard()
        return _guards[key]
