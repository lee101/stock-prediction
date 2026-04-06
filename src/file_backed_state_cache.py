"""Lightweight in-memory cache for file-backed state.

Avoids re-reading and re-parsing JSON when the underlying file has
not changed (based on ``os.stat`` results).
"""

from __future__ import annotations

from pathlib import Path
from typing import Generic, TypeVar


_T = TypeVar("_T")

_StatKey = tuple[float, int]  # (mtime_ns, size)


def _stat_key(path: Path) -> _StatKey | None:
    """Return a cache key derived from *path*'s stat, or *None* if missing."""
    try:
        st = path.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


class FileBackedStateCache(Generic[_T]):  # noqa: UP046
    """A simple per-key cache that tracks file freshness via stat.

    Usage::

        cache: FileBackedStateCache[MyState] = FileBackedStateCache()
        cached, key = cache.load("acct", path)
        if cached is not None:
            return cached
        # ... read & parse file ...
        return cache.store("acct", parsed_state, stat_key=key)
    """

    def __init__(self) -> None:
        self._data: dict[str, tuple[_T, _StatKey | None]] = {}

    def load(self, key: str, path: Path) -> tuple[_T | None, _StatKey | None]:
        """Return ``(cached_value, stat_key)`` for *key*.

        *cached_value* is the previously stored value if the file on disk has
        not changed since the last ``store`` call, otherwise ``None``.
        *stat_key* is the current stat of *path* (``None`` if the file is
        missing).
        """
        current = _stat_key(path)
        entry = self._data.get(key)
        if entry is not None:
            cached_val, cached_sk = entry
            if current is not None and cached_sk == current:
                return cached_val, current
        return None, current

    def store(self, key: str, value: _T, *, stat_key: _StatKey | None) -> _T:
        """Store *value* in the cache for *key* and return it."""
        self._data[key] = (value, stat_key)
        return value

    def store_for_path(self, key: str, value: _T, path: Path) -> _T:
        """Store *value* using a fresh stat from *path*."""
        return self.store(key, value, stat_key=_stat_key(path))
