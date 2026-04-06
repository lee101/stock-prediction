from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Generic, TypeVar

StateT = TypeVar("StateT")
StateStatKey = tuple[int, int]


@dataclass(frozen=True)
class FileBackedStateCacheEntry(Generic[StateT]):
    stat_key: StateStatKey | None
    state: StateT


class FileBackedStateCache(Generic[StateT]):
    def __init__(self, *, max_entries: int = 512) -> None:
        self._max_entries = max(int(max_entries), 1)
        self._entries: OrderedDict[str, FileBackedStateCacheEntry[StateT]] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def entry_count(self) -> int:
        with self._lock:
            return len(self._entries)

    @staticmethod
    def clone(state: StateT) -> StateT:
        return deepcopy(state)

    @staticmethod
    def state_stat_key(path: Path) -> StateStatKey | None:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        return (int(stat.st_mtime_ns), int(stat.st_size))

    def load(self, key: str, path: Path) -> tuple[StateT | None, StateStatKey | None]:
        stat_key = self.state_stat_key(path)
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None and cached.stat_key == stat_key:
                self._entries.move_to_end(key)
                return self.clone(cached.state), stat_key
            if cached is not None:
                self._entries.pop(key, None)
        return None, stat_key

    def store(self, key: str, state: StateT, *, stat_key: StateStatKey | None) -> StateT:
        cached_state = self.clone(state)
        with self._lock:
            self._entries[key] = FileBackedStateCacheEntry(stat_key=stat_key, state=cached_state)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
        return self.clone(cached_state)

    def store_for_path(self, key: str, state: StateT, path: Path) -> StateT:
        stat_key = self.state_stat_key(path)
        if stat_key is None:
            with self._lock:
                self._entries.pop(key, None)
            return self.clone(state)
        return self.store(key, state, stat_key=stat_key)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
