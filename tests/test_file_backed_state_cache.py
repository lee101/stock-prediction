from __future__ import annotations

from pathlib import Path
import threading

from src.file_backed_state_cache import FileBackedStateCache


def test_file_backed_state_cache_reuses_cached_clone_when_file_is_unchanged(tmp_path: Path) -> None:
    cache = FileBackedStateCache[dict[str, object]]()
    path = tmp_path / "state.json"
    path.write_text("{}", encoding="utf-8")

    original = {"nested": {"value": 1}}
    stored = cache.store_for_path("acct", original, path)

    assert stored == original
    assert stored is not original

    cached, stat_key = cache.load("acct", path)

    assert stat_key is not None
    assert cached == original
    assert cached is not original
    assert cached["nested"] is not original["nested"]


def test_file_backed_state_cache_invalidates_when_file_stat_changes(tmp_path: Path) -> None:
    cache = FileBackedStateCache[dict[str, object]]()
    path = tmp_path / "state.json"
    path.write_text("{}", encoding="utf-8")

    cache.store_for_path("acct", {"value": 1}, path)
    path.write_text("{\"changed\": true}", encoding="utf-8")

    cached, stat_key = cache.load("acct", path)

    assert cached is None
    assert stat_key is not None
    assert cache.entry_count == 0


def test_file_backed_state_cache_evicts_oldest_entry_when_capacity_is_exceeded(tmp_path: Path) -> None:
    cache = FileBackedStateCache[dict[str, object]](max_entries=2)
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    path_c = tmp_path / "c.json"
    for path in (path_a, path_b, path_c):
        path.write_text("{}", encoding="utf-8")

    cache.store_for_path("a", {"value": "a"}, path_a)
    cache.store_for_path("b", {"value": "b"}, path_b)

    cached_a, _ = cache.load("a", path_a)
    assert cached_a == {"value": "a"}

    cache.store_for_path("c", {"value": "c"}, path_c)

    evicted_b, _ = cache.load("b", path_b)
    retained_a, _ = cache.load("a", path_a)
    retained_c, _ = cache.load("c", path_c)

    assert cache.entry_count == 2
    assert evicted_b is None
    assert retained_a == {"value": "a"}
    assert retained_c == {"value": "c"}


def test_file_backed_state_cache_treats_deleted_file_during_load_stat_as_miss(tmp_path: Path, monkeypatch) -> None:
    cache = FileBackedStateCache[dict[str, object]]()
    path = tmp_path / "state.json"
    path.write_text("{}", encoding="utf-8")
    cache.store_for_path("acct", {"value": 1}, path)

    real_stat = Path.stat

    def flaky_stat(self: Path):
        if self == path:
            raise FileNotFoundError(str(self))
        return real_stat(self)

    monkeypatch.setattr(Path, "stat", flaky_stat)

    cached, stat_key = cache.load("acct", path)

    assert cached is None
    assert stat_key is None
    assert cache.entry_count == 0


def test_file_backed_state_cache_does_not_cache_when_file_disappears_during_store(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache = FileBackedStateCache[dict[str, object]]()
    path = tmp_path / "state.json"
    path.write_text("{}", encoding="utf-8")
    state = {"value": 1}
    gate = threading.Event()

    real_stat = Path.stat

    def flaky_stat(self: Path):
        if self == path and not gate.is_set():
            gate.set()
            raise FileNotFoundError(str(self))
        return real_stat(self)

    monkeypatch.setattr(Path, "stat", flaky_stat)

    stored = cache.store_for_path("acct", state, path)

    assert stored == state
    assert stored is not state
    assert cache.entry_count == 0

    cached, stat_key = cache.load("acct", path)

    assert cached is None
    assert stat_key is not None
