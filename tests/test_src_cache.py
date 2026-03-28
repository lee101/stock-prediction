from __future__ import annotations

import importlib
import sqlite3
from pathlib import Path

import diskcache

import src.cache as cache_mod


def test_cache_falls_back_to_memory_when_diskcache_init_fails(monkeypatch) -> None:
    original_cache_cls = diskcache.Cache

    class _FailingDiskCache:
        def __init__(self, *_args, **_kwargs) -> None:
            raise sqlite3.OperationalError("database or disk is full")

    try:
        monkeypatch.setattr(diskcache, "Cache", _FailingDiskCache)
        reloaded = importlib.reload(cache_mod)

        assert reloaded.cache.backend_name == "memory"

        calls = {"count": 0}

        @reloaded.sync_cache_decorator(expire=30)
        def _compute(value: int) -> int:
            calls["count"] += 1
            return value * 2

        assert _compute(3) == 6
        assert _compute(3) == 6
        assert calls["count"] == 1
    finally:
        monkeypatch.setattr(diskcache, "Cache", original_cache_cls)
        importlib.reload(cache_mod)


def test_cache_switches_to_memory_after_runtime_storage_error(tmp_path: Path) -> None:
    class _FailingBackend:
        def set(self, *_args, **_kwargs):
            raise sqlite3.OperationalError("database or disk is full")

        def close(self) -> None:
            return None

    cache_obj = cache_mod._ResilientCache(tmp_path / "cache")
    cache_obj._backend = _FailingBackend()

    cache_obj.set("alpha", 123, expire=30)

    assert cache_obj.backend_name == "memory"
    assert cache_obj.get("alpha") == 123
