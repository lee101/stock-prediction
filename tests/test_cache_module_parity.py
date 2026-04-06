from __future__ import annotations

import importlib


def test_cache_root_module_aliases_src_module():
    root_module = importlib.import_module("cache")
    src_module = importlib.import_module("src.cache")

    assert root_module is src_module
    assert root_module.cache is src_module.cache
    assert root_module.sync_cache_decorator is src_module.sync_cache_decorator
    assert root_module.async_cache_decorator is src_module.async_cache_decorator
