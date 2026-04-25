from __future__ import annotations

import importlib
import sys


def test_cache_root_module_aliases_src_module():
    sys.modules.pop("cache", None)
    sys.modules.pop("src.cache", None)

    root_module = importlib.import_module("cache")
    src_module = importlib.import_module("src.cache")

    assert root_module is src_module
    assert root_module.cache is src_module.cache
    assert root_module.sync_cache_decorator is src_module.sync_cache_decorator
    assert root_module.async_cache_decorator is src_module.async_cache_decorator


def test_cache_src_module_aliases_root_module_when_imported_first():
    sys.modules.pop("cache", None)
    sys.modules.pop("src.cache", None)

    src_module = importlib.import_module("src.cache")
    root_module = importlib.import_module("cache")

    assert root_module is src_module
    assert sys.modules["cache"] is src_module
    assert sys.modules["src.cache"] is src_module
