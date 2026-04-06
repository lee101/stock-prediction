from __future__ import annotations

import builtins
import importlib
import sys


def test_forecast_cache_metrics_root_module_aliases_src_module():
    sys.modules.pop("forecast_cache_metrics", None)
    sys.modules.pop("src.forecast_cache_metrics", None)

    root_module = importlib.import_module("forecast_cache_metrics")
    src_module = importlib.import_module("src.forecast_cache_metrics")

    assert root_module is src_module


def test_forecast_cache_metrics_root_import_uses_src_fallback_when_kronostraining_is_missing(
    monkeypatch,
):
    sys.modules.pop("forecast_cache_metrics", None)
    sys.modules.pop("src.forecast_cache_metrics", None)
    sys.modules.pop("kronostraining.metrics_utils", None)

    real_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "kronostraining.metrics_utils":
            raise ImportError("simulated missing optional dependency")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)

    root_module = importlib.import_module("forecast_cache_metrics")
    src_module = importlib.import_module("src.forecast_cache_metrics")

    assert root_module is src_module
    assert root_module.compute_mae_percent(2.0, [1.0, -3.0]) == 100.0
