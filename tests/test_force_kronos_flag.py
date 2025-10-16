from __future__ import annotations

import importlib


def test_resolve_best_model_forced_kronos(monkeypatch):
    module = importlib.import_module("backtest_test3_inline")
    monkeypatch.setenv("MARKETSIM_FORCE_KRONOS", "1")
    monkeypatch.setattr(module, "_forced_kronos_logged_symbols", set())

    def _fail(symbol: str):
        raise AssertionError("load_model_selection should not be called when Kronos forcing is enabled.")

    monkeypatch.setattr(module, "load_model_selection", _fail)

    assert module.resolve_best_model("TEST") == "kronos"
