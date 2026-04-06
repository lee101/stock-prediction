from __future__ import annotations

import importlib


def test_maxdiff_optimizer_root_module_aliases_src_module():
    root_module = importlib.import_module("maxdiff_optimizer")
    src_module = importlib.import_module("src.maxdiff_optimizer")

    assert root_module is src_module
    assert root_module.optimize_maxdiff_entry_exit is src_module.optimize_maxdiff_entry_exit
    assert root_module.optimize_maxdiff_always_on is src_module.optimize_maxdiff_always_on


def test_maxdiff_optimizer_root_and_src_share_backend_contract():
    root_module = importlib.import_module("maxdiff_optimizer")
    src_module = importlib.import_module("src.maxdiff_optimizer")

    assert root_module.ENTRY_EXIT_OPTIMIZER_BACKEND == src_module.ENTRY_EXIT_OPTIMIZER_BACKEND
    assert root_module.EntryExitOptimizationResult is src_module.EntryExitOptimizationResult
    assert root_module.AlwaysOnOptimizationResult is src_module.AlwaysOnOptimizationResult
