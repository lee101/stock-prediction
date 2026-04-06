from __future__ import annotations

import importlib


def test_optimization_utils_root_module_aliases_src_module():
    root_module = importlib.import_module("optimization_utils")
    src_module = importlib.import_module("src.optimization_utils")

    assert root_module is src_module
    assert (
        root_module.optimize_entry_exit_multipliers
        is src_module.optimize_entry_exit_multipliers
    )
    assert (
        root_module.optimize_always_on_multipliers
        is src_module.optimize_always_on_multipliers
    )


def test_optimization_utils_root_and_src_share_runtime_flags():
    root_module = importlib.import_module("optimization_utils")
    src_module = importlib.import_module("src.optimization_utils")

    assert root_module._USE_DIRECT == src_module._USE_DIRECT
    assert root_module._FAST_MODE == src_module._FAST_MODE
