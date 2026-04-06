from __future__ import annotations

import importlib


def test_torch_device_utils_root_module_aliases_src_module():
    root_module = importlib.import_module("torch_device_utils")
    src_module = importlib.import_module("src.torch_device_utils")

    assert root_module is src_module
    assert root_module.get_strategy_device is src_module.get_strategy_device
    assert root_module.resolve_runtime_device is src_module.resolve_runtime_device
    assert (
        root_module.move_module_to_runtime_device
        is src_module.move_module_to_runtime_device
    )
