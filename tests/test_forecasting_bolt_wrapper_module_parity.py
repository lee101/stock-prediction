from __future__ import annotations

import importlib
from types import ModuleType


def test_forecasting_bolt_wrapper_root_module_aliases_src_module():
    root_module = importlib.import_module("forecasting_bolt_wrapper")
    src_module = importlib.import_module("src.forecasting_bolt_wrapper")

    assert root_module is src_module
    assert root_module.ForecastingBoltWrapper is src_module.ForecastingBoltWrapper
    assert root_module.setup_forecasting_bolt_imports is src_module.setup_forecasting_bolt_imports


def test_forecasting_bolt_wrapper_setup_updates_shared_module_globals():
    root_module = importlib.import_module("forecasting_bolt_wrapper")
    src_module = importlib.import_module("src.forecasting_bolt_wrapper")

    fake_torch = ModuleType("fake_torch")
    fake_numpy = ModuleType("fake_numpy")
    original_torch = src_module.torch
    original_numpy = src_module.np
    try:
        root_module.setup_forecasting_bolt_imports(
            torch_module=fake_torch,
            numpy_module=fake_numpy,
        )

        assert root_module.torch is fake_torch
        assert root_module.np is fake_numpy
        assert src_module.torch is fake_torch
        assert src_module.np is fake_numpy
    finally:
        src_module.torch = original_torch
        src_module.np = original_numpy
