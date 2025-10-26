from __future__ import annotations

import sys
from types import ModuleType

import pytest

from faltrain import dependencies as deps


@pytest.fixture(autouse=True)
def _reset_registry():
    existing = {}
    for name in ("torch", "numpy", "pandas", "torch_alias"):
        if name in sys.modules:
            existing[name] = sys.modules[name]
    deps._reset_for_tests()
    yield
    deps._reset_for_tests()
    for name in ("torch", "numpy", "pandas", "torch_alias"):
        if name in existing:
            sys.modules[name] = existing[name]
        else:
            sys.modules.pop(name, None)


def test_bulk_register_populates_registry():
    torch_stub = ModuleType("torch")
    registered = deps.bulk_register_fal_dependencies({"torch": torch_stub})

    assert registered["torch"] is torch_stub
    assert deps.get_registered_dependency("torch") is torch_stub
    assert sys.modules["torch"] is torch_stub


def test_bulk_register_skips_none_values():
    numpy_stub = ModuleType("numpy")
    registered = deps.bulk_register_fal_dependencies({"numpy": numpy_stub, "pandas": None})

    assert "pandas" not in registered
    assert deps.get_registered_dependency("numpy") is numpy_stub
    with pytest.raises(KeyError):
        deps.get_registered_dependency("pandas")


def test_duplicate_registration_requires_same_module():
    first = ModuleType("torch")
    second = ModuleType("torch")

    deps.register_dependency("torch", first)
    assert deps.register_dependency("torch", first) is first

    with pytest.raises(ValueError):
        deps.register_dependency("torch", second)

    with pytest.raises(ValueError):
        deps.bulk_register_fal_dependencies({"torch": second})


def test_overwrite_replaces_sys_modules():
    initial = ModuleType("torch")
    replacement = ModuleType("torch")

    deps.register_dependency("torch", initial)
    deps.register_dependency("torch", replacement, overwrite=True)

    assert deps.get_registered_dependency("torch") is replacement
    assert sys.modules["torch"] is replacement


def test_registers_module_name_alias():
    module = ModuleType("torch_alias")
    deps.register_dependency("torch", module, overwrite=True)

    assert sys.modules["torch"] is module
    assert sys.modules["torch_alias"] is module
