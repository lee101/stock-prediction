from __future__ import annotations

import sys
from types import ModuleType

import pytest

import faltrain.dependencies as deps


@pytest.fixture(autouse=True)
def _reset_registry():
    deps._clear_registry_for_tests()
    try:
        yield
    finally:
        deps._clear_registry_for_tests()


def test_register_and_get_dependency():
    module = ModuleType("fal_fake_numpy")
    deps.register_fal_dependency("fal_fake_numpy", module)

    assert deps.get_fal_dependency("fal_fake_numpy") is module
    assert deps.is_dependency_registered("fal_fake_numpy") is True
    assert "fal_fake_numpy" in deps.registered_dependency_names()
    assert sys.modules["fal_fake_numpy"] is module
    sys.modules.pop("fal_fake_numpy", None)


def test_get_dependency_imports_when_missing(monkeypatch):
    module_name = "fal_fake_torch"
    module = ModuleType(module_name)
    sys.modules[module_name] = module
    try:
        resolved = deps.get_fal_dependency(module_name)
        assert resolved is module
        assert deps.is_dependency_registered(module_name) is True
    finally:
        sys.modules.pop(module_name, None)
