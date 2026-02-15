from __future__ import annotations

import importlib
import sys
from types import ModuleType

from src.runtime_imports import _reset_for_tests, setup_src_imports


def _make_stub_torch() -> ModuleType:
    module = ModuleType("torch")
    module.Tensor = type("Tensor", (), {})  # type: ignore[attr-defined]
    return module


def _make_stub_numpy() -> ModuleType:
    module = ModuleType("numpy")
    module.asarray = lambda data, **kwargs: data  # type: ignore[attr-defined]
    module.quantile = lambda data, qs, axis=0: qs  # type: ignore[attr-defined]
    return module


def test_setup_src_imports_updates_conversion_utils():
    _reset_for_tests()

    torch_stub = _make_stub_torch()
    numpy_stub = _make_stub_numpy()

    original_torch = sys.modules.get("torch")
    original_numpy = sys.modules.get("numpy")
    original_conversion = sys.modules.get("src.conversion_utils")

    sys.modules["torch"] = torch_stub
    sys.modules["numpy"] = numpy_stub
    sys.modules.pop("src.conversion_utils", None)

    setup_src_imports(torch_stub, numpy_stub, None)

    module = importlib.import_module("src.conversion_utils")

    assert getattr(module, "torch") is torch_stub

    # Clean up sys.modules to avoid leaking stubs into other tests.
    if original_torch is not None:
        sys.modules["torch"] = original_torch
    else:
        sys.modules.pop("torch", None)
    if original_numpy is not None:
        sys.modules["numpy"] = original_numpy
    else:
        sys.modules.pop("numpy", None)
    if original_conversion is not None:
        sys.modules["src.conversion_utils"] = original_conversion
    else:
        sys.modules.pop("src.conversion_utils", None)
