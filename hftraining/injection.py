"""
Utilities to allow external environments (e.g. FAL apps) to inject heavy
dependencies such as torch/numpy before the rest of the training stack loads.

The default behaviour falls back to local imports so the module works in
standalone scripts without any special setup.
"""

from __future__ import annotations

from types import ModuleType
from typing import Optional, Tuple

_torch: Optional[ModuleType] = None
_np: Optional[ModuleType] = None


def setup_training_imports(torch_module: ModuleType, numpy_module: ModuleType) -> None:
    """Register externally supplied torch/numpy modules."""
    global _torch, _np
    if torch_module is not None:
        _torch = torch_module
    if numpy_module is not None:
        _np = numpy_module


def _resolve() -> Tuple[ModuleType, ModuleType]:
    """Ensure torch/numpy modules are available, importing locally if needed."""
    global _torch, _np
    if _torch is None:
        import importlib

        _torch = importlib.import_module("torch")
    if _np is None:
        import importlib

        _np = importlib.import_module("numpy")
    return _torch, _np


def get_torch() -> ModuleType:
    """Return the torch module, importing it on demand."""
    return _resolve()[0]


def get_numpy() -> ModuleType:
    """Return the numpy module, importing it on demand."""
    return _resolve()[1]
