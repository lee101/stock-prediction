"""
Allow external orchestrators to inject heavy dependencies before the RL
training stack imports them. Falls back to local imports when used directly.
"""

from __future__ import annotations

from types import ModuleType
from typing import Optional, Tuple

_torch: Optional[ModuleType] = None
_np: Optional[ModuleType] = None


def setup_training_imports(torch_module: ModuleType, numpy_module: ModuleType) -> None:
    global _torch, _np
    if torch_module is not None:
        _torch = torch_module
    if numpy_module is not None:
        _np = numpy_module


def _resolve() -> Tuple[ModuleType, ModuleType]:
    global _torch, _np
    if _torch is None:
        import importlib

        _torch = importlib.import_module("torch")
    if _np is None:
        import importlib

        _np = importlib.import_module("numpy")
    return _torch, _np


def get_torch() -> ModuleType:
    return _resolve()[0]


def get_numpy() -> ModuleType:
    return _resolve()[1]

