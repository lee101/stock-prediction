from __future__ import annotations

import sys
from threading import RLock
from types import ModuleType
from typing import Dict, Mapping, Optional

_LOCK = RLock()
_REGISTRY: Dict[str, ModuleType] = {}


def register_dependency(name: str, module: ModuleType, *, overwrite: bool = False) -> ModuleType:
    """
    Record a shared dependency under ``name`` and expose it through ``sys.modules``.

    If the dependency is already registered with a different module object and
    ``overwrite`` is not requested, a ``ValueError`` is raised to avoid silently
    swapping implementations.
    """

    alias = module.__name__
    with _LOCK:
        existing = _REGISTRY.get(name)
        if existing is not None and existing is not module and not overwrite:
            raise ValueError(f"{name!r} is already registered with a different module.")
        _REGISTRY[name] = module
        sys.modules[name] = module
        sys.modules[alias] = module
    return module


def bulk_register_fal_dependencies(
    mapping: Mapping[str, Optional[ModuleType]], *, overwrite: bool = False
) -> Dict[str, ModuleType]:
    """
    Convenience helper that registers several dependencies at once, skipping ``None`` values.
    """

    registered: Dict[str, ModuleType] = {}
    for name, module in mapping.items():
        if module is None:
            continue
        registered[name] = register_dependency(name, module, overwrite=overwrite)
    return registered


def get_registered_dependency(name: str) -> ModuleType:
    """
    Return a previously registered dependency or raise ``KeyError`` when missing.
    """

    with _LOCK:
        if name in _REGISTRY:
            return _REGISTRY[name]
    raise KeyError(name)


def _reset_for_tests() -> None:  # pragma: no cover - exercised in tests
    with _LOCK:
        for name in list(_REGISTRY):
            module = _REGISTRY.pop(name)
            if sys.modules.get(name) is module:
                sys.modules.pop(name, None)
            alias = module.__name__
            if sys.modules.get(alias) is module:
                sys.modules.pop(alias, None)
