from __future__ import annotations

"""
Utilities for sharing heavy runtime dependencies (torch, numpy, pandas, …)
between the fal training app and the in-repo trainers.

The fal worker process eagerly imports the large packages as part of
`StockTrainerApp.setup`.  Trainers that run in the same process can then
grab the already-loaded modules via `get_fal_dependency` without paying the
import cost again or worrying about mismatched versions.
"""

import importlib
import sys
from threading import RLock
from types import ModuleType
from typing import Dict, Iterable, Tuple

_LOCK = RLock()
_REGISTRY: Dict[str, ModuleType] = {}


def register_fal_dependency(name: str, module: ModuleType) -> None:
    """Register a module that the fal runtime has already imported."""

    if not isinstance(module, ModuleType):
        raise TypeError(f"Expected ModuleType for {name!r}, got {type(module)}")

    with _LOCK:
        _REGISTRY[name] = module
        # Ensure downstream imports resolve to the injected module.
        sys.modules.setdefault(name, module)


def bulk_register_fal_dependencies(mapping: Dict[str, ModuleType]) -> None:
    for name, module in mapping.items():
        if module is not None:
            register_fal_dependency(name, module)


def get_fal_dependency(name: str, *, import_if_missing: bool = True) -> ModuleType:
    """Return an injected dependency, falling back to importing if allowed."""

    with _LOCK:
        module = _REGISTRY.get(name)
    if module is not None:
        return module

    if import_if_missing:
        module = importlib.import_module(name)
        register_fal_dependency(name, module)
        return module

    raise KeyError(
        f"{name!r} is not registered. Add it to StockTrainerApp.requirements or "
        "call register_fal_dependency during setup."
    )


def get_fal_dependencies(*names: str, import_if_missing: bool = True) -> Tuple[ModuleType, ...]:
    return tuple(get_fal_dependency(name, import_if_missing=import_if_missing) for name in names)


def is_dependency_registered(name: str) -> bool:
    with _LOCK:
        return name in _REGISTRY


def registered_dependency_names() -> Tuple[str, ...]:
    with _LOCK:
        return tuple(sorted(_REGISTRY))


def _clear_registry_for_tests() -> None:  # pragma: no cover - exercised in tests
    with _LOCK:
        _REGISTRY.clear()
