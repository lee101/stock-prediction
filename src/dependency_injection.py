"""
Legacy dependency-injection facade.

The original project exposed ``src.dependency_injection`` with helpers for
registering observers and resolving heavy numerical dependencies. The modern
codebase centralises this logic in ``src.runtime_imports``; some tools (and a
few third-party scripts) still import the old module path.  This shim restores
that surface area while delegating the actual setup work to
``runtime_imports``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Callable, Dict, MutableMapping

from .runtime_imports import setup_src_imports

_MODULES: Dict[str, object] = {}
_OBSERVERS: Dict[str, list[Callable[[object], None]]] = {}


def _notify(name: str, module: object) -> None:
    if module is None:
        return
    _MODULES[name] = module
    for callback in _OBSERVERS.get(name, []):
        try:
            callback(module)
        except Exception:
            continue


def injected_modules() -> MutableMapping[str, object]:
    """Return a mutable mapping of currently injected modules."""
    return _MODULES


def register_observer(name: str, callback: Callable[[object], None]) -> None:
    """Register a callback that fires whenever ``name`` is (re)injected."""
    _OBSERVERS.setdefault(name, []).append(callback)
    if name in _MODULES:
        callback(_MODULES[name])


def setup_imports(
    *,
    torch: object | None = None,
    numpy: object | None = None,
    pandas: object | None = None,
    **extra_modules: object | None,
) -> None:
    """Inject modules and fan out to the modern runtime-import hooks."""
    if torch is not None:
        _notify("torch", torch)
    if numpy is not None:
        _notify("numpy", numpy)
    if pandas is not None:
        _notify("pandas", pandas)
    for name, module in extra_modules.items():
        if module is not None:
            _notify(name, module)
    setup_src_imports(torch, numpy, pandas, **extra_modules)


def _resolve(name: str, fallback: str) -> object:
    module = _MODULES.get(name)
    if module is not None:
        return module
    imported = import_module(fallback)
    _notify(name, imported)
    return imported


def resolve_torch() -> object:
    """Return the injected torch module (importing it if required)."""
    return _resolve("torch", "torch")


def resolve_numpy() -> object:
    """Return the injected NumPy module (importing it if required)."""
    return _resolve("numpy", "numpy")


def resolve_pandas() -> object:
    """Return the injected pandas module (importing it if required)."""
    return _resolve("pandas", "pandas")


def _reset_for_tests() -> None:
    """Test-only helper retained for backwards compatibility."""
    _MODULES.clear()
    _OBSERVERS.clear()


__all__ = [
    "injected_modules",
    "register_observer",
    "resolve_numpy",
    "resolve_pandas",
    "resolve_torch",
    "setup_imports",
    "_reset_for_tests",
]
