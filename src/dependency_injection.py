from __future__ import annotations

"""
Shared helpers for injecting heavyweight numerical dependencies (torch, numpy,
pandas, …) when running inside fal workers.

FAL applications can call :func:`setup_imports` during ``App.setup`` to supply
pre-imported modules that were resolved inside the worker process. The helper
stores the modules locally so downstream packages observe the same instances.

Code that wishes to access these libraries should call ``resolve_torch()``,
``resolve_numpy()``, or ``resolve_pandas()`` instead of importing directly. The
resolvers first prefer the injected module, falling back to a regular import
when used outside of fal.
"""

import sys
from importlib import import_module
from threading import RLock
from types import ModuleType
from typing import Callable, Dict, List, Optional

_LOCK = RLock()
_MODULES: Dict[str, ModuleType] = {}
_OBSERVERS: Dict[str, List[Callable[[ModuleType], None]]] = {}


def setup_imports(
    torch_module: Optional[ModuleType] = None,
    numpy_module: Optional[ModuleType] = None,
    pandas_module: Optional[ModuleType] = None,
    **modules: Optional[ModuleType],
) -> None:
    """
    Record modules supplied by the fal runtime and register them for reuse.
    """

    mapping: Dict[str, ModuleType] = {}
    # Prefer explicit keyword names (torch_module etc.) but allow direct names.
    candidate_torch = modules.pop("torch", None)
    if candidate_torch is not None:
        torch_module = candidate_torch
    candidate_numpy = modules.pop("numpy", None)
    if candidate_numpy is not None:
        numpy_module = candidate_numpy
    candidate_pandas = modules.pop("pandas", None)
    if candidate_pandas is not None:
        pandas_module = candidate_pandas

    if torch_module is not None:
        mapping["torch"] = torch_module
    if numpy_module is not None:
        mapping["numpy"] = numpy_module
    if pandas_module is not None:
        mapping["pandas"] = pandas_module
    for name, module in list(modules.items()):
        if module is not None:
            mapping[name] = module

    if not mapping:
        return

    with _LOCK:
        for name, module in mapping.items():
            _MODULES[name] = module
    for name, module in mapping.items():
        sys.modules.setdefault(name, module)
        _notify_observers(name, module)


def _resolve_module(
    name: str,
    *,
    import_name: Optional[str] = None,
    import_if_missing: bool = True,
) -> ModuleType:
    with _LOCK:
        module = _MODULES.get(name)
    if module is not None:
        return module
    if not import_if_missing:
        raise RuntimeError(f"{name} has not been injected. Call setup_imports first.")
    module = import_module(import_name or name)
    setup_imports(**{name: module})
    return module


def resolve_torch(import_if_missing: bool = True) -> ModuleType:
    """
    Return the injected torch module, importing lazily when running locally.
    """

    return _resolve_module("torch", import_if_missing=import_if_missing)


def resolve_numpy(import_if_missing: bool = True) -> ModuleType:
    """
    Return the injected numpy module, importing lazily when running locally.
    """

    return _resolve_module("numpy", import_if_missing=import_if_missing)


def resolve_pandas(import_if_missing: bool = True) -> ModuleType:
    """
    Return the injected pandas module, importing lazily when running locally.
    """

    return _resolve_module("pandas", import_if_missing=import_if_missing)


def injected_modules() -> Dict[str, ModuleType]:
    """
    Return a snapshot of the currently injected module mapping.
    """

    with _LOCK:
        return dict(_MODULES)


def register_observer(name: str, observer: Callable[[ModuleType], None]) -> None:
    """
    Register a callback invoked whenever the named dependency is injected.
    """

    with _LOCK:
        observers = _OBSERVERS.setdefault(name, [])
        observers.append(observer)
        current = _MODULES.get(name)
    if current is not None:
        try:
            observer(current)
        except Exception:
            pass


def _notify_observers(name: str, module: ModuleType) -> None:
    if name not in _OBSERVERS:
        return
    with _LOCK:
        observers = list(_OBSERVERS[name])
    for observer in observers:
        try:
            observer(module)
        except Exception:
            pass


def _reset_for_tests() -> None:  # pragma: no cover - exercised in tests
    with _LOCK:
        _MODULES.clear()
        _OBSERVERS.clear()
