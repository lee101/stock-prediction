from __future__ import annotations

"""
Shared helpers for injecting heavyweight numerical dependencies (torch, numpy,
pandas, …) when running inside fal workers.

FAL applications can call :func:`setup_imports` during ``App.setup`` to supply
pre-imported modules that were resolved inside the worker process. The helper
stores the modules locally and attempts to register them with
``faltrain.dependencies`` so other packages that rely on the fal registry see
the same instances.

Code that wishes to access these libraries should call ``resolve_torch()``,
``resolve_numpy()``, or ``resolve_pandas()`` instead of importing directly. The
resolvers first prefer the injected module, falling back to a regular import
when used outside of fal.
"""

from importlib import import_module
from threading import RLock
from types import ModuleType
from typing import Callable, Dict, List, Optional

_LOCK = RLock()
_TORCH: Optional[ModuleType] = None
_NUMPY: Optional[ModuleType] = None
_PANDAS: Optional[ModuleType] = None
_OBSERVERS: Dict[str, List[Callable[[ModuleType], None]]] = {
    "torch": [],
    "numpy": [],
    "pandas": [],
}


def _register_with_fal(mapping: Dict[str, ModuleType]) -> None:
    """
    Register injected modules with faltrain.dependencies when available.

    The helper intentionally swallows import errors so the repo continues to
    function when faltrain is not installed (e.g. local scripts/tests).
    """

    if not mapping:
        return
    try:
        from faltrain.dependencies import bulk_register_fal_dependencies
    except Exception:
        return

    try:
        bulk_register_fal_dependencies(mapping)
    except Exception:
        # Fal registry registration is best effort; ignore failures so injection
        # never prevents local execution.
        pass


def setup_imports(
    torch_module: Optional[ModuleType] = None,
    numpy_module: Optional[ModuleType] = None,
    pandas_module: Optional[ModuleType] = None,
) -> None:
    """
    Record modules supplied by the fal runtime and register them for reuse.
    """

    mapping: Dict[str, ModuleType] = {}
    with _LOCK:
        global _TORCH, _NUMPY, _PANDAS

        if torch_module is not None:
            _TORCH = torch_module
            mapping["torch"] = torch_module
        if numpy_module is not None:
            _NUMPY = numpy_module
            mapping["numpy"] = numpy_module
        if pandas_module is not None:
            _PANDAS = pandas_module
            mapping["pandas"] = pandas_module

    _register_with_fal(mapping)
    for name, module in mapping.items():
        _notify_observers(name, module)


def resolve_torch(import_if_missing: bool = True) -> ModuleType:
    """
    Return the injected torch module, importing lazily when running locally.
    """

    with _LOCK:
        module = _TORCH
    if module is not None:
        return module
    if not import_if_missing:
        raise RuntimeError("Torch has not been injected. Call setup_imports first.")

    module = import_module("torch")
    setup_imports(torch_module=module)
    return module


def resolve_numpy(import_if_missing: bool = True) -> ModuleType:
    """
    Return the injected numpy module, importing lazily when running locally.
    """

    with _LOCK:
        module = _NUMPY
    if module is not None:
        return module
    if not import_if_missing:
        raise RuntimeError("NumPy has not been injected. Call setup_imports first.")

    module = import_module("numpy")
    setup_imports(numpy_module=module)
    return module


def resolve_pandas(import_if_missing: bool = True) -> ModuleType:
    """
    Return the injected pandas module, importing lazily when running locally.
    """

    with _LOCK:
        module = _PANDAS
    if module is not None:
        return module
    if not import_if_missing:
        raise RuntimeError("Pandas has not been injected. Call setup_imports first.")

    module = import_module("pandas")
    setup_imports(pandas_module=module)
    return module


def injected_modules() -> Dict[str, ModuleType]:
    """
    Return a snapshot of the currently injected module mapping.
    """

    with _LOCK:
        mapping = {}
        if _TORCH is not None:
            mapping["torch"] = _TORCH
        if _NUMPY is not None:
            mapping["numpy"] = _NUMPY
        if _PANDAS is not None:
            mapping["pandas"] = _PANDAS
    return mapping


def register_observer(name: str, observer: Callable[[ModuleType], None]) -> None:
    """
    Register a callback invoked whenever the named dependency is injected.
    """

    if name not in _OBSERVERS:
        raise ValueError(f"Unsupported dependency name: {name!r}")

    current: Optional[ModuleType]
    with _LOCK:
        _OBSERVERS[name].append(observer)
        current = _current_module(name)

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


def _current_module(name: str) -> Optional[ModuleType]:
    if name == "torch":
        return _TORCH
    if name == "numpy":
        return _NUMPY
    if name == "pandas":
        return _PANDAS
    return None
