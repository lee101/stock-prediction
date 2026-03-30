"""PufferLib C market trading environment with Chronos2 forecast integration."""

from __future__ import annotations

import importlib
import sys

from . import binding_fallback as _binding_fallback


def _ensure_binding():
    fullname = __name__ + ".binding"
    module = sys.modules.get(fullname)
    if module is not None:
        return module
    try:
        importlib.import_module(fullname)
    except Exception as exc:  # pragma: no cover - exercised in numpy/ABI mismatch envs
        _binding_fallback.BINDING_IMPORT_ERROR = exc
        sys.modules[fullname] = _binding_fallback
    return sys.modules[fullname]


def __getattr__(name: str):
    if name == "binding":
        return _ensure_binding()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_ensure_binding()

__all__ = ["binding"]
