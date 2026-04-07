"""Python bindings for the C++ market simulator.

This module exposes :class:`MarketEnvironment` from the underlying
``_market_sim_py_ext`` pybind11 extension. It supports both SCALAR and DPS
action modes, with the action dimensionality reported at runtime via
``get_action_dim()``.
"""

from __future__ import annotations

import ctypes
import os
import sys

import torch  # noqa: F401  (libtorch must be loaded before the ext)

# Preload libmarket_sim.so so the extension below can resolve its symbols.
# RPATH (set in setup.py) handles this for normal installs, but an explicit
# RTLD_GLOBAL dlopen makes editable / out-of-tree builds robust too.
_BUILD_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "build")
)
_libpath = os.path.join(_BUILD_DIR, "libmarket_sim.so")
if os.path.isfile(_libpath):
    try:
        ctypes.CDLL(_libpath, mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:  # pragma: no cover - defensive
        print(f"warning: failed to preload {_libpath}: {exc}", file=sys.stderr)

from . import _market_sim_py_ext as _ext  # type: ignore  # noqa: E402

MarketEnvironment = _ext.MarketEnvironment

__all__ = ["MarketEnvironment"]
