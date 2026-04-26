"""Compatibility alias for :mod:`pufferlib_market.hourly_replay`.

The implementation lives in ``pufferlib_market.hourly_replay``.  Keeping a
second copy at repo root caused simulator API and risk-guard behavior to drift,
so this module aliases the package implementation for legacy imports.
"""

from __future__ import annotations

import importlib
import sys

_impl = importlib.import_module("pufferlib_market.hourly_replay")
__all__ = list(getattr(_impl, "__all__", [name for name in dir(_impl) if not name.startswith("_")]))
globals().update({name: getattr(_impl, name) for name in __all__})

if __name__ not in {"__main__", "<run_path>"}:
    sys.modules[__name__] = _impl
