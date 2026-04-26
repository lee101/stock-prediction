"""Compatibility alias for :mod:`src.market_sim_early_exit`.

The implementation lives in ``src.market_sim_early_exit``.  Keeping a root copy
risks drifting the simulator early-exit gates or monkeypatching a separate
module instance in legacy imports.
"""

from __future__ import annotations

import importlib
import sys

_impl = importlib.import_module("src.market_sim_early_exit")
__all__ = list(getattr(_impl, "__all__", [name for name in dir(_impl) if not name.startswith("_")]))
globals().update({name: getattr(_impl, name) for name in __all__})

if __name__ not in {"__main__", "<run_path>"}:
    sys.modules[__name__] = _impl
