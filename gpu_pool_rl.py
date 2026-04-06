from __future__ import annotations

"""Compatibility shim for the GPU pool scheduler module."""

import runpy
import sys

from pufferlib_market import gpu_pool_rl as _impl


if __name__ == "__main__":
    runpy.run_module("pufferlib_market.gpu_pool_rl", run_name="__main__")
else:
    sys.modules[__name__] = _impl
