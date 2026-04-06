from __future__ import annotations

"""Compatibility shim for the RunPod client helpers module."""

import runpy
import sys

from src import runpod_client as _impl


if __name__ == "__main__":
    runpy.run_module("src.runpod_client", run_name="__main__")
else:
    sys.modules[__name__] = _impl
