from __future__ import annotations

"""Compatibility shim for the shared cache helpers module."""

import importlib
import sys

_impl = importlib.import_module("src.cache")
sys.modules[__name__] = _impl
