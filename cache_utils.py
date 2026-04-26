"""Compatibility shim for the canonical cache utilities module."""

from __future__ import annotations

import importlib
import sys


_impl = importlib.import_module("src.cache_utils")
sys.modules[__name__] = _impl
