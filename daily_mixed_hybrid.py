"""Compatibility shim for the canonical mixed daily hybrid module."""

from __future__ import annotations

import importlib
import sys


_impl = importlib.import_module("src.daily_mixed_hybrid")
sys.modules[__name__] = _impl
