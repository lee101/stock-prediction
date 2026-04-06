"""Compatibility shim for the canonical crypto order loop server module."""

from __future__ import annotations

import importlib
import sys


_impl = importlib.import_module("src.crypto_loop.crypto_order_loop_server")
sys.modules[__name__] = _impl
