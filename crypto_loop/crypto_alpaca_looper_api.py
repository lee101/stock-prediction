"""Compatibility shim for the canonical crypto Alpaca looper API module."""

from __future__ import annotations

import importlib
import sys


_impl = importlib.import_module("src.crypto_loop.crypto_alpaca_looper_api")
sys.modules[__name__] = _impl
