"""Compatibility shim for the canonical train_calibrator module."""

from __future__ import annotations

import importlib
import sys


_impl = importlib.import_module("rl_trading_agent_binance.train_calibrator")
sys.modules[__name__] = _impl
