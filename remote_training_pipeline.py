"""Compatibility shim for the canonical remote training pipeline module."""

from __future__ import annotations

import importlib
import sys


_impl = importlib.import_module("src.remote_training_pipeline")
sys.modules[__name__] = _impl
