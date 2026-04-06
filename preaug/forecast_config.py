from __future__ import annotations

"""Compatibility shim for the preaug forecast-config helpers."""

import sys

from src.preaug import forecast_config as _impl


sys.modules[__name__] = _impl
