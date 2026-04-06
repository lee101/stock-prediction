"""Simple JSON type aliases used across the trading server protocol."""

from __future__ import annotations

from typing import Any


JsonObject = dict[str, Any]

__all__ = ["JsonObject"]
