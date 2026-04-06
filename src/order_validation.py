"""Order validation helpers shared by the trading-server and clients."""

from __future__ import annotations

import json
import math
from typing import Any


MAX_ORDER_METADATA_ITEMS: int = 32
MAX_ORDER_METADATA_BYTES: int = 4096


def normalize_order_metadata(
    value: object,
    *,
    metadata_label: str = "metadata",
    max_items: int = MAX_ORDER_METADATA_ITEMS,
    max_bytes: int = MAX_ORDER_METADATA_BYTES,
) -> dict[str, Any]:
    """Validate and normalise an order metadata payload.

    Returns a plain ``dict[str, Any]`` that is safe to JSON-serialise.
    Raises :class:`ValueError` when *value* is invalid.
    """
    if value is None:
        return {}

    if not isinstance(value, dict):
        raise ValueError(f"{metadata_label} must be a dict, got {type(value).__name__}")

    if len(value) > max_items:
        raise ValueError(f"{metadata_label} has {len(value)} items, max is {max_items}")

    # Ensure the dict is JSON-round-trippable.
    try:
        encoded = json.dumps(value, allow_nan=False, default=str)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{metadata_label} is not JSON-serialisable: {exc}") from exc

    if len(encoded.encode()) > max_bytes:
        raise ValueError(f"{metadata_label} exceeds {max_bytes} bytes when JSON-encoded")

    return dict(value)


def normalize_positive_finite_float(
    value: float | int | str,
    *,
    field_name: str = "value",
) -> float:
    """Return *value* as a positive finite ``float``.

    Raises :class:`ValueError` for non-numeric, non-positive, or
    non-finite inputs.
    """
    try:
        fval = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric, got {value!r}") from exc

    if not math.isfinite(fval):
        raise ValueError(f"{field_name} must be finite, got {fval}")
    if fval <= 0:
        raise ValueError(f"{field_name} must be positive, got {fval}")
    return fval


def safe_json_float(value: float | int | Any) -> float:
    """Convert *value* to a float that is safe for JSON serialisation.

    Non-finite values (``inf``, ``-inf``, ``nan``) are replaced with ``0.0``
    so that ``json.dumps`` never raises.
    """
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(fval):
        return 0.0
    return fval
