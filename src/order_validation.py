from __future__ import annotations

import json
import math
from typing import cast

from src.json_types import JsonObject


MAX_ORDER_METADATA_ITEMS = 32
MAX_ORDER_METADATA_BYTES = 4096


def normalize_positive_finite_float(value: object, *, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive finite number") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite number")
    return parsed


def safe_json_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def normalize_order_metadata(
    metadata: object,
    *,
    metadata_label: str = "metadata",
    max_items: int = MAX_ORDER_METADATA_ITEMS,
    max_bytes: int = MAX_ORDER_METADATA_BYTES,
) -> JsonObject:
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise ValueError(f"{metadata_label} must be a JSON object")
    item_count = len(metadata)
    if item_count > max_items:
        raise ValueError(
            f"{metadata_label} may contain at most {max_items} entries "
            f"(got {item_count})"
        )
    if any(not isinstance(key, str) for key in metadata):
        raise ValueError(f"{metadata_label} keys must be strings")
    try:
        encoded = json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{metadata_label} must be JSON-serializable without NaN or Infinity values"
        ) from exc
    encoded_size = len(encoded.encode("utf-8"))
    if encoded_size > max_bytes:
        raise ValueError(
            f"{metadata_label} exceeds {max_bytes} bytes when serialized "
            f"(got {encoded_size} bytes)"
        )
    normalized = json.loads(encoded)
    if not isinstance(normalized, dict):
        raise ValueError(f"{metadata_label} must be a JSON object")
    return cast(JsonObject, normalized)
