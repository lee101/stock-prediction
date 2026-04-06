from __future__ import annotations

import math

import pytest

from src.order_validation import (
    MAX_ORDER_METADATA_BYTES,
    MAX_ORDER_METADATA_ITEMS,
    normalize_order_metadata,
    normalize_positive_finite_float,
    safe_json_float,
)


def test_normalize_positive_finite_float_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="qty must be a positive finite number"):
        normalize_positive_finite_float(math.inf, field_name="qty")


def test_normalize_order_metadata_enforces_explicit_limits() -> None:
    with pytest.raises(
        ValueError,
        match=rf"metadata may contain at most {MAX_ORDER_METADATA_ITEMS} entries",
    ):
        normalize_order_metadata({f"k{i}": i for i in range(MAX_ORDER_METADATA_ITEMS + 1)})

    oversized = {"payload": "x" * (MAX_ORDER_METADATA_BYTES + 128)}
    with pytest.raises(
        ValueError,
        match=rf"metadata exceeds {MAX_ORDER_METADATA_BYTES} bytes when serialized",
    ):
        normalize_order_metadata(oversized)


def test_normalize_order_metadata_supports_custom_label() -> None:
    with pytest.raises(ValueError, match="trading server metadata must be a JSON object"):
        normalize_order_metadata([], metadata_label="trading server metadata")


def test_safe_json_float_filters_non_finite_values() -> None:
    assert safe_json_float("1.5") == 1.5
    assert safe_json_float(math.nan) is None


def test_normalize_order_metadata_preserves_nested_json_shape() -> None:
    metadata = {
        "nested": {"confidence": 0.5, "tags": ["alpha", "beta"]},
        "active": True,
    }

    assert normalize_order_metadata(metadata) == metadata
