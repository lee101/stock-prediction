"""Utility functions for comparing trading-related values."""

from __future__ import annotations

from typing import Any


def normalize_side(side: Any) -> str:
    """Normalize buy/sell/long/short-like inputs (including Enum objects)."""
    if side is None:
        return ""
    if isinstance(side, str):
        raw = side.strip().lower()
        if "." in raw:
            raw = raw.rsplit(".", 1)[-1]
        return raw

    value = getattr(side, "value", None)
    if isinstance(value, str) and value.strip():
        raw = value.strip().lower()
        if "." in raw:
            raw = raw.rsplit(".", 1)[-1]
        return raw

    name = getattr(side, "name", None)
    if isinstance(name, str) and name.strip():
        raw = name.strip().lower()
        if "." in raw:
            raw = raw.rsplit(".", 1)[-1]
        return raw

    raw = str(side).strip().lower()
    if "." in raw:
        raw = raw.rsplit(".", 1)[-1]
    return raw


def is_same_side(side1: Any, side2: Any) -> bool:
    """
    Compare position sides accounting for different nomenclature.
    Handles 'buy'/'long' and 'sell'/'short' equivalence.

    Args:
        side1: First position side
        side2: Second position side
    Returns:
        bool: True if sides are equivalent
    """
    buy_variants = {'buy', 'long'}
    sell_variants = {'sell', 'short'}

    side1 = normalize_side(side1)
    side2 = normalize_side(side2)

    if side1 in buy_variants and side2 in buy_variants:
        return True
    if side1 in sell_variants and side2 in sell_variants:
        return True
    return False


def is_buy_side(side: Any) -> bool:
    return normalize_side(side) in {'buy', 'long'}


def is_sell_side(side: Any) -> bool:
    return normalize_side(side) in {'sell', 'short'}
