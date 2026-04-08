from __future__ import annotations

import math
from typing import Any


_EPSILON = 1e-9
_MIN_UNEXPECTED_POSITION_VALUE_USD = 1.0


def normalize_symbols(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_items = value.replace(",", " ").split()
    elif isinstance(value, (list, tuple)):
        raw_items = [str(item) for item in value]
    else:
        return []
    return [item.strip().upper() for item in raw_items if item and item.strip()]


def safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def detail_has_activity(detail: dict[str, Any]) -> bool:
    if detail.get("actions"):
        return True
    for key in ("allocation_pct", "target_value"):
        numeric = safe_float(detail.get(key))
        if numeric is not None and abs(numeric) > _EPSILON:
            return True
    current_value = safe_float(detail.get("current_value"))
    if current_value is not None and abs(current_value) >= _MIN_UNEXPECTED_POSITION_VALUE_USD:
        return True
    if current_value is None:
        current_qty = safe_float(detail.get("current_qty"))
        if current_qty is not None and abs(current_qty) > _EPSILON:
            return True
    return False


def snapshot_allowed_market_symbols(
    snapshot: dict[str, Any],
    allowed_symbols: list[str],
) -> set[str]:
    allowed: set[str] = set()
    for detail in snapshot.get("symbols_detail") or []:
        if not isinstance(detail, dict):
            continue
        symbol = str(detail.get("symbol") or "").strip().upper()
        market_symbol = str(detail.get("market_symbol") or "").strip().upper()
        if symbol in allowed_symbols and market_symbol:
            allowed.add(market_symbol)
    return allowed


def iter_snapshot_order_symbols(snapshot: dict[str, Any]) -> list[str]:
    orders = snapshot.get("orders")
    if not isinstance(orders, dict):
        return []
    symbols: list[str] = []
    for bucket in orders.values():
        if not isinstance(bucket, list):
            continue
        for order in bucket:
            if not isinstance(order, dict):
                continue
            symbol = str(order.get("symbol") or "").strip().upper()
            if symbol:
                symbols.append(symbol)
    return symbols


def find_unexpected_snapshot_activity(
    snapshot: dict[str, Any],
    allowed_symbols: list[str],
) -> tuple[list[str], list[str]]:
    unexpected_symbols: set[str] = set()
    for detail in snapshot.get("symbols_detail") or []:
        if not isinstance(detail, dict):
            continue
        symbol = str(detail.get("symbol") or "").strip().upper()
        if symbol and symbol not in allowed_symbols and detail_has_activity(detail):
            unexpected_symbols.add(symbol)

    unexpected_order_symbols: set[str] = set()
    allowed_market_symbols = snapshot_allowed_market_symbols(snapshot, allowed_symbols)
    if allowed_market_symbols:
        for order_symbol in iter_snapshot_order_symbols(snapshot):
            if order_symbol not in allowed_market_symbols:
                unexpected_order_symbols.add(order_symbol)

    return sorted(unexpected_symbols), sorted(unexpected_order_symbols)
