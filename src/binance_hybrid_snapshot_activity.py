from __future__ import annotations

import math
from typing import Any


_EPSILON = 1e-9
_MIN_UNEXPECTED_POSITION_VALUE_USD = 1.0


def normalize_symbols(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_items = value.replace(",", " ").split()
    elif isinstance(value, (list, tuple)):
        raw_items = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                raw_items.append(text)
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


def iter_snapshot_active_orders(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    orders = snapshot.get("orders")
    if not isinstance(orders, dict):
        return []

    active_orders: list[dict[str, Any]] = []
    used_target_buckets = False
    for bucket_name in ("open_after_cleanup", "placed"):
        bucket = orders.get(bucket_name)
        if not isinstance(bucket, list):
            continue
        used_target_buckets = True
        for order in bucket:
            if isinstance(order, dict):
                active_orders.append(order)

    if used_target_buckets:
        return active_orders

    for bucket in orders.values():
        if not isinstance(bucket, list):
            continue
        for order in bucket:
            if isinstance(order, dict):
                active_orders.append(order)
    return active_orders


def find_missing_exit_order_coverage(
    snapshot: dict[str, Any],
    allowed_symbols: list[str],
    *,
    min_position_value_usd: float = 12.0,
) -> tuple[list[str], list[str]]:
    sell_qty_by_market_symbol: dict[str, float] = {}
    for order in iter_snapshot_active_orders(snapshot):
        symbol = str(order.get("symbol") or "").strip().upper()
        side = str(order.get("side") or "").strip().upper()
        if not symbol or side != "SELL":
            continue
        orig_qty = safe_float(order.get("origQty"))
        if orig_qty is None:
            orig_qty = safe_float(order.get("qty"))
        if orig_qty is None:
            orig_qty = safe_float(order.get("orig_qty"))
        executed_qty = safe_float(order.get("executedQty"))
        if executed_qty is None:
            executed_qty = safe_float(order.get("executed_qty"))
        executed_qty = executed_qty or 0.0
        remaining_qty = max(0.0, float(orig_qty or 0.0) - float(executed_qty))
        if remaining_qty <= _EPSILON:
            continue
        sell_qty_by_market_symbol[symbol] = sell_qty_by_market_symbol.get(symbol, 0.0) + remaining_qty

    missing_exit_price_symbols: set[str] = set()
    missing_exit_order_symbols: set[str] = set()
    for detail in snapshot.get("symbols_detail") or []:
        if not isinstance(detail, dict):
            continue
        symbol = str(detail.get("symbol") or "").strip().upper()
        if not symbol or symbol not in allowed_symbols:
            continue
        current_qty = safe_float(detail.get("current_qty")) or 0.0
        current_value = safe_float(detail.get("current_value")) or 0.0
        if current_qty <= _EPSILON or current_value < float(min_position_value_usd):
            continue
        exit_price = safe_float(detail.get("exit_price"))
        if exit_price is None or exit_price <= 0.0:
            missing_exit_price_symbols.add(symbol)
            continue
        market_symbol = str(detail.get("market_symbol") or "").strip().upper()
        if not market_symbol:
            missing_exit_order_symbols.add(symbol)
            continue
        covered_qty = sell_qty_by_market_symbol.get(market_symbol, 0.0)
        uncovered_qty = max(0.0, current_qty - covered_qty)
        current_price = current_value / current_qty if current_qty > _EPSILON else 0.0
        uncovered_value = uncovered_qty * current_price
        if uncovered_qty > current_qty * 0.02 and uncovered_value >= float(min_position_value_usd):
            missing_exit_order_symbols.add(symbol)

    return sorted(missing_exit_price_symbols), sorted(missing_exit_order_symbols)


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
