from __future__ import annotations

import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable, Iterable, Sequence

from loguru import logger

import alpaca_wrapper
from src.symbol_utils import is_crypto_symbol

POLL_SECONDS = max(float(os.getenv("CANCEL_MULTI_ORDERS_POLL_SECONDS", "30")), 1.0)


def _normalize_symbol(symbol: object) -> str:
    return str(symbol or "").replace("/", "").replace("-", "").upper()


def _normalize_side(side: object) -> str:
    raw = getattr(side, "value", side)
    return str(raw or "").strip().lower()


def _coerce_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _order_created_at(order: object) -> datetime:
    for field in ("created_at", "submitted_at", "updated_at"):
        value = getattr(order, field, None)
        if value is None:
            continue
        if isinstance(value, datetime):
            return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        try:
            parsed = datetime.fromisoformat(str(value))
        except ValueError:
            continue
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    return datetime.min.replace(tzinfo=timezone.utc)


def _position_is_flat(position: object) -> bool:
    symbol = _normalize_symbol(getattr(position, "symbol", ""))
    qty = abs(_coerce_float(getattr(position, "qty", 0.0)))
    price = _coerce_float(getattr(position, "current_price", 0.0))
    notional = qty * price
    if is_crypto_symbol(symbol):
        return qty < 1e-8 or notional < 1.0
    return qty < 1.0 and notional < 1.0


def cancel_duplicate_opening_orders(
    orders: Sequence[object],
    positions: Sequence[object],
    *,
    cancel_order_fn: Callable[[object], object],
) -> list[str]:
    positions_by_symbol = {
        _normalize_symbol(getattr(position, "symbol", "")): position
        for position in positions
    }
    duplicate_groups: dict[tuple[str, str], list[object]] = defaultdict(list)

    for order in orders:
        symbol = _normalize_symbol(getattr(order, "symbol", ""))
        if not symbol:
            continue
        position = positions_by_symbol.get(symbol)
        if position is not None and not _position_is_flat(position):
            continue
        side = _normalize_side(getattr(order, "side", ""))
        if side not in {"buy", "sell"}:
            continue
        duplicate_groups[(symbol, side)].append(order)

    cancelled_ids: list[str] = []
    for (_symbol, _side), grouped_orders in duplicate_groups.items():
        if len(grouped_orders) <= 1:
            continue
        grouped_orders = sorted(grouped_orders, key=_order_created_at)
        for order in grouped_orders[:-1]:
            cancel_order_fn(order)
            cancelled_ids.append(str(getattr(order, "id", "")))

    return cancelled_ids


def run_once() -> list[str]:
    orders = list(alpaca_wrapper.get_orders())
    positions = list(alpaca_wrapper.get_all_positions())
    return cancel_duplicate_opening_orders(
        orders,
        positions,
        cancel_order_fn=alpaca_wrapper.cancel_order,
    )


def main() -> None:
    logger.info("Starting duplicate opening-order guard loop (poll={}s)", POLL_SECONDS)
    while True:
        try:
            cancelled = run_once()
            if cancelled:
                logger.info("Cancelled duplicate opening orders: {}", cancelled)
        except Exception as exc:
            logger.exception("Duplicate opening-order guard failed: {}", exc)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
