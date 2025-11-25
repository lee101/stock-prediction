from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable, Tuple


def _parse_iso_timestamp(value: str | None) -> datetime | None:
    """Best-effort ISO timestamp parser that always returns an aware datetime."""
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(value)
    except Exception:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def _sum_order_qty(orders: Iterable[object]) -> float:
    total = 0.0
    for order in orders:
        try:
            total += abs(float(getattr(order, "qty", 0.0) or 0.0))
        except Exception:
            continue
    return total


def _effective_entry_quantities(
    *,
    status: dict,
    current_qty: float,
    open_order_qty: float,
    target_qty: float,
    now: datetime,
    pending_ttl_seconds: int,
) -> Tuple[float, float, float, bool]:
    """
    Compute effective exposure for an entry watcher, accounting for:
      * current filled position
      * open orders on the same side
      * recently submitted orders (pending_qty) that may not yet reflect in positions

    Returns:
        effective_qty: current_qty + open_order_qty + pending_qty_adjusted
        remaining_qty: max(target_qty - effective_qty, 0)
        pending_qty: updated pending reservations after decay/fill reconciliation
        target_reached: True if effective_qty is within 1% of target
    """
    target = max(float(target_qty), 0.0)
    position_qty = max(float(current_qty), 0.0)
    pending_qty = max(float(status.get("pending_qty", 0.0) or 0.0), 0.0)

    # Expire stale pending reservations
    pending_expiry = _parse_iso_timestamp(status.get("pending_expires_at"))
    if pending_expiry and now >= pending_expiry:
        pending_qty = 0.0

    # If no open orders but we have pending_qty, the order was likely canceled externally
    # Clear pending to allow re-ordering
    if open_order_qty == 0 and pending_qty > 0:
        pending_qty = 0.0

    # Reconcile pending with newly observed fills
    prev_position = max(float(status.get("position_qty", 0.0) or 0.0), 0.0)
    filled_delta = max(position_qty - prev_position, 0.0)
    if filled_delta > 0:
        pending_qty = max(pending_qty - filled_delta, 0.0)

    effective_qty = position_qty + open_order_qty + pending_qty
    remaining_qty = max(target - effective_qty, 0.0)
    target_reached = effective_qty >= target * 0.99 if target > 0 else False

    # Refresh pending expiry when we still carry reservations
    if pending_qty > 0:
        status["pending_expires_at"] = (now + timedelta(seconds=pending_ttl_seconds)).isoformat()

    status["pending_qty"] = pending_qty
    status["effective_qty"] = effective_qty
    status["remaining_qty"] = remaining_qty

    return effective_qty, remaining_qty, pending_qty, target_reached
