from __future__ import annotations

from datetime import datetime
import math
from typing import Optional, Sequence

import pandas as pd

from src.hourly_trader_utils import OrderIntent, infer_working_order_kind

DEFAULT_PRICE_TOL_PCT = 0.0003
DEFAULT_QTY_TOL_PCT = 0.05
DEFAULT_QTY_TOL_NOTIONAL_USD = 100.0


def coerce_order_timestamp(value: object) -> Optional[datetime]:
    if value is None or value == "":
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).to_pydatetime()


def order_created_at(order: object) -> Optional[datetime]:
    for field in ("created_at", "submitted_at", "updated_at"):
        dt = coerce_order_timestamp(getattr(order, field, None))
        if dt is not None:
            return dt
    return None


def desired_order_keys(intents: Sequence[OrderIntent]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for intent in intents:
        if float(intent.qty) <= 0.0:
            continue
        keys.add((str(intent.kind).lower(), str(intent.side).lower()))
    return keys


def _order_qty(order: object) -> float:
    try:
        qty = float(getattr(order, "qty", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return qty if math.isfinite(qty) and qty > 0.0 else 0.0


def _order_limit_price(order: object) -> Optional[float]:
    raw = getattr(order, "limit_price", None)
    if raw in (None, ""):
        return None
    try:
        price = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(price) or price <= 0.0:
        return None
    return price


def order_matches_intent(
    order: object,
    intent: OrderIntent,
    *,
    price_tol_pct: float = DEFAULT_PRICE_TOL_PCT,
    qty_tol_pct: float = DEFAULT_QTY_TOL_PCT,
    qty_tol_notional_usd: float = DEFAULT_QTY_TOL_NOTIONAL_USD,
) -> bool:
    existing_price = _order_limit_price(order)
    desired_price = float(intent.limit_price)
    if existing_price is None or not math.isfinite(desired_price) or desired_price <= 0.0:
        return False

    price_diff_pct = abs(existing_price - desired_price) / desired_price
    if price_diff_pct >= float(price_tol_pct):
        return False

    existing_qty = _order_qty(order)
    desired_qty = float(intent.qty)
    if existing_qty <= 0.0 or not math.isfinite(desired_qty) or desired_qty <= 0.0:
        return False

    qty_diff_pct = abs(existing_qty - desired_qty) / desired_qty
    notional_diff = abs(existing_qty - desired_qty) * desired_price
    return (qty_diff_pct < float(qty_tol_pct)) or (notional_diff < float(qty_tol_notional_usd))


def orders_to_cancel_for_live_symbol(
    orders: Sequence[object],
    *,
    position_qty: float,
    intents: Sequence[OrderIntent],
    price_tol_pct: float = DEFAULT_PRICE_TOL_PCT,
    qty_tol_pct: float = DEFAULT_QTY_TOL_PCT,
    qty_tol_notional_usd: float = DEFAULT_QTY_TOL_NOTIONAL_USD,
) -> list[tuple[object, str]]:
    desired_by_key: dict[tuple[str, str], list[OrderIntent]] = {}
    unmatched_by_key: dict[tuple[str, str], list[OrderIntent]] = {}
    for intent in intents:
        if float(intent.qty) <= 0.0:
            continue
        key = (str(intent.kind).lower(), str(intent.side).lower())
        desired_by_key.setdefault(key, []).append(intent)
        unmatched_by_key.setdefault(key, []).append(intent)

    to_cancel: list[tuple[object, str]] = []

    def _sort_key(order: object) -> tuple[bool, float]:
        created_at = order_created_at(order)
        if created_at is None:
            return (False, 0.0)
        return (True, created_at.timestamp())

    for order in sorted(orders, key=_sort_key, reverse=True):
        side = str(getattr(order, "side", "") or "").lower()
        key = (infer_working_order_kind(side=side, position_qty=float(position_qty)), side)
        desired_intents = desired_by_key.get(key)
        if not desired_intents:
            to_cancel.append((order, "no_matching_intent"))
            continue

        remaining_intents = unmatched_by_key.get(key, [])
        matched_idx = next(
            (
                idx
                for idx, intent in enumerate(remaining_intents)
                if order_matches_intent(
                    order,
                    intent,
                    price_tol_pct=price_tol_pct,
                    qty_tol_pct=qty_tol_pct,
                    qty_tol_notional_usd=qty_tol_notional_usd,
                )
            ),
            None,
        )
        if matched_idx is not None:
            remaining_intents.pop(matched_idx)
            continue

        if any(
            order_matches_intent(
                order,
                intent,
                price_tol_pct=price_tol_pct,
                qty_tol_pct=qty_tol_pct,
                qty_tol_notional_usd=qty_tol_notional_usd,
            )
            for intent in desired_intents
        ):
            to_cancel.append((order, "duplicate_working_order"))
            continue

        to_cancel.append((order, "stale_working_order"))

    return to_cancel


__all__ = [
    "coerce_order_timestamp",
    "desired_order_keys",
    "order_matches_intent",
    "order_created_at",
    "orders_to_cancel_for_live_symbol",
]
