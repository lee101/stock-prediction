"""Pure helpers shared by the hourly live trader and its simulators.

Keep this module free of broker/network dependencies so it can be imported in
offline backtests without requiring Alpaca credentials.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Optional, Tuple


@dataclass
class TradingPlan:
    symbol: str
    buy_price: float
    sell_price: float
    buy_amount: float
    sell_amount: float
    timestamp: datetime


@dataclass(frozen=True)
class OrderIntent:
    side: str  # "buy" or "sell"
    qty: float
    limit_price: float
    kind: str  # "entry" or "exit"


def build_plan_from_action(action: dict, *, intensity_scale: float) -> TradingPlan:
    buy_amount = max(0.0, min(100.0, float(action["buy_amount"]) * float(intensity_scale)))
    sell_amount = max(0.0, min(100.0, float(action["sell_amount"]) * float(intensity_scale)))
    return TradingPlan(
        symbol=str(action["symbol"]).upper(),
        buy_price=float(action["buy_price"]),
        sell_price=float(action["sell_price"]),
        buy_amount=buy_amount,
        sell_amount=sell_amount,
        timestamp=action["timestamp"],
    )


def directional_entry_amount(action: Mapping[str, Any] | object, *, is_short: bool) -> float:
    """Return side-aware entry amount from model action payload.

    Long entries should use `buy_amount`; short entries should use `sell_amount`.
    `trade_amount` is used as a fallback for legacy payloads.
    """

    primary = "sell_amount" if is_short else "buy_amount"
    secondary = "trade_amount"
    tertiary = "buy_amount" if is_short else "sell_amount"
    for key in (primary, secondary, tertiary):
        value = _read_optional_numeric_field(action, key)
        if value is not None:
            return max(0.0, value)
    return 0.0


def entry_intensity_fraction(
    action: Mapping[str, Any] | object,
    *,
    is_short: bool,
    trade_amount_scale: float,
    intensity_power: float = 1.0,
    min_intensity_fraction: float = 0.0,
    side_multiplier: float = 1.0,
) -> tuple[float, float]:
    """Return `(signal_amount, intensity_fraction)` for position sizing."""

    amount = directional_entry_amount(action, is_short=is_short)
    scale = float(trade_amount_scale)
    if scale <= 0:
        return amount, 0.0
    raw_fraction = min(max(amount / scale, 0.0), 1.0)

    power = float(intensity_power)
    if power <= 0:
        adjusted = raw_fraction
    elif power == 1.0:
        adjusted = raw_fraction
    else:
        adjusted = raw_fraction**power

    adjusted *= max(float(side_multiplier), 0.0)
    if raw_fraction > 0:
        adjusted = max(adjusted, max(float(min_intensity_fraction), 0.0))
    adjusted = min(max(adjusted, 0.0), 1.0)
    return amount, adjusted


def ensure_valid_levels(buy_price: float, sell_price: float, *, min_gap_pct: float) -> Optional[Tuple[float, float]]:
    if buy_price <= 0 or sell_price <= 0:
        return None
    if sell_price <= buy_price:
        sell_price = buy_price * (1.0 + float(min_gap_pct))
        if sell_price <= buy_price:
            return None
    return float(buy_price), float(sell_price)


def build_order_intents(
    plan: TradingPlan,
    *,
    position_qty: float,
    allocation_usd: Optional[float],
    buy_price: float,
    sell_price: float,
    can_long: bool,
    can_short: bool,
    allow_short: bool,
    exit_only: bool,
    allow_position_adds: bool = True,
    always_full_exit: bool = False,
) -> list[OrderIntent]:
    intents: list[OrderIntent] = []

    if exit_only:
        if position_qty > 0:
            intents.append(
                OrderIntent(
                    side="sell",
                    qty=float(position_qty),
                    limit_price=float(sell_price),
                    kind="exit",
                )
            )
        elif position_qty < 0:
            intents.append(
                OrderIntent(
                    side="buy",
                    qty=float(abs(position_qty)),
                    limit_price=float(buy_price),
                    kind="exit",
                )
            )
        return intents

    # Exits are always permitted (for safety), regardless of can_long/can_short.
    if position_qty > 0:
        sell_qty_exit = float(position_qty) if always_full_exit else float(position_qty) * (float(plan.sell_amount) / 100.0)
        if sell_qty_exit > 0:
            intents.append(
                OrderIntent(
                    side="sell",
                    qty=sell_qty_exit,
                    limit_price=float(sell_price),
                    kind="exit",
                )
            )
    elif position_qty < 0:
        buy_qty_cover = (
            float(abs(position_qty))
            if always_full_exit
            else float(abs(position_qty)) * (float(plan.buy_amount) / 100.0)
        )
        if buy_qty_cover > 0:
            intents.append(
                OrderIntent(
                    side="buy",
                    qty=buy_qty_cover,
                    limit_price=float(buy_price),
                    kind="exit",
                )
            )

    allocation = float(allocation_usd) if allocation_usd is not None else None
    buy_notional = 0.0 if allocation is None else allocation * (float(plan.buy_amount) / 100.0)
    sell_notional = 0.0 if allocation is None else allocation * (float(plan.sell_amount) / 100.0)

    buy_qty_entry = (buy_notional / float(buy_price)) if buy_notional > 0 else 0.0
    sell_qty_entry = (sell_notional / float(sell_price)) if sell_notional > 0 else 0.0

    if position_qty > 0:
        if not allow_position_adds:
            return intents
        # While long, only allow long adds (no short entries).
        if not can_long:
            buy_qty_entry = 0.0
        sell_qty_entry = 0.0
        if buy_qty_entry > 0:
            intents.append(
                OrderIntent(
                    side="buy",
                    qty=buy_qty_entry,
                    limit_price=float(buy_price),
                    kind="entry",
                )
            )
        return intents

    if position_qty < 0:
        if not allow_position_adds:
            return intents
        # While short, only allow short adds (no long entries).
        buy_qty_entry = 0.0
        if not (allow_short and can_short):
            sell_qty_entry = 0.0
        if sell_qty_entry > 0:
            intents.append(
                OrderIntent(
                    side="sell",
                    qty=sell_qty_entry,
                    limit_price=float(sell_price),
                    kind="entry",
                )
            )
        return intents

    # Flat: allow at most one entry direction.
    if not can_long:
        buy_qty_entry = 0.0
    if not (allow_short and can_short):
        sell_qty_entry = 0.0

    if buy_qty_entry <= 0 and sell_qty_entry <= 0:
        return intents

    if buy_qty_entry > 0 and sell_qty_entry > 0:
        take_buy = buy_notional >= sell_notional
    else:
        take_buy = buy_qty_entry > 0

    if take_buy and buy_qty_entry > 0:
        intents.append(
            OrderIntent(
                side="buy",
                qty=buy_qty_entry,
                limit_price=float(buy_price),
                kind="entry",
            )
        )
    elif sell_qty_entry > 0:
        intents.append(
            OrderIntent(
                side="sell",
                qty=sell_qty_entry,
                limit_price=float(sell_price),
                kind="entry",
            )
        )
    return intents


def _read_numeric_field(action: Mapping[str, Any] | object, key: str) -> float:
    value = _read_optional_numeric_field(action, key)
    return 0.0 if value is None else value


def _read_optional_numeric_field(action: Mapping[str, Any] | object, key: str) -> Optional[float]:
    value: Any
    missing = object()
    if isinstance(action, Mapping):
        value = action.get(key, missing)
    else:
        value = getattr(action, key, missing)
    if value is missing:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


__all__ = [
    "OrderIntent",
    "TradingPlan",
    "build_order_intents",
    "build_plan_from_action",
    "directional_entry_amount",
    "entry_intensity_fraction",
    "ensure_valid_levels",
]
