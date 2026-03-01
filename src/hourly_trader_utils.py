"""Pure helpers shared by the hourly live trader and its simulators.

Keep this module free of broker/network dependencies so it can be imported in
offline backtests without requiring Alpaca credentials.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple


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


__all__ = [
    "OrderIntent",
    "TradingPlan",
    "build_order_intents",
    "build_plan_from_action",
    "ensure_valid_levels",
]
