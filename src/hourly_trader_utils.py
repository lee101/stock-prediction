"""Pure helpers shared by the hourly live trader and its simulators.

Keep this module free of broker/network dependencies so it can be imported in
offline backtests without requiring Alpaca credentials.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence, Tuple


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


@dataclass(frozen=True)
class EntryAllocationCandidate:
    symbol: str
    edge: float
    intensity_fraction: float
    slot_budget: float


def infer_working_order_kind(*, side: str, position_qty: float) -> str:
    normalized_side = str(side).lower()
    qty = float(position_qty)
    if qty > 0.0:
        return "exit" if normalized_side == "sell" else "entry"
    if qty < 0.0:
        return "exit" if normalized_side == "buy" else "entry"
    return "entry"


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


def normalize_entry_allocator_mode(value: str | None) -> str:
    raw = str(value or "legacy").strip().lower()
    if raw in {"legacy", "concentrated"}:
        return raw
    return "legacy"


def allocate_concentrated_entry_budget(
    candidates: Sequence[EntryAllocationCandidate],
    *,
    max_positions: int,
    deployable_budget: float,
    edge_power: float = 2.0,
    max_single_position_fraction: float = 0.6,
) -> list[float]:
    """Allocate spare budget into the strongest selected entries.

    Each candidate keeps its existing `slot_budget * intensity_fraction` sizing as
    a floor, then any unused slot budget is reallocated by weighted waterfilling.
    This preserves the current signal semantics while allowing a small number of
    strong candidates to absorb the risk budget that would otherwise sit idle.
    """

    if not candidates:
        return []

    position_count = max(int(max_positions), 1)
    candidate_count = len(candidates)
    budget_cap = max(float(deployable_budget), 0.0)
    if budget_cap <= 0.0:
        return [0.0] * candidate_count

    slot_budgets = [max(float(candidate.slot_budget), 0.0) for candidate in candidates]
    theoretical_budget = sum(slot_budgets) * (float(position_count) / float(candidate_count))
    if theoretical_budget <= 0.0:
        return [0.0] * candidate_count
    budget_cap = min(budget_cap, theoretical_budget)

    single_fraction = min(max(float(max_single_position_fraction), 0.0), 1.0)
    if single_fraction <= 0.0:
        return [0.0] * candidate_count

    allocations: list[float] = []
    max_caps: list[float] = []
    scores: list[float] = []

    for candidate, slot_budget in zip(candidates, slot_budgets, strict=False):
        intensity = min(max(float(candidate.intensity_fraction), 0.0), 1.0)
        base_alloc = slot_budget * intensity
        max_cap = min(slot_budget * float(position_count), budget_cap * single_fraction)
        allocations.append(min(base_alloc, max_cap))
        max_caps.append(max_cap)
        scores.append(_entry_allocator_score(candidate, edge_power=edge_power))

    total_alloc = sum(allocations)
    if total_alloc > budget_cap and total_alloc > 0.0:
        scale = budget_cap / total_alloc
        return [alloc * scale for alloc in allocations]

    remaining = budget_cap - total_alloc
    while remaining > 1e-9:
        active = [idx for idx, alloc in enumerate(allocations) if max_caps[idx] - alloc > 1e-9]
        if not active:
            break

        weight_sum = sum(scores[idx] for idx in active)
        if weight_sum <= 0.0:
            weights = {idx: 1.0 / float(len(active)) for idx in active}
        else:
            weights = {idx: scores[idx] / weight_sum for idx in active}

        distributed = 0.0
        for idx in active:
            headroom = max_caps[idx] - allocations[idx]
            if headroom <= 0.0:
                continue
            add = min(remaining * weights[idx], headroom)
            if add <= 0.0:
                continue
            allocations[idx] += add
            distributed += add

        if distributed <= 1e-9:
            break
        remaining -= distributed

    return allocations


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
    dust_notional_usd: float = 1.0,
) -> list[OrderIntent]:
    intents: list[OrderIntent] = []

    # Treat dust positions as flat (e.g. 0.000000244 ETH ~ $0.0005).
    position_notional = abs(float(position_qty)) * max(float(buy_price), float(sell_price))
    effective_qty = float(position_qty) if position_notional >= dust_notional_usd else 0.0

    if exit_only:
        if effective_qty > 0:
            intents.append(
                OrderIntent(
                    side="sell",
                    qty=float(abs(position_qty)),
                    limit_price=float(sell_price),
                    kind="exit",
                )
            )
        elif effective_qty < 0:
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
    if effective_qty > 0:
        sell_qty_exit = float(abs(position_qty)) if always_full_exit else float(abs(position_qty)) * (float(plan.sell_amount) / 100.0)
        if sell_qty_exit > 0:
            intents.append(
                OrderIntent(
                    side="sell",
                    qty=sell_qty_exit,
                    limit_price=float(sell_price),
                    kind="exit",
                )
            )
    elif effective_qty < 0:
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

    if effective_qty > 0:
        if not allow_position_adds:
            return intents
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

    if effective_qty < 0:
        if not allow_position_adds:
            return intents
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

    # Flat: generate entry for the stronger side.
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


def _entry_allocator_score(candidate: EntryAllocationCandidate, *, edge_power: float) -> float:
    intensity = min(max(float(candidate.intensity_fraction), 0.0), 1.0)
    if intensity <= 0.0:
        return 0.0
    power = float(edge_power)
    if power <= 0.0:
        return intensity
    edge = max(float(candidate.edge), 1e-9)
    if power == 1.0:
        return intensity * edge
    return intensity * (edge**power)


__all__ = [
    "EntryAllocationCandidate",
    "OrderIntent",
    "TradingPlan",
    "allocate_concentrated_entry_budget",
    "build_order_intents",
    "build_plan_from_action",
    "directional_entry_amount",
    "entry_intensity_fraction",
    "ensure_valid_levels",
    "normalize_entry_allocator_mode",
]
