"""Utilities for summarizing realized trade PnL from Alpaca fills.

The functions here keep the logic reusable for CLI tools and tests.  We avoid
Alpaca-specific types by normalizing fills into simple dataclasses that capture
just the fields we need (symbol, side, quantity, price, fee, timestamp).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Sequence


@dataclass
class FillEvent:
    symbol: str
    side: str  # "buy" or "sell"
    qty: float
    price: float
    fee: float
    transacted_at: datetime


@dataclass
class TradePnL:
    symbol: str
    direction: str  # long|short
    quantity: float
    opened_at: datetime
    closed_at: datetime
    entry_price: float
    exit_price: float
    gross: float
    fees: float
    net: float

    @property
    def duration_seconds(self) -> float:
        return (self.closed_at - self.opened_at).total_seconds()


def _coerce_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _parse_timestamp(raw: str) -> datetime:
    if not raw:
        return datetime.now(timezone.utc)
    normalized = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except Exception:
        parsed = datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def normalize_fill_activities(activities: Iterable[dict]) -> List[FillEvent]:
    fills: List[FillEvent] = []
    for activity in activities or []:
        if activity.get("activity_type") not in {"FILL", None}:
            # Some responses omit the activity_type key when already filtered; allow None
            continue

        symbol = (activity.get("symbol") or "").strip().upper()
        side_raw = (activity.get("side") or activity.get("type") or "").strip().lower()
        if side_raw not in {"buy", "sell"}:
            continue

        qty = _coerce_float(activity.get("qty"))
        price = _coerce_float(activity.get("price") or activity.get("price_per_share"))
        fee = _coerce_float(activity.get("fee") or activity.get("commission"))
        ts = activity.get("transaction_time") or activity.get("event_at") or activity.get("processed_at")
        transacted_at = _parse_timestamp(ts)

        if qty <= 0 or price <= 0:
            continue

        fills.append(
            FillEvent(
                symbol=symbol,
                side=side_raw,
                qty=qty,
                price=price,
                fee=fee,
                transacted_at=transacted_at,
            )
        )
    fills.sort(key=lambda f: (f.symbol, f.transacted_at))
    return fills


def _cashflow(fill: FillEvent, qty_portion: float | None = None) -> float:
    qty_used = fill.qty if qty_portion is None else qty_portion
    notional = fill.price * qty_used
    fee = fill.fee * (qty_used / fill.qty)
    if fill.side == "buy":
        return -(notional + fee)
    return notional - fee


def compute_round_trips(fills: Sequence[FillEvent]) -> List[TradePnL]:
    """Collapse a sequence of fills into realized round-trip trades.

    A trade starts when inventory for a symbol moves away from zero and ends
    when it returns to zero. Trades that flip direction in a single fill are
    split so the closing portion is realized and the remainder opens a new
    trade.
    """

    trades: List[TradePnL] = []
    state_per_symbol = {}

    QTY_EPS = 1e-6

    for fill in sorted(fills, key=lambda f: f.transacted_at):
        sign = 1.0 if fill.side == "buy" else -1.0
        signed_qty_total = sign * fill.qty

        symbol_state = state_per_symbol.get(fill.symbol)
        if symbol_state is None:
            symbol_state = {
                "qty": 0.0,
                "opened_at": None,
                "direction": None,
                "entry_notional": 0.0,
                "entry_qty": 0.0,
                "entry_fees": 0.0,
            }
            state_per_symbol[fill.symbol] = symbol_state

        remaining_qty = signed_qty_total
        remaining_cashflow = _cashflow(fill)
        remaining_fee = fill.fee

        while abs(remaining_qty) > QTY_EPS:
            qty = symbol_state["qty"]

            # No open position; start a new trade with whatever remains
            if abs(qty) <= QTY_EPS:
                symbol_state.update(
                    {
                        "qty": remaining_qty,
                        "opened_at": fill.transacted_at,
                        "direction": "long" if remaining_qty > 0 else "short",
                        "entry_notional": fill.price * abs(remaining_qty),
                        "entry_qty": abs(remaining_qty),
                        "entry_fees": remaining_fee,
                    }
                )
                remaining_qty = 0.0
                continue

            # Same direction as current position -> expand entry
            if math.copysign(1, qty) == math.copysign(1, remaining_qty):
                symbol_state["qty"] += remaining_qty
                symbol_state["entry_notional"] += fill.price * abs(remaining_qty)
                symbol_state["entry_qty"] += abs(remaining_qty)
                symbol_state["entry_fees"] += remaining_fee
                remaining_qty = 0.0
                continue

            # Opposite direction: partially or fully close the position
            close_qty = min(abs(qty), abs(remaining_qty))
            proportion = close_qty / abs(remaining_qty)
            cashflow_used = remaining_cashflow * proportion
            fee_used = remaining_fee * proportion

            # Realize PnL for the closed portion immediately (even if position remains open)
            entry_qty_total = max(symbol_state["entry_qty"], 1e-12)
            entry_price_avg = symbol_state["entry_notional"] / entry_qty_total
            exit_price = fill.price
            proportional_entry_fees = symbol_state["entry_fees"] * (close_qty / entry_qty_total)

            # realized PnL before fees
            if symbol_state["direction"] == "long":
                gross = (exit_price - entry_price_avg) * close_qty
            else:
                gross = (entry_price_avg - exit_price) * close_qty

            fees_total = fee_used + proportional_entry_fees
            net = gross - fees_total

            symbol_state["qty"] -= math.copysign(close_qty, qty)

            trade = TradePnL(
                symbol=fill.symbol,
                direction=symbol_state["direction"],
                quantity=close_qty,
                opened_at=symbol_state["opened_at"],
                closed_at=fill.transacted_at,
                entry_price=entry_price_avg,
                exit_price=exit_price,
                gross=gross,
                fees=fees_total,
                net=net,
            )
            trades.append(trade)

            # Reduce entry buckets to reflect the portion closed
            symbol_state["entry_notional"] -= entry_price_avg * close_qty
            symbol_state["entry_qty"] -= close_qty
            symbol_state["entry_fees"] -= proportional_entry_fees

            remaining_qty = remaining_qty - math.copysign(close_qty, remaining_qty)
            remaining_cashflow = remaining_cashflow - cashflow_used
            remaining_fee = remaining_fee - fee_used

            if abs(symbol_state["qty"]) <= QTY_EPS:
                symbol_state.update(
                    {
                        "qty": 0.0,
                        "opened_at": None,
                        "direction": None,
                        "entry_notional": 0.0,
                        "entry_qty": 0.0,
                        "entry_fees": 0.0,
                    }
                )
            elif abs(remaining_qty) > QTY_EPS and math.copysign(1, remaining_qty) != math.copysign(1, symbol_state["qty"]):
                symbol_state.update(
                    {
                        "qty": remaining_qty,
                        "opened_at": fill.transacted_at,
                        "direction": "long" if remaining_qty > 0 else "short",
                        "entry_notional": fill.price * abs(remaining_qty),
                        "entry_qty": abs(remaining_qty),
                        "entry_fees": remaining_fee,
                    }
                )
                remaining_qty = 0.0
                remaining_cashflow = 0.0
                remaining_fee = 0.0


    return trades
