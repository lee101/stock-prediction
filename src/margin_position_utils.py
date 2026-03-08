from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class DirectionalSignal:
    side: str
    entry_price: float
    exit_price: float
    entry_amount: float
    exit_amount: float


def position_side_from_qty(qty: float, *, step_size: float = 0.0) -> str:
    signed_qty = float(qty)
    step = max(0.0, float(step_size))
    if step > 0.0 and abs(signed_qty) + 1e-12 < step:
        return ""
    if signed_qty > 0.0:
        return "long"
    if signed_qty < 0.0:
        return "short"
    return ""


def position_notional(qty: float, market_price: float) -> float:
    return abs(float(qty)) * max(0.0, float(market_price))


def directional_signal(signal: Mapping[str, float], *, side: str) -> DirectionalSignal:
    normalized = str(side or "").strip().lower()
    if normalized == "long":
        return DirectionalSignal(
            side="long",
            entry_price=max(0.0, float(signal.get("buy_price", 0.0))),
            exit_price=max(0.0, float(signal.get("sell_price", 0.0))),
            entry_amount=max(0.0, float(signal.get("buy_amount", 0.0))),
            exit_amount=max(0.0, float(signal.get("sell_amount", 0.0))),
        )
    if normalized == "short":
        return DirectionalSignal(
            side="short",
            entry_price=max(0.0, float(signal.get("sell_price", 0.0))),
            exit_price=max(0.0, float(signal.get("buy_price", 0.0))),
            entry_amount=max(0.0, float(signal.get("sell_amount", 0.0))),
            exit_amount=max(0.0, float(signal.get("buy_amount", 0.0))),
        )
    raise ValueError(f"Unsupported side {side!r}; expected 'long' or 'short'.")


def choose_flat_entry_side(signal: Mapping[str, float], *, allow_short: bool) -> str:
    long_signal = directional_signal(signal, side="long")
    short_signal = directional_signal(signal, side="short")
    long_active = long_signal.entry_amount > 0.0 and long_signal.entry_price > 0.0
    short_active = allow_short and short_signal.entry_amount > 0.0 and short_signal.entry_price > 0.0
    if long_active and short_active:
        return "short" if short_signal.entry_amount > long_signal.entry_amount else "long"
    if long_active:
        return "long"
    if short_active:
        return "short"
    return ""


def side_max_leverage(
    side: str,
    *,
    long_max_leverage: float,
    short_max_leverage: float,
) -> float:
    normalized = str(side or "").strip().lower()
    if normalized == "long":
        return max(0.0, float(long_max_leverage))
    if normalized == "short":
        return max(0.0, float(short_max_leverage))
    raise ValueError(f"Unsupported side {side!r}; expected 'long' or 'short'.")


def remaining_entry_notional(
    *,
    side: str,
    equity: float,
    current_qty: float,
    market_price: float,
    long_max_leverage: float,
    short_max_leverage: float,
) -> float:
    normalized = str(side or "").strip().lower()
    max_lev = side_max_leverage(
        side,
        long_max_leverage=long_max_leverage,
        short_max_leverage=short_max_leverage,
    )
    if max_lev <= 0.0:
        return 0.0
    price = max(0.0, float(market_price))
    if price <= 0.0:
        return 0.0
    target_notional = max(0.0, float(equity)) * max_lev
    target_signed_notional = target_notional if normalized == "long" else -target_notional
    current_signed_notional = float(current_qty) * price
    remaining_signed_notional = target_signed_notional - current_signed_notional
    if normalized == "short":
        return max(0.0, -remaining_signed_notional)
    return max(0.0, remaining_signed_notional)


__all__ = [
    "DirectionalSignal",
    "choose_flat_entry_side",
    "directional_signal",
    "position_notional",
    "position_side_from_qty",
    "remaining_entry_notional",
    "side_max_leverage",
]
