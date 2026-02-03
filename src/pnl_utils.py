from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple


@dataclass
class PositionState:
    qty: float = 0.0
    avg_price: float = 0.0


@dataclass
class SymbolPnl:
    realized: float = 0.0
    unrealized: float = 0.0
    fees: float = 0.0
    net_qty: float = 0.0
    avg_price: float = 0.0
    current_price: Optional[float] = None

    @property
    def gross(self) -> float:
        return self.realized + self.unrealized

    @property
    def net(self) -> float:
        return self.gross - self.fees


@dataclass
class PnlSummary:
    realized: float
    unrealized: float
    fees: float
    gross: float
    net: float
    positions: Dict[str, PositionState]
    per_symbol: Dict[str, SymbolPnl]


def _apply_buy(
    state: PositionState,
    qty: float,
    price: float,
) -> float:
    """Apply a buy fill. Returns realized PnL from any short cover."""
    realized = 0.0
    if state.qty >= 0:
        new_qty = state.qty + qty
        if state.qty == 0:
            state.avg_price = price
        else:
            state.avg_price = ((state.avg_price * state.qty) + (price * qty)) / new_qty
        state.qty = new_qty
        return realized

    cover_qty = min(qty, abs(state.qty))
    realized += (state.avg_price - price) * cover_qty
    state.qty += cover_qty
    if state.qty == 0:
        state.avg_price = 0.0
    remaining = qty - cover_qty
    if remaining > 0:
        state.avg_price = price
        state.qty = remaining
    return realized


def _apply_sell(
    state: PositionState,
    qty: float,
    price: float,
) -> float:
    """Apply a sell fill. Returns realized PnL from any long close."""
    realized = 0.0
    if state.qty <= 0:
        new_qty = state.qty - qty
        if state.qty == 0:
            state.avg_price = price
        else:
            state.avg_price = (
                (state.avg_price * abs(state.qty)) + (price * qty)
            ) / abs(new_qty)
        state.qty = new_qty
        return realized

    close_qty = min(qty, state.qty)
    realized += (price - state.avg_price) * close_qty
    state.qty -= close_qty
    if state.qty == 0:
        state.avg_price = 0.0
    remaining = qty - close_qty
    if remaining > 0:
        state.avg_price = price
        state.qty = -remaining
    return realized


def compute_trade_pnl(
    fills: Iterable[dict],
    *,
    current_prices: Dict[str, float],
    fee_rate_by_symbol: Dict[str, float],
) -> PnlSummary:
    positions: Dict[str, PositionState] = {}
    per_symbol: Dict[str, SymbolPnl] = {}
    realized_total = 0.0
    fees_total = 0.0

    for fill in fills:
        symbol = fill["symbol"]
        side = fill["side"].lower()
        qty = float(fill["qty"])
        price = float(fill["price"])
        if qty <= 0 or price <= 0:
            continue

        state = positions.setdefault(symbol, PositionState())
        symbol_pnl = per_symbol.setdefault(symbol, SymbolPnl())
        fee_rate = float(fee_rate_by_symbol.get(symbol, 0.0))
        fee = price * qty * fee_rate
        fees_total += fee
        symbol_pnl.fees += fee

        if side == "buy":
            realized = _apply_buy(state, qty, price)
        elif side == "sell":
            realized = _apply_sell(state, qty, price)
        else:
            continue

        realized_total += realized
        symbol_pnl.realized += realized

    unrealized_total = 0.0
    for symbol, state in positions.items():
        if state.qty == 0:
            continue
        current_price = current_prices.get(symbol)
        if current_price is None:
            continue
        symbol_pnl = per_symbol.setdefault(symbol, SymbolPnl())
        symbol_pnl.current_price = current_price
        symbol_pnl.net_qty = state.qty
        symbol_pnl.avg_price = state.avg_price

        if state.qty > 0:
            unrealized = (current_price - state.avg_price) * state.qty
        else:
            unrealized = (state.avg_price - current_price) * abs(state.qty)
        symbol_pnl.unrealized = unrealized
        unrealized_total += unrealized

    gross = realized_total + unrealized_total
    net = gross - fees_total

    return PnlSummary(
        realized=realized_total,
        unrealized=unrealized_total,
        fees=fees_total,
        gross=gross,
        net=net,
        positions=positions,
        per_symbol=per_symbol,
    )
