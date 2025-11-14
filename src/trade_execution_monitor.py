from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from jsonshelve import FlatShelf

from stock.state import get_state_file, resolve_state_suffix
from src.trade_stock_state_utils import normalize_side_for_key, state_key


@dataclass(frozen=True)
class TradeEvent:
    symbol: str
    side: str  # "buy" or "sell" representing action (order side)
    quantity: float
    price: float
    timestamp: datetime


@dataclass(frozen=True)
class ClosureEvent:
    entry_side: str
    qty: float
    pnl: float
    timestamp: datetime


class PositionLots:
    """FIFO lot tracker supporting both long and short exposure."""

    def __init__(self) -> None:
        self._lots: List[tuple[float, float]] = []  # (qty, price) qty>0 -> long, qty<0 -> short

    def apply_trade(self, event: TradeEvent) -> List[ClosureEvent]:
        remaining = event.quantity if event.side.lower() == "buy" else -event.quantity
        closures: List[ClosureEvent] = []

        def _has_opposite_sign(lot_qty: float) -> bool:
            return (remaining < 0 and lot_qty > 0) or (remaining > 0 and lot_qty < 0)

        idx = 0
        while remaining and idx < len(self._lots) and _has_opposite_sign(self._lots[idx][0]):
            lot_qty, lot_price = self._lots[idx]
            closable = min(abs(remaining), abs(lot_qty))
            if remaining < 0 and lot_qty > 0:
                # selling long
                pnl = (event.price - lot_price) * closable
                lot_qty -= closable
                remaining += closable
                closures.append(
                    ClosureEvent(entry_side="buy", qty=closable, pnl=pnl, timestamp=event.timestamp)
                )
            elif remaining > 0 and lot_qty < 0:
                # buying to cover short
                pnl = (lot_price - event.price) * closable
                lot_qty += closable
                remaining -= closable
                closures.append(
                    ClosureEvent(entry_side="sell", qty=closable, pnl=pnl, timestamp=event.timestamp)
                )
            if abs(lot_qty) < 1e-9:
                self._lots.pop(idx)
            else:
                self._lots[idx] = (lot_qty, lot_price)
                idx += 1

        if abs(remaining) > 1e-9:
            self._lots.append((remaining, event.price))

        return closures


class TradeHistoryWriter:
    def __init__(self, *, state_suffix: Optional[str] = None) -> None:
        self.state_suffix = resolve_state_suffix(state_suffix)
        self.store = FlatShelf(get_state_file("trade_history", self.state_suffix))
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            try:
                self.store.load()
            except Exception:
                pass
            self._loaded = True

    def append(self, symbol: str, side: str, qty: float, pnl: float, timestamp: datetime) -> None:
        self._ensure_loaded()
        key = state_key(symbol, normalize_side_for_key(side))
        history = list(self.store.get(key, []))
        history.append(
            {
                "symbol": symbol,
                "side": normalize_side_for_key(side),
                "qty": float(qty),
                "pnl": float(pnl),
                "closed_at": timestamp.astimezone(timezone.utc).isoformat(),
                "reason": "execution_listener",
                "mode": "normal",
            }
        )
        self.store[key] = history[-100:]


class TradeExecutionMonitor:
    def __init__(self, *, state_suffix: Optional[str] = None) -> None:
        self._lots_by_symbol: Dict[str, PositionLots] = {}
        self._writer = TradeHistoryWriter(state_suffix=state_suffix)

    def process_event(self, event: TradeEvent) -> List[ClosureEvent]:
        lots = self._lots_by_symbol.setdefault(event.symbol, PositionLots())
        closures = lots.apply_trade(event)
        for closure in closures:
            if closure.qty <= 0:
                continue
            self._writer.append(event.symbol, closure.entry_side, closure.qty, closure.pnl, closure.timestamp)
        return closures


def trade_event_from_dict(payload: Dict[str, object]) -> TradeEvent:
    symbol = str(payload.get("symbol"))
    side = str(payload.get("side", "")).lower()
    quantity = float(payload.get("quantity", 0.0))
    price = float(payload.get("price", 0.0))
    timestamp_raw = payload.get("timestamp")
    if isinstance(timestamp_raw, str):
        timestamp = datetime.fromisoformat(timestamp_raw.replace("Z", "+00:00"))
    else:
        timestamp = datetime.now(timezone.utc)
    return TradeEvent(symbol=symbol, side=side, quantity=quantity, price=price, timestamp=timestamp)


def load_events_from_file(path: Path) -> List[TradeEvent]:
    events: List[TradeEvent] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            events.append(trade_event_from_dict(payload))
    return events


__all__ = [
    "TradeEvent",
    "TradeExecutionMonitor",
    "load_events_from_file",
    "trade_event_from_dict",
]
