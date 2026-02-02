from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from stock.state import get_state_dir, resolve_state_suffix, get_paper_suffix


STATE_SUFFIX = resolve_state_suffix()
PAPER_SUFFIX = get_paper_suffix()
STATE_PATH = get_state_dir() / f"binanceneural_pnl_state{PAPER_SUFFIX}{STATE_SUFFIX or ''}.json"


@dataclass
class SymbolPnlState:
    position_qty: float = 0.0
    cost_basis: float = 0.0
    realized_pnl: float = 0.0
    last_realized_pnl: float = 0.0
    mode: str = "normal"
    last_update: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "position_qty": self.position_qty,
            "cost_basis": self.cost_basis,
            "realized_pnl": self.realized_pnl,
            "last_realized_pnl": self.last_realized_pnl,
            "mode": self.mode,
            "last_update": self.last_update,
        }


def _load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_symbol_state(symbol: str) -> SymbolPnlState:
    data = _load_state()
    rec = data.get(symbol.upper(), {})
    return SymbolPnlState(
        position_qty=float(rec.get("position_qty", 0.0)),
        cost_basis=float(rec.get("cost_basis", 0.0)),
        realized_pnl=float(rec.get("realized_pnl", 0.0)),
        last_realized_pnl=float(rec.get("last_realized_pnl", 0.0)),
        mode=str(rec.get("mode", "normal")),
        last_update=rec.get("last_update"),
    )


def record_fill(symbol: str, side: str, price: float, quantity: float) -> SymbolPnlState:
    symbol = symbol.upper()
    side = side.lower()
    state = get_symbol_state(symbol)

    qty = max(0.0, float(quantity))
    price = float(price)
    if qty <= 0 or price <= 0:
        return state

    if side.startswith("b"):
        new_qty = state.position_qty + qty
        if new_qty <= 0:
            state.position_qty = 0.0
            state.cost_basis = 0.0
        else:
            state.cost_basis = (
                state.cost_basis * state.position_qty + price * qty
            ) / new_qty
            state.position_qty = new_qty
    elif side.startswith("s"):
        sell_qty = min(qty, state.position_qty)
        realized = (price - state.cost_basis) * sell_qty
        state.position_qty = max(0.0, state.position_qty - sell_qty)
        if state.position_qty <= 0:
            state.position_qty = 0.0
            state.cost_basis = 0.0
        state.realized_pnl += realized
        state.last_realized_pnl = realized
        if realized < 0:
            state.mode = "probe"
        elif realized > 0 and state.mode == "probe":
            state.mode = "normal"
    else:
        return state

    state.last_update = _now_iso()
    data = _load_state()
    data[symbol] = state.to_dict()
    _save_state(data)
    return state


def get_probe_mode(symbol: str) -> bool:
    state = get_symbol_state(symbol)
    return state.mode == "probe"


__all__ = ["get_probe_mode", "get_symbol_state", "record_fill", "SymbolPnlState", "STATE_PATH"]
