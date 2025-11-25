from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Tuple

DEFAULT_PATH = Path("strategy_state/price_guard.json")


@dataclass
class GuardState:
    last_buy_price: Optional[float] = None
    last_buy_ts: Optional[datetime] = None
    last_sell_price: Optional[float] = None
    last_sell_ts: Optional[datetime] = None

    @classmethod
    def load(cls, symbol: str, path: Path = DEFAULT_PATH) -> "GuardState":
        try:
            data = json.loads(path.read_text())
        except Exception:
            data = {}
        rec = data.get(symbol.upper(), {})
        return cls(
            last_buy_price=rec.get("last_buy_price"),
            last_buy_ts=_parse_iso(rec.get("last_buy_ts")),
            last_sell_price=rec.get("last_sell_price"),
            last_sell_ts=_parse_iso(rec.get("last_sell_ts")),
        )

    def save(self, symbol: str, path: Path = DEFAULT_PATH) -> None:
        try:
            data = json.loads(path.read_text())
        except Exception:
            data = {}
        data[symbol.upper()] = {
            "last_buy_price": self.last_buy_price,
            "last_buy_ts": self.last_buy_ts.isoformat() if self.last_buy_ts else None,
            "last_sell_price": self.last_sell_price,
            "last_sell_ts": self.last_sell_ts.isoformat() if self.last_sell_ts else None,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True))


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def record_buy(symbol: str, price: float, now: Optional[datetime] = None) -> None:
    state = GuardState.load(symbol)
    state.last_buy_price = float(price)
    state.last_buy_ts = now or datetime.now(timezone.utc)
    state.save(symbol)


def record_sell(symbol: str, price: float, now: Optional[datetime] = None) -> None:
    state = GuardState.load(symbol)
    state.last_sell_price = float(price)
    state.last_sell_ts = now or datetime.now(timezone.utc)
    state.save(symbol)


def enforce_gap(
    symbol: str,
    buy_price: float,
    sell_price: float,
    min_gap_pct: float = 0.001,
    window: timedelta = timedelta(minutes=45),
) -> Tuple[float, float]:
    """Ensure buy < sell within a recent window; maintain min_gap_pct separation.

    - If a new sell would be <= recent buy, lift sell to buy*(1+gap).
    - If a new buy would be >= recent sell, lower buy to sell*(1-gap).
    - Also enforces direct buy<sell with the same gap.
    """
    state = GuardState.load(symbol)
    now = datetime.now(timezone.utc)
    adj_buy, adj_sell = buy_price, sell_price

    if state.last_buy_price and state.last_buy_ts and now - state.last_buy_ts <= window:
        floor = state.last_buy_price * (1 + min_gap_pct)
        if adj_sell <= floor:
            adj_sell = floor
    if state.last_sell_price and state.last_sell_ts and now - state.last_sell_ts <= window:
        ceiling = state.last_sell_price * (1 - min_gap_pct)
        if adj_buy >= ceiling:
            adj_buy = ceiling

    # Enforce direct gap between current buy/sell as well
    if adj_sell <= adj_buy:
        adj_sell = adj_buy * (1 + min_gap_pct)

    return adj_buy, adj_sell
