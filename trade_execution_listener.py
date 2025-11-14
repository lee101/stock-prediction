"""Background listener that syncs live execution fills back into trade state.

The trading loop occasionally misses final fill events when orders are closed
out-of-band (manual interventions, forced liquidations, etc.). This utility
polls Alpaca for open positions and recently filled orders, derives realised
PnL for completed trades, and feeds the results through trade_stock_e2e's
_record_trade_outcome helper so probe-learning state stays in sync.
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

from alpaca.trading.requests import GetOrdersRequest

import alpaca_wrapper
from jsonshelve import FlatShelf
from src.trading_obj_utils import filter_to_realistic_positions
from stock.state import ensure_state_dir, get_state_file, resolve_state_suffix
from trade_stock_e2e import _normalize_side_for_key, _record_trade_outcome


logger = logging.getLogger("trade_execution_listener")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

STATE_PATH = get_state_file("execution_listener_state", resolve_state_suffix())


@dataclass
class PositionSnapshot:
    symbol: str
    side: str  # "buy" for long, "sell" for short
    qty: float
    avg_entry_price: float
    tracked_at: datetime

    @classmethod
    def from_position(cls, position) -> "PositionSnapshot | None":
        raw_side = getattr(position, "side", "")
        normalized_side = _normalize_side_for_key(raw_side)
        if normalized_side not in {"buy", "sell"}:
            return None
        try:
            qty = abs(float(getattr(position, "qty", 0.0) or 0.0))
            avg_entry_price = float(getattr(position, "avg_entry_price", 0.0) or 0.0)
        except (TypeError, ValueError):
            return None
        if qty <= 0 or avg_entry_price <= 0:
            return None
        return cls(
            symbol=str(getattr(position, "symbol", "")).upper(),
            side=normalized_side,
            qty=qty,
            avg_entry_price=avg_entry_price,
            tracked_at=datetime.now(timezone.utc),
        )

    def key(self) -> str:
        return f"{self.symbol}|{self.side}"

    def to_dict(self) -> Dict[str, object]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "avg_entry_price": self.avg_entry_price,
            "tracked_at": self.tracked_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "PositionSnapshot | None":
        try:
            tracked = data.get("tracked_at")
            tracked_at = datetime.fromisoformat(str(tracked)) if tracked else datetime.now(timezone.utc)
            if tracked_at.tzinfo is None:
                tracked_at = tracked_at.replace(tzinfo=timezone.utc)
            return cls(
                symbol=str(data["symbol"]).upper(),
                side=_normalize_side_for_key(str(data["side"])),
                qty=float(data["qty"]),
                avg_entry_price=float(data["avg_entry_price"]),
                tracked_at=tracked_at,
            )
        except Exception:
            return None


class ExecutionListenerState:
    def __init__(self):
        ensure_state_dir()
        self._store = FlatShelf(str(STATE_PATH))
        self.open_positions: Dict[str, PositionSnapshot] = {}
        self.pending_closures: Dict[str, PositionSnapshot] = {}
        self.processed_order_ids: List[str] = []
        self._load()

    def _load(self) -> None:
        try:
            self._store.load()
        except Exception as exc:
            logger.warning("Execution listener state unavailable (cold start): %s", exc)
            return
        data = self._store.get("state", {}) or {}
        self.open_positions = {
            key: snap
            for key, raw in data.get("open_positions", {}).items()
            if (snap := PositionSnapshot.from_dict(raw)) is not None
        }
        self.pending_closures = {
            key: snap
            for key, raw in data.get("pending_closures", {}).items()
            if (snap := PositionSnapshot.from_dict(raw)) is not None
        }
        processed = data.get("processed_order_ids", []) or []
        self.processed_order_ids = [str(order_id) for order_id in processed][-500:]

    def save(self) -> None:
        payload = {
            "open_positions": {k: v.to_dict() for k, v in self.open_positions.items()},
            "pending_closures": {k: v.to_dict() for k, v in self.pending_closures.items()},
            "processed_order_ids": self.processed_order_ids[-500:],
        }
        try:
            self._store.load()
        except Exception:
            pass
        self._store["state"] = payload


def _capture_open_positions() -> Dict[str, PositionSnapshot]:
    raw_positions = alpaca_wrapper.get_all_positions()
    filtered = filter_to_realistic_positions(raw_positions)
    snapshots: Dict[str, PositionSnapshot] = {}
    for position in filtered:
        snapshot = PositionSnapshot.from_position(position)
        if snapshot is None:
            continue
        snapshots[snapshot.key()] = snapshot
    return snapshots


def _closing_side(entry_side: str) -> str:
    return "sell" if entry_side == "buy" else "buy"


def _fetch_recent_filled_orders(limit: int = 200):
    try:
        orders = alpaca_wrapper.alpaca_api.get_orders(
            filter=GetOrdersRequest(status="filled", limit=limit, direction="desc")
        )
    except Exception as exc:
        logger.error("Failed to fetch filled orders: %s", exc)
        return []
    parsed = []
    for order in orders or []:
        order_id = str(getattr(order, "id", ""))
        symbol = str(getattr(order, "symbol", "")).upper()
        side = _normalize_side_for_key(getattr(order, "side", ""))
        try:
            price = float(getattr(order, "filled_avg_price", 0.0) or 0.0)
        except (TypeError, ValueError):
            price = 0.0
        filled_at_raw = getattr(order, "filled_at", None) or getattr(order, "updated_at", None)
        if filled_at_raw:
            try:
                filled_at = datetime.fromisoformat(str(filled_at_raw))
            except ValueError:
                filled_at = None
        else:
            filled_at = None
        if filled_at and filled_at.tzinfo is None:
            filled_at = filled_at.replace(tzinfo=timezone.utc)
        parsed.append(
            {
                "id": order_id,
                "symbol": symbol,
                "side": side,
                "price": price,
                "filled_at": filled_at,
            }
        )
    return parsed


def _find_closing_fill(
    symbol: str,
    closing_side: str,
    tracked_at: datetime,
    processed_ids: Iterable[str],
    orders: List[Dict[str, object]],
) -> Optional[Dict[str, object]]:
    processed = set(processed_ids)
    for order in orders:
        if order.get("symbol") != symbol:
            continue
        if order.get("side") != closing_side:
            continue
        if order.get("id") in processed:
            continue
        filled_at = order.get("filled_at")
        if isinstance(filled_at, datetime) and filled_at < tracked_at:
            continue
        price = float(order.get("price") or 0.0)
        if price <= 0:
            continue
        return order
    return None


class _SyntheticPosition:
    def __init__(self, symbol: str, side: str, qty: float, pnl: float):
        self.symbol = symbol
        self.side = side
        self.qty = qty
        self.unrealized_pl = pnl


def _compute_realised_pnl(snapshot: PositionSnapshot, closing_price: float) -> float:
    if snapshot.side == "buy":
        return (closing_price - snapshot.avg_entry_price) * snapshot.qty
    return (snapshot.avg_entry_price - closing_price) * snapshot.qty


def _process_pending_closures(state: ExecutionListenerState, orders: List[Dict[str, object]]) -> None:
    pending_keys = list(state.pending_closures.keys())
    for key in pending_keys:
        snapshot = state.pending_closures[key]
        closing_side = _closing_side(snapshot.side)
        closing_order = _find_closing_fill(
            snapshot.symbol,
            closing_side,
            snapshot.tracked_at,
            state.processed_order_ids,
            orders,
        )
        if closing_order is None:
            continue
        closing_price = float(closing_order.get("price", 0.0) or 0.0)
        pnl = _compute_realised_pnl(snapshot, closing_price)
        synthetic = _SyntheticPosition(snapshot.symbol, snapshot.side, snapshot.qty, pnl)
        try:
            _record_trade_outcome(synthetic, reason="execution_listener")
        except Exception as exc:
            logger.error("Failed to persist trade outcome for %s: %s", snapshot.symbol, exc)
            continue
        state.processed_order_ids.append(str(closing_order.get("id")))
        state.pending_closures.pop(key, None)
        logger.info(
            "%s %s: recorded realised PnL %.2f via execution listener",
            snapshot.symbol,
            snapshot.side,
            pnl,
        )


def run_listener_once(state: ExecutionListenerState) -> None:
    current_positions = _capture_open_positions()
    previous_keys = set(state.open_positions.keys())
    current_keys = set(current_positions.keys())

    newly_closed = previous_keys - current_keys
    for key in newly_closed:
        state.pending_closures[key] = state.open_positions[key]
    state.open_positions = current_positions

    orders = _fetch_recent_filled_orders()
    _process_pending_closures(state, orders)
    state.save()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync live execution fills into trade state stores")
    parser.add_argument("--interval", type=int, default=30, help="Polling interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run a single iteration and exit")
    args = parser.parse_args()

    state = ExecutionListenerState()
    while True:
        start = time.time()
        try:
            run_listener_once(state)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            logger.exception("Execution listener iteration failed: %s", exc)
        elapsed = time.time() - start
        if args.once:
            break
        sleep_duration = max(1, args.interval - math.floor(elapsed))
        time.sleep(sleep_duration)


if __name__ == "__main__":
    main()
