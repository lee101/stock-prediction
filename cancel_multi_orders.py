from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys
from time import sleep
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from src.hourly_trader_utils import infer_working_order_kind
from src.symbol_utils import is_crypto_symbol


def _normalize_symbol(value: object) -> str:
    return str(value or "").replace("/", "").replace("-", "").upper()


def _float_attr(value: object, *names: str) -> float:
    for name in names:
        raw = getattr(value, name, None)
        if raw in (None, ""):
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return 0.0


def _position_is_flat(position: object | None) -> bool:
    if position is None:
        return True

    symbol = _normalize_symbol(getattr(position, "symbol", None))
    qty = abs(_float_attr(position, "qty"))
    if is_crypto_symbol(symbol):
        market_value = abs(_float_attr(position, "market_value", "usd"))
        current_price = abs(_float_attr(position, "current_price", "avg_entry_price"))
        if market_value > 0.0:
            return market_value < 1.0
        if current_price > 0.0:
            return qty * current_price < 1.0
        return qty < 1e-8
    return qty < 1.0


def _entry_duplicate_groups(
    orders: Sequence[object],
    positions: Iterable[object],
) -> dict[tuple[str, str], list[object]]:
    positions_by_symbol = {
        _normalize_symbol(getattr(position, "symbol", None)): position
        for position in positions
        if _normalize_symbol(getattr(position, "symbol", None))
    }
    duplicates: dict[tuple[str, str], list[object]] = defaultdict(list)
    for order in orders:
        symbol = _normalize_symbol(getattr(order, "symbol", None))
        side = str(getattr(order, "side", "") or "").strip().lower()
        if not symbol or side not in {"buy", "sell"}:
            continue
        position = positions_by_symbol.get(symbol)
        if not _position_is_flat(position):
            continue
        position_qty = _float_attr(position, "qty")
        kind = infer_working_order_kind(side=side, position_qty=position_qty)
        if kind != "entry":
            continue
        duplicates[(symbol, side)].append(order)
    return {
        key: value
        for key, value in duplicates.items()
        if len(value) > 1
    }


def _order_sort_key(order: object) -> tuple[int, str]:
    created_at = getattr(order, "created_at", None)
    if created_at is None:
        return (0, "")
    return (1, str(created_at))


def cancel_duplicate_opening_orders(
    orders: Sequence[object],
    positions: Iterable[object],
    *,
    cancel_order_fn,
) -> list[str]:
    cancelled_ids: list[str] = []
    for (symbol, side), symbol_orders in sorted(_entry_duplicate_groups(orders, positions).items()):
        symbol_orders = sorted(symbol_orders, key=_order_sort_key)
        for order in symbol_orders[:-1]:
            order_id = str(getattr(order, "id", ""))
            logger.info("canceling duplicate opening order {} for {} {}", order_id, symbol, side)
            cancel_order_fn(order)
            cancelled_ids.append(order_id)
    return cancelled_ids


def _load_broker_functions():
    from alpaca_wrapper import cancel_order, get_all_positions, get_open_orders

    return get_open_orders, get_all_positions, cancel_order


def run_loop(*, poll_seconds: int = 5 * 60) -> None:
    get_open_orders, get_all_positions, cancel_order_fn = _load_broker_functions()
    while True:
        orders = list(get_open_orders())
        positions = list(get_all_positions())
        cancel_duplicate_opening_orders(
            orders,
            positions,
            cancel_order_fn=cancel_order_fn,
        )
        sleep(poll_seconds)


def main() -> None:
    run_loop()


if __name__ == "__main__":
    main()
