from __future__ import annotations

import argparse
import logging
import time
from typing import Iterable, List, Optional

import alpaca_wrapper
from src.stock_utils import pairs_equal
from src.symbol_utils import is_crypto_symbol


logger = logging.getLogger("alpaca_exit_only")


def _parse_symbols(raw: Optional[str]) -> List[str]:
    if not raw:
        return ["NFLX"]
    return [token.strip().upper() for token in raw.split(",") if token.strip()]


def _cancel_open_orders(symbol: str, orders: Iterable) -> int:
    cancelled = 0
    for order in orders:
        order_symbol = str(getattr(order, "symbol", "") or "")
        if not pairs_equal(order_symbol, symbol):
            continue
        try:
            alpaca_wrapper.cancel_order(order)
            cancelled += 1
        except Exception:
            logger.exception("Failed canceling order %s for %s", getattr(order, "id", "?"), symbol)
    return cancelled


def _find_position(positions: Iterable, symbol: str):
    for position in positions:
        if pairs_equal(str(getattr(position, "symbol", "") or ""), symbol):
            return position
    return None


def _run_cycle(
    symbols: Iterable[str],
    *,
    pct_above_market: float,
    cancel_open_orders: bool,
    allow_after_hours: bool,
    dry_run: bool,
) -> None:
    positions = alpaca_wrapper.get_all_positions()
    orders = alpaca_wrapper.get_orders(use_cache=False) if cancel_open_orders else []
    clock = None

    for symbol in symbols:
        symbol = symbol.upper()
        if not is_crypto_symbol(symbol) and not allow_after_hours:
            if clock is None:
                clock = alpaca_wrapper.get_clock()
            if not getattr(clock, "is_open", True):
                logger.info("Market closed; skipping %s", symbol)
                continue

        if cancel_open_orders:
            cancelled = _cancel_open_orders(symbol, orders)
            if cancelled:
                logger.info("Canceled %d open orders for %s", cancelled, symbol)

        position = _find_position(positions, symbol)
        if position is None:
            logger.info("No open position for %s", symbol)
            continue

        qty = float(getattr(position, "qty", 0.0) or 0.0)
        side = str(getattr(position, "side", "") or "")
        logger.info("Exit-only %s position found qty=%.6f side=%s", symbol, qty, side)

        if dry_run:
            continue

        result = alpaca_wrapper.close_position_near_market(position, pct_above_market=pct_above_market)
        if not result:
            logger.warning("Close order not submitted for %s", symbol)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exit-only Alpaca positions for selected symbols.")
    parser.add_argument("--symbols", default="NFLX", help="Comma-separated symbols to close.")
    parser.add_argument("--pct-above-market", type=float, default=0.0)
    parser.add_argument("--allow-after-hours", action="store_true")
    parser.add_argument("--no-cancel-open-orders", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    symbols = _parse_symbols(args.symbols)
    if args.pct_above_market < 0:
        raise ValueError("--pct-above-market must be >= 0 for exit-only orders.")
    cancel_open_orders = not args.no_cancel_open_orders

    while True:
        _run_cycle(
            symbols,
            pct_above_market=args.pct_above_market,
            cancel_open_orders=cancel_open_orders,
            allow_after_hours=args.allow_after_hours,
            dry_run=args.dry_run,
        )
        if args.once:
            break
        time.sleep(max(5.0, float(args.poll_seconds)))


if __name__ == "__main__":
    main()
