#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from alpaca_trade_api.stream import Stream
except Exception:  # pragma: no cover - optional dependency at runtime
    Stream = None  # type: ignore

from env_real import ALP_KEY_ID, ALP_KEY_ID_PROD, ALP_SECRET_KEY, ALP_SECRET_KEY_PROD
from src.trade_execution_monitor import (
    PositionLots,
    TradeExecutionMonitor,
    load_events_from_file,
    trade_event_from_dict,
)

logger = logging.getLogger(__name__)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Listen for trade executions and update local PnL state.")
    parser.add_argument(
        "--mode",
        choices=["alpaca", "stdin", "file"],
        default="stdin",
        help="Event source: Alpaca trade_updates stream, stdin JSON lines, or file JSON lines.",
    )
    parser.add_argument(
        "--events-file",
        type=Path,
        default=None,
        help="Path to newline-delimited JSON event file when --mode=file.",
    )
    parser.add_argument(
        "--state-suffix",
        default=None,
        help="Optional TRADE_STATE_SUFFIX override (defaults to environment).",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Force Alpaca paper trading endpoint when --mode=alpaca.",
    )
    return parser.parse_args(argv)


def _run_stdin(listener: TradeExecutionMonitor) -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        event = trade_event_from_dict(payload)
        listener.process_event(event)


def _run_file(listener: TradeExecutionMonitor, events_file: Path) -> None:
    for event in load_events_from_file(events_file):
        listener.process_event(event)


async def _run_alpaca(listener: TradeExecutionMonitor, paper: bool) -> None:  # pragma: no cover - network path
    if Stream is None:
        raise RuntimeError("alpaca-trade-api is not installed; cannot use --mode=alpaca")

    key_id = ALP_KEY_ID if paper else ALP_KEY_ID_PROD
    secret_key = ALP_SECRET_KEY if paper else ALP_SECRET_KEY_PROD
    stream = Stream(key_id, secret_key, paper=paper)

    @stream.on_trade_updates
    async def _(data):
        raw_qty = getattr(data, "qty", None) or getattr(getattr(data, "order", None), "filled_qty", None)
        raw_price = getattr(data, "price", None) or getattr(getattr(data, "order", None), "filled_avg_price", None)
        payload = {
            "symbol": getattr(data, "symbol", None) or getattr(getattr(data, "order", None), "symbol", None),
            "side": getattr(getattr(data, "order", None), "side", None),
            "quantity": float(raw_qty) if raw_qty is not None else 0.0,
            "price": float(raw_price) if raw_price is not None else 0.0,
            "timestamp": getattr(data, "timestamp", datetime.now(timezone.utc).isoformat()),
        }
        event = trade_event_from_dict(payload)
        if event is not None:
            listener.process_event(event)

    await stream._run_forever()  # type: ignore[attr-defined]


def _seed_positions_from_alpaca(listener: TradeExecutionMonitor) -> None:
    """Seed the listener with current Alpaca positions so closes are recognised after restart."""
    try:
        from alpaca_wrapper import get_all_positions
        positions = get_all_positions()
        for pos in positions:
            symbol = getattr(pos, "symbol", None)
            qty = float(getattr(pos, "qty", 0))
            avg_price = float(getattr(pos, "avg_entry_price", 0))
            if not symbol or qty == 0 or avg_price == 0:
                continue
            lots = listener._lots_by_symbol.setdefault(symbol, PositionLots())
            lots._lots.append((qty, avg_price))
            logger.info("Seeded %s: qty=%.4f @ %.2f", symbol, qty, avg_price)
    except Exception as exc:
        logger.warning("Could not seed positions from Alpaca: %s", exc)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    listener = TradeExecutionMonitor(state_suffix=args.state_suffix)
    if args.mode == "alpaca":
        _seed_positions_from_alpaca(listener)
    if args.mode == "stdin":
        _run_stdin(listener)
    elif args.mode == "file":
        if args.events_file is None:
            raise SystemExit("--events-file is required when --mode=file")
        _run_file(listener, args.events_file)
    else:
        asyncio.run(_run_alpaca(listener, args.paper))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
