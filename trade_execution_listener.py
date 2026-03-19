#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from alpaca_trade_api.stream import Stream
except Exception:  # pragma: no cover - optional dependency at runtime
    Stream = None  # type: ignore

from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD
from src.trade_execution_monitor import (
    TradeExecutionMonitor,
    load_events_from_file,
    trade_event_from_dict,
)


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
        if event is None:
            continue
        listener.process_event(event)


def _run_file(listener: TradeExecutionMonitor, events_file: Path) -> None:
    for event in load_events_from_file(events_file):
        listener.process_event(event)


async def _run_alpaca(listener: TradeExecutionMonitor, paper: bool) -> None:  # pragma: no cover - network path
    if Stream is None:
        raise RuntimeError("alpaca-trade-api is not installed; cannot use --mode=alpaca")

    stream = Stream(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD, paper=paper)

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


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    listener = TradeExecutionMonitor(state_suffix=args.state_suffix)
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
