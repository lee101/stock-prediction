#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from alpaca_trade_api.stream import Stream
except Exception:  # pragma: no cover - optional dependency at runtime
    Stream = None  # type: ignore

from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD
from src.trade_execution_monitor import (
    TradeEvent,
    TradeExecutionMonitor,
    load_events_from_file,
    trade_event_from_dict,
)

LOG = logging.getLogger("trade_execution_listener")
SIGNAL_MAX_AGE_SECONDS = 300
EXCLUDED_SYMBOLS = frozenset({"NYT"})
_PLACEHOLDER_VALUES = {
    "",
    "AKAONQRN6CZJFGTHN3DWDFPIBA",
    "GYwPufjn8TrKNHV4jwWoMs7cwmDPiP4U1Xsu8UHzXDz4",
}


def _emit(event: str, **fields) -> None:
    payload = {"event": event, "ts": datetime.now(timezone.utc).isoformat(), **fields}
    LOG.debug(json.dumps(payload, sort_keys=True, default=str))


def _normalize_timestamp(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _is_stale(event: TradeEvent, *, max_age_seconds: int = SIGNAL_MAX_AGE_SECONDS) -> bool:
    age_seconds = (datetime.now(timezone.utc) - _normalize_timestamp(event.timestamp)).total_seconds()
    return age_seconds > float(max_age_seconds)


def _is_excluded(event: TradeEvent) -> bool:
    return str(event.symbol or "").upper() in EXCLUDED_SYMBOLS


def _should_process(event: TradeEvent, *, signal_max_age: int = SIGNAL_MAX_AGE_SECONDS) -> bool:
    return not _is_excluded(event) and not _is_stale(event, max_age_seconds=signal_max_age)


def _process_event_with_logging(
    listener: TradeExecutionMonitor,
    event: TradeEvent,
    *,
    dry_run: bool = False,
    signal_max_age: int = SIGNAL_MAX_AGE_SECONDS,
) -> None:
    _emit("signal_received", symbol=event.symbol, side=event.side, quantity=event.quantity, price=event.price)

    if _is_excluded(event):
        _emit("signal_excluded", symbol=event.symbol)
        return
    if _is_stale(event, max_age_seconds=signal_max_age):
        _emit("signal_rejected_stale", symbol=event.symbol, max_age_seconds=signal_max_age)
        return
    if dry_run:
        _emit("dry_run_would_process", symbol=event.symbol)
        return

    try:
        closures = listener.process_event(event)
    except Exception as exc:  # noqa: BLE001
        _emit("process_event_failed", symbol=event.symbol, error=str(exc))
        return
    _emit("signal_processed", symbol=event.symbol, closures=len(closures))


def _run_stdin(
    listener: TradeExecutionMonitor,
    *,
    dry_run: bool = False,
    heartbeat_interval: int = 60,
    signal_max_age: int = SIGNAL_MAX_AGE_SECONDS,
) -> None:
    last_heartbeat = time.monotonic()
    for raw_line in sys.stdin:
        now = time.monotonic()
        if heartbeat_interval > 0 and (now - last_heartbeat) >= float(heartbeat_interval):
            _emit("heartbeat")
            last_heartbeat = now

        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            _emit("json_parse_failed", error=str(exc), line_preview=line[:120])
            continue
        try:
            event = trade_event_from_dict(payload)
        except Exception as exc:  # noqa: BLE001
            _emit("event_parse_failed", error=str(exc))
            continue
        _process_event_with_logging(listener, event, dry_run=dry_run, signal_max_age=signal_max_age)


def _run_file(
    listener: TradeExecutionMonitor,
    events_file: Path,
    *,
    dry_run: bool = False,
    signal_max_age: int = SIGNAL_MAX_AGE_SECONDS,
) -> None:
    for event in load_events_from_file(events_file):
        _process_event_with_logging(listener, event, dry_run=dry_run, signal_max_age=signal_max_age)


def _check_config() -> None:
    valid = True
    for value in (str(ALP_KEY_ID_PROD or "").strip(), str(ALP_SECRET_KEY_PROD or "").strip()):
        if value in _PLACEHOLDER_VALUES or len(value) < 8:
            valid = False
    if "NYT" not in EXCLUDED_SYMBOLS:
        valid = False
    raise SystemExit(0 if valid else 1)


async def _run_alpaca(listener: TradeExecutionMonitor, paper: bool) -> None:  # pragma: no cover - network path
    if Stream is None:
        raise RuntimeError("alpaca-trade-api is not installed; cannot use --mode=alpaca")

    stream = Stream(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD, paper=paper)

    @stream.on_trade_updates
    async def _(data):
        payload = {
            "symbol": getattr(data, "symbol", None) or getattr(getattr(data, "order", None), "symbol", None),
            "side": getattr(getattr(data, "order", None), "side", None),
            "quantity": float(getattr(data, "qty", getattr(data, "filled_qty", 0.0)) or 0.0),
            "price": float(getattr(data, "price", getattr(data, "filled_avg_price", 0.0)) or 0.0),
            "timestamp": getattr(data, "timestamp", datetime.now(timezone.utc).isoformat()),
        }
        event = trade_event_from_dict(payload)
        _process_event_with_logging(listener, event, dry_run=False)

    await stream._run_forever()  # type: ignore[attr-defined]


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Listen for trade executions and update local PnL state.")
    parser.add_argument("--mode", choices=["alpaca", "stdin", "file"], default="stdin")
    parser.add_argument("--events-file", type=Path, default=None)
    parser.add_argument("--state-suffix", default=None)
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--signal-max-age", type=int, default=SIGNAL_MAX_AGE_SECONDS)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    if args.check_config:
        _check_config()

    listener = TradeExecutionMonitor(state_suffix=args.state_suffix)
    if args.mode == "stdin":
        _run_stdin(listener, dry_run=args.dry_run, signal_max_age=args.signal_max_age)
    elif args.mode == "file":
        if args.events_file is None:
            raise SystemExit("--events-file is required when --mode=file")
        _run_file(listener, args.events_file, dry_run=args.dry_run, signal_max_age=args.signal_max_age)
    else:
        asyncio.run(_run_alpaca(listener, args.paper))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
