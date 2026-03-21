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

# ---------------------------------------------------------------------------
# Structured JSON logger
# ---------------------------------------------------------------------------

_log = logging.getLogger("trade_execution_listener")


def _setup_logging() -> None:
    """Configure structured JSON logging to stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)


def _emit(level: str, event: str, **kwargs: object) -> None:
    """Emit a structured JSON log line."""
    record: dict = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "event": event,
    }
    record.update(kwargs)
    line = json.dumps(record, default=str)
    if level == "error":
        _log.error(line)
    elif level == "warning":
        _log.warning(line)
    elif level == "debug":
        _log.debug(line)
    else:
        _log.info(line)


# ---------------------------------------------------------------------------
# Exclusion list — symbols that MUST NOT be traded (NYT rallied +51.6% SHORT)
# ---------------------------------------------------------------------------

EXCLUDED_SYMBOLS: frozenset[str] = frozenset({"NYT"})

# Signals older than this are rejected to prevent stale-order execution.
SIGNAL_MAX_AGE_SECONDS: int = 300  # 5 minutes

# Reconnect settings for Alpaca WebSocket stream.
_RECONNECT_MAX_ATTEMPTS: int = 10
_RECONNECT_BASE_DELAY_S: float = 2.0


# ---------------------------------------------------------------------------
# Signal validation helpers
# ---------------------------------------------------------------------------


def _is_stale(event: TradeEvent, max_age_seconds: int = SIGNAL_MAX_AGE_SECONDS) -> bool:
    """Return True if the event timestamp is older than max_age_seconds."""
    now = datetime.now(timezone.utc)
    # Ensure event.timestamp is tz-aware
    ts = event.timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age = (now - ts).total_seconds()
    return age > max_age_seconds


def _is_excluded(event: TradeEvent) -> bool:
    """Return True if the symbol is in the hard-exclude list."""
    return event.symbol.upper() in EXCLUDED_SYMBOLS


def _should_process(event: TradeEvent, max_age_seconds: int = SIGNAL_MAX_AGE_SECONDS) -> bool:
    """Validate event and return True if it should be processed."""
    if _is_excluded(event):
        _emit(
            "warning",
            "signal_excluded",
            symbol=event.symbol,
            reason="symbol_in_exclusion_list",
        )
        return False
    if _is_stale(event, max_age_seconds=max_age_seconds):
        ts_str = event.timestamp.isoformat() if event.timestamp else "unknown"
        _emit(
            "warning",
            "signal_rejected_stale",
            symbol=event.symbol,
            signal_ts=ts_str,
            max_age_seconds=max_age_seconds,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Event processing with logging
# ---------------------------------------------------------------------------


def _process_event_with_logging(
    listener: TradeExecutionMonitor,
    event: TradeEvent,
    dry_run: bool = False,
    max_age_seconds: int = SIGNAL_MAX_AGE_SECONDS,
) -> None:
    """Process a single TradeEvent after validation, with structured logging."""
    _emit(
        "info",
        "signal_received",
        symbol=event.symbol,
        side=event.side,
        quantity=event.quantity,
        price=event.price,
        signal_ts=event.timestamp.isoformat(),
        dry_run=dry_run,
    )

    if not _should_process(event, max_age_seconds=max_age_seconds):
        return

    if dry_run:
        _emit(
            "info",
            "dry_run_would_process",
            symbol=event.symbol,
            side=event.side,
            quantity=event.quantity,
            price=event.price,
        )
        return

    try:
        closures = listener.process_event(event)
        for closure in closures:
            _emit(
                "info",
                "position_closed",
                symbol=event.symbol,
                entry_side=closure.entry_side,
                qty=closure.qty,
                pnl=closure.pnl,
                closed_at=closure.timestamp.isoformat(),
            )
    except Exception as exc:  # noqa: BLE001
        _emit(
            "error",
            "process_event_failed",
            symbol=event.symbol,
            side=event.side,
            error=str(exc),
            exc_type=type(exc).__name__,
        )


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------


def _heartbeat(extra: Optional[dict] = None) -> None:
    """Emit a periodic heartbeat log line for liveness monitoring."""
    _emit("info", "heartbeat", **(extra or {}))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log signals but do not submit or record any trades.",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate configuration (API keys, symbol exclusions) and exit.",
    )
    parser.add_argument(
        "--signal-max-age",
        type=int,
        default=SIGNAL_MAX_AGE_SECONDS,
        help=f"Reject signals older than this many seconds (default {SIGNAL_MAX_AGE_SECONDS}).",
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=int,
        default=60,
        help="Emit a heartbeat log line every N seconds (0 = disabled, default 60).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def _check_config(dry_run: bool = False) -> None:
    """Validate configuration and exit 0 on success, 1 on failure."""
    errors: list[str] = []

    if not ALP_KEY_ID_PROD or ALP_KEY_ID_PROD.startswith("AKAONQRN"):
        errors.append("ALP_KEY_ID_PROD is not set or still using placeholder default.")
    if not ALP_SECRET_KEY_PROD or ALP_SECRET_KEY_PROD.startswith("GYwPufjn"):
        errors.append("ALP_SECRET_KEY_PROD is not set or still using placeholder default.")

    # Verify EXCLUDED_SYMBOLS is non-empty and contains NYT
    if "NYT" not in EXCLUDED_SYMBOLS:
        errors.append("EXCLUDED_SYMBOLS does not contain NYT — this symbol must be excluded.")

    if errors:
        for err in errors:
            _emit("error", "config_invalid", message=err)
        raise SystemExit(1)

    _emit(
        "info",
        "config_ok",
        excluded_symbols=sorted(EXCLUDED_SYMBOLS),
        signal_max_age_seconds=SIGNAL_MAX_AGE_SECONDS,
        dry_run=dry_run,
        api_key_id_prefix=ALP_KEY_ID_PROD[:6] + "..." if ALP_KEY_ID_PROD else "",
    )
    raise SystemExit(0)


# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------


def _run_stdin(
    listener: TradeExecutionMonitor,
    dry_run: bool = False,
    max_age_seconds: int = SIGNAL_MAX_AGE_SECONDS,
    heartbeat_interval: int = 60,
) -> None:
    last_heartbeat = time.monotonic()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            _emit("error", "json_parse_failed", raw=line[:200], error=str(exc))
            continue
        try:
            event = trade_event_from_dict(payload)
        except Exception as exc:  # noqa: BLE001
            _emit("error", "event_parse_failed", payload=payload, error=str(exc))
            continue
        _process_event_with_logging(listener, event, dry_run=dry_run, max_age_seconds=max_age_seconds)

        now = time.monotonic()
        if heartbeat_interval > 0 and (now - last_heartbeat) >= heartbeat_interval:
            _heartbeat({"mode": "stdin"})
            last_heartbeat = now


def _run_file(
    listener: TradeExecutionMonitor,
    events_file: Path,
    dry_run: bool = False,
    max_age_seconds: int = SIGNAL_MAX_AGE_SECONDS,
) -> None:
    _emit("info", "file_mode_start", path=str(events_file))
    try:
        events = load_events_from_file(events_file)
    except Exception as exc:  # noqa: BLE001
        _emit("error", "file_load_failed", path=str(events_file), error=str(exc))
        raise
    _emit("info", "file_loaded", path=str(events_file), event_count=len(events))
    for event in events:
        _process_event_with_logging(listener, event, dry_run=dry_run, max_age_seconds=max_age_seconds)


async def _run_alpaca(
    listener: TradeExecutionMonitor,
    paper: bool,
    dry_run: bool = False,
    heartbeat_interval: int = 60,
) -> None:  # pragma: no cover - network path
    if Stream is None:
        raise RuntimeError("alpaca-trade-api is not installed; cannot use --mode=alpaca")

    attempt = 0
    last_heartbeat = time.monotonic()

    while True:
        attempt += 1
        _emit("info", "alpaca_stream_connecting", attempt=attempt, paper=paper)
        try:
            stream = Stream(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD, paper=paper)

            @stream.on_trade_updates
            async def _(data):
                nonlocal last_heartbeat
                try:
                    payload = {
                        "symbol": (
                            getattr(data, "symbol", None)
                            or getattr(getattr(data, "order", None), "symbol", None)
                        ),
                        "side": getattr(getattr(data, "order", None), "side", None),
                        "quantity": float(
                            getattr(data, "qty", getattr(data, "filled_qty", 0.0)) or 0.0
                        ),
                        "price": float(
                            getattr(data, "price", getattr(data, "filled_avg_price", 0.0)) or 0.0
                        ),
                        "timestamp": getattr(
                            data, "timestamp", datetime.now(timezone.utc).isoformat()
                        ),
                    }
                    event = trade_event_from_dict(payload)
                    _process_event_with_logging(listener, event, dry_run=dry_run)
                except Exception as exc:  # noqa: BLE001
                    _emit(
                        "error",
                        "trade_update_callback_failed",
                        error=str(exc),
                        exc_type=type(exc).__name__,
                    )

                now = time.monotonic()
                if heartbeat_interval > 0 and (now - last_heartbeat) >= heartbeat_interval:
                    _heartbeat({"mode": "alpaca", "paper": paper})
                    last_heartbeat = now

            await stream._run_forever()  # type: ignore[attr-defined]

        except asyncio.CancelledError:
            _emit("info", "alpaca_stream_cancelled")
            raise
        except Exception as exc:  # noqa: BLE001
            delay = min(_RECONNECT_BASE_DELAY_S * (2 ** (attempt - 1)), 120.0)
            _emit(
                "error",
                "alpaca_stream_error",
                attempt=attempt,
                error=str(exc),
                exc_type=type(exc).__name__,
                reconnect_in_seconds=delay,
            )
            if attempt >= _RECONNECT_MAX_ATTEMPTS:
                _emit(
                    "error",
                    "alpaca_stream_max_retries_exceeded",
                    max_attempts=_RECONNECT_MAX_ATTEMPTS,
                )
                raise
            await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    _setup_logging()
    args = _parse_args(argv)
    max_age = args.signal_max_age

    if args.check_config:
        _check_config(dry_run=args.dry_run)
        return  # _check_config raises SystemExit; this is unreachable but defensive

    if args.dry_run:
        _emit("info", "dry_run_mode_active", note="No trades will be recorded.")

    _emit(
        "info",
        "listener_start",
        mode=args.mode,
        dry_run=args.dry_run,
        paper=args.paper,
        signal_max_age_seconds=max_age,
        excluded_symbols=sorted(EXCLUDED_SYMBOLS),
    )

    listener = TradeExecutionMonitor(state_suffix=args.state_suffix)

    if args.mode == "stdin":
        _run_stdin(
            listener,
            dry_run=args.dry_run,
            max_age_seconds=max_age,
            heartbeat_interval=args.heartbeat_interval,
        )
    elif args.mode == "file":
        if args.events_file is None:
            raise SystemExit("--events-file is required when --mode=file")
        _run_file(listener, args.events_file, dry_run=args.dry_run, max_age_seconds=max_age)
    else:
        asyncio.run(
            _run_alpaca(
                listener,
                args.paper,
                dry_run=args.dry_run,
                heartbeat_interval=args.heartbeat_interval,
            )
        )


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
