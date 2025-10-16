from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .config import DashboardConfig
from .db import DashboardDatabase, MetricEntry

logger = logging.getLogger(__name__)

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
TIMESTAMP_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC")

TRADE_POSITION_RE = re.compile(
    r"(?P<symbol>[A-Z./]+): Current position: (?P<current_qty>-?\d+(?:\.\d+)?) qty "
    r"\(\$(?P<current_value>[\d,\.]+)\), Target: (?P<target_qty>-?\d+(?:\.\d+)?) qty "
    r"\(\$(?P<target_value>[\d,\.]+)\)"
)
TRADE_TARGET_RE = re.compile(
    r"Target quantity for (?P<symbol>[A-Z./]+): (?P<target_qty>-?\d+(?:\.\d+)?) at price (?P<price>-?\d+(?:\.\d+)?)"
)
TRADE_PRED_HIGH_RE = re.compile(
    r"Placing .*order for (?P<symbol>[A-Z./]+).*predicted_high=(?P<predicted_high>-?\d+(?:\.\d+)?)",
    flags=re.IGNORECASE,
)
TRADE_PRED_LOW_RE = re.compile(
    r"takeprofit.*predicted_low=(?P<predicted_low>-?\d+(?:\.\d+)?)",
    flags=re.IGNORECASE,
)

ALPACA_RETRIEVED_RE = re.compile(r"Retrieved (?P<count>\d+) total positions", flags=re.IGNORECASE)
ALPACA_FILTERED_RE = re.compile(r"After filtering, (?P<count>\d+) positions remain", flags=re.IGNORECASE)
ALPACA_OPEN_ORDERS_RE = re.compile(r"Found (?P<count>\d+) open orders", flags=re.IGNORECASE)
ALPACA_MATCH_RE = re.compile(r"Found matching position for (?P<symbol>[A-Z./]+)", flags=re.IGNORECASE)
ALPACA_BACKOUT_RE = re.compile(
    r"Position side: (?P<side>long|short), pct_above_market: (?P<pct>-?\d+(?:\.\d+)?), "
    r"minutes_since_start: (?P<minutes>-?\d+(?:\.\d+)?), progress: (?P<progress>-?\d+(?:\.\d+)?)",
    flags=re.IGNORECASE,
)


def _strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def _parse_timestamp(line: str) -> Optional[datetime]:
    match = TIMESTAMP_RE.search(line)
    if not match:
        return None
    ts = datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S")
    return ts.replace(tzinfo=timezone.utc)


def _extract_message(line: str) -> str:
    parts = line.split("|", 4)
    if len(parts) >= 5:
        return parts[4].strip()
    return line.strip()


def _to_float(value: str) -> Optional[float]:
    try:
        return float(value.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def _record_metrics(
    db: DashboardDatabase,
    recorded_at: datetime,
    source: str,
    symbol: Optional[str],
    message: str,
    items: Sequence[Tuple[str, Optional[float]]],
) -> int:
    stored = 0
    message_snippet = message.strip()
    if len(message_snippet) > 500:
        message_snippet = f"{message_snippet[:497]}..."
    for metric, value in items:
        if value is None:
            continue
        db.record_metric(
            MetricEntry(
                recorded_at=recorded_at,
                source=source,
                symbol=symbol.upper() if symbol else None,
                metric=metric,
                value=value,
                message=message_snippet,
            )
        )
        stored += 1
    return stored


def _read_new_lines(path: Path, offset: int) -> Tuple[int, List[str]]:
    if not path.exists():
        return 0, []
    file_size = path.stat().st_size
    start = offset if offset <= file_size else 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        handle.seek(start)
        lines = handle.readlines()
        new_offset = handle.tell()
    return new_offset, lines


def _process_trade_log(path: Path, db: DashboardDatabase) -> int:
    offset = db.get_log_offset(path)
    new_offset, lines = _read_new_lines(path, offset)
    processed = 0
    for raw_line in lines:
        clean_line = _strip_ansi(raw_line).strip()
        if not clean_line:
            continue
        recorded_at = _parse_timestamp(clean_line)
        if not recorded_at:
            continue
        message = _extract_message(clean_line)

        position_match = TRADE_POSITION_RE.search(message)
        if position_match:
            symbol = position_match.group("symbol")
            metrics = [
                ("current_qty", _to_float(position_match.group("current_qty"))),
                ("current_value", _to_float(position_match.group("current_value"))),
                ("target_qty", _to_float(position_match.group("target_qty"))),
                ("target_value", _to_float(position_match.group("target_value"))),
            ]
            processed += _record_metrics(db, recorded_at, "trade_stock_e2e", symbol, message, metrics)
            continue

        target_match = TRADE_TARGET_RE.search(message)
        if target_match:
            symbol = target_match.group("symbol")
            metrics = [
                ("target_qty", _to_float(target_match.group("target_qty"))),
                ("target_price", _to_float(target_match.group("price"))),
            ]
            processed += _record_metrics(db, recorded_at, "trade_stock_e2e", symbol, message, metrics)
            continue

        pred_high_match = TRADE_PRED_HIGH_RE.search(message)
        if pred_high_match:
            symbol = pred_high_match.group("symbol")
            metrics = [("predicted_high", _to_float(pred_high_match.group("predicted_high")))]
            processed += _record_metrics(db, recorded_at, "trade_stock_e2e", symbol, message, metrics)
            continue

        pred_low_match = TRADE_PRED_LOW_RE.search(message)
        if pred_low_match:
            # Attempt to capture symbol from context within message if present
            symbol_match = re.search(r"for ([A-Z./]+)", message)
            symbol = symbol_match.group(1) if symbol_match else None
            metrics = [("predicted_low", _to_float(pred_low_match.group("predicted_low")))]
            processed += _record_metrics(db, recorded_at, "trade_stock_e2e", symbol, message, metrics)
            continue

    if new_offset != offset:
        db.update_log_offset(path, new_offset)
    return processed


def _process_alpaca_log(path: Path, db: DashboardDatabase) -> int:
    offset = db.get_log_offset(path)
    new_offset, lines = _read_new_lines(path, offset)
    processed = 0
    last_symbol: Optional[str] = None
    for raw_line in lines:
        clean_line = _strip_ansi(raw_line).strip()
        if not clean_line:
            continue
        recorded_at = _parse_timestamp(clean_line)
        if not recorded_at:
            continue
        message = _extract_message(clean_line)

        retrieved_match = ALPACA_RETRIEVED_RE.search(message)
        if retrieved_match:
            metrics = [("total_positions", _to_float(retrieved_match.group("count")))]
            processed += _record_metrics(db, recorded_at, "alpaca_cli", None, message, metrics)
            last_symbol = None
            continue

        filtered_match = ALPACA_FILTERED_RE.search(message)
        if filtered_match:
            metrics = [("filtered_positions", _to_float(filtered_match.group("count")))]
            processed += _record_metrics(db, recorded_at, "alpaca_cli", None, message, metrics)
            continue

        open_orders_match = ALPACA_OPEN_ORDERS_RE.search(message)
        if open_orders_match:
            metrics = [("open_orders", _to_float(open_orders_match.group("count")))]
            processed += _record_metrics(db, recorded_at, "alpaca_cli", None, message, metrics)
            continue

        match_symbol = ALPACA_MATCH_RE.search(message)
        if match_symbol:
            last_symbol = match_symbol.group("symbol").upper()
            metrics = [("backout_match", 1.0)]
            processed += _record_metrics(db, recorded_at, "alpaca_cli", last_symbol, message, metrics)
            continue

        backout_match = ALPACA_BACKOUT_RE.search(message)
        if backout_match:
            symbol = last_symbol
            metrics = [
                ("pct_above_market", _to_float(backout_match.group("pct"))),
                ("minutes_since_start", _to_float(backout_match.group("minutes"))),
                ("progress", _to_float(backout_match.group("progress"))),
            ]
            processed += _record_metrics(db, recorded_at, "alpaca_cli", symbol, message, metrics)
            continue

        if "no positions found" in message.lower():
            last_symbol = None

    if new_offset != offset:
        db.update_log_offset(path, new_offset)
    return processed


def collect_log_metrics(config: DashboardConfig, db: DashboardDatabase) -> int:
    total_metrics = 0
    for name, path in config.log_files.items():
        try:
            if name == "trade":
                total_metrics += _process_trade_log(path, db)
            elif name == "alpaca":
                total_metrics += _process_alpaca_log(path, db)
            else:
                logger.warning("No parser registered for log type '%s' (%s)", name, path)
        except Exception:
            logger.exception("Failed processing log '%s' at %s", name, path)
    return total_metrics


__all__ = ["collect_log_metrics"]
