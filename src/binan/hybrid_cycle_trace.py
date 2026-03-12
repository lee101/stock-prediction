from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT_TRACE_DIR = REPO / "strategy_state" / "hybrid_trade_cycles"
TRACE_TAG = "hybrid-cycle"


def _trace_logging_disabled() -> bool:
    argv0 = Path(sys.argv[0]).name.lower()
    return bool(
        os.environ.get("PYTEST_CURRENT_TEST")
        or os.environ.get("PYTEST_VERSION")
        or os.environ.get("BINANCE_HYBRID_DISABLE_TRACE") == "1"
        or "pytest" in argv0
    )


def _trace_log_path(
    *,
    log_dir: Path = DEFAULT_TRACE_DIR,
    tag: str = TRACE_TAG,
    now: datetime | None = None,
) -> Path:
    ts = now or datetime.now(timezone.utc)
    return log_dir / f"{tag}_{ts.strftime('%Y%m%d')}.jsonl"


def append_cycle_snapshot(
    snapshot: Mapping[str, Any],
    *,
    log_dir: Path = DEFAULT_TRACE_DIR,
    tag: str = TRACE_TAG,
) -> Path | None:
    if _trace_logging_disabled():
        return None
    now = datetime.now(timezone.utc)
    record = dict(snapshot)
    record.setdefault("event", "cycle_snapshot")
    record.setdefault("ts", now.isoformat())
    record.setdefault("cycle_started_at", record["ts"])
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        path = _trace_log_path(log_dir=log_dir, tag=tag, now=now)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str, sort_keys=True) + "\n")
        return path
    except Exception:
        return None


def _normalize_ts(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_cycle_snapshots(
    *,
    log_dir: Path = DEFAULT_TRACE_DIR,
    start: Any = None,
    end: Any = None,
    live_only: bool = False,
) -> list[dict[str, Any]]:
    start_ts = _normalize_ts(start)
    end_ts = _normalize_ts(end)
    snapshots: list[dict[str, Any]] = []
    if not log_dir.exists():
        return snapshots
    for path in sorted(log_dir.glob(f"{TRACE_TAG}_*.jsonl")):
        with open(path, encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                if record.get("event") != "cycle_snapshot":
                    continue
                cycle_ts = _normalize_ts(record.get("cycle_started_at") or record.get("ts"))
                if cycle_ts is None:
                    continue
                if start_ts is not None and cycle_ts < start_ts:
                    continue
                if end_ts is not None and cycle_ts > end_ts:
                    continue
                if live_only and str(record.get("mode", "")).lower() != "live":
                    continue
                normalized = dict(record)
                normalized["cycle_started_at"] = cycle_ts.isoformat()
                snapshots.append(normalized)
    snapshots.sort(key=lambda row: row.get("cycle_started_at", ""))
    return snapshots


def normalize_exchange_order(order: Mapping[str, Any]) -> dict[str, Any]:
    created_ts = _normalize_ts(order.get("time"))
    updated_ts = _normalize_ts(order.get("updateTime"))
    return {
        "order_id": _safe_int(order.get("order_id", order.get("orderId"))),
        "symbol": str(order.get("symbol", "") or "").upper(),
        "side": str(order.get("side", "") or "").upper(),
        "type": str(order.get("type", "") or "").upper(),
        "status": str(order.get("status", "") or "").upper(),
        "price": _safe_float(order.get("price")),
        "orig_qty": _safe_float(order.get("orig_qty", order.get("origQty", order.get("qty")))),
        "executed_qty": _safe_float(order.get("executed_qty", order.get("executedQty"))),
        "quote_qty": _safe_float(order.get("quote_qty", order.get("cummulativeQuoteQty"))),
        "time": created_ts.isoformat() if created_ts is not None else None,
        "update_time": updated_ts.isoformat() if updated_ts is not None else None,
    }


def normalize_exchange_trade(trade: Mapping[str, Any]) -> dict[str, Any]:
    trade_ts = _normalize_ts(trade.get("time"))
    side = trade.get("side")
    if side is None and "isBuyer" in trade:
        side = "BUY" if bool(trade.get("isBuyer")) else "SELL"
    return {
        "order_id": _safe_int(trade.get("order_id", trade.get("orderId"))),
        "symbol": str(trade.get("symbol", "") or "").upper(),
        "side": str(side or "").upper(),
        "price": _safe_float(trade.get("price")),
        "qty": _safe_float(trade.get("qty")),
        "quote_qty": _safe_float(trade.get("quote_qty", trade.get("quoteQty"))),
        "time": trade_ts.isoformat() if trade_ts is not None else None,
    }


def extract_expected_orders(snapshots: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    expected: list[dict[str, Any]] = []
    for snapshot in snapshots:
        cycle_ts = str(snapshot.get("cycle_started_at") or snapshot.get("ts") or "")
        cycle_id = str(snapshot.get("cycle_id", "") or "")
        cycle_kind = str(snapshot.get("cycle_kind", "") or "")
        for detail in snapshot.get("symbols_detail", []) or []:
            if not isinstance(detail, dict):
                continue
            symbol = str(detail.get("symbol", "") or "")
            market_symbol = str(detail.get("market_symbol", "") or "").upper()
            for action in detail.get("actions", []) or []:
                if not isinstance(action, dict):
                    continue
                side = str(action.get("side", "") or "").upper()
                if side not in {"BUY", "SELL"}:
                    continue
                status = str(action.get("status", "") or "")
                order_payload: dict[str, Any] | None = None
                source = ""
                if isinstance(action.get("placed_order"), dict):
                    order_payload = action["placed_order"]
                    source = "placed"
                elif status == "already_working":
                    matched = action.get("matched_open_orders") or []
                    if matched and isinstance(matched[0], dict):
                        order_payload = matched[0]
                        source = "already_working"
                if order_payload is None:
                    continue
                expected.append(
                    {
                        "cycle_id": cycle_id,
                        "cycle_started_at": cycle_ts,
                        "cycle_kind": cycle_kind,
                        "mode": str(snapshot.get("mode", "") or ""),
                        "trace_symbol": symbol,
                        "action_kind": str(action.get("kind", "") or ""),
                        "action_status": status,
                        "source": source,
                        "symbol": str(order_payload.get("symbol", market_symbol) or market_symbol).upper(),
                        "side": str(order_payload.get("side", side) or side).upper(),
                        "order_id": _safe_int(order_payload.get("order_id", order_payload.get("orderId"))),
                        "price": _safe_float(order_payload.get("price", action.get("desired_price"))),
                        "qty": _safe_float(
                            order_payload.get(
                                "orig_qty",
                                order_payload.get("origQty", order_payload.get("qty", action.get("desired_qty"))),
                            )
                        ),
                    }
                )
    return expected


def match_expected_orders(
    expected_orders: Sequence[Mapping[str, Any]],
    actual_orders: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    normalized_actual = [normalize_exchange_order(order) for order in actual_orders]
    actual_by_id = {
        order["order_id"]: order
        for order in normalized_actual
        if order.get("order_id") is not None
    }
    results: list[dict[str, Any]] = []
    matched_ids: set[int] = set()

    for expected in expected_orders:
        expected_id = _safe_int(expected.get("order_id"))
        actual = actual_by_id.get(expected_id) if expected_id is not None else None
        matched = actual is not None
        if matched and expected_id is not None:
            matched_ids.add(expected_id)
        results.append(
            {
                "cycle_id": str(expected.get("cycle_id", "") or ""),
                "cycle_started_at": str(expected.get("cycle_started_at", "") or ""),
                "symbol": str(expected.get("symbol", "") or "").upper(),
                "side": str(expected.get("side", "") or "").upper(),
                "order_id": expected_id,
                "expected_price": _safe_float(expected.get("price")),
                "expected_qty": _safe_float(expected.get("qty")),
                "matched": matched,
                "actual": actual,
            }
        )

    unexpected = [
        order for order in normalized_actual
        if order.get("order_id") is not None and int(order["order_id"]) not in matched_ids
    ]
    matched_count = sum(1 for row in results if row["matched"])
    return {
        "matched_count": matched_count,
        "expected_count": len(results),
        "missing_count": len(results) - matched_count,
        "results": results,
        "unexpected_orders": unexpected,
    }


def build_cycle_windows(snapshots: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        snapshots,
        key=lambda row: str(row.get("cycle_started_at") or row.get("ts") or ""),
    )
    windows: list[dict[str, Any]] = []
    for index, snapshot in enumerate(ordered):
        start_ts = _normalize_ts(snapshot.get("cycle_started_at") or snapshot.get("ts"))
        if start_ts is None:
            continue
        next_snapshot = ordered[index + 1] if index + 1 < len(ordered) else None
        end_ts = _normalize_ts(next_snapshot.get("cycle_started_at") or next_snapshot.get("ts")) if next_snapshot else None
        windows.append(
            {
                "cycle_id": str(snapshot.get("cycle_id", "") or ""),
                "start": start_ts.isoformat(),
                "end": end_ts.isoformat() if end_ts is not None else None,
            }
        )
    return windows


def order_price_touch_summary(
    side: str,
    price: float,
    bars_5m: pd.DataFrame,
) -> dict[str, Any]:
    if bars_5m.empty:
        return {"touched": False, "first_touch_ts": None}
    normalized_side = str(side or "").upper()
    try:
        target_price = float(price)
    except (TypeError, ValueError):
        return {"touched": False, "first_touch_ts": None}
    if not math.isfinite(target_price) or target_price <= 0:
        return {"touched": False, "first_touch_ts": None}

    if normalized_side == "BUY":
        touched = bars_5m.loc[bars_5m["low"] <= target_price]
    else:
        touched = bars_5m.loc[bars_5m["high"] >= target_price]
    if touched.empty:
        return {"touched": False, "first_touch_ts": None}
    first_ts = _normalize_ts(touched.iloc[0]["timestamp"])
    return {
        "touched": True,
        "first_touch_ts": first_ts.isoformat() if first_ts is not None else None,
    }


def summarize_touch_results(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    touched = 0
    untouched = 0
    filled_without_touch = 0
    for row in rows:
        touch = bool(row.get("touched"))
        if touch:
            touched += 1
        else:
            untouched += 1
        if row.get("filled") and not touch:
            filled_without_touch += 1
    return {
        "touched": touched,
        "untouched": untouched,
        "filled_without_touch": filled_without_touch,
    }
