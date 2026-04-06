#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.binan.binance_margin import get_all_margin_orders, get_margin_trades
from src.binan.binance_wrapper import get_client
from src.binan.history_dedupe import dedupe_margin_orders, dedupe_margin_trades
from src.binan.hybrid_cycle_trace import (
    DEFAULT_TRACE_DIR,
    build_cycle_windows,
    extract_expected_orders,
    load_cycle_snapshots,
    match_expected_orders,
    normalize_exchange_trade,
    order_price_touch_summary,
    summarize_touch_results,
)


FINAL_ORDER_STATUSES = {"FILLED", "CANCELED", "EXPIRED", "REJECTED"}


def _to_utc_ts(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _chunk_windows(
    start_ts: pd.Timestamp, end_ts: pd.Timestamp, *, max_span: pd.Timedelta
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start_ts
    while cursor < end_ts:
        chunk_end = min(cursor + max_span, end_ts)
        windows.append((cursor, chunk_end))
        cursor = chunk_end
    return windows


def pull_margin_orders(symbols: list[str], start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        for chunk_start, chunk_end in _chunk_windows(start_ts, end_ts, max_span=pd.Timedelta(days=1)):
            raw = get_all_margin_orders(
                symbol,
                start_time=int(chunk_start.timestamp() * 1000),
                end_time=int(chunk_end.timestamp() * 1000),
                limit=500,
            )
            for order in raw:
                row = dict(order)
                row["symbol"] = symbol
                rows.append(row)
    return dedupe_margin_orders(rows)


def pull_margin_trades(symbols: list[str], start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        for chunk_start, chunk_end in _chunk_windows(start_ts, end_ts, max_span=pd.Timedelta(days=1)):
            raw = get_margin_trades(
                symbol,
                start_time=int(chunk_start.timestamp() * 1000),
                end_time=int(chunk_end.timestamp() * 1000),
                limit=1000,
            )
            for trade in raw:
                row = dict(trade)
                row["symbol"] = symbol
                rows.append(row)
    return dedupe_margin_trades(rows)


def fetch_5m_bars(symbol: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    client = get_client()
    rows: list[dict[str, Any]] = []
    cursor_ms = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)
    step_ms = 5 * 60 * 1000
    while cursor_ms < end_ms:
        payload = client.get_klines(
            symbol=symbol,
            interval="5m",
            startTime=cursor_ms,
            endTime=end_ms,
            limit=1000,
        )
        if not payload:
            break
        for item in payload:
            rows.append(
                {
                    "timestamp": pd.Timestamp(int(item[0]), unit="ms", tz="UTC"),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                }
            )
        last_open_ms = int(payload[-1][0])
        next_cursor = last_open_ms + step_ms
        if next_cursor <= cursor_ms:
            break
        cursor_ms = next_cursor
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(rows).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return frame


def build_touch_report(
    match_payload: dict[str, Any],
    trade_rows: list[dict[str, Any]],
    cycle_windows: list[dict[str, Any]],
    end_ts: pd.Timestamp,
) -> list[dict[str, Any]]:
    trades = [normalize_exchange_trade(trade) for trade in trade_rows]
    trades_by_order_id: dict[int, list[dict[str, Any]]] = {}
    for trade in trades:
        order_id = trade.get("order_id")
        if order_id is None:
            continue
        trades_by_order_id.setdefault(int(order_id), []).append(trade)

    window_by_cycle = {row["cycle_id"]: row for row in cycle_windows}
    bars_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
    touch_rows: list[dict[str, Any]] = []

    for result in match_payload["results"]:
        actual = result.get("actual")
        if not actual:
            continue
        cycle_window = window_by_cycle.get(result["cycle_id"], {})
        analysis_start = _to_utc_ts(actual.get("time") or result["cycle_started_at"])
        cycle_end_raw = cycle_window.get("end")
        cycle_end = _to_utc_ts(cycle_end_raw) if cycle_end_raw else end_ts
        actual_update = actual.get("update_time")
        if actual_update and actual.get("status") in FINAL_ORDER_STATUSES:
            analysis_end = min(_to_utc_ts(actual_update), cycle_end)
        else:
            analysis_end = cycle_end
        if analysis_end <= analysis_start:
            analysis_end = analysis_start + timedelta(minutes=5)
        cache_key = (actual["symbol"], analysis_start.isoformat(), analysis_end.isoformat())
        bars_5m = bars_cache.get(cache_key)
        if bars_5m is None:
            bars_5m = fetch_5m_bars(actual["symbol"], analysis_start, analysis_end)
            bars_cache[cache_key] = bars_5m
        touch = order_price_touch_summary(actual["side"], float(actual["price"] or 0.0), bars_5m)
        order_id = actual.get("order_id")
        order_trades = trades_by_order_id.get(int(order_id), []) if order_id is not None else []
        executed_qty = float(actual.get("executed_qty") or 0.0)
        touch_rows.append(
            {
                "cycle_id": result["cycle_id"],
                "symbol": actual["symbol"],
                "side": actual["side"],
                "order_id": order_id,
                "status": actual.get("status"),
                "filled": executed_qty > 0 or bool(order_trades),
                "touched": bool(touch["touched"]),
                "first_touch_ts": touch["first_touch_ts"],
                "analysis_start": analysis_start.isoformat(),
                "analysis_end": analysis_end.isoformat(),
            }
        )
    return touch_rows


def build_report(
    *,
    snapshots: list[dict[str, Any]],
    expected_orders: list[dict[str, Any]],
    actual_orders: list[dict[str, Any]],
    actual_trades: list[dict[str, Any]],
    touch_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    match_payload = match_expected_orders(expected_orders, actual_orders)
    touch_summary = summarize_touch_results(touch_rows)
    return {
        "summary": {
            "snapshots": len(snapshots),
            "expected_orders": match_payload["expected_count"],
            "matched_orders": match_payload["matched_count"],
            "missing_orders": match_payload["missing_count"],
            "unexpected_orders": len(match_payload["unexpected_orders"]),
            "trade_rows": len(actual_trades),
            **touch_summary,
        },
        "matches": match_payload["results"],
        "unexpected_orders": match_payload["unexpected_orders"],
        "touch_rows": touch_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate hybrid cycle snapshots against live Binance margin orders and 5m price touches."
    )
    parser.add_argument("--start", required=True, help="UTC start timestamp")
    parser.add_argument("--end", default=None, help="UTC end timestamp (defaults to now)")
    parser.add_argument("--log-dir", default=str(DEFAULT_TRACE_DIR), help="Hybrid cycle JSONL directory")
    parser.add_argument("--live-only", action="store_true", help="Only validate live snapshots")
    parser.add_argument(
        "--symbols", nargs="*", default=None, help="Optional symbol override, e.g. BTCUSDT ETHUSDT SOLUSDT"
    )
    parser.add_argument(
        "--order-lookback-hours",
        type=float,
        default=48.0,
        help="Extra lookback window for matching already-working orders created before the first validated cycle.",
    )
    parser.add_argument("--output-json", default=None, help="Optional path to write the full validation report")
    args = parser.parse_args()

    start_ts = _to_utc_ts(args.start)
    end_ts = _to_utc_ts(args.end) if args.end else pd.Timestamp.now(tz="UTC")
    log_dir = Path(args.log_dir)

    snapshots = load_cycle_snapshots(log_dir=log_dir, start=start_ts, end=end_ts, live_only=args.live_only)
    if not snapshots:
        raise SystemExit(f"No cycle snapshots found in {log_dir} between {start_ts} and {end_ts}")

    expected_orders = extract_expected_orders(snapshots)
    symbols = sorted({row["symbol"] for row in expected_orders if row.get("symbol")})
    if args.symbols:
        symbols = [str(symbol).upper() for symbol in args.symbols]
    if not symbols:
        raise SystemExit("No symbols resolved from snapshots; pass --symbols explicitly.")

    order_query_start = start_ts - pd.Timedelta(hours=max(0.0, float(args.order_lookback_hours)))
    actual_orders = pull_margin_orders(symbols, order_query_start, end_ts)
    actual_trades = pull_margin_trades(symbols, start_ts, end_ts)
    match_payload = match_expected_orders(expected_orders, actual_orders)
    cycle_windows = build_cycle_windows(snapshots)
    touch_rows = build_touch_report(match_payload, actual_trades, cycle_windows, end_ts)
    report = build_report(
        snapshots=snapshots,
        expected_orders=expected_orders,
        actual_orders=actual_orders,
        actual_trades=actual_trades,
        touch_rows=touch_rows,
    )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, default=str))

    summary = report["summary"]
    print(f"Snapshots: {summary['snapshots']}")
    print(f"Expected orders: {summary['expected_orders']}")
    print(f"Matched orders: {summary['matched_orders']}")
    print(f"Missing orders: {summary['missing_orders']}")
    print(f"Unexpected orders: {summary['unexpected_orders']}")
    print(f"Trade rows: {summary['trade_rows']}")
    print(f"Touched: {summary['touched']}")
    print(f"Untouched: {summary['untouched']}")
    print(f"Filled without touch: {summary['filled_without_touch']}")


if __name__ == "__main__":
    main()
