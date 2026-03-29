#!/usr/bin/env python3
"""Audit recent worksteal actions against live Binance orders and 5m touches."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from binance.client import Client

from binance_worksteal.trade_live import get_binance_pair
from src.binan.hybrid_cycle_trace import order_price_touch_summary


STABLE_ASSETS = {"USDT", "FDUSD", "BUSD", "USDC"}


def _utc_ts(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _load_worksteal_actions(log_path: Path, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _load_jsonl(log_path):
        ts = _utc_ts(row.get("timestamp"))
        if ts is None or ts < start_ts or ts > end_ts:
            continue
        if bool(row.get("dry_run")):
            continue
        symbol = str(row.get("symbol") or "").upper().strip()
        side = str(row.get("side") or "").lower().strip()
        if not symbol or side not in {"staged_buy", "staged_sell", "buy", "sell"}:
            continue
        rows.append(
            {
                "timestamp": ts.isoformat(),
                "symbol": symbol,
                "pair": get_binance_pair(symbol, prefer_fdusd=False),
                "side": side,
                "price": float(row.get("price") or 0.0),
                "quantity": float(row.get("quantity") or 0.0),
                "reason": str(row.get("reason") or ""),
            }
        )
    rows.sort(key=lambda item: item["timestamp"])
    return rows


def _order_timestamp(order: dict[str, Any]) -> pd.Timestamp | None:
    raw_ts = order.get("updateTime") or order.get("time")
    if raw_ts is None:
        return None
    try:
        return pd.Timestamp(int(raw_ts), unit="ms", tz="UTC")
    except (TypeError, ValueError):
        return None


def _price_distance_bps(a: float, b: float) -> float:
    ref = max(abs(float(a or 0.0)), abs(float(b or 0.0)), 1e-9)
    return abs(float(a or 0.0) - float(b or 0.0)) / ref * 10_000.0


def _matching_exchange_orders(
    *,
    action: dict[str, Any],
    exchange_orders: list[dict[str, Any]],
    end_ts: pd.Timestamp,
) -> list[dict[str, Any]]:
    desired_side = "BUY" if action["side"] in {"staged_buy", "buy"} else "SELL"
    action_ts = pd.Timestamp(action["timestamp"])
    matches: list[dict[str, Any]] = []
    for order in exchange_orders:
        order_ts = _order_timestamp(order)
        if order_ts is None or order_ts < action_ts - pd.Timedelta(minutes=2) or order_ts > end_ts:
            continue
        if str(order.get("symbol") or "").upper() != action["pair"]:
            continue
        if str(order.get("side") or "").upper() != desired_side:
            continue
        if _price_distance_bps(float(order.get("price") or 0.0), action["price"]) > 25.0:
            continue
        matches.append(
            {
                "timestamp": order_ts.isoformat(),
                "status": str(order.get("status") or ""),
                "price": float(order.get("price") or 0.0),
                "orig_qty": float(order.get("origQty") or 0.0),
                "executed_qty": float(order.get("executedQty") or 0.0),
                "order_id": order.get("orderId"),
            }
        )
    matches.sort(key=lambda item: item["timestamp"])
    return matches


def _fetch_margin_orders(client: Client, pairs: list[str], start_ts: pd.Timestamp) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        try:
            orders = client.get_all_margin_orders(symbol=pair, isIsolated="FALSE", limit=50)
        except Exception as exc:
            rows.append(
                {
                    "symbol": pair,
                    "side": "ERROR",
                    "status": str(exc),
                    "time": None,
                    "updateTime": None,
                    "price": 0.0,
                    "origQty": 0.0,
                    "executedQty": 0.0,
                    "orderId": None,
                }
            )
            continue
        for order in orders:
            order_ts = _order_timestamp(order)
            if order_ts is None or order_ts < start_ts:
                continue
            rows.append(order)
        time.sleep(0.15)
    rows.sort(key=lambda item: (_order_timestamp(item) or pd.Timestamp.min.tz_localize("UTC")).isoformat())
    return rows


def _fetch_5m_bars(client: Client, pair: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    try:
        klines = client.get_klines(
            symbol=pair,
            interval="5m",
            startTime=int(start_ts.timestamp() * 1000),
            endTime=int(end_ts.timestamp() * 1000),
            limit=1000,
        )
    except Exception:
        return pd.DataFrame()

    rows = []
    for kline in klines:
        rows.append(
            {
                "timestamp": pd.Timestamp(kline[0], unit="ms", tz="UTC"),
                "open": float(kline[1]),
                "high": float(kline[2]),
                "low": float(kline[3]),
                "close": float(kline[4]),
                "volume": float(kline[5]),
            }
        )
    return pd.DataFrame(rows)


def _tracked_pairs_from_account(client: Client) -> list[str]:
    pairs: set[str] = set()
    try:
        account = client.get_margin_account()
    except Exception:
        account = {}
    for row in account.get("userAssets", []) if isinstance(account, dict) else []:
        asset = str(row.get("asset") or "").upper().strip()
        if not asset or asset in STABLE_ASSETS:
            continue
        free_qty = float(row.get("free") or 0.0)
        net_qty = float(row.get("netAsset") or 0.0)
        borrowed_qty = float(row.get("borrowed") or 0.0)
        if max(abs(free_qty), abs(net_qty), abs(borrowed_qty)) <= 1e-8:
            continue
        pairs.add(f"{asset}USDT")
    try:
        open_orders = client.get_open_margin_orders(isIsolated="FALSE")
    except Exception:
        open_orders = []
    for order in open_orders if isinstance(open_orders, list) else []:
        pair = str(order.get("symbol") or "").upper().strip()
        if pair:
            pairs.add(pair)
    return sorted(pairs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit worksteal recent-day live actions")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--log-path", default="binance_worksteal/trade_log.jsonl")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    from env_real import BINANCE_API_KEY, BINANCE_SECRET

    end_ts = pd.Timestamp(datetime.now(timezone.utc))
    start_ts = end_ts - pd.Timedelta(hours=int(args.hours))
    output_path = Path(args.output) if args.output else ROOT / "reports" / f"worksteal_recent_day_audit_{end_ts.strftime('%Y%m%d_%H%M%S')}.json"

    client = Client(BINANCE_API_KEY, BINANCE_SECRET)
    actions = _load_worksteal_actions(Path(args.log_path), start_ts, end_ts)
    action_pairs = {row["pair"] for row in actions}
    tracked_pairs = set(_tracked_pairs_from_account(client))
    pairs = sorted(action_pairs | tracked_pairs)
    exchange_orders = _fetch_margin_orders(client, pairs, start_ts)

    bars_by_pair: dict[str, pd.DataFrame] = {}
    for pair in sorted(action_pairs):
        bars_by_pair[pair] = _fetch_5m_bars(client, pair, start_ts, end_ts + pd.Timedelta(minutes=5))
        time.sleep(0.15)

    action_rows = []
    matched_order_ids: set[Any] = set()
    for action in actions:
        pair = action["pair"]
        bars = bars_by_pair.get(pair, pd.DataFrame())
        after_action = bars[bars["timestamp"] >= pd.Timestamp(action["timestamp"])] if not bars.empty else bars
        touch = {"touched": False, "first_touch_ts": None}
        if action["side"] in {"staged_buy", "staged_sell"}:
            touch_side = "BUY" if action["side"] == "staged_buy" else "SELL"
            touch = order_price_touch_summary(touch_side, action["price"], after_action)
        matches = _matching_exchange_orders(action=action, exchange_orders=exchange_orders, end_ts=end_ts)
        for match in matches:
            matched_order_ids.add(match["order_id"])
        action_rows.append(
            {
                **action,
                "market_touched": bool(touch.get("touched")),
                "first_touch_ts": touch.get("first_touch_ts"),
                "exchange_matches": matches,
            }
        )

    external_orders = []
    for order in exchange_orders:
        order_id = order.get("orderId")
        pair = str(order.get("symbol") or "").upper().strip()
        if order_id in matched_order_ids:
            continue
        if pair in action_pairs:
            continue
        order_ts = _order_timestamp(order)
        external_orders.append(
            {
                "timestamp": order_ts.isoformat() if order_ts is not None else None,
                "pair": pair,
                "side": str(order.get("side") or ""),
                "status": str(order.get("status") or ""),
                "price": float(order.get("price") or 0.0),
                "orig_qty": float(order.get("origQty") or 0.0),
                "executed_qty": float(order.get("executedQty") or 0.0),
                "order_id": order_id,
            }
        )

    report = {
        "window": {
            "start": start_ts.isoformat(),
            "end": end_ts.isoformat(),
            "hours": int(args.hours),
        },
        "worksteal_live_actions": action_rows,
        "external_account_orders": external_orders,
        "summary": {
            "n_worksteal_actions": len(action_rows),
            "n_worksteal_staged_orders_touched": sum(
                1 for row in action_rows if row["side"] in {"staged_buy", "staged_sell"} and row["market_touched"]
            ),
            "n_worksteal_actions_with_exchange_match": sum(1 for row in action_rows if row["exchange_matches"]),
            "n_external_account_orders": len(external_orders),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str))

    print(f"window={start_ts.isoformat()} -> {end_ts.isoformat()}")
    print(f"worksteal_actions={len(action_rows)}")
    print(f"external_account_orders={len(external_orders)}")
    print(f"report={output_path}")
    for row in action_rows:
        status = row["exchange_matches"][0]["status"] if row["exchange_matches"] else "NO_MATCH"
        touch = row["first_touch_ts"] or "no_touch"
        print(
            f"{row['timestamp']} {row['symbol']} {row['side']} price={row['price']:.6f} "
            f"touch={touch} exchange={status}"
        )
    if external_orders:
        print("external:")
        for row in external_orders:
            print(
                f"{row['timestamp']} {row['pair']} {row['side']} "
                f"status={row['status']} price={row['price']:.6f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
