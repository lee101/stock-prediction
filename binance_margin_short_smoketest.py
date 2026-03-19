#!/usr/bin/env python3
"""Inspect and optionally execute a tiny cross-margin short roundtrip."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.binan import binance_wrapper
from src.binan.binance_margin import (
    create_margin_market_buy,
    create_margin_market_sell,
    get_borrow_repay_records,
    get_margin_asset_balance,
    get_open_margin_orders,
    margin_repay,
)
from src.binan.margin_smoke import (
    MarginAssetSnapshot,
    build_excess_flatten_qty,
    build_liability_cleanup_qty,
    build_market_qty_from_notional,
)
from binanceneural.execution import resolve_symbol_rules, split_binance_symbol


def _coerce_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return numeric


def _fetch_snapshot(asset: str) -> MarginAssetSnapshot:
    entry = get_margin_asset_balance(asset) or {}
    return MarginAssetSnapshot(
        free=entry.get("free", 0.0),
        borrowed=entry.get("borrowed", 0.0),
        interest=entry.get("interest", 0.0),
        net_asset=entry.get("netAsset", 0.0),
    )


def _latest_records(asset: str, *, start_time_ms: int) -> dict[str, list[dict[str, Any]]]:
    borrows = get_borrow_repay_records("BORROW", asset=asset, start_time=start_time_ms, limit=20)
    repays = get_borrow_repay_records("REPAY", asset=asset, start_time=start_time_ms, limit=20)
    keep_keys = ("type", "status", "amount", "asset", "timestamp", "txId")

    def _normalize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{k: row.get(k) for k in keep_keys} for row in rows]

    return {"borrow": _normalize(borrows[-10:]), "repay": _normalize(repays[-10:])}


def _sleep(seconds: float) -> None:
    if seconds > 0.0:
        time.sleep(seconds)


def _repay_asset_liability(asset: str, *, settle_delay: float, max_attempts: int = 3) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for _ in range(max(1, int(max_attempts))):
        snapshot = _fetch_snapshot(asset)
        if snapshot.liability <= 0.000001 or snapshot.repayable <= 0.000001:
            break
        repay_amount = snapshot.repayable
        try:
            result = margin_repay(asset, repay_amount)
        except Exception:
            safe_amount = snapshot.repayable * 0.999
            if safe_amount <= 0.000001:
                break
            result = margin_repay(asset, safe_amount)
            repay_amount = safe_amount
        actions.append(
            {
                "action": "repay_asset_liability",
                "asset": asset,
                "requested_amount": repay_amount,
                "result": result,
            }
        )
        _sleep(settle_delay)
    return actions


def _cleanup_existing_liability(
    *,
    symbol: str,
    asset: str,
    market_price: float,
    rules,
    settle_delay: float,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    snapshot = _fetch_snapshot(asset)
    cleanup_qty = build_liability_cleanup_qty(
        snapshot=snapshot,
        market_price=market_price,
        step_size=rules.step_size,
        min_qty=rules.min_qty,
        min_notional=rules.min_notional,
    )
    if cleanup_qty > 0.0:
        result = create_margin_market_buy(
            symbol,
            cleanup_qty,
            side_effect_type="MARGIN_BUY",
        )
        actions.append(
            {
                "action": "cleanup_buy",
                "symbol": symbol,
                "qty": cleanup_qty,
                "market_price": market_price,
                "result": result,
            }
        )
        _sleep(settle_delay)
    actions.extend(_repay_asset_liability(asset, settle_delay=settle_delay))
    post_repay = _fetch_snapshot(asset)
    flatten_qty = build_excess_flatten_qty(
        snapshot=post_repay,
        market_price=market_price,
        step_size=rules.step_size,
        min_qty=rules.min_qty,
        min_notional=rules.min_notional,
    )
    if flatten_qty > 0.0:
        result = create_margin_market_sell(
            symbol,
            flatten_qty,
            side_effect_type="NO_SIDE_EFFECT",
        )
        actions.append(
            {
                "action": "flatten_excess_asset",
                "symbol": symbol,
                "qty": flatten_qty,
                "market_price": market_price,
                "result": result,
            }
        )
        _sleep(settle_delay)
    return actions


def _write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="DOGEUSDT", help="Cross-margin Binance symbol.")
    parser.add_argument("--target-notional", type=float, default=5.10, help="Target quote notional for the short smoke test.")
    parser.add_argument("--sleep-seconds", type=float, default=2.0, help="Seconds to wait after each exchange action.")
    parser.add_argument("--skip-clean-before", action="store_true", help="Skip cleaning any pre-existing base-asset liability before the short smoke test.")
    parser.add_argument("--cleanup-only", action="store_true", help="Only clean existing liability; do not place the short roundtrip.")
    parser.add_argument("--execute", action="store_true", help="Actually submit live margin orders. Without this flag the script only prints the plan.")
    parser.add_argument("--output", type=Path, help="Optional JSON artifact path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    symbol = args.symbol.upper()
    base_asset, quote_asset = split_binance_symbol(symbol)
    start_time_ms = int((datetime.now(timezone.utc) - timedelta(hours=2)).timestamp() * 1000)

    rules = resolve_symbol_rules(symbol)
    market_price = _coerce_float(binance_wrapper.get_symbol_price(symbol))
    if market_price <= 0.0:
        raise RuntimeError(f"Failed to fetch a usable market price for {symbol}.")

    before = _fetch_snapshot(base_asset)
    cleanup_qty = build_liability_cleanup_qty(
        snapshot=before,
        market_price=market_price,
        step_size=rules.step_size,
        min_qty=rules.min_qty,
        min_notional=rules.min_notional,
    )
    short_qty = build_market_qty_from_notional(
        target_notional=args.target_notional,
        market_price=market_price,
        step_size=rules.step_size,
        min_qty=rules.min_qty,
        min_notional=rules.min_notional,
    )
    plan: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "base_asset": base_asset,
        "quote_asset": quote_asset,
        "market_price": market_price,
        "rules": {
            "tick_size": _coerce_float(rules.tick_size),
            "step_size": _coerce_float(rules.step_size),
            "min_qty": _coerce_float(rules.min_qty),
            "min_notional": _coerce_float(rules.min_notional),
        },
        "before": before.as_dict(),
        "planned_cleanup_qty": cleanup_qty,
        "planned_short_qty": short_qty,
        "planned_short_notional": short_qty * market_price,
        "execute": bool(args.execute),
        "actions": [],
    }

    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        if args.output:
            _write_output(args.output, plan)
        return 0

    if not args.skip_clean_before:
        plan["actions"].extend(
            _cleanup_existing_liability(
                symbol=symbol,
                asset=base_asset,
                market_price=market_price,
                rules=rules,
                settle_delay=args.sleep_seconds,
            )
        )

    after_preclean = _fetch_snapshot(base_asset)
    plan["after_preclean"] = after_preclean.as_dict()

    if not args.cleanup_only:
        if after_preclean.liability > max(_coerce_float(rules.step_size), 0.000001):
            raise RuntimeError(
                f"Refusing to short {symbol} with outstanding {base_asset} liability still present: "
                f"{after_preclean.liability:.8f}"
            )
        if short_qty <= 0.0:
            raise RuntimeError(f"Resolved short quantity is zero for {symbol}.")
        entry = create_margin_market_sell(
            symbol,
            short_qty,
            side_effect_type="AUTO_BORROW_REPAY",
        )
        plan["actions"].append(
            {
                "action": "short_entry",
                "symbol": symbol,
                "qty": short_qty,
                "result": entry,
            }
        )
        _sleep(args.sleep_seconds)
        mid_snapshot = _fetch_snapshot(base_asset)
        plan["after_short_entry"] = mid_snapshot.as_dict()

        exit_order = create_margin_market_buy(
            symbol,
            short_qty,
            side_effect_type="AUTO_REPAY",
        )
        plan["actions"].append(
            {
                "action": "short_exit",
                "symbol": symbol,
                "qty": short_qty,
                "result": exit_order,
            }
        )
        _sleep(args.sleep_seconds)
        plan["actions"].extend(_cleanup_existing_liability(
            symbol=symbol,
            asset=base_asset,
            market_price=_coerce_float(binance_wrapper.get_symbol_price(symbol)),
            rules=rules,
            settle_delay=args.sleep_seconds,
        ))

    after = _fetch_snapshot(base_asset)
    plan["after"] = after.as_dict()
    plan["open_orders"] = get_open_margin_orders(symbol)
    plan["recent_records"] = _latest_records(base_asset, start_time_ms=start_time_ms)

    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.output:
        _write_output(args.output, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
