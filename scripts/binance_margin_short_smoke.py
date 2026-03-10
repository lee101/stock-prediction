#!/usr/bin/env python3
"""Run a tiny live Binance cross-margin short/open-close smoke test and save an artifact."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from binanceneural.execution import resolve_symbol_rules, split_binance_symbol
from src.binan.binance_margin import (
    create_margin_market_buy,
    create_margin_market_sell,
    get_all_margin_orders,
    get_borrow_repay_records,
    get_margin_asset_balance,
    get_margin_trades,
    get_max_borrowable,
    get_open_margin_orders,
)
from src.binan.binance_wrapper import get_symbol_price
from src.binance_margin_smoke import (
    AssetBalanceSnapshot,
    balance_is_flat,
    build_short_smoke_plan,
    residual_repay_qty,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _format_snapshot(snapshot: AssetBalanceSnapshot) -> str:
    return (
        f"{snapshot.asset}: free={snapshot.free:.8f} borrowed={snapshot.borrowed:.8f} "
        f"interest={snapshot.interest:.8f} net={snapshot.net_asset:.8f}"
    )


def _order_executed_qty(order: dict[str, Any] | None, *, fallback: float) -> float:
    if not isinstance(order, dict):
        return float(fallback)
    for key in ("executedQty", "origQty"):
        raw = order.get(key)
        try:
            qty = float(raw)
        except (TypeError, ValueError):
            continue
        if qty > 0:
            return qty
    return float(fallback)


def _artifact_path(artifact_dir: Path, *, symbol: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return artifact_dir / f"{stamp}_{symbol.lower()}_short_smoke.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="ETHUSDT", help="Cross-margin symbol to test (default ETHUSDT).")
    parser.add_argument("--target-notional", type=float, default=5.0, help="Target quote notional for the short.")
    parser.add_argument("--sleep-seconds", type=float, default=2.0, help="Pause between entry/exit and verification.")
    parser.add_argument(
        "--artifact-dir",
        default="experiments/live_margin_short_smoke",
        help="Directory for JSON evidence artifacts.",
    )
    parser.add_argument(
        "--allow-existing-base-balance",
        action="store_true",
        help="Allow running even when the base asset already has a non-flat margin balance.",
    )
    parser.add_argument(
        "--borrowed-tolerance",
        type=float,
        default=1e-8,
        help="Maximum borrowed balance allowed after cleanup.",
    )
    parser.add_argument(
        "--net-tolerance",
        type=float,
        default=None,
        help="Maximum absolute net asset allowed after cleanup. Defaults to the symbol step size.",
    )
    parser.add_argument("--execute", action="store_true", help="Actually place the live short and cleanup orders.")
    args = parser.parse_args()

    symbol = str(args.symbol).upper()
    base_asset, quote_asset = split_binance_symbol(symbol)
    rules = resolve_symbol_rules(symbol)
    reference_price = get_symbol_price(symbol)
    if reference_price is None or reference_price <= 0:
        raise SystemExit(f"Failed to resolve a live price for {symbol}.")

    max_borrowable = get_max_borrowable(base_asset)
    if max_borrowable <= 0:
        raise SystemExit(f"{symbol} cannot be borrowed on this account right now (max_borrowable={max_borrowable}).")

    base_before = AssetBalanceSnapshot.from_margin_entry(base_asset, get_margin_asset_balance(base_asset))
    quote_before = AssetBalanceSnapshot.from_margin_entry(quote_asset, get_margin_asset_balance(quote_asset))
    flat_tolerance = args.net_tolerance if args.net_tolerance is not None else (rules.step_size or args.borrowed_tolerance)

    plan = build_short_smoke_plan(
        symbol=symbol,
        base_asset=base_asset,
        reference_price=reference_price,
        target_notional=args.target_notional,
        min_notional=rules.min_notional,
        min_qty=rules.min_qty,
        step_size=rules.step_size,
    )

    print(json.dumps(
        {
            "symbol": symbol,
            "reference_price": reference_price,
            "max_borrowable": max_borrowable,
            "plan": plan.to_dict(),
            "base_before": base_before.to_dict(),
            "quote_before": quote_before.to_dict(),
            "execute": bool(args.execute),
        },
        indent=2,
        sort_keys=True,
        default=_json_default,
    ))

    if not args.allow_existing_base_balance and not balance_is_flat(
        base_before,
        borrowed_tolerance=args.borrowed_tolerance,
        net_tolerance=flat_tolerance,
    ):
        raise SystemExit(
            f"Refusing to run on non-flat {base_asset} margin balance without --allow-existing-base-balance: "
            f"{_format_snapshot(base_before)}"
        )

    if not args.execute:
        return 0

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    start_ms = int((started_at.timestamp() - 120.0) * 1000)

    entry_order: dict[str, Any] | None = None
    exit_order: dict[str, Any] | None = None
    cleanup_order: dict[str, Any] | None = None
    emergency_cleanup_error: str | None = None

    try:
        entry_order = create_margin_market_sell(symbol, plan.qty, side_effect_type="AUTO_BORROW_REPAY")
        time.sleep(max(0.0, args.sleep_seconds))

        exit_qty = _order_executed_qty(entry_order, fallback=plan.qty)
        exit_order = create_margin_market_buy(symbol, exit_qty, side_effect_type="AUTO_REPAY")
        time.sleep(max(0.0, args.sleep_seconds))
    except Exception:
        if entry_order is not None and exit_order is None:
            try:
                emergency_qty = _order_executed_qty(entry_order, fallback=plan.qty)
                exit_order = create_margin_market_buy(symbol, emergency_qty, side_effect_type="AUTO_REPAY")
                time.sleep(max(0.0, args.sleep_seconds))
            except Exception as exc:  # pragma: no cover - live emergency path
                emergency_cleanup_error = str(exc)
        raise

    base_after = AssetBalanceSnapshot.from_margin_entry(base_asset, get_margin_asset_balance(base_asset))
    if not balance_is_flat(base_after, borrowed_tolerance=args.borrowed_tolerance, net_tolerance=flat_tolerance):
        extra_qty = residual_repay_qty(
            base_after,
            step_size=rules.step_size,
            min_qty=rules.min_qty,
            safety_steps=1,
        )
        if extra_qty > 0:
            cleanup_order = create_margin_market_buy(symbol, extra_qty, side_effect_type="AUTO_REPAY")
            time.sleep(max(0.0, args.sleep_seconds))
            base_after = AssetBalanceSnapshot.from_margin_entry(base_asset, get_margin_asset_balance(base_asset))

    quote_after = AssetBalanceSnapshot.from_margin_entry(quote_asset, get_margin_asset_balance(quote_asset))
    finished_at = datetime.now(timezone.utc)
    end_ms = int(finished_at.timestamp() * 1000)
    artifact = {
        "started_at": started_at,
        "finished_at": finished_at,
        "symbol": symbol,
        "base_asset": base_asset,
        "quote_asset": quote_asset,
        "reference_price": reference_price,
        "max_borrowable": max_borrowable,
        "plan": plan.to_dict(),
        "base_before": base_before.to_dict(),
        "quote_before": quote_before.to_dict(),
        "entry_order": entry_order,
        "exit_order": exit_order,
        "cleanup_order": cleanup_order,
        "emergency_cleanup_error": emergency_cleanup_error,
        "base_after": base_after.to_dict(),
        "quote_after": quote_after.to_dict(),
        "open_orders": get_open_margin_orders(symbol),
        "orders_since_start": get_all_margin_orders(symbol, start_time=start_ms, end_time=end_ms, limit=20),
        "trades_since_start": get_margin_trades(symbol, start_time=start_ms, end_time=end_ms, limit=20),
        "borrow_records_since_start": get_borrow_repay_records(
            "BORROW",
            asset=base_asset,
            start_time=start_ms,
            end_time=end_ms,
            limit=20,
        ),
        "repay_records_since_start": get_borrow_repay_records(
            "REPAY",
            asset=base_asset,
            start_time=start_ms,
            end_time=end_ms,
            limit=20,
        ),
    }
    artifact["success"] = bool(
        not artifact["open_orders"]
        and balance_is_flat(
            base_after,
            borrowed_tolerance=args.borrowed_tolerance,
            net_tolerance=flat_tolerance,
        )
        and any(not trade.get("isBuyer", False) for trade in artifact["trades_since_start"])
        and any(bool(trade.get("isBuyer", False)) for trade in artifact["trades_since_start"])
        and len(artifact["borrow_records_since_start"]) >= 1
        and len(artifact["repay_records_since_start"]) >= 1
    )

    path = _artifact_path(artifact_dir, symbol=symbol)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True, default=_json_default))
    print(f"artifact={path}")
    print(f"base_after={_format_snapshot(base_after)}")
    print(f"success={artifact['success']}")
    return 0 if artifact["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
