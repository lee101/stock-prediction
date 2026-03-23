#!/usr/bin/env python3
"""Safely unwind Binance cross-margin positions with audit logging."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from binanceneural.execution import quantize_down, resolve_symbol_rules
from src.binan import binance_wrapper as bw
from src.binan.binance_margin import (
    create_margin_order,
    get_margin_account,
)
from src.binan.binance_wrapper import _STABLECOIN_ASSETS, _coerce_balance_value

DUST_USD = 1.0


def _get_price(asset):
    if asset in _STABLECOIN_ASSETS:
        return 1.0
    price = bw.get_symbol_price(f"{asset}USDT")
    if price and price > 0:
        return price
    return None


def _log_event(log_path, event):
    event["ts"] = datetime.now(timezone.utc).isoformat()
    with open(log_path, "a") as f:
        f.write(json.dumps(event) + "\n")


def scan_positions():
    account = get_margin_account()
    if not account:
        print("ERROR: failed to fetch margin account")
        return []
    positions = []
    for entry in account.get("userAssets", []):
        asset = str(entry.get("asset", "")).upper()
        if not asset or asset in _STABLECOIN_ASSETS:
            continue
        free = _coerce_balance_value(entry.get("free"))
        locked = _coerce_balance_value(entry.get("locked"))
        borrowed = _coerce_balance_value(entry.get("borrowed"))
        interest = _coerce_balance_value(entry.get("interest"))
        net = _coerce_balance_value(entry.get("netAsset"))
        if free <= 0:
            continue
        price = _get_price(asset)
        if price is None:
            continue
        notional = free * price
        if notional < DUST_USD:
            continue
        positions.append({
            "asset": asset,
            "free": free,
            "locked": locked,
            "borrowed": borrowed,
            "interest": interest,
            "net": net,
            "price": price,
            "notional": notional,
        })
    positions.sort(key=lambda x: -x["notional"])
    return positions


def display_positions(positions):
    if not positions:
        print("No significant positions found.")
        return
    print(f"\n{'Asset':<8} {'Free Qty':>14} {'Price':>12} {'Notional':>12} {'Borrowed':>12}")
    print("-" * 62)
    total = 0.0
    for p in positions:
        print(
            f"{p['asset']:<8} {p['free']:>14.6f} {p['price']:>12.4f} "
            f"${p['notional']:>10,.2f} {p['borrowed']:>12.6f}"
        )
        total += p["notional"]
    print("-" * 62)
    print(f"{'TOTAL':<8} {'':>14} {'':>12} ${total:>10,.2f}")


def _get_sell_qty(pair, free_qty):
    rules = resolve_symbol_rules(pair)
    qty = quantize_down(free_qty, rules.step_size)
    return qty, rules.step_size or 0, rules.min_qty or 0


def unwind(positions, symbol_filter, live, log_path):
    if symbol_filter:
        allowed = {s.upper() for s in symbol_filter}
        positions = [p for p in positions if p["asset"] in allowed]

    display_positions(positions)

    if not positions:
        return

    if not live:
        print("\n[DRY RUN] would sell the above positions. Use --live to execute.")
        for p in positions:
            pair = f"{p['asset']}USDT"
            qty, step, min_qty = _get_sell_qty(pair, p["free"])
            print(f"  SELL {qty} {pair} (step={step}, min_qty={min_qty})")
            _log_event(log_path, {
                "action": "dry_run_sell",
                "asset": p["asset"],
                "pair": pair,
                "qty": qty,
                "price": p["price"],
                "notional": p["notional"],
            })
        return

    print("\n[LIVE] executing market sells with AUTO_REPAY...")
    for p in positions:
        pair = f"{p['asset']}USDT"
        qty, step, min_qty = _get_sell_qty(pair, p["free"])
        if qty <= 0 or (min_qty > 0 and qty < min_qty):
            print(f"  SKIP {p['asset']}: qty {qty} below min {min_qty}")
            _log_event(log_path, {
                "action": "skip",
                "asset": p["asset"],
                "reason": f"qty {qty} < min {min_qty}",
            })
            continue
        print(f"  SELL {qty} {pair} ...", end=" ")
        try:
            order = create_margin_order(
                pair, "SELL", "MARKET", qty,
                side_effect_type="AUTO_REPAY",
            )
            print(f"OK orderId={order.get('orderId')}")
            _log_event(log_path, {
                "action": "sell",
                "asset": p["asset"],
                "pair": pair,
                "qty": qty,
                "order": order,
            })
        except Exception as exc:
            print(f"FAILED: {exc}")
            _log_event(log_path, {
                "action": "sell_error",
                "asset": p["asset"],
                "pair": pair,
                "qty": qty,
                "error": str(exc),
            })


def main():
    parser = argparse.ArgumentParser(description="Unwind Binance cross-margin positions")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True, help="Print only (default)")
    mode.add_argument("--live", action="store_true", help="Execute real trades")
    parser.add_argument("--symbols", nargs="+", help="Only unwind these assets (e.g. LINK)")
    args = parser.parse_args()

    state_dir = Path(__file__).resolve().parent.parent / "strategy_state"
    state_dir.mkdir(exist_ok=True)
    datestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    log_path = state_dir / f"unwind_log_{datestamp}.jsonl"

    print(f"Scanning cross-margin account...")
    positions = scan_positions()
    unwind(positions, args.symbols, args.live, log_path)
    print(f"\nAudit log: {log_path}")


if __name__ == "__main__":
    main()
