#!/usr/bin/env python3
"""Live daily trading agent using LLM allocation via OpenPaths."""
import argparse
import json
import time
import sys
import os
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gstockagent.config import GStockConfig, OPENPATHS_API_KEY
from gstockagent.llm_client import call_llm, parse_allocation
from gstockagent.prompt import build_prompt, load_daily_bars

try:
    from src.binan.binance_wrapper import get_account_balances, get_asset_balance
    from src.binan.binance_margin import (
        get_margin_account, get_margin_balances,
        create_margin_order, get_max_borrowable,
    )
    from binanceneural.execution import resolve_symbol_rules, quantize_price, quantize_qty
    HAS_BINANCE = True
except ImportError:
    HAS_BINANCE = False


def get_current_portfolio(symbols):
    if not HAS_BINANCE:
        return {}, 0.0
    try:
        margin = get_margin_account()
        balances = margin.get("userAssets", [])
        positions = {}
        total_usdt = 0.0
        for b in balances:
            asset = b["asset"]
            net = float(b["netAsset"])
            if asset == "USDT":
                total_usdt += net
                continue
            if net > 0 and asset in symbols:
                positions[asset] = {
                    "qty": net,
                    "entry_price": 0,  # would need order history
                }
        return positions, total_usdt
    except Exception as e:
        print(f"[WARN] portfolio fetch failed: {e}")
        return {}, 0.0


def get_live_prices(symbols):
    prices = {}
    try:
        import requests
        r = requests.get("https://api.binance.com/api/v3/ticker/price", timeout=10)
        data = r.json()
        price_map = {d["symbol"]: float(d["price"]) for d in data}
        for sym in symbols:
            for suffix in ["USDT", "FDUSD"]:
                key = f"{sym}{suffix}"
                if key in price_map:
                    prices[sym] = price_map[key]
                    break
    except Exception as e:
        print(f"[WARN] price fetch failed: {e}")
    return prices


def execute_rebalance(allocations, prices, total_capital, leverage):
    if not HAS_BINANCE:
        print("[DRY RUN] Would execute:")
        for sym, spec in allocations.items():
            pct = spec.get("allocation_pct", 0)
            direction = spec.get("direction", "long")
            notional = total_capital * leverage * pct / 100
            print(f"  {sym}: {direction} ${notional:.0f} ({pct}%)")
        return

    for sym, spec in allocations.items():
        pct = float(spec.get("allocation_pct", 0))
        direction = spec.get("direction", "long")
        exit_price = float(spec.get("exit_price", 0))

        if pct <= 0 or sym not in prices:
            continue

        notional = total_capital * leverage * pct / 100
        price = prices[sym]
        qty = notional / price

        market_sym = f"{sym}USDT"
        try:
            rules = resolve_symbol_rules(market_sym)
            qty = quantize_qty(qty, rules.step_size)
            price = quantize_price(price, rules.tick_size, "BUY" if direction == "long" else "SELL")

            if qty * price < rules.min_notional:
                print(f"  {sym}: skip (below min notional)")
                continue

            side = "BUY" if direction == "long" else "SELL"
            order = create_margin_order(
                market_sym, side, "LIMIT", qty,
                price=price, side_effect_type="MARGIN_BUY" if side == "BUY" else "AUTO_REPAY",
                time_in_force="GTC",
            )
            print(f"  {sym}: {side} {qty} @ {price} = ${qty*price:.0f} -> order {order.get('orderId')}")

            if exit_price > 0:
                exit_side = "SELL" if direction == "long" else "BUY"
                ep = quantize_price(exit_price, rules.tick_size, exit_side)
                exit_order = create_margin_order(
                    market_sym, exit_side, "LIMIT", qty,
                    price=ep,
                    side_effect_type="AUTO_REPAY" if exit_side == "SELL" else "MARGIN_BUY",
                    time_in_force="GTC",
                )
                print(f"  {sym}: exit {exit_side} {qty} @ {ep} -> order {exit_order.get('orderId')}")

        except Exception as e:
            print(f"  {sym}: ERROR {e}")


def run_once(config: GStockConfig, dry_run: bool = True):
    now = datetime.now(timezone.utc)
    print(f"\n{'='*60}")
    print(f"[{now.isoformat()}] gstockagent cycle start")
    print(f"  model={config.model} lev={config.leverage}x max_pos={config.max_positions}")

    prices = get_live_prices(config.symbols)
    if not prices:
        print("[ERROR] no prices available")
        return

    positions, usdt_balance = get_current_portfolio(config.symbols)
    total_value = usdt_balance + sum(
        pos["qty"] * prices.get(sym, 0) for sym, pos in positions.items()
    )
    if total_value <= 0:
        total_value = config.initial_capital

    print(f"  portfolio: ${total_value:.2f} ({len(positions)} positions, ${usdt_balance:.2f} USDT)")

    prompt = build_prompt(
        config.symbols, config.data_dir, config.forecast_cache_dir,
        pd.Timestamp(now), positions, prices, total_value,
        config.leverage, config.max_positions,
    )

    date_str = now.strftime("%Y-%m-%d")
    print(f"  calling {config.model}...")
    try:
        resp = call_llm(prompt, config.model, config.temperature, date_str=date_str, use_cache=False)
        alloc = parse_allocation(resp)
    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}")
        return

    if not alloc:
        print("[WARN] empty allocation, holding current positions")
        return

    total_pct = sum(float(v.get("allocation_pct", 0)) for v in alloc.values() if isinstance(v, dict))
    print(f"  allocation: {len(alloc)} symbols, {total_pct:.0f}% deployed")
    for sym, spec in alloc.items():
        if isinstance(spec, dict):
            print(f"    {sym}: {spec.get('direction','?')} {spec.get('allocation_pct',0)}% "
                  f"exit={spec.get('exit_price','?')} stop={spec.get('stop_price','?')}")

    if dry_run:
        execute_rebalance(alloc, prices, total_value, config.leverage)
    else:
        execute_rebalance(alloc, prices, total_value, config.leverage)

    print(f"[{datetime.now(timezone.utc).isoformat()}] cycle complete")


def main():
    parser = argparse.ArgumentParser(description="gstockagent live trader")
    parser.add_argument("--model", default="gemini-3.1-lite",
                        help="LLM model (gemini-3.1-lite, glm-5, gemini-3.1-pro for prod)")
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--interval-hours", type=int, default=24)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--live", action="store_true", help="actually execute trades")
    parser.add_argument("--once", action="store_true", help="run once and exit")
    parser.add_argument("--daemon", action="store_true", help="run as daemon loop")
    args = parser.parse_args()

    if not OPENPATHS_API_KEY:
        print("ERROR: set OPENPATHS_API_KEY env var")
        sys.exit(1)

    config = GStockConfig(
        model=args.model,
        leverage=args.leverage,
        max_positions=args.max_positions,
    )

    dry_run = not args.live

    if args.once or not args.daemon:
        run_once(config, dry_run=dry_run)
        return

    print(f"Starting daemon: interval={args.interval_hours}h model={args.model}")
    while True:
        try:
            run_once(config, dry_run=dry_run)
        except Exception as e:
            print(f"[ERROR] cycle failed: {e}")
        interval_secs = args.interval_hours * 3600
        print(f"Sleeping {args.interval_hours}h until next cycle...")
        time.sleep(interval_secs)


if __name__ == "__main__":
    main()
