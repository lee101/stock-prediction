#!/usr/bin/env python3
"""Smoke test Binance API keys by fetching data and placing/canceling a tiny limit order."""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Any, Dict

from src.binan import binance_wrapper
from binanceneural.execution import quantize_price, quantize_qty, resolve_symbol_rules, split_binance_symbol


def _coerce_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return numeric


def _resolve_percent_price_bounds(filters: Dict[str, Dict[str, Any]], last_price: float) -> tuple[float | None, float | None]:
    for key in ("PERCENT_PRICE_BY_SIDE", "PERCENT_PRICE"):
        entry = filters.get(key)
        if not isinstance(entry, dict):
            continue
        mult_up = _coerce_float(entry.get("multiplierUp"))
        mult_down = _coerce_float(entry.get("multiplierDown"))
        if mult_up > 0 and mult_down > 0 and last_price > 0:
            return last_price * mult_down, last_price * mult_up
    return None, None


def _build_limit_price(symbol: str, side: str, last_price: float, multiplier: float) -> float:
    filters = binance_wrapper.get_symbol_filters(symbol)
    min_allowed, max_allowed = _resolve_percent_price_bounds(filters, last_price)

    raw_price = last_price * multiplier
    if min_allowed is not None:
        raw_price = max(raw_price, min_allowed)
    if max_allowed is not None:
        raw_price = min(raw_price, max_allowed)

    rules = resolve_symbol_rules(symbol)
    return quantize_price(raw_price, tick_size=rules.tick_size, side=side)


def _build_quantity(symbol: str, side: str, notional: float, price: float) -> tuple[float, float]:
    rules = resolve_symbol_rules(symbol)
    notional_eff = notional
    if rules.min_notional and notional_eff < rules.min_notional:
        notional_eff = rules.min_notional

    qty = notional_eff / price if price > 0 else 0.0
    qty = quantize_qty(qty, step_size=rules.step_size)

    if rules.min_qty and qty < rules.min_qty:
        qty = 0.0
    if not math.isfinite(qty):
        qty = 0.0
    return qty, notional_eff


def main() -> int:
    parser = argparse.ArgumentParser(description="Binance API smoke test (data fetch + tiny limit order + cancel).")
    parser.add_argument("--symbol", default="SOLUSDT", help="Binance spot symbol (default SOLUSDT).")
    parser.add_argument("--side", default="sell", choices=["buy", "sell"], help="Order side (default sell).")
    parser.add_argument("--notional", type=float, default=1.0, help="Target notional in quote currency.")
    parser.add_argument("--price-multiplier", type=float, default=1.15, help="Limit price multiplier vs last price.")
    parser.add_argument("--skip-order", action="store_true", help="Only fetch data; skip order placement.")
    parser.add_argument("--sleep-seconds", type=float, default=1.0, help="Seconds to wait before cancel.")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    side = args.side.upper()

    try:
        last_price = binance_wrapper.get_symbol_price(symbol)
    except Exception as exc:
        print(f"Failed to fetch symbol price: {exc}")
        return 2
    if last_price is None or last_price <= 0:
        print(f"Failed to fetch last price for {symbol}.")
        return 2

    print(f"{symbol} last price: {last_price:.6f}")

    if args.skip_order:
        return 0

    try:
        client = binance_wrapper.get_client()
        client.get_account()
    except Exception as exc:
        print(f"Failed to fetch account info (check API key/region): {exc}")
        return 3

    limit_price = _build_limit_price(symbol, side, last_price, args.price_multiplier)
    qty, notional_eff = _build_quantity(symbol, side, args.notional, limit_price)
    if qty <= 0:
        print("Quantity resolved to 0; check balances and minimum notional/qty rules.")
        return 4

    base, quote = split_binance_symbol(symbol)
    if side == "SELL":
        free_base = binance_wrapper.get_asset_free_balance(base)
        if free_base < qty:
            print(
                f"Insufficient {base} balance for sell test: have {free_base:.6f}, need {qty:.6f}."
            )
            return 5
    else:
        free_quote = binance_wrapper.get_asset_free_balance(quote)
        if free_quote < notional_eff:
            print(
                f"Insufficient {quote} balance for buy test: have {free_quote:.2f}, need {notional_eff:.2f}."
            )
            return 5

    print(
        f"Placing {side} {symbol} limit order qty={qty:.6f} "
        f"price={limit_price:.6f} notional~{qty * limit_price:.4f}"
    )

    try:
        order = binance_wrapper.create_order(symbol, side, qty, limit_price)
    except Exception as exc:
        print(f"Order placement failed: {exc}")
        return 6

    order_id = order.get("orderId")
    if not order_id:
        print(f"Order created but no orderId returned: {order}")
        return 7

    print(f"Order placed: orderId={order_id}")
    time.sleep(max(0.0, args.sleep_seconds))

    try:
        client.cancel_order(symbol=symbol, orderId=order_id)
    except Exception as exc:
        print(f"Failed to cancel order {order_id}: {exc}")
        return 8

    print(f"Order {order_id} canceled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
