from __future__ import annotations

import os
from typing import Any

from loguru import logger


def binance_live_executor(order: dict[str, Any]) -> dict[str, Any] | None:
    if str(os.getenv("ALLOW_BINANCE_LIVE_TRADING", "")).strip() not in {"1", "true", "TRUE"}:
        logger.error("ALLOW_BINANCE_LIVE_TRADING not set")
        return None

    symbol = order["symbol"]
    side = order["side"].upper()
    qty = float(order["qty"])
    price = float(order["limit_price"])

    try:
        from binanceneural.execution import resolve_symbol_rules, quantize_price, quantize_qty
        rules = resolve_symbol_rules(symbol)
        if rules.tick_size:
            price = quantize_price(price, tick_size=rules.tick_size, side=side.lower())
        if rules.step_size:
            qty = quantize_qty(qty, step_size=rules.step_size)
    except Exception as e:
        logger.warning(f"Could not quantize {symbol}: {e}")

    try:
        from src.binan.binance_wrapper import create_order
        result = create_order(symbol, side, qty, price)
    except Exception as e:
        logger.error(f"Binance order failed for {symbol}: {e}")
        return None

    if result is None:
        return None

    return {
        "broker_order_id": str(result.get("orderId", "")),
        "status": str(result.get("status", "NEW")),
        "symbol": symbol,
        "adjusted_price": price,
        "adjusted_qty": qty,
    }
