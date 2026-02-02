from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from loguru import logger

from src.binan import binance_wrapper
from src.stock_utils import binance_remap_symbols


@dataclass
class SymbolRules:
    min_notional: Optional[float] = None
    min_qty: Optional[float] = None
    step_size: Optional[float] = None
    tick_size: Optional[float] = None
    min_price: Optional[float] = None


@dataclass
class OrderSizingResult:
    buy_qty: float
    sell_qty: float
    buy_notional: float
    sell_notional: float
    notes: list[str] = field(default_factory=list)


def _coerce_float(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return numeric


def resolve_binance_symbol(symbol: str) -> str:
    return binance_remap_symbols(symbol.replace("/", ""))


def split_binance_symbol(symbol: str) -> Tuple[str, str]:
    normalized = symbol.replace("/", "").upper()
    for quote in ("USDT", "FDUSD", "BUSD", "USDC"):
        if normalized.endswith(quote):
            return normalized[: -len(quote)], quote
    if len(normalized) <= 3:
        return normalized, ""
    return normalized[:-3], normalized[-3:]


def resolve_symbol_rules(symbol: str) -> SymbolRules:
    pair = resolve_binance_symbol(symbol)
    filters = binance_wrapper.get_symbol_filters(pair)
    rules = SymbolRules(min_notional=binance_wrapper.get_min_notional(pair))

    lot = filters.get("LOT_SIZE")
    if lot:
        rules.min_qty = _coerce_float(lot.get("minQty")) or None
        rules.step_size = _coerce_float(lot.get("stepSize")) or None

    price_filter = filters.get("PRICE_FILTER")
    if price_filter:
        rules.tick_size = _coerce_float(price_filter.get("tickSize")) or None
        rules.min_price = _coerce_float(price_filter.get("minPrice")) or None

    return rules


def quantize_down(value: float, step: Optional[float]) -> float:
    if step is None or step <= 0:
        return value
    return math.floor(value / step) * step


def quantize_up(value: float, step: Optional[float]) -> float:
    if step is None or step <= 0:
        return value
    return math.ceil(value / step) * step


def quantize_price(price: float, *, tick_size: Optional[float], side: str) -> float:
    if tick_size is None or tick_size <= 0:
        return price
    side_norm = side.lower()
    if side_norm.startswith("b"):
        return quantize_down(price, tick_size)
    if side_norm.startswith("s"):
        return quantize_up(price, tick_size)
    return quantize_down(price, tick_size)


def quantize_qty(quantity: float, *, step_size: Optional[float]) -> float:
    if step_size is None or step_size <= 0:
        return quantity
    return quantize_down(quantity, step_size)


def get_free_balances(symbol: str) -> Tuple[float, float]:
    pair = resolve_binance_symbol(symbol)
    base, quote = split_binance_symbol(pair)
    balances = binance_wrapper.get_account_balances()
    base_entry = binance_wrapper.get_asset_balance(base, balances=balances) or {}
    quote_entry = binance_wrapper.get_asset_balance(quote, balances=balances) or {}
    base_free = _coerce_float(base_entry.get("free"))
    quote_free = _coerce_float(quote_entry.get("free"))
    return quote_free, base_free


def compute_order_quantities(
    *,
    symbol: str,
    buy_amount: float,
    sell_amount: float,
    buy_price: float,
    sell_price: float,
    quote_free: float,
    base_free: float,
    allocation_usdt: Optional[float] = None,
    rules: Optional[SymbolRules] = None,
) -> OrderSizingResult:
    notes: list[str] = []
    buy_intensity = max(0.0, min(1.0, buy_amount / 100.0))
    sell_intensity = max(0.0, min(1.0, sell_amount / 100.0))

    if buy_price <= 0 or sell_price <= 0:
        return OrderSizingResult(0.0, 0.0, 0.0, 0.0, ["invalid_prices"])

    max_quote = float(quote_free)
    if allocation_usdt is not None and allocation_usdt > 0:
        max_quote = min(max_quote, float(allocation_usdt))

    buy_notional = max_quote * buy_intensity
    buy_qty = buy_notional / buy_price if buy_price > 0 else 0.0
    sell_qty = float(base_free) * sell_intensity

    if rules is None:
        rules = resolve_symbol_rules(symbol)

    buy_price_adj = quantize_price(buy_price, tick_size=rules.tick_size, side="buy")
    sell_price_adj = quantize_price(sell_price, tick_size=rules.tick_size, side="sell")

    if buy_price_adj != buy_price:
        notes.append("buy_price_quantized")
    if sell_price_adj != sell_price:
        notes.append("sell_price_quantized")

    buy_qty = quantize_qty(buy_qty, step_size=rules.step_size)
    sell_qty = quantize_qty(sell_qty, step_size=rules.step_size)

    if rules.min_qty is not None:
        if buy_qty < rules.min_qty:
            buy_qty = 0.0
            notes.append("buy_below_min_qty")
        if sell_qty < rules.min_qty:
            sell_qty = 0.0
            notes.append("sell_below_min_qty")

    buy_notional = buy_qty * buy_price_adj
    sell_notional = sell_qty * sell_price_adj

    if rules.min_notional is not None:
        if buy_notional and buy_notional < rules.min_notional:
            buy_qty = 0.0
            buy_notional = 0.0
            notes.append("buy_below_min_notional")
        if sell_notional and sell_notional < rules.min_notional:
            sell_qty = 0.0
            sell_notional = 0.0
            notes.append("sell_below_min_notional")

    if buy_qty <= 0:
        notes.append("buy_zero")
    if sell_qty <= 0:
        notes.append("sell_zero")

    if notes:
        logger.debug("%s sizing notes: %s", symbol, ",".join(notes))

    return OrderSizingResult(
        buy_qty=float(buy_qty),
        sell_qty=float(sell_qty),
        buy_notional=float(buy_notional),
        sell_notional=float(sell_notional),
        notes=notes,
    )


__all__ = [
    "OrderSizingResult",
    "SymbolRules",
    "compute_order_quantities",
    "get_free_balances",
    "quantize_price",
    "quantize_qty",
    "resolve_binance_symbol",
    "resolve_symbol_rules",
    "split_binance_symbol",
]
