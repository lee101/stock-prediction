from __future__ import annotations

import os
from typing import Iterable

from loguru import logger

from src.binan import binance_wrapper
from src.binan.binance_conversion import (
    build_stable_quote_conversion_plan,
    execute_stable_quote_conversion,
)
from src.stock_utils import binance_remap_symbols


_STABLE_QUOTES = ("USDT", "FDUSD", "USDC", "BUSD", "TUSD", "USDP", "DAI", "U")
_CONVERTIBLE_QUOTES = frozenset({"USDT", "FDUSD"})
_ZERO_FEE_FDUSD_PAIRS = {
    "BTCUSD": "BTCFDUSD",
    "ETHUSD": "ETHFDUSD",
}


def _normalize_symbol(symbol: str) -> str:
    return str(symbol or "").replace("/", "").replace("-", "").replace("_", "").strip().upper()


def _coerce_float(value: object) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def split_binance_symbol(symbol: str) -> tuple[str, str]:
    normalized = _normalize_symbol(symbol)
    for quote in _STABLE_QUOTES:
        if normalized.endswith(quote) and len(normalized) > len(quote):
            return normalized[: -len(quote)], quote
    if len(normalized) <= 3:
        return normalized, ""
    return normalized[:-3], normalized[-3:]


def preferred_spot_execution_symbol(symbol: str) -> str:
    normalized = _normalize_symbol(symbol)
    if not normalized:
        return normalized

    for quote in _STABLE_QUOTES:
        if normalized.endswith(quote) and len(normalized) > len(quote):
            return normalized

    if os.getenv("BINANCE_TLD", "").strip().lower() == "us":
        return binance_remap_symbols(normalized)

    preferred = _ZERO_FEE_FDUSD_PAIRS.get(normalized)
    if preferred:
        return preferred
    return binance_remap_symbols(normalized)


def get_spendable_quote_balance(execution_symbol: str, *, balances: Iterable[dict] | None = None) -> float:
    normalized = preferred_spot_execution_symbol(execution_symbol)
    _, quote_asset = split_binance_symbol(normalized)
    if not quote_asset:
        return 0.0

    balance_list = list(balances) if balances is not None else binance_wrapper.get_account_balances()
    quote_entry = binance_wrapper.get_asset_balance(quote_asset, balances=balance_list) or {}
    spendable = _coerce_float(quote_entry.get("free"))
    if quote_asset not in _CONVERTIBLE_QUOTES:
        return spendable

    sibling_asset = "USDT" if quote_asset == "FDUSD" else "FDUSD"
    sibling_entry = binance_wrapper.get_asset_balance(sibling_asset, balances=balance_list) or {}
    return spendable + _coerce_float(sibling_entry.get("free"))


def ensure_stable_quote_balance(
    execution_symbol: str,
    needed_amount: float,
    *,
    dry_run: bool = False,
    balances: Iterable[dict] | None = None,
) -> bool:
    required = _coerce_float(needed_amount)
    if required <= 0.0:
        return True

    normalized = preferred_spot_execution_symbol(execution_symbol)
    _, quote_asset = split_binance_symbol(normalized)
    if not quote_asset:
        return False

    balance_list = list(balances) if balances is not None else binance_wrapper.get_account_balances()
    quote_entry = binance_wrapper.get_asset_balance(quote_asset, balances=balance_list) or {}
    quote_free = _coerce_float(quote_entry.get("free"))
    if quote_free + 1e-9 >= required:
        return True

    if quote_asset not in _CONVERTIBLE_QUOTES:
        logger.warning(
            "Insufficient {} balance for {}: available={:.8f} required={:.8f}",
            quote_asset,
            normalized,
            quote_free,
            required,
        )
        return False

    source_asset = "USDT" if quote_asset == "FDUSD" else "FDUSD"
    source_entry = binance_wrapper.get_asset_balance(source_asset, balances=balance_list) or {}
    source_free = _coerce_float(source_entry.get("free"))
    shortfall = max(0.0, required - quote_free)
    if source_free + 1e-9 < shortfall:
        logger.warning(
            "Insufficient {} to convert into {} for {}: available={:.8f} shortfall={:.8f}",
            source_asset,
            quote_asset,
            normalized,
            source_free,
            shortfall,
        )
        return False

    plan = build_stable_quote_conversion_plan(
        from_asset=source_asset,
        to_asset=quote_asset,
        amount=shortfall,
        available_pairs=["FDUSDUSDT"],
    )
    if plan is None:
        logger.error(
            "No stablecoin conversion path available for {} -> {} while funding {}",
            source_asset,
            quote_asset,
            normalized,
        )
        return False

    logger.info(
        "Funding {} by converting {:.8f} {} -> {}",
        normalized,
        shortfall,
        source_asset,
        quote_asset,
    )
    if dry_run:
        return True

    execute_stable_quote_conversion(plan, dry_run=False)
    return True


__all__ = [
    "ensure_stable_quote_balance",
    "get_spendable_quote_balance",
    "preferred_spot_execution_symbol",
    "split_binance_symbol",
]
