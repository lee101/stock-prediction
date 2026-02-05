from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from src.symbol_utils import is_crypto_symbol

# Longable tech stocks (explicit allowlist)
LONGABLE_STOCKS: Tuple[str, ...] = (
    "NVDA",
    "GOOG",
    "MSFT",
)

# Shortable stock allowlist (explicit)
SHORTABLE_STOCKS: Tuple[str, ...] = (
    "YELP",
    "EBAY",
    "TRIP",
    "MTCH",
    "KIND",
    "ANGI",
    "Z",
    "EXPE",
    "BKNG",
    "NWSA",
    "NYT",
)

# Default crypto list for constrained experiments (can be overridden via CLI)
DEFAULT_LONG_CRYPTO: Tuple[str, ...] = (
    "BTCUSD",
    "ETHUSD",
    "SOLUSD",
)


def normalize_symbols(symbols: Iterable[str]) -> List[str]:
    cleaned: List[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        if not symbol:
            continue
        token = str(symbol).strip().upper()
        if not token:
            continue
        if token in seen:
            continue
        cleaned.append(token)
        seen.add(token)
    return cleaned


def build_longable_symbols(
    *,
    crypto_symbols: Sequence[str] | None = None,
    stock_symbols: Sequence[str] | None = None,
) -> List[str]:
    crypto = list(crypto_symbols) if crypto_symbols is not None else list(DEFAULT_LONG_CRYPTO)
    stocks = list(stock_symbols) if stock_symbols is not None else list(LONGABLE_STOCKS)
    merged = normalize_symbols(list(crypto) + list(stocks))
    return merged


def build_shortable_symbols(
    *,
    stock_symbols: Sequence[str] | None = None,
) -> List[str]:
    stocks = list(stock_symbols) if stock_symbols is not None else list(SHORTABLE_STOCKS)
    return normalize_symbols(stocks)


def split_symbols_by_constraint(
    symbols: Iterable[str],
    *,
    longable_stocks: Sequence[str] | None = None,
    shortable_stocks: Sequence[str] | None = None,
) -> tuple[List[str], List[str]]:
    longable_stock_set = {s.upper() for s in (longable_stocks or LONGABLE_STOCKS)}
    shortable_stock_set = {s.upper() for s in (shortable_stocks or SHORTABLE_STOCKS)}
    longable: List[str] = []
    shortable: List[str] = []
    for symbol in normalize_symbols(symbols):
        if is_crypto_symbol(symbol):
            longable.append(symbol)
            continue
        if symbol in longable_stock_set:
            longable.append(symbol)
        if symbol in shortable_stock_set:
            shortable.append(symbol)
    return longable, shortable


__all__ = [
    "DEFAULT_LONG_CRYPTO",
    "LONGABLE_STOCKS",
    "SHORTABLE_STOCKS",
    "build_longable_symbols",
    "build_shortable_symbols",
    "normalize_symbols",
    "split_symbols_by_constraint",
]
