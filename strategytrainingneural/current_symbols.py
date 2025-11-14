"""
Symbol universe helpers for neural strategy training.

These utilities reuse the existing AST-based extractor that the
strategytraining package already uses to keep `trade_stock_e2e.py` as
the single source of truth for the live symbol list.  The helpers here
add convenience wrappers for splitting the universe into stock vs
crypto cohorts so that we can train dedicated allocators for each
market regime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from strategytraining.symbol_sources import load_trade_stock_symbols


def load_current_symbols(script_path: str | Path = "trade_stock_e2e.py") -> List[str]:
    """
    Return the ordered list of symbols referenced by the trading script.
    """

    return load_trade_stock_symbols(script_path)


def split_by_asset_class(symbols: Sequence[str]) -> Tuple[List[str], List[str]]:
    """
    Separate the provided symbols into stock vs crypto cohorts.

    The heuristic matches the ones elsewhere in the repo: anything ending
    with ``-USD`` (case insensitive) is treated as crypto.
    """

    stock: List[str] = []
    crypto: List[str] = []
    for symbol in symbols:
        normalized = symbol.upper()
        if normalized.endswith("-USD"):
            crypto.append(symbol)
        else:
            stock.append(symbol)
    return stock, crypto


def filter_symbols(
    symbols: Iterable[str],
    *,
    include: Sequence[str] | None = None,
    exclude_crypto: bool = False,
    exclude_stocks: bool = False,
) -> List[str]:
    """
    Filter the symbols according to include lists and asset-class toggles.
    """

    include_set = {item.upper() for item in include} if include else None
    filtered: List[str] = []
    for symbol in symbols:
        upper = symbol.upper()
        if include_set is not None and upper not in include_set:
            continue
        is_crypto = upper.endswith("-USD")
        if exclude_crypto and is_crypto:
            continue
        if exclude_stocks and not is_crypto:
            continue
        filtered.append(symbol)
    return filtered


__all__ = ["load_current_symbols", "split_by_asset_class", "filter_symbols"]
