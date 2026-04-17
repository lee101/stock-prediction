from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

from src.symbol_file_utils import load_symbols_from_file


MAX_STOCK_SYMBOL_LENGTH = 20
SAFE_STOCK_SYMBOL_RE = re.compile(
    rf"^[A-Z0-9](?:[A-Z0-9.-]{{0,{MAX_STOCK_SYMBOL_LENGTH - 2}}}[A-Z0-9])?$"
)


def normalize_stock_symbol(raw_symbol: object) -> str:
    symbol = str(raw_symbol).strip().upper()
    if not symbol:
        raise ValueError("symbol is required")
    if ".." in symbol or "/" in symbol or "\\" in symbol:
        raise ValueError(f"Unsupported symbol: {raw_symbol}")
    if not SAFE_STOCK_SYMBOL_RE.fullmatch(symbol):
        raise ValueError(f"Unsupported symbol: {raw_symbol}")
    return symbol


def _symbol_items(values: Iterable[object] | Sequence[object]) -> list[object]:
    if isinstance(values, (str, bytes)):
        return [values]
    return list(values)


def normalize_stock_symbol_list(symbols: Iterable[object]) -> list[str]:
    return [normalize_stock_symbol(symbol) for symbol in _symbol_items(symbols)]


def normalize_symbols(raw_symbols: Sequence[object]) -> tuple[list[str], list[str], list[str]]:
    normalized: list[str] = []
    removed_duplicate_symbols: list[str] = []
    ignored_symbol_inputs: list[str] = []
    seen: set[str] = set()
    removed_seen: set[str] = set()

    for raw_symbol in _symbol_items(raw_symbols):
        stripped = str(raw_symbol).strip()
        if not stripped:
            ignored_symbol_inputs.append("<blank>")
            continue
        symbol = normalize_stock_symbol(stripped)
        if symbol in seen:
            if symbol not in removed_seen:
                removed_duplicate_symbols.append(symbol)
                removed_seen.add(symbol)
            continue
        normalized.append(symbol)
        seen.add(symbol)

    if not normalized:
        raise ValueError("No valid symbols configured after normalization")

    return normalized, removed_duplicate_symbols, ignored_symbol_inputs


def load_symbols_file(path: str | Path) -> list[str]:
    values = load_symbols_from_file(
        Path(path),
        normalize_symbol=normalize_stock_symbol,
        dedupe=False,
    )
    if not values:
        raise ValueError(f"No valid symbols found in {path}")
    return values
