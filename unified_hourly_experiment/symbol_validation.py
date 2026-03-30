from __future__ import annotations

import re

_SYMBOL_RE = re.compile(r"^[A-Z0-9]+(?:[._-][A-Z0-9]+)*$")
_MAX_SYMBOL_LENGTH = 16


def normalize_symbol(raw: str) -> str:
    symbol = str(raw or "").strip().upper()
    if not symbol:
        raise ValueError("At least one symbol is required.")
    if len(symbol) > _MAX_SYMBOL_LENGTH:
        raise ValueError(f"Unsupported symbol {raw!r}: symbol is longer than {_MAX_SYMBOL_LENGTH} characters.")
    if not _SYMBOL_RE.fullmatch(symbol):
        raise ValueError(
            f"Unsupported symbol {raw!r}: only alphanumerics plus '.', '_' or '-' separators are allowed."
        )
    return symbol


def parse_symbols(raw: str) -> list[str]:
    symbols = [normalize_symbol(token) for token in str(raw).split(",") if token.strip()]
    if not symbols:
        raise ValueError("At least one symbol is required.")
    deduped: list[str] = []
    for symbol in symbols:
        if symbol not in deduped:
            deduped.append(symbol)
    return deduped


__all__ = ["normalize_symbol", "parse_symbols"]
