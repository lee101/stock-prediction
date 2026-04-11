from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from pathlib import Path


SYMBOL_FILE_COMMENT_MARKER = "#"
type SymbolNormalizer = Callable[[str], str]


class SymbolFileOrdering(StrEnum):
    PRESERVE_INPUT = "preserve_input"
    SORTED = "sorted"


def _dedupe_preserve_order(symbols: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        if symbol in seen:
            continue
        seen.add(symbol)
        deduped.append(symbol)
    return deduped


def parse_symbols_text(
    raw_text: str,
    *,
    normalize_symbol: SymbolNormalizer | None = None,
) -> list[str]:
    values: list[str] = []
    normalizer = normalize_symbol or str.upper
    for raw_line in str(raw_text).splitlines():
        line = raw_line.split(SYMBOL_FILE_COMMENT_MARKER, 1)[0].strip()
        if not line:
            continue
        for token in line.replace(",", " ").split():
            symbol = normalizer(token.strip())
            if symbol:
                values.append(symbol)
    return values


def load_symbols_from_file(
    path: Path,
    *,
    normalize_symbol: SymbolNormalizer | None = None,
    dedupe: bool = True,
    ordering: SymbolFileOrdering = SymbolFileOrdering.PRESERVE_INPUT,
    encoding: str = "utf-8",
) -> list[str]:
    values = parse_symbols_text(
        Path(path).read_text(encoding=encoding),
        normalize_symbol=normalize_symbol,
    )
    if dedupe:
        values = _dedupe_preserve_order(values)
    if ordering is SymbolFileOrdering.SORTED:
        values = sorted(values)
    return values
