from __future__ import annotations

from pathlib import Path


def load_symbols_from_file(path: Path) -> list[str]:
    values: list[str] = []
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        for token in line.replace(",", " ").split():
            symbol = token.strip().upper()
            if symbol:
                values.append(symbol)
    deduped: list[str] = []
    seen: set[str] = set()
    for symbol in values:
        if symbol in seen:
            continue
        seen.add(symbol)
        deduped.append(symbol)
    return deduped
