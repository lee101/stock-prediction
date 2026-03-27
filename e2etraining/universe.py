from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable


_STOCK_SYMBOL_RE = re.compile(r"^[A-Z]{1,5}$")


def _normalize_symbols(values: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in values:
        symbol = str(raw).strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        normalized.append(symbol)
    return normalized


def _load_symbols_from_file(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        if isinstance(payload.get("available_symbols"), list):
            return _normalize_symbols(payload["available_symbols"])
        raise ValueError(f"Unsupported universe file format in {path}")
    if isinstance(payload, list):
        values: list[str] = []
        for item in payload:
            if isinstance(item, dict) and item.get("symbol") is not None:
                values.append(str(item["symbol"]))
            elif isinstance(item, str):
                values.append(item)
        return _normalize_symbols(values)
    raise ValueError(f"Unsupported universe payload in {path}")


def _discover_stock_like_symbols(data_root: Path) -> list[str]:
    values = []
    for path in sorted(Path(data_root).glob("*.csv")):
        symbol = path.stem.upper()
        if _STOCK_SYMBOL_RE.fullmatch(symbol):
            values.append(symbol)
    return _normalize_symbols(values)


def load_stock_universe(
    *,
    data_root: Path,
    universe_file: Path | None = None,
    include_symbols: Iterable[str] = (),
    exclude_symbols: Iterable[str] = (),
    max_assets: int | None = None,
) -> list[str]:
    include = _normalize_symbols(include_symbols)
    exclude = set(_normalize_symbols(exclude_symbols))
    available_from_data = set(_discover_stock_like_symbols(data_root))

    if include:
        selected = [symbol for symbol in include if symbol in available_from_data and symbol not in exclude]
    else:
        selected_source = (
            _load_symbols_from_file(universe_file)
            if universe_file is not None and Path(universe_file).exists()
            else sorted(available_from_data)
        )
        selected = [symbol for symbol in selected_source if symbol in available_from_data and symbol not in exclude]

    if max_assets is not None:
        selected = selected[: max(0, int(max_assets))]
    if not selected:
        raise ValueError("No stock symbols selected for e2e training")
    return selected
