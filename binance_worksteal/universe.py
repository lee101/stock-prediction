"""YAML-based universe config for work-stealing strategy."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SymbolInfo:
    symbol: str          # e.g. "BTCUSD"
    usdt_pair: str       # e.g. "BTCUSDT"
    fee_tier: str        # "fdusd" or "usdt"
    margin_eligible: bool
    has_lora: bool
    min_notional: float


def _normalize_universe_symbol(raw_symbol: Any, *, path: Path, index: int) -> str:
    if not isinstance(raw_symbol, str):
        raise ValueError(
            f"Symbol entry at index {index} in {path} must use a string symbol value: {raw_symbol!r}"
        )
    symbol = raw_symbol.strip().upper()
    if symbol.endswith("USDT"):
        symbol = symbol[:-1]
    if not symbol:
        raise ValueError(f"Symbol entry at index {index} in {path} has an empty 'symbol' value")
    return symbol


def _coerce_min_notional(raw_value: Any, *, path: Path, symbol: str) -> float:
    if isinstance(raw_value, bool):
        raise ValueError(f"Invalid min_notional for {symbol} in {path}: {raw_value!r}")
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid min_notional for {symbol} in {path}: {raw_value!r}") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid min_notional for {symbol} in {path}: {raw_value!r}")
    return value


def _coerce_bool_field(
    raw_value: Any,
    *,
    path: Path,
    symbol: str,
    field_name: str,
) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        value = raw_value.strip().lower()
        if value in {"true", "yes", "on", "1"}:
            return True
        if value in {"false", "no", "off", "0"}:
            return False
    if isinstance(raw_value, int):
        if raw_value in (0, 1):
            return bool(raw_value)
    raise ValueError(f"Invalid {field_name} for {symbol} in {path}: {raw_value!r}")


def _parse_symbol_entry(entry: Any, *, path: Path, index: int) -> SymbolInfo:
    if isinstance(entry, str):
        symbol = _normalize_universe_symbol(entry, path=path, index=index)
        base = symbol.replace("USD", "")
        return SymbolInfo(
            symbol=symbol,
            usdt_pair=f"{base}USDT",
            fee_tier="usdt",
            margin_eligible=True,
            has_lora=False,
            min_notional=10.0,
        )
    if not isinstance(entry, dict):
        raise ValueError(
            f"Symbol entry at index {index} in {path} must be a mapping or string: {entry!r}"
        )
    if "symbol" not in entry:
        raise ValueError(f"Symbol entry missing 'symbol' field at index {index} in {path}: {entry!r}")

    symbol = _normalize_universe_symbol(entry.get("symbol"), path=path, index=index)
    base = symbol.replace("USD", "")
    raw_usdt_pair = entry.get("usdt_pair", f"{base}USDT")
    if not isinstance(raw_usdt_pair, str):
        raise ValueError(f"Invalid usdt_pair for {symbol} in {path}: {raw_usdt_pair!r}")
    usdt_pair = raw_usdt_pair.strip().upper()
    if not usdt_pair:
        raise ValueError(f"Invalid usdt_pair for {symbol} in {path}: {raw_usdt_pair!r}")
    fee_tier = str(entry.get("fee_tier", "usdt")).strip().lower()
    if fee_tier not in ("fdusd", "usdt"):
        raise ValueError(f"Invalid fee_tier '{fee_tier}' for {symbol}, must be 'fdusd' or 'usdt'")

    return SymbolInfo(
        symbol=symbol,
        usdt_pair=usdt_pair,
        fee_tier=fee_tier,
        margin_eligible=_coerce_bool_field(
            entry.get("margin_eligible", True),
            path=path,
            symbol=symbol,
            field_name="margin_eligible",
        ),
        has_lora=_coerce_bool_field(
            entry.get("has_lora", False),
            path=path,
            symbol=symbol,
            field_name="has_lora",
        ),
        min_notional=_coerce_min_notional(entry.get("min_notional", 10.0), path=path, symbol=symbol),
    )


def load_universe(yaml_path: str | Path) -> list[SymbolInfo]:
    path = Path(yaml_path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except FileNotFoundError:
        raise FileNotFoundError(f"Universe file not found: {path}")
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid universe YAML in {path}: {exc}") from exc
    if not isinstance(data, dict) or "symbols" not in data:
        raise ValueError("Universe YAML must have a top-level 'symbols' key")

    raw_symbols = data["symbols"]
    if not isinstance(raw_symbols, list):
        raise ValueError(f"Universe YAML 'symbols' value must be a list in {path}")

    seen: set[str] = set()
    universe: list[SymbolInfo] = []
    for index, entry in enumerate(raw_symbols):
        symbol_info = _parse_symbol_entry(entry, path=path, index=index)
        if symbol_info.symbol in seen:
            raise ValueError(f"Duplicate symbol: {symbol_info.symbol}")
        seen.add(symbol_info.symbol)
        universe.append(symbol_info)
    return universe


def get_symbols(universe: list[SymbolInfo]) -> list[str]:
    return [s.symbol for s in universe]


def get_fee(symbol: str, universe: list[SymbolInfo]) -> float:
    for s in universe:
        if s.symbol == symbol:
            return 0.0 if s.fee_tier == "fdusd" else 0.001
    return 0.001


def validate_universe(universe: list[SymbolInfo], data_dir: str | Path | None = None) -> list[str]:
    """Return list of warning strings."""
    warns: list[str] = []
    if not universe:
        warns.append("Universe is empty")
        return warns

    syms = [s.symbol for s in universe]
    if len(syms) != len(set(syms)):
        warns.append("Duplicate symbols detected")

    if data_dir is not None:
        dp = Path(data_dir)
        for s in universe:
            base = s.symbol.replace("USD", "")
            candidates = [f"{base}USDT.csv", f"{s.symbol}.csv", f"{base}USD.csv"]
            if not any((dp / c).exists() for c in candidates):
                warns.append(f"No data file for {s.symbol}")

    return warns
