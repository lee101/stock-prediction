"""YAML-based universe config for work-stealing strategy."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class SymbolInfo:
    symbol: str          # e.g. "BTCUSD"
    usdt_pair: str       # e.g. "BTCUSDT"
    fee_tier: str        # "fdusd" or "usdt"
    margin_eligible: bool
    has_lora: bool
    min_notional: float


def load_universe(yaml_path: str | Path) -> list[SymbolInfo]:
    path = Path(yaml_path)
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Universe file not found: {path}")
    if not isinstance(data, dict) or "symbols" not in data:
        raise ValueError("Universe YAML must have a top-level 'symbols' key")

    seen: set[str] = set()
    universe: list[SymbolInfo] = []
    for entry in data["symbols"]:
        sym = str(entry.get("symbol", "")).strip().upper()
        if not sym:
            raise ValueError(f"Symbol entry missing 'symbol' field: {entry}")
        if sym in seen:
            raise ValueError(f"Duplicate symbol: {sym}")
        seen.add(sym)

        usdt_pair = str(entry.get("usdt_pair", sym.replace("USD", "") + "USDT"))
        fee_tier = str(entry.get("fee_tier", "usdt")).lower()
        if fee_tier not in ("fdusd", "usdt"):
            raise ValueError(f"Invalid fee_tier '{fee_tier}' for {sym}, must be 'fdusd' or 'usdt'")

        universe.append(SymbolInfo(
            symbol=sym,
            usdt_pair=usdt_pair,
            fee_tier=fee_tier,
            margin_eligible=bool(entry.get("margin_eligible", True)),
            has_lora=bool(entry.get("has_lora", False)),
            min_notional=float(entry.get("min_notional", 10.0)),
        ))
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
