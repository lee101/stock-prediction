from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

STABLE_QUOTES = ("FDUSD", "USDT", "USD")
DEFAULT_BINANCE_HOURLY_SYMBOLS = (
    "BTCFDUSD",
    "ETHFDUSD",
    "AAVEFDUSD",
    "DOGEUSD",
    "SUIUSDT",
    "SOLFDUSD",
)
DEFAULT_SHORTABLE_SYMBOLS = DEFAULT_BINANCE_HOURLY_SYMBOLS
FREE_FEE_SYMBOLS = {"BTCFDUSD", "ETHFDUSD"}


@dataclass(frozen=True)
class SymbolMetadata:
    symbol: str
    shortable: bool
    quote: str
    trade_fee_bps: float


def normalize_symbol(value: object) -> str:
    return str(value or "").replace("/", "").replace("-", "").strip().upper()


def parse_symbols(raw: str | Iterable[str] | None, *, default: Iterable[str] = DEFAULT_BINANCE_HOURLY_SYMBOLS) -> list[str]:
    if raw is None:
        return [normalize_symbol(symbol) for symbol in default]
    if isinstance(raw, str):
        symbols = [token for token in raw.split(",") if token.strip()]
    else:
        symbols = [str(token) for token in raw if str(token).strip()]
    normalized = [normalize_symbol(symbol) for symbol in symbols if normalize_symbol(symbol)]
    return normalized or [normalize_symbol(symbol) for symbol in default]


def infer_quote(symbol: str) -> str:
    normalized = normalize_symbol(symbol)
    for quote in STABLE_QUOTES:
        if normalized.endswith(quote):
            return quote
    return ""


def build_symbol_metadata(
    symbols: Iterable[str],
    *,
    shortable_symbols: Iterable[str] | None = None,
    default_trade_fee_bps: float = 2.0,
) -> list[SymbolMetadata]:
    shortable_set = {
        normalize_symbol(symbol)
        for symbol in (
            shortable_symbols
            if shortable_symbols is not None
            else DEFAULT_SHORTABLE_SYMBOLS
        )
    }
    metadata: list[SymbolMetadata] = []
    for raw_symbol in symbols:
        symbol = normalize_symbol(raw_symbol)
        if not symbol:
            continue
        metadata.append(
            SymbolMetadata(
                symbol=symbol,
                shortable=symbol in shortable_set,
                quote=infer_quote(symbol),
                trade_fee_bps=0.0 if symbol in FREE_FEE_SYMBOLS else float(default_trade_fee_bps),
            )
        )
    if not metadata:
        raise ValueError("No symbols were provided.")
    return metadata
