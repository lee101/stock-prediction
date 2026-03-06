"""Trade direction constraints (long/short) by symbol.

This module centralizes symbol-level constraints used by simulators and live
traders so that backtests and production execution stay aligned.

Defaults are intentionally conservative: shorting is disabled unless explicitly
enabled by the caller (e.g., via a simulator config flag).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from src.symbol_utils import is_crypto_symbol


DEFAULT_SHORT_ONLY_STOCKS = frozenset(
    {
        "DBX",
        "ANGI",
        "BKNG",
        "EBAY",
        "EXPE",
        "KIND",
        "MTCH",
        "NWSA",
        "NYT",
        "TRIP",
        "YELP",
        "Z",
    }
)

DEFAULT_LONG_ONLY_STOCKS = frozenset(
    {
        "AAPL",
        "AMD",
        "AMZN",
        "GOOG",
        "GOOGL",
        "META",
        "MSFT",
        "NET",
        "NVDA",
        "PLTR",
        "TSLA",
    }
)

DEFAULT_ALPACA_CORE_LONG_STOCKS: tuple[str, ...] = ("NVDA", "PLTR", "GOOG", "TSLA")
DEFAULT_ALPACA_CORE_SHORT_STOCKS: tuple[str, ...] = ("DBX", "TRIP", "MTCH", "NYT", "YELP")
DEFAULT_ALPACA_LIVE8_STOCKS: tuple[str, ...] = (
    *DEFAULT_ALPACA_CORE_LONG_STOCKS,
    *DEFAULT_ALPACA_CORE_SHORT_STOCKS[:4],
)


def _normalize_symbols(values: Optional[Iterable[str]]) -> frozenset[str]:
    if not values:
        return frozenset()
    cleaned: set[str] = set()
    for raw in values:
        if not raw:
            continue
        token = str(raw).strip().upper()
        if token:
            cleaned.add(token)
    return frozenset(cleaned)


@dataclass(frozen=True)
class TradeDirections:
    """Allowed trade directions for a symbol."""

    can_long: bool
    can_short: bool


def resolve_trade_directions(
    symbol: str,
    *,
    allow_short: bool,
    long_only_symbols: Optional[Sequence[str]] = None,
    short_only_symbols: Optional[Sequence[str]] = None,
    use_default_groups: bool = True,
) -> TradeDirections:
    """Resolve whether a symbol can be traded long and/or short.

    Rules:
    - Crypto is always long-only (no shorts).
    - If ``allow_short`` is False: all symbols are treated as long-only.
    - If a symbol is in ``short_only_symbols``: long entries are disabled.
    - If a symbol is in ``long_only_symbols``: short entries are disabled.

    Note: ``short_only_symbols`` and ``long_only_symbols`` only constrain *entry
    direction*. The opposite side is still required to *close* an open position.
    """

    sym = str(symbol or "").strip().upper()
    if not sym:
        return TradeDirections(can_long=False, can_short=False)

    if is_crypto_symbol(sym):
        return TradeDirections(can_long=True, can_short=False)

    if not allow_short:
        # Conservative default: never short unless explicitly enabled.
        return TradeDirections(can_long=True, can_short=False)

    long_only = _normalize_symbols(long_only_symbols)
    short_only = _normalize_symbols(short_only_symbols)
    if use_default_groups:
        long_only = long_only.union(DEFAULT_LONG_ONLY_STOCKS)
        short_only = short_only.union(DEFAULT_SHORT_ONLY_STOCKS)

    if sym in short_only and sym in long_only:
        # Explicit conflict: safest option is to disable trading entirely.
        return TradeDirections(can_long=False, can_short=False)
    if sym in short_only:
        return TradeDirections(can_long=False, can_short=True)
    if sym in long_only:
        return TradeDirections(can_long=True, can_short=False)
    return TradeDirections(can_long=True, can_short=True)


def is_long_only_symbol(
    symbol: str,
    *,
    allow_short: bool = True,
    long_only_symbols: Optional[Sequence[str]] = None,
    short_only_symbols: Optional[Sequence[str]] = None,
    use_default_groups: bool = True,
) -> bool:
    directions = resolve_trade_directions(
        symbol,
        allow_short=allow_short,
        long_only_symbols=long_only_symbols,
        short_only_symbols=short_only_symbols,
        use_default_groups=use_default_groups,
    )
    return directions.can_long and not directions.can_short


def is_short_only_symbol(
    symbol: str,
    *,
    allow_short: bool = True,
    long_only_symbols: Optional[Sequence[str]] = None,
    short_only_symbols: Optional[Sequence[str]] = None,
    use_default_groups: bool = True,
) -> bool:
    directions = resolve_trade_directions(
        symbol,
        allow_short=allow_short,
        long_only_symbols=long_only_symbols,
        short_only_symbols=short_only_symbols,
        use_default_groups=use_default_groups,
    )
    return directions.can_short and not directions.can_long


def trade_direction_name(
    symbol: str,
    *,
    allow_short: bool = True,
    long_only_symbols: Optional[Sequence[str]] = None,
    short_only_symbols: Optional[Sequence[str]] = None,
    use_default_groups: bool = True,
) -> str:
    directions = resolve_trade_directions(
        symbol,
        allow_short=allow_short,
        long_only_symbols=long_only_symbols,
        short_only_symbols=short_only_symbols,
        use_default_groups=use_default_groups,
    )
    if directions.can_long and directions.can_short:
        return "both"
    if directions.can_long:
        return "long"
    if directions.can_short:
        return "short"
    return "none"


__all__ = [
    "DEFAULT_ALPACA_CORE_LONG_STOCKS",
    "DEFAULT_ALPACA_CORE_SHORT_STOCKS",
    "DEFAULT_ALPACA_LIVE8_STOCKS",
    "DEFAULT_LONG_ONLY_STOCKS",
    "DEFAULT_SHORT_ONLY_STOCKS",
    "TradeDirections",
    "is_long_only_symbol",
    "is_short_only_symbol",
    "resolve_trade_directions",
    "trade_direction_name",
]
