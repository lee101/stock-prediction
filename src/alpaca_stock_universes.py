from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from src.trade_directions import DEFAULT_ALPACA_CORE_LONG_STOCKS, DEFAULT_ALPACA_CORE_SHORT_STOCKS


def _normalize_symbols(values: Optional[Iterable[str]]) -> tuple[str, ...]:
    if not values:
        return ()
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in values:
        token = str(raw or "").strip().upper()
        if not token or token in seen:
            continue
        ordered.append(token)
        seen.add(token)
    return tuple(ordered)


@dataclass(frozen=True)
class AlpacaStockUniverse:
    name: str
    long_symbols: tuple[str, ...] = ()
    short_symbols: tuple[str, ...] = ()
    description: str = ""

    @property
    def symbols(self) -> tuple[str, ...]:
        return _normalize_symbols((*self.long_symbols, *self.short_symbols))


DEFAULT_ALPACA_AI_TECH_LONG_STOCKS: tuple[str, ...] = (
    "NVDA",
    "PLTR",
    "AMD",
    "GOOG",
    "MSFT",
    "META",
    "AMZN",
    "AAPL",
    "TSLA",
    "NET",
    "NFLX",
)

DEFAULT_ALPACA_SOFTWARE_SHORT_STOCKS: tuple[str, ...] = (
    "YELP",
    "EBAY",
    "TRIP",
    "MTCH",
    "KIND",
    "ANGI",
    "Z",
    "EXPE",
    "BKNG",
    "NWSA",
    "NYT",
)

DEFAULT_ALPACA_STOCK19_LONG_STOCKS: tuple[str, ...] = (
    "NVDA",
    "NET",
    "AMD",
    "GOOG",
    "MSFT",
    "META",
    "AMZN",
    "AAPL",
    "TSLA",
)

DEFAULT_ALPACA_STOCK19_SHORT_STOCKS: tuple[str, ...] = (
    "YELP",
    "ANGI",
    "Z",
    "MTCH",
    "TRIP",
    "BKNG",
    "EBAY",
    "EXPE",
    "NWSA",
    "NYT",
)

DEFAULT_ALPACA_STOCK21_PLUS_PLTR_NFLX_LONG_STOCKS: tuple[str, ...] = (
    *DEFAULT_ALPACA_STOCK19_LONG_STOCKS,
    "PLTR",
    "NFLX",
)


STOCK_UNIVERSES: dict[str, AlpacaStockUniverse] = {
    "live8": AlpacaStockUniverse(
        name="live8",
        long_symbols=tuple(DEFAULT_ALPACA_CORE_LONG_STOCKS),
        short_symbols=tuple(DEFAULT_ALPACA_CORE_SHORT_STOCKS[:4]),
        description="Legacy Alpaca live8 mix: core longs plus four short-only software/media names.",
    ),
    "ai_tech_long11": AlpacaStockUniverse(
        name="ai_tech_long11",
        long_symbols=DEFAULT_ALPACA_AI_TECH_LONG_STOCKS,
        description="Broader long-only AI/tech stock basket for Alpaca hourly experiments.",
    ),
    "software_short11": AlpacaStockUniverse(
        name="software_short11",
        short_symbols=DEFAULT_ALPACA_SOFTWARE_SHORT_STOCKS,
        description="User-curated short-only software, travel, classifieds, and media names.",
    ),
    "stock19": AlpacaStockUniverse(
        name="stock19",
        long_symbols=DEFAULT_ALPACA_STOCK19_LONG_STOCKS,
        short_symbols=DEFAULT_ALPACA_STOCK19_SHORT_STOCKS,
        description="Existing stock19 Alpaca hourly universe used in prior stock-only runs.",
    ),
    "stock21_plus_pltr_nflx": AlpacaStockUniverse(
        name="stock21_plus_pltr_nflx",
        long_symbols=DEFAULT_ALPACA_STOCK21_PLUS_PLTR_NFLX_LONG_STOCKS,
        short_symbols=DEFAULT_ALPACA_STOCK19_SHORT_STOCKS,
        description="Stock19 plus PLTR and NFLX on the long side for broader AI/software stock experiments.",
    ),
    "ai_long_short22": AlpacaStockUniverse(
        name="ai_long_short22",
        long_symbols=DEFAULT_ALPACA_AI_TECH_LONG_STOCKS,
        short_symbols=DEFAULT_ALPACA_SOFTWARE_SHORT_STOCKS,
        description="Expanded AI-tech long basket plus the full shortable software/media set.",
    ),
}


def available_stock_universe_names() -> tuple[str, ...]:
    return tuple(sorted(STOCK_UNIVERSES))


def resolve_stock_universe(name: Optional[str]) -> Optional[AlpacaStockUniverse]:
    token = str(name or "").strip().lower().replace("-", "_")
    if not token:
        return None
    universe = STOCK_UNIVERSES.get(token)
    if universe is not None:
        return universe
    raise ValueError(
        f"Unknown stock universe '{name}'. Available: {', '.join(available_stock_universe_names())}"
    )


def merge_symbols_with_stock_universe(
    *,
    base_symbols: Optional[Sequence[str]],
    stock_universe: Optional[str],
    long_only_symbols: Optional[Sequence[str]] = None,
    short_only_symbols: Optional[Sequence[str]] = None,
    universe_only: bool = False,
) -> tuple[list[str], list[str], list[str]]:
    universe = resolve_stock_universe(stock_universe)
    base = () if universe_only else _normalize_symbols(base_symbols)
    long_only = _normalize_symbols(long_only_symbols)
    short_only = _normalize_symbols(short_only_symbols)
    if universe is None:
        return list(base), list(long_only), list(short_only)
    symbols = _normalize_symbols((*base, *universe.symbols))
    merged_long_only = _normalize_symbols((*long_only, *universe.long_symbols))
    merged_short_only = _normalize_symbols((*short_only, *universe.short_symbols))
    return list(symbols), list(merged_long_only), list(merged_short_only)


__all__ = [
    "AlpacaStockUniverse",
    "DEFAULT_ALPACA_AI_TECH_LONG_STOCKS",
    "DEFAULT_ALPACA_SOFTWARE_SHORT_STOCKS",
    "DEFAULT_ALPACA_STOCK19_LONG_STOCKS",
    "DEFAULT_ALPACA_STOCK19_SHORT_STOCKS",
    "DEFAULT_ALPACA_STOCK21_PLUS_PLTR_NFLX_LONG_STOCKS",
    "STOCK_UNIVERSES",
    "available_stock_universe_names",
    "merge_symbols_with_stock_universe",
    "resolve_stock_universe",
]
