"""Symbol universe and trading constraints for LLM hourly trader."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SymbolConfig:
    symbol: str
    asset_class: str  # "crypto" or "stock"
    allowed_directions: list[str] = field(default_factory=lambda: ["long"])
    # Crypto trades 24/7; stocks only during market hours
    maker_fee: float = 0.0008  # 8bp default


# Direction logic:
#   - AI / growth stocks: long only
#   - Old / declining / non-AI: short only
#   - Crypto: long only (no shorting on spot)

SYMBOL_UNIVERSE: dict[str, SymbolConfig] = {
    # --- Crypto (long only, 24/7) ---
    "BTCUSD": SymbolConfig("BTCUSD", "crypto", ["long"], maker_fee=0.0008),
    "ETHUSD": SymbolConfig("ETHUSD", "crypto", ["long"], maker_fee=0.0008),
    "SOLUSD": SymbolConfig("SOLUSD", "crypto", ["long"], maker_fee=0.0008),
    "LINKUSD": SymbolConfig("LINKUSD", "crypto", ["long"], maker_fee=0.0008),
    "UNIUSD": SymbolConfig("UNIUSD", "crypto", ["long"], maker_fee=0.0008),
    # --- AI / Growth stocks (long only) ---
    "NVDA": SymbolConfig("NVDA", "stock", ["long"], maker_fee=0.0005),
    "META": SymbolConfig("META", "stock", ["long"], maker_fee=0.0005),
    "AAPL": SymbolConfig("AAPL", "stock", ["long"], maker_fee=0.0005),
    "TSLA": SymbolConfig("TSLA", "stock", ["long"], maker_fee=0.0005),
    "MSFT": SymbolConfig("MSFT", "stock", ["long"], maker_fee=0.0005),
    "GOOG": SymbolConfig("GOOG", "stock", ["long"], maker_fee=0.0005),
    "PLTR": SymbolConfig("PLTR", "stock", ["long"], maker_fee=0.0005),
    "NFLX": SymbolConfig("NFLX", "stock", ["long"], maker_fee=0.0005),
    "NET": SymbolConfig("NET", "stock", ["long"], maker_fee=0.0005),
    # --- Old / non-AI stocks (short only) ---
    "DBX": SymbolConfig("DBX", "stock", ["short"], maker_fee=0.0005),
    "TRIP": SymbolConfig("TRIP", "stock", ["short"], maker_fee=0.0005),
    "NYT": SymbolConfig("NYT", "stock", ["short"], maker_fee=0.0005),
    "YELP": SymbolConfig("YELP", "stock", ["short"], maker_fee=0.0005),
}

# Symbols with usable forecast + bar data (fc_end >= 2026-02-06)
# Stocks with forecasts only to 2026-02-13 can do ~12-day backtests
# Crypto with forecasts to 2026-03-09 can do full 30-day
FORECAST_CUTOFFS = {
    "BTCUSD": "2026-03-09", "ETHUSD": "2026-03-09", "SOLUSD": "2026-03-09",
    "LINKUSD": "2026-03-09", "UNIUSD": "2026-03-09",
    "NVDA": "2026-03-09", "NFLX": "2026-03-09", "AAPL": "2026-03-06",
    "META": "2026-02-13", "MSFT": "2026-02-13", "GOOG": "2026-02-13",
    "PLTR": "2026-02-13", "TSLA": "2026-02-13", "NET": "2026-02-13",
    "DBX": "2026-02-13", "TRIP": "2026-02-13", "NYT": "2026-02-13",
    "YELP": "2026-02-13",
}


@dataclass
class BacktestConfig:
    initial_cash: float = 10_000.0
    max_hold_hours: int = 6
    max_position_pct: float = 0.25  # max 25% of equity in one position
    rate_limit_seconds: float = 4.2  # Gemini free tier: 15 req/min
    model: str = "gemini-3.1-flash-lite-preview"
    prompt_variant: str = "default"
    parallel_workers: int = 1  # >1 for parallel API calls (codex/deepseek)
    end_timestamp: Optional[str] = None  # fixed UTC end timestamp for reproducible reruns
    cache_only: bool = False  # fail if prompt/model pair is not already cached
    thinking_level: Optional[str] = None  # "high"/"medium"/"low" for Gemini; any truthy for Anthropic thinking


# Stock market hours (Eastern): 9:30-16:00, but our hourly bars
# are aligned to UTC. We filter bars to trading hours only.
STOCK_TRADING_HOURS_UTC = (14, 21)  # 9am-4pm ET = 14:00-21:00 UTC (EST+5)
