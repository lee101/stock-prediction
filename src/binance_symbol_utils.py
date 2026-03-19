from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


DEFAULT_STABLE_QUOTES: Tuple[str, ...] = (
    "FDUSD",
    "USDT",
    "USDC",
    "BUSD",
    "TUSD",
    "USDP",
    "U",
    "USD",
)

DEFAULT_FORECAST_CACHE_QUOTES: Tuple[str, ...] = (
    "USDT",
    "FDUSD",
    "USDC",
    "BUSD",
    "TUSD",
    "USDP",
    "U",
)

_BASE_SYMBOL_ALIASES = {
    "RNDR": "RENDER",
    "RENDER": "RNDR",
}


def normalize_compact_symbol(symbol: str) -> str:
    """Normalize a symbol to Binance-style compact format (e.g., 'BTC/USDT' -> 'BTCUSDT')."""
    if not isinstance(symbol, str):
        raise TypeError(f"symbol must be str, got {type(symbol).__name__}")
    return (
        symbol.replace("/", "")
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
        .strip()
        .upper()
    )


def split_stable_quote_symbol(
    symbol: str,
    *,
    stable_quotes: Sequence[str] = DEFAULT_STABLE_QUOTES,
) -> Tuple[str, str]:
    """Split a compact symbol into (base, quote) for known stable-quote suffixes.

    Returns ("", "") only for empty/whitespace symbols.
    """
    normalized = normalize_compact_symbol(symbol)
    if not normalized:
        return "", ""
    # Prefer longer matches (e.g., FDUSD before USD).
    quotes = sorted({q.upper() for q in stable_quotes if q}, key=len, reverse=True)
    for quote in quotes:
        if normalized.endswith(quote) and len(normalized) > len(quote):
            return normalized[: -len(quote)], quote
    return normalized, ""


def proxy_symbol_to_usd(
    symbol: str,
    *,
    stable_quotes: Sequence[str] = DEFAULT_STABLE_QUOTES,
) -> str:
    """Map stable-quote crypto pairs to a USD proxy symbol.

    Examples:
        BTCUSDT -> BTCUSD
        SOLFDUSD -> SOLUSD
        BTCUSD -> BTCUSD
        AAPL -> AAPL
    """
    normalized = normalize_compact_symbol(symbol)
    if not normalized:
        return normalized
    base, quote = split_stable_quote_symbol(normalized, stable_quotes=stable_quotes)
    if quote and quote != "USD":
        return f"{base}USD"
    return normalized


def stable_quote_aliases_from_usd(
    symbol_usd: str,
    *,
    stable_quotes: Sequence[str] = DEFAULT_STABLE_QUOTES,
) -> List[str]:
    """Return stable-quote aliases for a USD-quoted crypto symbol (e.g., BTCUSD -> BTCUSDT,...)."""
    normalized = normalize_compact_symbol(symbol_usd)
    base, quote = split_stable_quote_symbol(normalized, stable_quotes=("USD",))
    if quote != "USD" or not base:
        return []
    aliases: List[str] = []
    seen = set()
    for q in stable_quotes:
        q_norm = normalize_compact_symbol(q)
        if not q_norm or q_norm == "USD":
            continue
        alias = f"{base}{q_norm}"
        if alias not in seen:
            aliases.append(alias)
            seen.add(alias)
    return aliases


def forecast_cache_symbol_candidates(
    symbol: str,
    *,
    stable_quotes: Sequence[str] = DEFAULT_FORECAST_CACHE_QUOTES,
) -> List[str]:
    """Return forecast-cache lookup candidates for Binance-style crypto symbols.

    The order intentionally prefers the exact symbol first, then a USD proxy,
    then the preferred stable-quote aliases (USDT before FDUSD) so live
    ``BNBUSD``-style symbols can consume newer ``BNBUSDT`` forecast caches.
    """
    normalized = normalize_compact_symbol(symbol)
    if not normalized:
        return []

    out: List[str] = []
    seen = set()

    def _push(value: str) -> None:
        candidate = normalize_compact_symbol(value)
        if candidate and candidate not in seen:
            out.append(candidate)
            seen.add(candidate)

    _push(normalized)

    if (
        normalized.endswith("USD")
        and len(normalized) > len("USD")
        and not normalized.endswith(("FDUSD", "USDT", "USDC", "TUSD", "USDP"))
    ):
        usd_proxy = normalized
    else:
        split_quotes = tuple(dict.fromkeys([*stable_quotes, "USD"]))
        usd_proxy = proxy_symbol_to_usd(normalized, stable_quotes=split_quotes)
    _push(usd_proxy)

    if usd_proxy.endswith("USD"):
        for alias in stable_quote_aliases_from_usd(usd_proxy, stable_quotes=stable_quotes):
            _push(alias)
        base = usd_proxy[: -len("USD")]
    else:
        base = normalized

    alias_base = _BASE_SYMBOL_ALIASES.get(base)
    if alias_base:
        alias_proxy = f"{alias_base}USD"
        _push(alias_proxy)
        for alias in stable_quote_aliases_from_usd(alias_proxy, stable_quotes=stable_quotes):
            _push(alias)

    return out


def unique_symbols(items: Iterable[str]) -> List[str]:
    """Order-preserving de-duplication for symbol lists."""
    out: List[str] = []
    seen = set()
    for item in items:
        normalized = normalize_compact_symbol(item)
        if not normalized or normalized in seen:
            continue
        out.append(normalized)
        seen.add(normalized)
    return out


__all__ = [
    "DEFAULT_FORECAST_CACHE_QUOTES",
    "DEFAULT_STABLE_QUOTES",
    "forecast_cache_symbol_candidates",
    "normalize_compact_symbol",
    "proxy_symbol_to_usd",
    "split_stable_quote_symbol",
    "stable_quote_aliases_from_usd",
    "unique_symbols",
]
