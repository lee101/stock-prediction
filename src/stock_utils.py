import os

from src.fixtures import crypto_symbols, all_crypto_symbols

# keep the base tickers handy for downstream checks
supported_cryptos = sorted({symbol[:-3] for symbol in crypto_symbols})

_STABLE_QUOTES = ("USD", "USDT", "USDC", "FDUSD", "BUSD", "TUSD", "DAI")


def remap_symbols(symbol: str) -> str:
    """Normalize crypto symbols to Alpaca-style slash pairs (e.g., BTCUSD -> BTC/USD).

    For known crypto symbols we remap deterministically using fixture lists.
    For unknown-but-stable-quote pairs (e.g., AVAXUSD) we still remap so callers
    don't accidentally send invalid symbols like AVAXUSD to Alpaca endpoints that
    require BASE/QUOTE formatting.
    """
    if not symbol:
        return symbol

    normalized = symbol.replace("/", "").replace("-", "").upper()

    # Avoid turning stable quote tokens (e.g., "USDT", "BUSD") into "U/SDT" etc.
    if normalized in _STABLE_QUOTES:
        return normalized

    # Check both active and all crypto symbols to ensure proper remapping.
    if normalized in crypto_symbols or normalized in all_crypto_symbols:
        return f"{normalized[:-3]}/{normalized[-3:]}"

    for quote in _STABLE_QUOTES:
        if normalized.endswith(quote) and len(normalized) > len(quote):
            base = normalized[: -len(quote)]
            if base.isalpha():
                return f"{base}/{quote}"

    # Already a stock ticker, or an unsupported format.
    return symbol

def pairs_equal(symbol1: str, symbol2: str) -> bool:
    """Compare two symbols, handling different formats (BTCUSD vs BTC/USD)"""
    # Normalize both symbols by removing slashes
    s1 = symbol1.replace("/", "").upper()
    s2 = symbol2.replace("/", "").upper()

    return remap_symbols(s1) == remap_symbols(s2)


def unmap_symbols(symbol: str) -> str:
    """Normalize slash pairs back to concatenated symbols (e.g., BTC/USD -> BTCUSD)."""
    if not symbol or "/" not in symbol:
        return symbol

    base, quote = symbol.split("/", 1)
    base = base.strip().upper()
    quote = quote.strip().upper()
    candidate = f"{base}{quote}"

    if candidate in crypto_symbols or candidate in all_crypto_symbols:
        return candidate
    if quote in _STABLE_QUOTES and base.isalpha():
        return candidate
    return symbol


def binance_remap_symbols(symbol: str) -> str:
    """Map internal USD-quoted crypto symbols (e.g., BTCUSD) to a Binance spot pair.

    Historical default is USDT. Override via BINANCE_DEFAULT_QUOTE (e.g., FDUSD).
    When using Binance.US (BINANCE_TLD=us), keep USD pairs unchanged.
    """
    if not symbol:
        return symbol

    normalized = symbol.replace("/", "").replace("-", "").replace("_", "").strip().upper()
    if not normalized:
        return normalized
    if normalized in _STABLE_QUOTES:
        return normalized

    # If a symbol already has a stable quote, keep it as-is.
    for quote in ("USDT", "FDUSD", "USDC", "BUSD", "TUSD", "USDP", "DAI"):
        if normalized.endswith(quote) and len(normalized) > len(quote):
            return normalized

    # On Binance.US, USD-quoted spot pairs are the norm (e.g., BTCUSD).
    if os.getenv("BINANCE_TLD", "").strip().lower() == "us":
        return normalized

    # For our internal crypto fixtures (USD-quoted), map to a preferred stable quote.
    if normalized in crypto_symbols or normalized in all_crypto_symbols:
        if normalized.endswith("USD") and len(normalized) > 3:
            base = normalized[:-3]
            preferred = os.getenv("BINANCE_DEFAULT_QUOTE", "USDT").strip().upper() or "USDT"
            if preferred == "USD":
                return normalized
            if preferred not in _STABLE_QUOTES:
                preferred = "USDT"
            return f"{base}{preferred}"

    # Fallback: leave unchanged (stock tickers, already-normalized pairs, etc.).
    return normalized
