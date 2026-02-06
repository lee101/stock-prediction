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
    if symbol in crypto_symbols:
        return f"{symbol[:-3]}USDT"
    return symbol
