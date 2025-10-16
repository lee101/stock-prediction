from src.fixtures import crypto_symbols

# keep the base tickers handy for downstream checks
supported_cryptos = sorted({symbol[:-3] for symbol in crypto_symbols})


def remap_symbols(symbol: str) -> str:
    if symbol in crypto_symbols:
        return f"{symbol[:-3]}/{symbol[-3:]}"
    return symbol

def pairs_equal(symbol1: str, symbol2: str) -> bool:
    """Compare two symbols, handling different formats (BTCUSD vs BTC/USD)"""
    # Normalize both symbols by removing slashes
    s1 = symbol1.replace("/", "").upper()
    s2 = symbol2.replace("/", "").upper()

    return remap_symbols(s1) == remap_symbols(s2)


def unmap_symbols(symbol: str) -> str:
    if "/" in symbol:
        base, quote = symbol.split("/", 1)
        candidate = f"{base}{quote}"
        if candidate in crypto_symbols:
            return candidate
    return symbol


def binance_remap_symbols(symbol: str) -> str:
    if symbol in crypto_symbols:
        return f"{symbol[:-3]}USDT"
    return symbol
