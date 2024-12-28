from src.fixtures import crypto_symbols
# USD currencies
#AAVE, BAT, BCH, BTC, DAI, ETH, GRT, LINK, LTC, MATIC, MKR, NEAR, PAXG, SHIB, SOL, UNI, USDT

# supported
supported_cryptos = [
    'BTC',
    'ETH',
    'GRT',
    'MATIC',
    'PAXG',
    'MKR',
    'UNI',
    'NEAR',
    'MKR',
]

# add paxg and mkr to get resiliency from crypto
def remap_symbols(symbol):
    crypto_remap = {
        "ETHUSD": "ETH/USD",
        "LTCUSD": "LTC/USD",
        "BTCUSD": "BTC/USD",
        "PAXGUSD": "PAXG/USD",
        "UNIUSD": "UNI/USD",
    }
    if symbol in crypto_symbols:
        return crypto_remap[symbol]
    return symbol

def pairs_equal(symbol1: str, symbol2: str) -> bool:
    """Compare two symbols, handling different formats (BTCUSD vs BTC/USD)"""
    # Normalize both symbols by removing slashes
    s1 = symbol1.replace("/", "").upper()
    s2 = symbol2.replace("/", "").upper()

    return remap_symbols(s1) == remap_symbols(s2)

def unmap_symbols(symbol):
    crypto_remap = {
        "ETH/USD": "ETHUSD",
        "LTC/USD": "LTCUSD",
        "BTC/USD": "BTCUSD",
        "PAXG/USD": "PAXGUSD",
        "UNI/USD": "UNIUSD",
    }
    if symbol in crypto_remap:
        return crypto_remap[symbol]
    return symbol

def binance_remap_symbols(symbol):
    crypto_remap = {
        "ETHUSD": "ETHUSDT",
        "LTCUSD": "LTCUSDT",
        "BTCUSD": "BTCUSDT",
        "PAXGUSD": "PAXGUSDT",
        "UNIUSD": "UNIUSDT",
    }
    if symbol in crypto_symbols:
        return crypto_remap[symbol]
    return symbol
