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
def remap_symbols(symbol):
    crypto_remap = {
        "ETHUSD": "ETH/USD",
        "LTCUSD": "LTC/USD",
        "BTCUSD": "BTC/USD",
    }
    if symbol in crypto_symbols:
        return crypto_remap[symbol]
    return symbol

def unmap_symbols(symbol):
    crypto_remap = {
        "ETH/USD": "ETHUSD",
        "LTC/USD": "LTCUSD",
        "BTC/USD": "BTCUSD",
    }
    if symbol in crypto_remap:
        return crypto_remap[symbol]
    return symbol

def binance_remap_symbols(symbol):
    crypto_remap = {
        "ETHUSD": "ETHUSDT",
        "LTCUSD": "LTCUSDT",
        "BTCUSD": "BTCUSDT",
    }
    if symbol in crypto_symbols:
        return crypto_remap[symbol]
    return symbol
