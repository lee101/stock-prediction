from src.fixtures import crypto_symbols


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
