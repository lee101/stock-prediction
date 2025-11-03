# All crypto symbols that could be identified as crypto (for checks/identification)
all_crypto_symbols = [
    'ADAUSD',
    'ALGOUSD',
    'ATOMUSD',
    'AVAXUSD',
    'BNBUSD',
    'BTCUSD',
    'DOGEUSD',
    'DOTUSD',
    'ETHUSD',
    'LINKUSD',
    'LTCUSD',
    'MATICUSD',
    'PAXGUSD',
    'SHIBUSD',
    'SOLUSD',
    'TRXUSD',
    'UNIUSD',
    'VETUSD',
    'XLMUSD',
    'XRPUSD',
]

# Active crypto symbols we actually want to trade (have reliable data)
active_crypto_symbols = [
    'BNBUSD',
    'BTCUSD',
    'ETHUSD',
    'PAXGUSD',
    'UNIUSD',
]

# Backwards compatibility - default to active symbols for trading
crypto_symbols = active_crypto_symbols
