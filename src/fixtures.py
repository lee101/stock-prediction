# All crypto symbols that could be identified as crypto (for checks/identification)
all_crypto_symbols = [
    'AAVEUSD',
    'ADAUSD',
    'ALGOUSD',
    'APTUSD',
    'ATOMUSD',
    'AVAXUSD',
    'BNBUSD',
    'BCHUSD',
    'BTCUSD',
    'DOGEUSD',
    'DOTUSD',
    'AEURUSD',
    'ETHUSD',
    'LINKUSD',
    'LTCUSD',
    'MATICUSD',
    'POLUSD',
    'SHIBUSD',
    'SKYUSD',
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
    'LINKUSD',
    'SKYUSD',
    'SOLUSD',
    'UNIUSD',
]

# Backwards compatibility - default to active symbols for trading
crypto_symbols = active_crypto_symbols

# All stock symbols we have training data for (from trainingdatabp/)
all_stock_symbols = [
    # Major tech
    'AAPL', 'MSFT', 'GOOG', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC',
    'AVGO', 'ADBE', 'CRM', 'NFLX', 'ORCL', 'CSCO', 'QCOM', 'TXN', 'MU', 'AMAT',
    # Financials
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'V', 'MA', 'BLK', 'SCHW', 'COF',
    # Healthcare
    'JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'LLY', 'TMO', 'DHR', 'ABT', 'BMY',
    # Consumer
    'WMT', 'COST', 'HD', 'LOW', 'TGT', 'SBUX', 'MCD', 'NKE', 'PG', 'KO', 'PEP',
    # Industrial
    'CAT', 'BA', 'GE', 'HON', 'UPS', 'MMM', 'DE', 'LMT', 'RTX', 'NOC',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY', 'VLO', 'MPC', 'PSX',
    # ETFs
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'EEM', 'GLD', 'SLV', 'XLE', 'XLF', 'XLK',
    # High-growth / Volatile
    'SHOP', 'SQ', 'PLTR', 'COIN', 'HOOD', 'SOFI', 'ROKU', 'SNOW', 'NET', 'DDOG',
    'CRWD', 'ZS', 'OKTA', 'PANW', 'NOW', 'TEAM', 'WDAY', 'VEEV', 'MDB', 'ESTC',
]

# Active stock symbols we actually trade
active_stock_symbols = [
    # Confirmed profitable from backtest
    'NFLX', 'TSLA', 'AVGO', 'AMZN', 'NVDA', 'META', 'AMD', 'AAPL', 'COST', 'ADBE',
]
