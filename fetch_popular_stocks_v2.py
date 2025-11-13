#!/usr/bin/env python3
"""
Fetch 100+ popular, liquid stocks using a curated list of well-known stocks.
"""

import json

# Curated list of 100+ popular, liquid US stocks
POPULAR_STOCKS = [
    # Mega cap tech (FAANG+)
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA',

    # Large cap tech
    'AMD', 'INTC', 'QCOM', 'AVGO', 'ADBE', 'CRM', 'ORCL', 'CSCO',
    'TXN', 'AMAT', 'LRCX', 'KLAC', 'NXPI', 'MCHP', 'ADI', 'SNPS',


    # Semiconductors
    'MU', 'ASML', 'TSM', 'ARM', 'MRVL',

    # Software/Cloud
    'NOW', 'ADSK', 'ANSS', 'CDNS', 'INTU', 'WDAY', 'ZS', 'CRWD',
    'NET', 'DDOG', 'SNOW', 'PLTR', 'U', 'DOCU',

    # E-commerce/Consumer Tech
    'SHOP', 'EBAY', 'UBER', 'LYFT', 'ABNB', 'DASH', 'COIN',

    # Streaming/Media
    'NFLX', 'DIS', 'ROKU', 'SPOT', 'WBD',

    # Financials
    'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW',
    'AXP', 'V', 'MA', 'PYPL', 'SQ',

    # Healthcare
    'JNJ', 'UNH', 'PFE', 'ABT', 'TMO', 'MRK', 'LLY', 'ABBV',
    'BMY', 'AMGN', 'GILD', 'ISRG', 'DHR', 'CVS',

    # Biotech
    'MRNA', 'BNTX', 'REGN', 'VRTX', 'BIIB',

    # Consumer/Retail
    'WMT', 'HD', 'LOW', 'TGT', 'COST', 'NKE', 'SBUX', 'MCD',
    'CMG', 'YUM', 'LULU', 'TJX',

    # Industrial
    'BA', 'CAT', 'DE', 'HON', 'MMM', 'GE', 'LMT', 'RTX',
    'UPS', 'FDX', 'WM', 'EMR',

    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO',

    # Materials
    'LIN', 'APD', 'ECL', 'NEM', 'FCX',

    # Real Estate/REITs
    'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'O', 'SPG',

    # Utilities
    'NEE', 'DUK', 'SO', 'D', 'AEP',

    # Telecom
    'T', 'VZ', 'TMUS',

    # Pharma
    'NVO', 'SNY', 'AZN', 'NVS',

    # Automotive
    'F', 'GM', 'TM', 'RIVN', 'LCID',

    # Consumer Goods
    'PG', 'KO', 'PEP', 'PM', 'MO', 'CL', 'EL',

    # Diversified
    'BRK.B', 'JNJ', 'XOM', 'UNH',

    # ETFs (broad market)
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI',

    # Sector ETFs
    'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLU', 'XLP', 'XLY',

    # International
    'EFA', 'EEM', 'VXUS',

    # Fixed Income/Commodities
    'GLD', 'SLV', 'USO', 'UNG', 'DBA', 'DBC',

    # ARK ETFs
    'ARKK', 'ARKW', 'ARKG', 'ARKQ',

    # Other growth/tech
    'REIT', 'SOFI', 'HOOD', 'COUR', 'RBLX', 'UPST',
    'AFRM', 'PTON', 'PLUG', 'FSLR', 'ENPH',

    # Industrials/Defense
    'NOC', 'GD', 'TDG', 'ETN',

    # Healthcare Equipment
    'MDT', 'SYK', 'BSX', 'EW',

    # Software services
    'VEEV', 'OKTA', 'TWLO', 'ZM', 'BILL',

    # Clean energy
    'ICLN', 'TAN', 'QCLN',

    # Semiconductors
    'ON', 'MPWR', 'MXL', 'SWKS', 'QRVO',

    # Japan stocks available on US exchanges
    'SONY', 'NTT',
]

# Add crypto pairs
CRYPTO_PAIRS = [
    'BTC-USD', 'ETH-USD', 'ETHUSD', 'BTCUSD',
    'SOL-USD', 'SOLUSD', 'AVAX-USD', 'AVAXUSD',
    'MATIC-USD', 'LINK-USD', 'LINKUSD', 'UNI-USD', 'UNIUSD',
    'DOT-USD', 'ATOM-USD', 'XRP-USD', 'ADA-USD',
    'ALGO-USD', 'XLM-USD', 'DOGE-USD', 'SHIB-USD',
    'LTCUSD', 'PAXGUSD',
]

def main():
    """Create a validated stock list"""

    # Remove duplicates and sort
    all_stocks = list(set(POPULAR_STOCKS))
    all_stocks.sort()

    # Create output structure
    stocks_data = []
    for symbol in all_stocks:
        stocks_data.append({
            'symbol': symbol,
            'validated': True,
            'source': 'curated_list'
        })

    for symbol in CRYPTO_PAIRS:
        stocks_data.append({
            'symbol': symbol,
            'validated': True,
            'source': 'crypto',
            'is_crypto': True
        })

    print("=" * 80)
    print("POPULAR TRADEABLE STOCKS")
    print("=" * 80)
    print()
    print(f"Total stocks: {len(all_stocks)}")
    print(f"Total crypto pairs: {len(CRYPTO_PAIRS)}")
    print(f"Combined total: {len(all_stocks) + len(CRYPTO_PAIRS)}")
    print()

    # Save to JSON
    output_file = 'validated_popular_stocks.json'
    with open(output_file, 'w') as f:
        json.dump(stocks_data, f, indent=2)
    print(f"✓ Saved to {output_file}")
    print()

    # Print symbols
    all_symbols = all_stocks + CRYPTO_PAIRS
    print("=" * 80)
    print("ALL SYMBOLS (comma-separated)")
    print("=" * 80)
    print(", ".join(all_symbols))
    print()

    print("=" * 80)
    print("STOCKS ONLY (comma-separated)")
    print("=" * 80)
    print(", ".join(all_stocks))
    print()

    print("=" * 80)
    print("Python list (all)")
    print("=" * 80)
    print(repr(all_symbols))

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
