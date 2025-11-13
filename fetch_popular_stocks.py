#!/usr/bin/env python3
"""
Fetch 100+ popular, liquid stocks from Alpaca and validate they're tradeable.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetStatus, AssetClass
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import json
import os

# Import Alpaca credentials
try:
    from env_real import ALP_KEY_ID, ALP_SECRET_KEY, PAPER
except ImportError:
    ALP_KEY_ID = os.getenv('APCA_API_KEY_ID')
    ALP_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')
    PAPER = os.getenv('APCA_PAPER', 'true').lower() == 'true'

def fetch_popular_stocks(target_count=100):
    """Fetch popular, liquid stocks from Alpaca"""

    # Initialize Alpaca clients
    if not ALP_KEY_ID or not ALP_SECRET_KEY:
        raise ValueError("Alpaca API credentials not found. Check env_real.py or environment variables.")

    trading_client = TradingClient(ALP_KEY_ID, ALP_SECRET_KEY, paper=PAPER)
    data_client = StockHistoricalDataClient(ALP_KEY_ID, ALP_SECRET_KEY)

    print("=" * 80)
    print("FETCHING POPULAR STOCKS FROM ALPACA")
    print("=" * 80)
    print()

    # Get all active US equity assets
    print("Fetching all active US equity assets...")
    request = GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
    assets = trading_client.get_all_assets(request)

    print(f"Found {len(assets)} total active assets")
    print()

    # Filter for tradeable stocks
    print("Filtering for tradeable stocks...")
    tradeable_stocks = []

    for asset in assets:
        # Must be tradeable
        if not asset.tradable:
            continue

        # Must be on major exchange
        if asset.exchange not in ['NYSE', 'NASDAQ', 'ARCA', 'NYSEARCA']:
            continue

        # Must be fractionable (indicates liquid stock)
        if not asset.fractionable:
            continue

        # Must be marginable (indicates established stock)
        if not asset.marginable:
            continue

        # Must support easy to borrow (for shorting)
        if not asset.easy_to_borrow:
            continue

        # Skip penny stocks and preferred shares
        symbol = asset.symbol
        if '.' in symbol or '^' in symbol or '-' in symbol:
            continue

        tradeable_stocks.append({
            'symbol': symbol,
            'name': asset.name,
            'exchange': asset.exchange,
            'tradable': asset.tradable,
            'fractionable': asset.fractionable,
            'marginable': asset.marginable,
            'shortable': asset.shortable,
            'easy_to_borrow': asset.easy_to_borrow
        })

    print(f"Found {len(tradeable_stocks)} tradeable stocks after filtering")
    print()

    # Sort by symbol (companies with shorter symbols tend to be more established)
    tradeable_stocks.sort(key=lambda x: (len(x['symbol']), x['symbol']))

    # Get top stocks by checking recent volume/activity
    print("Validating stock activity with recent bars...")
    validated_stocks = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    for i, stock in enumerate(tradeable_stocks[:200]):  # Check top 200
        symbol = stock['symbol']

        try:
            # Get recent bars to validate activity
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )
            bars = data_client.get_stock_bars(request).df

            if len(bars) >= 3:  # Has recent activity
                avg_volume = bars['volume'].mean()
                avg_price = bars['close'].mean()

                # Filter out very low volume or penny stocks
                if avg_volume > 100000 and avg_price > 5:
                    stock['avg_volume'] = float(avg_volume)
                    stock['avg_price'] = float(avg_price)
                    validated_stocks.append(stock)

                    if len(validated_stocks) % 10 == 0:
                        print(f"  Validated {len(validated_stocks)} stocks... (checked {i+1}/{min(200, len(tradeable_stocks))})")

                    if len(validated_stocks) >= target_count:
                        break
        except Exception as e:
            # Skip stocks we can't get data for
            continue

    print()
    print(f"✓ Validated {len(validated_stocks)} liquid, tradeable stocks")
    print()

    # Sort by volume (most liquid first)
    validated_stocks.sort(key=lambda x: x.get('avg_volume', 0), reverse=True)

    return validated_stocks

def display_stocks(stocks):
    """Display stock list"""
    print("=" * 80)
    print("VALIDATED POPULAR STOCKS")
    print("=" * 80)
    print()

    print(f"{'#':<4} {'Symbol':<8} {'Exchange':<10} {'Avg Volume':>15} {'Avg Price':>12} {'Name':<30}")
    print("-" * 80)

    for i, stock in enumerate(stocks, 1):
        print(f"{i:<4} {stock['symbol']:<8} {stock['exchange']:<10} "
              f"{stock['avg_volume']:>15,.0f} ${stock['avg_price']:>11,.2f} "
              f"{stock['name'][:30]:<30}")

    print()
    print(f"Total: {len(stocks)} stocks")

def save_stocks(stocks, filename='validated_popular_stocks.json'):
    """Save stock list to JSON"""
    with open(filename, 'w') as f:
        json.dump(stocks, f, indent=2)
    print(f"\n✓ Saved to {filename}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Fetch popular stocks from Alpaca')
    parser.add_argument('--count', type=int, default=100, help='Number of stocks to fetch')
    parser.add_argument('--output', default='validated_popular_stocks.json', help='Output file')
    args = parser.parse_args()

    try:
        stocks = fetch_popular_stocks(target_count=args.count)
        display_stocks(stocks)
        save_stocks(stocks, args.output)

        # Print symbols for easy copy
        print()
        print("=" * 80)
        print("SYMBOLS (comma-separated)")
        print("=" * 80)
        symbols = [s['symbol'] for s in stocks]
        print(", ".join(symbols))
        print()

        print("=" * 80)
        print("SYMBOLS (Python list)")
        print("=" * 80)
        print(repr(symbols))

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == '__main__':
    sys.exit(main())
