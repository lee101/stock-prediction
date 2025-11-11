"""
Fetch top stocks from Alpaca by market cap/volume

This script queries Alpaca's assets API to get a list of tradeable US stocks,
filters them by tradability and status, and returns the top N stocks.
"""

import sys
from pathlib import Path
from typing import List, Dict
import pandas as pd
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alpaca_wrapper import alpaca_api


def fetch_top_stocks(limit: int = 200, output_file: str = None) -> List[str]:
    """
    Fetch top N tradeable stocks from Alpaca.

    Args:
        limit: Number of top stocks to return
        output_file: Optional path to save the stock list as CSV

    Returns:
        List of stock symbols
    """
    logger.info(f"Fetching tradeable assets from Alpaca...")

    try:
        # Get all assets from Alpaca
        assets = alpaca_api.get_all_assets()

        # Filter for:
        # 1. Active stocks
        # 2. Tradeable on Alpaca
        # 3. US exchanges (NYSE, NASDAQ, AMEX)
        # 4. Not fractional_only (want full shares tradeable)
        stock_assets = []
        for asset in assets:
            if (asset.status == 'active' and
                asset.tradable and
                asset.asset_class == 'us_equity' and
                asset.exchange in ['NYSE', 'NASDAQ', 'AMEX', 'ARCA', 'BATS']):

                stock_assets.append({
                    'symbol': asset.symbol,
                    'name': asset.name,
                    'exchange': asset.exchange,
                    'easy_to_borrow': getattr(asset, 'easy_to_borrow', True),
                    'shortable': getattr(asset, 'shortable', False),
                    'marginable': getattr(asset, 'marginable', False),
                })

        logger.info(f"Found {len(stock_assets)} tradeable stocks")

        # Convert to dataframe for easier manipulation
        df = pd.DataFrame(stock_assets)

        # Prioritize by:
        # 1. Easy to borrow (for shorting)
        # 2. Marginable (for leverage)
        # 3. Alphabetically (we'll sort by volume later with historical data if needed)
        df['priority_score'] = (
            df['easy_to_borrow'].astype(int) * 4 +
            df['marginable'].astype(int) * 2 +
            df['shortable'].astype(int) * 1
        )

        df = df.sort_values(['priority_score', 'symbol'], ascending=[False, True])

        # Take top N
        top_stocks = df.head(limit)
        symbols = top_stocks['symbol'].tolist()

        logger.info(f"Selected top {len(symbols)} stocks")
        logger.info(f"Sample: {symbols[:10]}")

        # Save to file if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            top_stocks.to_csv(output_path, index=False)
            logger.info(f"Saved stock list to {output_path}")

        return symbols

    except Exception as e:
        logger.error(f"Error fetching stocks from Alpaca: {e}")
        logger.warning("Falling back to default stock list")

        # Fallback to a reasonable default list of major stocks
        from alpaca_wrapper import EXTENDED_STOCK_SYMBOLS
        return sorted(EXTENDED_STOCK_SYMBOLS)[:limit]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch top stocks from Alpaca")
    parser.add_argument("--limit", type=int, default=200, help="Number of stocks to fetch")
    parser.add_argument("--output", type=str, default="strategytraining/top_stocks.csv",
                       help="Output CSV file path")

    args = parser.parse_args()

    symbols = fetch_top_stocks(limit=args.limit, output_file=args.output)

    print(f"\nFetched {len(symbols)} stocks")
    print(f"First 20: {symbols[:20]}")
    print(f"Last 20: {symbols[-20:]}")
