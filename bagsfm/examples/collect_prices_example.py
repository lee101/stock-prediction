#!/usr/bin/env python3
"""
Example: Collect price data and build OHLC bars.

This shows how to:
1. Set up price data collection
2. Collect prices at regular intervals
3. Aggregate into OHLC bars
4. Save data to disk

Usage:
    python -m bagsfm.examples.collect_prices_example
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bagsfm import DataCollector, BagsConfig, DataConfig, TokenConfig
from bagsfm.config import SOL_MINT, CODEX_MINT


# Tokens to track
TOKENS = [
    TokenConfig(
        symbol="SOL",
        mint=SOL_MINT,
        decimals=9,
        name="Solana",
    ),
    TokenConfig(
        symbol="CODEX",
        mint=CODEX_MINT,
        decimals=9,
        name="CODEX",
    ),
]


async def main():
    """Collect prices and show OHLC aggregation."""
    print(f"\n{'='*60}")
    print("Price Collection Example")
    print(f"{'='*60}\n")

    # Create configuration
    bags_config = BagsConfig()
    data_config = DataConfig(
        tracked_tokens=TOKENS,
        collection_interval_minutes=1,  # 1 minute for demo
        ohlc_interval_minutes=1,
    )

    # Create collector
    collector = DataCollector(bags_config, data_config)

    # Load any existing data
    num_prices, num_bars = collector.load_from_disk()
    print(f"Loaded {num_bars} existing OHLC bars from disk")

    # Collect some prices
    print("\nCollecting prices (5 iterations, 10 seconds apart)...")
    print("-" * 60)

    for i in range(5):
        print(f"\n[{datetime.utcnow().strftime('%H:%M:%S')}] Collection {i+1}/5")

        prices = await collector.collect_all_prices()

        for price in prices:
            print(f"  {price.token_symbol}: {price.price_sol:.8f} SOL")

            # Check for completed OHLC bars
            current_bar = collector.aggregator.get_current_bar(price.token_mint)
            if current_bar:
                print(f"    Current bar: O={current_bar.open:.8f} H={current_bar.high:.8f} "
                      f"L={current_bar.low:.8f} C={current_bar.close:.8f} "
                      f"(ticks: {current_bar.num_ticks})")

        if i < 4:
            await asyncio.sleep(10)

    # Show completed bars
    print(f"\n{'='*60}")
    print("Completed OHLC Bars")
    print(f"{'='*60}")

    all_bars = collector.aggregator.get_all_bars()
    for mint, bars in all_bars.items():
        token_name = next((t.symbol for t in TOKENS if t.mint == mint), mint[:8])
        print(f"\n{token_name}:")
        for bar in bars[-5:]:  # Show last 5 bars
            print(f"  {bar.timestamp.strftime('%H:%M')}: "
                  f"O={bar.open:.8f} H={bar.high:.8f} "
                  f"L={bar.low:.8f} C={bar.close:.8f}")

    # Save to disk
    collector.save_to_disk()
    print(f"\nData saved to {data_config.data_path}/")

    # Show as DataFrame
    print(f"\n{'='*60}")
    print("As DataFrame")
    print(f"{'='*60}\n")

    for token in TOKENS:
        df = collector.get_ohlc_dataframe(token.mint)
        if not df.empty:
            print(f"{token.symbol}:")
            print(df.tail().to_string())
            print()


if __name__ == "__main__":
    asyncio.run(main())
