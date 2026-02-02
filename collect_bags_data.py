#!/usr/bin/env python3
"""
Collect price data from Bags.fm API.

Runs continuously, collecting prices every 10 minutes and
aggregating into OHLC bars for later forecasting.

Usage:
    # Run indefinitely
    python collect_bags_data.py

    # Run for 24 hours
    python collect_bags_data.py --duration 24

    # Faster collection (every 5 min)
    python collect_bags_data.py --interval 5
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bagsfm import DataCollector, BagsConfig, DataConfig, TokenConfig
from bagsfm.config import SOL_MINT, CODEX_MINT, BLON_MINT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Reduce httpx noise
logging.getLogger("httpx").setLevel(logging.WARNING)


# Tokens to collect data for
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
        name="CODEX (Bags.fm)",
    ),
    TokenConfig(
        symbol="ELIZA",
        mint="CZRsbB6BrHsAmGKeoxyfwzCyhttXvhfEukXCWnseBAGS",
        decimals=9,
        name="Eliza Town",
    ),
    TokenConfig(
        symbol="GAS",
        mint="7pskt3A1Zsjhngazam7vHWjWHnfgiRump916Xj7ABAGS",
        decimals=9,
        name="Gas Town",
    ),
    TokenConfig(
        symbol="VIBE",
        mint="W6n4FdEd7D6SetV5FxMMo8kNrduoj9qLWUu2v64BAGS",
        decimals=9,
        name="Vibecraft",
    ),
    TokenConfig(
        symbol="GROK",
        mint="5ghpHpgiew7WuuKY1f3CSMgGHmDiZUeMjZrgDrfzBAGS",
        decimals=9,
        name="GrokPrompt",
    ),
    TokenConfig(
        symbol="RO",
        mint="BFTPbgPoeLjcmH1QCUAZxCafT56AgwRKGmJvbCv6BAGS",
        decimals=9,
        name="Ralph Orchestrator",
    ),
    TokenConfig(
        symbol="ISONYC",
        mint="6rZfGFiHydm1RQfE1i7hhSdsZAE57bAK7ydQvA49BAGS",
        decimals=9,
        name="ISO NYC",
    ),
    TokenConfig(
        symbol="BLON",
        mint=BLON_MINT,
        decimals=9,
        name="BLON",
    ),
]


def on_prices_collected(prices):
    """Callback after each price collection."""
    if prices:
        logger.info(f"Collected {len(prices)} prices:")
        for p in prices:
            logger.info(f"  {p.token_symbol}: {p.price_sol:.8f} SOL")


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Collect Bags.fm price data")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration in hours (None for indefinite)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Collection interval in minutes [default: 10]",
    )
    args = parser.parse_args()

    # Create configuration
    bags_config = BagsConfig()
    data_config = DataConfig(
        tracked_tokens=TOKENS,
        collection_interval_minutes=args.interval,
        ohlc_interval_minutes=args.interval,
        context_bars=512,
    )

    # Create collector
    collector = DataCollector(bags_config, data_config)

    # Load recent history only (avoid full-file scan for very large CSVs)
    max_rows = max(2000, data_config.context_bars * len(TOKENS) * 4)
    num_prices, num_bars = collector.load_from_disk(
        max_rows=max_rows,
        required_mints=[t.mint for t in TOKENS],
    )
    logger.info(f"Loaded {num_bars} existing OHLC bars (tail {max_rows} rows)")

    # Show current state
    for token in TOKENS:
        bars = collector.get_context_bars(token.mint, n=5)
        if bars:
            logger.info(f"{token.symbol}: {len(collector.aggregator.get_bars(token.mint))} bars total")
            latest = bars[-1]
            logger.info(f"  Latest: {latest.timestamp} C={latest.close:.8f}")

    print(f"\n{'='*60}")
    print(f"Bags.fm Data Collection")
    print(f"{'='*60}")
    print(f"Tokens: {', '.join(t.symbol for t in TOKENS)}")
    print(f"Interval: {args.interval} minutes")
    print(f"Duration: {args.duration} hours" if args.duration else "Duration: Indefinite")
    print(f"Data dir: {data_config.data_path}")
    print(f"{'='*60}")
    print(f"Press Ctrl+C to stop\n")

    try:
        await collector.run_collection_loop(
            duration_hours=args.duration,
            callback=on_prices_collected,
        )
    except KeyboardInterrupt:
        logger.info("Stopping collection...")
    finally:
        collector.save_to_disk()
        await collector.aclose()
        logger.info("Data saved.")

        # Show final stats
        for token in TOKENS:
            bars = collector.aggregator.get_bars(token.mint)
            logger.info(f"{token.symbol}: {len(bars)} total bars collected")


if __name__ == "__main__":
    asyncio.run(main())
