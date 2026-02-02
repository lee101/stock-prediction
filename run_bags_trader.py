#!/usr/bin/env python3
"""
Run the Bags.fm Solana trading bot.

Usage:
    # Dry run mode (default) - no actual trades
    python run_bags_trader.py

    # Live trading mode
    python run_bags_trader.py --live

    # Run for 24 hours
    python run_bags_trader.py --duration 24

    # Custom interval (5 minutes)
    python run_bags_trader.py --interval 5

Environment variables:
    BAGS_API_KEY - Bags.fm API key
    SOLANA_PRIVATE_KEY - Base58 encoded private key for trading wallet
    SOLANA_RPC_URL - Solana RPC URL (optional, defaults to mainnet)
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bagsfm import (
    BagsTrader,
    TradingConfig,
    BagsConfig,
    DataConfig,
    ForecastConfig,
    TokenConfig,
)
from bagsfm.config import (
    SOL_MINT,
    USDC_MINT,
    CODEX_MINT,
    BLON_MINT,
    BAGS_TOTAL_FEE_BPS,
    BAGS_CREATOR_FEE_BPS,
)


# Example tokens to trade (add your own here)
# Note: Only include tokens that have historical OHLC data in bagstraining/ohlc_data.csv
EXAMPLE_TOKENS = {
    "SOL": TokenConfig(
        symbol="SOL",
        mint=SOL_MINT,
        decimals=9,
        name="Solana",
        min_trade_amount=0.01,
    ),
    # Add CODEX (Bags.fm token) - has historical data
    "CODEX": TokenConfig(
        symbol="CODEX",
        mint=CODEX_MINT,
        decimals=9,
        name="CODEX",
        min_trade_amount=1.0,
        entry_fee_bps=BAGS_TOTAL_FEE_BPS,
        exit_fee_bps=BAGS_TOTAL_FEE_BPS,
        creator_rebate_bps=BAGS_CREATOR_FEE_BPS,
    ),
    "BLON": TokenConfig(
        symbol="BLON",
        mint=BLON_MINT,
        decimals=9,
        name="BLON",
        min_trade_amount=1.0,
        entry_fee_bps=BAGS_TOTAL_FEE_BPS,
        exit_fee_bps=BAGS_TOTAL_FEE_BPS,
    ),
    # USDC removed - no historical OHLC data available
    # Uncomment after collecting USDC price data:
    # "USDC": TokenConfig(
    #     symbol="USDC",
    #     mint=USDC_MINT,
    #     decimals=6,
    #     name="USD Coin",
    #     min_trade_amount=1.0,
    # ),
    # Add more tokens as needed
    # "BONK": TokenConfig(
    #     symbol="BONK",
    #     mint="DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
    #     decimals=5,
    #     name="Bonk",
    #     min_trade_amount=1000000.0,
    # ),
}

def _select_tokens(trade_mints: str | None, codex_only: bool) -> list[TokenConfig]:
    if codex_only:
        return [EXAMPLE_TOKENS["CODEX"]]

    if trade_mints:
        mint_set = {mint.strip() for mint in trade_mints.split(",") if mint.strip()}
        selected = [token for token in EXAMPLE_TOKENS.values() if token.mint in mint_set]
        if not selected:
            raise ValueError("No matching tokens found for trade_mints")
        return selected

    return list(EXAMPLE_TOKENS.values())


def create_config(
    dry_run: bool = True,
    interval_minutes: int = 10,
    tokens: list[TokenConfig] | None = None,
    strategy: str = "forecast",
    daily_range_min_bps: float = 100.0,
    daily_range_max_actions: int = 2,
    daily_range_use_previous_day: bool = True,
    max_daily_trades: int = 20,
) -> TradingConfig:
    """Create trading configuration.

    Args:
        dry_run: If True, don't execute actual trades
        interval_minutes: How often to check for trades

    Returns:
        TradingConfig instance
    """
    # Bags API config
    bags_config = BagsConfig()

    # Data collection config
    tracked_tokens = tokens or list(EXAMPLE_TOKENS.values())
    data_config = DataConfig(
        tracked_tokens=tracked_tokens,
        collection_interval_minutes=interval_minutes,
        ohlc_interval_minutes=interval_minutes,
        context_bars=256,  # Use 256 bars of history for forecasting
    )

    # Forecasting config
    forecast_config = ForecastConfig(
        prediction_length=6,  # Predict 6 bars ahead
        context_length=256,
    )

    # Main trading config
    config = TradingConfig(
        bags=bags_config,
        data=data_config,
        forecast=forecast_config,
        strategy="daily-range" if strategy == "daily-range" else "forecast",
        check_interval_minutes=interval_minutes,
        min_predicted_return=0.005,  # 0.5% min predicted return
        min_confidence=0.3,
        daily_range_min_bps=daily_range_min_bps,
        daily_range_max_actions_per_day=daily_range_max_actions,
        daily_range_use_previous_day=daily_range_use_previous_day,
        position_size_pct=0.25,  # 25% of portfolio per position
        max_positions=4,
        max_position_sol=1.0,
        slippage_bps=100,  # 1% slippage tolerance
        max_daily_loss_pct=0.05,  # 5% daily loss limit
        max_daily_trades=max_daily_trades,
        dry_run=dry_run,
    )

    return config


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Bags.fm Solana Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Run in dry-run mode (no actual trades) [default]",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in live mode (execute actual trades)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration to run in hours (None for indefinite)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Check interval in minutes [default: 10]",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level [default: INFO]",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Only collect price data, don't trade",
    )
    parser.add_argument(
        "--trade-mints",
        type=str,
        default=None,
        help="Comma-separated list of token mints to trade",
    )
    parser.add_argument(
        "--codex-only",
        action="store_true",
        help="Trade only CODEX",
    )
    parser.add_argument(
        "--strategy",
        choices=["forecast", "daily-range"],
        default="forecast",
        help="Trading strategy [default: forecast]",
    )
    parser.add_argument(
        "--daily-range-min-bps",
        type=float,
        default=100.0,
        help="Min prior-day range in bps for daily-range [default: 100]",
    )
    parser.add_argument(
        "--daily-range-max-actions",
        type=int,
        default=2,
        help="Max actions per token per day for daily-range [default: 2]",
    )
    parser.add_argument(
        "--daily-range-use-previous-day",
        action="store_true",
        default=True,
        help="Use prior-day levels for daily-range [default]",
    )
    parser.add_argument(
        "--daily-range-use-same-day",
        action="store_true",
        help="Use same-day levels for daily-range (override previous-day)",
    )
    parser.add_argument(
        "--max-daily-trades",
        type=int,
        default=20,
        help="Max trades per day [default: 20]",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Reduce noise from httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Create config
    dry_run = not args.live
    tokens = _select_tokens(args.trade_mints, args.codex_only)
    use_previous = args.daily_range_use_previous_day and not args.daily_range_use_same_day
    config = create_config(
        dry_run=dry_run,
        interval_minutes=args.interval,
        tokens=tokens,
        strategy=args.strategy,
        daily_range_min_bps=args.daily_range_min_bps,
        daily_range_max_actions=args.daily_range_max_actions,
        daily_range_use_previous_day=use_previous,
        max_daily_trades=args.max_daily_trades,
    )

    print(f"\n{'='*60}")
    print(f"Bags.fm Trading Bot")
    print(f"{'='*60}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE TRADING'}")
    print(f"Check interval: {args.interval} minutes")
    print(f"Duration: {args.duration} hours" if args.duration else "Duration: Indefinite")
    print(f"Tokens: {', '.join(t.symbol for t in tokens)}")
    print(f"{'='*60}\n")

    if not dry_run:
        print("\n⚠️  WARNING: LIVE TRADING MODE")
        print("Real trades will be executed!")
        print("Press Ctrl+C within 10 seconds to cancel...\n")
        try:
            await asyncio.sleep(10)
        except KeyboardInterrupt:
            print("\nCancelled.")
            return

    if args.collect_only:
        # Just collect data
        from bagsfm import DataCollector, create_collector

        max_rows = max(2000, config.data.context_bars * len(tokens) * 4)
        collector = await create_collector(
            bags_config=config.bags,
            data_config=config.data,
            max_rows=max_rows,
            required_mints=[token.mint for token in tokens],
        )

        print("Collecting price data...")
        await collector.run_collection_loop(
            duration_hours=args.duration,
            callback=lambda prices: print(f"Collected {len(prices)} prices"),
        )
    else:
        # Run full trader
        trader = BagsTrader(config)

        try:
            await trader.run(duration_hours=args.duration)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            trader.shutdown()

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
