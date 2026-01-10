#!/usr/bin/env python3
"""Close all positions to prepare for chronos_full strategy.

Usage:
    PAPER=1 python close_all_positions.py --crypto-only  # Close crypto now
    PAPER=1 python close_all_positions.py                 # Close all (market hours only for stocks)
"""

import argparse
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def is_paper_mode() -> bool:
    return os.environ.get("PAPER", "1").lower() in ("1", "true", "yes")


def close_all_positions(crypto_only: bool = False, dry_run: bool = False):
    """Close all positions."""
    import alpaca_wrapper
    from src.trading_obj_utils import filter_to_realistic_positions

    positions = filter_to_realistic_positions(alpaca_wrapper.get_all_positions())

    logger.info("=" * 60)
    logger.info("CLOSING POSITIONS")
    logger.info("=" * 60)
    logger.info("Mode: %s", "PAPER" if is_paper_mode() else "LIVE")
    logger.info("Crypto Only: %s", crypto_only)
    logger.info("Dry Run: %s", dry_run)
    logger.info("=" * 60)

    closed = 0
    failed = 0
    skipped = 0

    for p in positions:
        symbol = getattr(p, "symbol", "")
        qty = float(getattr(p, "qty", 0))
        side = getattr(p, "side", "").lower()
        mkt_val = float(getattr(p, "market_value", 0))

        is_crypto = symbol.endswith("USD")

        if crypto_only and not is_crypto:
            logger.info("SKIP (stock): %s", symbol)
            skipped += 1
            continue

        close_side = "sell" if "long" in side else "buy"
        close_qty = abs(qty)

        logger.info(
            "CLOSING: %s %s %.4f (value: $%.2f)",
            close_side.upper(), symbol, close_qty, mkt_val
        )

        if dry_run:
            logger.info("  [DRY RUN] Would close")
            closed += 1
            continue

        try:
            if is_crypto:
                # Use limit order at market for crypto
                quote = alpaca_wrapper.latest_data(symbol)
                if quote:
                    if close_side == "sell":
                        price = float(getattr(quote, "bid_price", 0) or 0)
                    else:
                        price = float(getattr(quote, "ask_price", 0) or 0)

                    if price > 0:
                        result = alpaca_wrapper.open_order_at_price(
                            symbol=symbol,
                            qty=close_qty,
                            side=close_side,
                            price=price,
                        )
                        if result:
                            logger.info("  SUCCESS: Order placed at $%.4f", price)
                            closed += 1
                        else:
                            logger.error("  FAILED: Order returned None")
                            failed += 1
                    else:
                        logger.error("  FAILED: No valid price")
                        failed += 1
                else:
                    logger.error("  FAILED: No quote data")
                    failed += 1
            else:
                # For stocks, try market order (only works during market hours)
                result = alpaca_wrapper.open_market_order_violently(
                    symbol=symbol,
                    qty=close_qty,
                    side=close_side,
                    retries=2,
                )
                if result:
                    logger.info("  SUCCESS: Market order placed")
                    closed += 1
                else:
                    logger.warning("  FAILED: Market closed - will retry at open")
                    failed += 1

            time.sleep(0.5)

        except Exception as e:
            logger.error("  ERROR: %s", e)
            failed += 1

    logger.info("=" * 60)
    logger.info("SUMMARY: Closed=%d, Failed=%d, Skipped=%d", closed, failed, skipped)
    logger.info("=" * 60)

    return closed, failed, skipped


def main():
    parser = argparse.ArgumentParser(description="Close all positions")
    parser.add_argument(
        "--crypto-only",
        action="store_true",
        help="Only close crypto positions (can be done anytime)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be closed without actually closing",
    )
    args = parser.parse_args()

    if not is_paper_mode():
        logger.error("PAPER mode not set! Set PAPER=1 to run.")
        sys.exit(1)

    close_all_positions(crypto_only=args.crypto_only, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
