#!/usr/bin/env python3
"""Multi-asset trading script for Neural Hourly Trading V5.

Aggregates opportunities across crypto and stocks to pick the highest
expected return opportunity each hour.

Use PAPER=1 environment variable for paper trading (default).

Example:
    # Trade all assets (crypto + stocks)
    PAPER=1 python trade_hourlyv5_multi.py --daemon

    # Trade only crypto
    PAPER=1 python trade_hourlyv5_multi.py --asset-class crypto --daemon

    # Trade only stocks (during market hours)
    PAPER=1 python trade_hourlyv5_multi.py --asset-class stocks --daemon

    # Execute top 3 opportunities instead of just 1
    PAPER=1 python trade_hourlyv5_multi.py --top-n 3 --daemon
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import torch

import alpaca_wrapper
from alpaca_data_wrapper import append_recent_crypto_data
from alpaca_wrapper import _get_min_order_notional
from src.date_utils import is_nyse_trading_day_now
from src.exit_plan_tracker import get_exit_tracker
from src.hourly_aggregator import HourlyAggregator, SymbolOpportunity
from src.price_guard import enforce_gap, record_buy, record_sell
from src.process_utils import (
    spawn_close_position_at_maxdiff_takeprofit,
    spawn_open_position_at_maxdiff_takeprofit,
)

logger = logging.getLogger("hourlyv5_multi")


# Default checkpoint paths
DEFAULT_CRYPTO_CHECKPOINT = "neuralhourlytradingv5/checkpoints/best_hourlyv5_epoch049_20250629_010505.pt"
DEFAULT_STOCK_CHECKPOINT = "neuralhourlystocksv5/checkpoints/best_hourlyv5_epoch049.pt"


def get_checkpoint_path(path: str, default: str, name: str) -> Optional[str]:
    """Get checkpoint path, returning None if not found."""
    if path:
        if Path(path).exists():
            return path
        logger.warning(f"Checkpoint not found: {path}")
        return None

    if Path(default).exists():
        return default

    # Try to find best checkpoint in directory
    checkpoint_dir = Path(default).parent
    if checkpoint_dir.exists():
        best_checkpoints = list(checkpoint_dir.glob("best_*.pt"))
        if best_checkpoints:
            return str(sorted(best_checkpoints)[-1])

    logger.warning(f"No {name} checkpoint found")
    return None


def update_crypto_data(symbols: List[str]) -> None:
    """Update crypto data from Alpaca."""
    try:
        append_recent_crypto_data(symbols)
    except Exception as e:
        logger.warning(f"Failed to update crypto data: {e}")


def manage_existing_positions(
    aggregator: HourlyAggregator,
    dry_run: bool = False,
) -> None:
    """Check existing positions and spawn close watchers if needed.

    This ensures positions opened by the multi-asset trader get proper
    take-profit exits, even if the daemon restarts.
    Also clears stale exit plans for positions that were closed.
    """
    try:
        positions = alpaca_wrapper.get_all_positions()
    except Exception as e:
        logger.error(f"Failed to get positions: {e}")
        return

    # Get current position symbols
    position_symbols = {pos.symbol for pos in positions if float(pos.qty) > 0}

    # Clear exit plans for positions that no longer exist
    exit_tracker = get_exit_tracker()
    for plan in exit_tracker.get_all_plans():
        if plan.symbol not in position_symbols:
            logger.info(
                f"Position {plan.symbol} no longer exists (likely hit TP). "
                f"Clearing exit plan."
            )
            if not dry_run:
                exit_tracker.clear_exit_plan(plan.symbol)

    for pos in positions:
        symbol = pos.symbol
        position_qty = float(pos.qty)

        if position_qty <= 0:
            continue

        # Check if this symbol is one we trade
        is_crypto = symbol in aggregator.crypto_symbols
        is_stock = symbol in aggregator.stock_symbols

        if not (is_crypto or is_stock):
            continue

        # Get current plan for this symbol to determine target sell price
        try:
            opp = aggregator.get_opportunity_for_symbol(symbol)

            if opp is None or opp.position_length == 0:
                logger.debug(f"No active plan for {symbol}, skipping close watcher")
                continue

            target_sell_price = opp.sell_price
            position_length = opp.position_length

        except Exception as e:
            logger.warning(f"Failed to get plan for {symbol}: {e}")
            continue

        logger.info(
            f"{'[DRY RUN] Would spawn' if dry_run else 'Spawning'} close watcher for "
            f"{symbol}: qty={position_qty:.4f}, TP @ ${target_sell_price:.4f}"
        )

        if dry_run:
            continue

        try:
            # side="buy" because we originally BOUGHT (went long)
            # The CLI will compute exit_side="sell" to close the long
            spawn_close_position_at_maxdiff_takeprofit(
                symbol=symbol,
                side="buy",  # Entry side, not exit side!
                takeprofit_price=target_sell_price,
                target_qty=position_qty,
                expiry_minutes=position_length * 60,
                entry_strategy="hourlyv5_multi_close",
            )
            record_sell(symbol, target_sell_price)
        except Exception as e:
            logger.error(f"Error spawning close watcher for {symbol}: {e}")


def execute_opportunity(
    opp: SymbolOpportunity,
    dry_run: bool = False,
    total_capital: float = 0.0,
) -> bool:
    """Execute a trading opportunity.

    Args:
        opp: The opportunity to execute
        dry_run: If True, only log what would happen
        total_capital: Available cash for position sizing

    Returns:
        True if trade was executed (or would be in dry_run)
    """
    symbol = opp.symbol

    # Calculate position value
    position_value = total_capital * opp.position_size

    # Get minimum notional for this asset
    try:
        min_notional = _get_min_order_notional(symbol)
    except Exception:
        min_notional = 25.0  # Default for stocks

    if position_value < min_notional:
        logger.warning(
            f"Position value ${position_value:.2f} below minimum ${min_notional} for {symbol}"
        )
        return False

    # CRITICAL SAFETY: Validate buy_price < sell_price
    if opp.buy_price >= opp.sell_price:
        logger.error(
            f"CRITICAL: Inverted prices for {symbol}! "
            f"buy=${opp.buy_price:.4f} >= sell=${opp.sell_price:.4f}. "
            f"Trade BLOCKED."
        )
        return False

    # Enforce minimum spread (fee zone protection)
    if opp.asset_class == "crypto":
        min_spread_pct = 0.0016  # 16 bps (2x 8bps maker fee)
    else:
        min_spread_pct = 0.0004  # 4 bps (2x 2bps maker fee)

    actual_spread_pct = (opp.sell_price - opp.buy_price) / opp.buy_price
    if actual_spread_pct < min_spread_pct:
        logger.warning(
            f"Spread too small for {symbol}: {actual_spread_pct*100:.3f}% < {min_spread_pct*100:.3f}%. "
            f"Trade blocked."
        )
        return False

    # Get adjusted prices from price guard
    adj_buy, adj_sell = enforce_gap(symbol, opp.buy_price, opp.sell_price)

    if adj_buy >= adj_sell:
        logger.error(
            f"Price adjustment resulted in invalid spread for {symbol}: "
            f"buy=${adj_buy:.4f} >= sell=${adj_sell:.4f}"
        )
        return False

    # Calculate quantity
    target_qty = position_value / adj_buy

    logger.info(
        f"{'[DRY RUN] Would open' if dry_run else 'Opening'} {opp.asset_class} position: "
        f"{symbol} @ ${adj_buy:.4f}, qty={target_qty:.6f}, "
        f"TP @ ${adj_sell:.4f}, hold {opp.position_length}h, "
        f"exp_ret={opp.expected_return_pct:.4%}"
    )

    if dry_run:
        return True

    try:
        # Get exit tracker and set exit plan FIRST
        # This clears any stale exit plan and sets the new deadline
        exit_tracker = get_exit_tracker()
        exit_tracker.set_exit_plan(
            symbol=symbol,
            exit_price=adj_sell,
            position_length_hours=opp.position_length,
            entry_qty=target_qty,
            entry_strategy="hourlyv5_multi",
        )

        # Spawn order watcher for entry
        spawn_open_position_at_maxdiff_takeprofit(
            symbol=symbol,
            side="buy",
            limit_price=adj_buy,
            target_qty=target_qty,
            expiry_minutes=opp.position_length * 60,
            entry_strategy="hourlyv5_multi",
        )

        record_buy(symbol, adj_buy)

        # Also spawn the close watcher immediately
        # This ensures take-profit is set even if daemon restarts
        # side="buy" because we're BUYING to open (going long)
        # The CLI computes exit_side="sell" to close the long
        spawn_close_position_at_maxdiff_takeprofit(
            symbol=symbol,
            side="buy",  # Entry side, not exit side!
            takeprofit_price=adj_sell,
            target_qty=target_qty,
            expiry_minutes=opp.position_length * 60,
            entry_strategy="hourlyv5_multi_close",
        )

        record_sell(symbol, adj_sell)
        return True

    except Exception as e:
        logger.error(f"Error executing trade for {symbol}: {e}")
        return False


def run_once(
    aggregator: HourlyAggregator,
    top_n: int = 1,
    min_expected_return_bps: int = 10,
    asset_class: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """Run single trading iteration.

    Args:
        aggregator: HourlyAggregator instance
        top_n: Number of top opportunities to execute
        min_expected_return_bps: Minimum expected return in basis points
        asset_class: Filter to "crypto" or "stocks" only
        dry_run: If True, only log what would happen
    """
    now = datetime.now(timezone.utc)
    logger.info(f"Running multi-asset trading iteration at {now}")

    # Check for expired exit plans and trigger time-based exits
    exit_tracker = get_exit_tracker()
    backed_out = exit_tracker.check_and_execute_expired(dry_run=dry_run)
    if backed_out:
        logger.info(f"Triggered time-based exits for: {backed_out}")

    # First, manage any existing positions (ensure close watchers are set)
    manage_existing_positions(aggregator, dry_run=dry_run)

    # Update crypto data
    if asset_class is None or asset_class == "crypto":
        update_crypto_data(aggregator.crypto_symbols)

    # Get all opportunities
    opportunities = aggregator.get_all_opportunities()

    # Filter by asset class if specified
    if asset_class == "crypto":
        opportunities = [o for o in opportunities if o.asset_class == "crypto"]
    elif asset_class == "stocks":
        opportunities = [o for o in opportunities if o.asset_class == "stock"]

    logger.info(f"Found {len(opportunities)} opportunities (asset_class={asset_class or 'all'})")

    # Rank and filter
    ranked = aggregator.rank_by_expected_return(opportunities)

    # Filter by minimum expected return
    min_return = min_expected_return_bps / 10000
    filtered = [o for o in ranked if o.expected_return_pct >= min_return]

    logger.info(f"After filtering: {len(filtered)} opportunities >= {min_expected_return_bps}bps")

    if not filtered:
        logger.info("No opportunities meet minimum return threshold")
        return

    # Get account cash
    try:
        account = alpaca_wrapper.get_account()
        cash = float(account.cash)
    except Exception as e:
        logger.error(f"Failed to get account: {e}")
        return

    logger.info(f"Available cash: ${cash:.2f}")

    # Execute top N opportunities
    # Divide capital equally among top N
    capital_per_trade = cash / top_n

    executed = 0
    for i, opp in enumerate(filtered[:top_n]):
        logger.info(
            f"Opportunity #{i+1}: {opp.symbol} ({opp.asset_class}) "
            f"exp_ret={opp.expected_return_pct:.4%}, "
            f"risk_adj={opp.risk_adjusted_return:.4%}, "
            f"size={opp.position_size:.2%}"
        )

        if execute_opportunity(opp, dry_run=dry_run, total_capital=capital_per_trade):
            executed += 1

    logger.info(f"Executed {executed}/{min(top_n, len(filtered))} opportunities")


def run_daemon(
    aggregator: HourlyAggregator,
    top_n: int = 1,
    min_expected_return_bps: int = 10,
    asset_class: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """Run continuous trading loop aligned to UTC hours.

    Args:
        aggregator: HourlyAggregator instance
        top_n: Number of top opportunities to execute
        min_expected_return_bps: Minimum expected return in basis points
        asset_class: Filter to "crypto" or "stocks" only
        dry_run: If True, only log what would happen
    """
    logger.info(f"Starting multi-asset daemon (asset_class={asset_class or 'all'}, top_n={top_n})")

    while True:
        try:
            # Calculate time to next hour + 5 minutes (let data settle)
            now = datetime.now(timezone.utc)
            next_hour = (now + timedelta(hours=1)).replace(minute=5, second=0, microsecond=0)
            sleep_seconds = (next_hour - now).total_seconds()

            logger.info(f"Sleeping {sleep_seconds:.0f}s until {next_hour}")
            time.sleep(max(0, sleep_seconds))

            # Check market hours for stocks-only mode
            if asset_class == "stocks" and not is_nyse_trading_day_now():
                logger.info("Market closed, skipping stocks iteration")
                continue

            # Run trading iteration
            run_once(
                aggregator=aggregator,
                top_n=top_n,
                min_expected_return_bps=min_expected_return_bps,
                asset_class=asset_class,
                dry_run=dry_run,
            )

        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Error in daemon loop: {e}")
            time.sleep(60)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-asset Neural Hourly Trading V5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Trade all assets (crypto + stocks)
    PAPER=1 python trade_hourlyv5_multi.py --daemon

    # Trade only crypto
    PAPER=1 python trade_hourlyv5_multi.py --asset-class crypto --daemon

    # Trade only stocks (during market hours)
    PAPER=1 python trade_hourlyv5_multi.py --asset-class stocks --daemon

    # Execute top 3 opportunities
    PAPER=1 python trade_hourlyv5_multi.py --top-n 3 --daemon
        """,
    )
    parser.add_argument(
        "--crypto-checkpoint",
        type=str,
        default=None,
        help="Path to crypto model checkpoint",
    )
    parser.add_argument(
        "--stock-checkpoint",
        type=str,
        default=None,
        help="Path to stock model checkpoint",
    )
    parser.add_argument(
        "--asset-class",
        type=str,
        choices=["crypto", "stocks", "all"],
        default="all",
        help="Which asset class to trade (default: all)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=1,
        help="Number of top opportunities to execute (default: 1)",
    )
    parser.add_argument(
        "--min-return-bps",
        type=int,
        default=10,
        help="Minimum expected return in basis points (default: 10)",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run continuously, trading every hour",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't execute trades, just log what would happen",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Resolve asset class
    asset_class = None if args.asset_class == "all" else args.asset_class

    # Get checkpoint paths
    crypto_checkpoint = None
    stock_checkpoint = None

    if asset_class in (None, "crypto"):
        crypto_checkpoint = get_checkpoint_path(
            args.crypto_checkpoint,
            DEFAULT_CRYPTO_CHECKPOINT,
            "crypto"
        )

    if asset_class in (None, "stocks"):
        stock_checkpoint = get_checkpoint_path(
            args.stock_checkpoint,
            DEFAULT_STOCK_CHECKPOINT,
            "stock"
        )

    # Validate we have at least one model
    if crypto_checkpoint is None and stock_checkpoint is None:
        logger.error("No model checkpoints found. Please train models first.")
        sys.exit(1)

    logger.info(f"Crypto checkpoint: {crypto_checkpoint}")
    logger.info(f"Stock checkpoint: {stock_checkpoint}")
    logger.info(f"Asset class filter: {asset_class or 'all'}")
    logger.info(f"Top N opportunities: {args.top_n}")
    logger.info(f"Min expected return: {args.min_return_bps} bps")
    logger.info(f"Paper trading: {os.environ.get('PAPER', '1')}")

    # Create aggregator
    aggregator = HourlyAggregator(
        crypto_checkpoint=crypto_checkpoint,
        stock_checkpoint=stock_checkpoint,
        device=args.device,
    )

    if args.daemon:
        run_daemon(
            aggregator=aggregator,
            top_n=args.top_n,
            min_expected_return_bps=args.min_return_bps,
            asset_class=asset_class,
            dry_run=args.dry_run,
        )
    else:
        run_once(
            aggregator=aggregator,
            top_n=args.top_n,
            min_expected_return_bps=args.min_return_bps,
            asset_class=asset_class,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
