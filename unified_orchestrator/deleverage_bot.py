#!/usr/bin/env python3
"""End-of-day deleverage bot.

Checks current stock leverage and sells positions to bring leverage
to 2x max before market close. Uses limit orders slightly below
market price (not market orders) to avoid slippage.

Usage:
  # Dry run (check what would happen)
  python -m unified_orchestrator.deleverage_bot

  # Live execution
  python -m unified_orchestrator.deleverage_bot --live

  # One-shot (run once then exit)
  python -m unified_orchestrator.deleverage_bot --live --once

  # Continuous: checks every 5 min during final 60 min before close
  python -m unified_orchestrator.deleverage_bot --live
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from loguru import logger

from unified_orchestrator.state import build_snapshot
from unified_orchestrator.orchestrator import (
    CRYPTO_SYMBOLS,
    MAX_OVERNIGHT_LEVERAGE,
    MARGIN_INTEREST_ANNUAL,
    deleverage_to_target,
)


def _minutes_to_close() -> int | None:
    """Minutes until 16:00 ET market close. None if market closed."""
    now_et = datetime.now(ZoneInfo("America/New_York"))
    # Weekends
    if now_et.weekday() >= 5:
        return None
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    if now_et < market_open or now_et >= market_close:
        return None
    return int((market_close - now_et).total_seconds() / 60)


def _get_current_leverage():
    """Get current stock leverage from Alpaca."""
    now = datetime.now(timezone.utc)
    snapshot = build_snapshot(now)
    equity = max(snapshot.total_stock_value, 1.0)
    crypto_set = set(CRYPTO_SYMBOLS)
    stock_positions = {
        sym: pos for sym, pos in snapshot.alpaca_positions.items()
        if sym not in crypto_set and pos.qty > 0
    }
    long_val = sum(p.market_value for p in stock_positions.values())
    leverage = long_val / equity
    margin_cost_day = max(0, long_val - equity) * MARGIN_INTEREST_ANNUAL / 365
    return snapshot, leverage, long_val, equity, margin_cost_day, stock_positions


def run_deleverage(dry_run: bool = True, target: float = MAX_OVERNIGHT_LEVERAGE):
    """Check leverage and deleverage if needed."""
    snapshot, lev, long_val, equity, margin_cost, positions = _get_current_leverage()

    logger.info(f"Stock leverage: {lev:.2f}x | Long: ${long_val:,.0f} | "
                f"Equity: ${equity:,.0f} | Margin cost: ${margin_cost:.2f}/day")
    for sym, pos in positions.items():
        pnl_pct = (pos.current_price - pos.avg_price) / pos.avg_price * 100 if pos.avg_price > 0 else 0
        logger.info(f"  {sym}: {pos.qty:.2f} shares @ ${pos.current_price:.2f} "
                    f"(${pos.market_value:,.0f}, {pnl_pct:+.1f}%)")

    if lev <= target:
        logger.info(f"Leverage {lev:.2f}x <= {target:.1f}x target — no action needed")
        return []

    logger.info(f"DELEVERAGING: {lev:.2f}x → {target:.1f}x")
    orders = deleverage_to_target(
        snapshot,
        target_leverage=target,
        dry_run=dry_run,
        use_limit=True,
        limit_offset_pct=0.05,  # 0.05% below current — tight limit, fills fast
    )
    return orders


def main():
    parser = argparse.ArgumentParser(description="End-of-day deleverage bot")
    parser.add_argument("--live", action="store_true", help="Execute trades (default: dry run)")
    parser.add_argument("--once", action="store_true", help="Run once then exit")
    parser.add_argument("--target", type=float, default=MAX_OVERNIGHT_LEVERAGE,
                        help=f"Target leverage (default: {MAX_OVERNIGHT_LEVERAGE}x)")
    parser.add_argument("--check-interval", type=int, default=300,
                        help="Seconds between checks (default: 300 = 5 min)")
    parser.add_argument("--deleverage-window", type=int, default=60,
                        help="Minutes before close to start deleveraging (default: 60)")
    args = parser.parse_args()

    dry_run = not args.live
    logger.info(f"Deleverage bot started ({'DRY RUN' if dry_run else 'LIVE'})")
    logger.info(f"Target: {args.target:.1f}x | Window: {args.deleverage_window} min before close")

    while True:
        mtc = _minutes_to_close()
        if mtc is None:
            if args.once:
                logger.info("Market closed — exiting")
                break
            logger.info("Market closed — sleeping 60s")
            time.sleep(60)
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"Minutes to close: {mtc}")

        if mtc <= args.deleverage_window:
            orders = run_deleverage(dry_run=dry_run, target=args.target)
            if orders:
                logger.info(f"Placed {len(orders)} deleverage orders")
                # After placing orders, wait a bit for fills then re-check
                if not dry_run and not args.once:
                    logger.info("Waiting 30s for fills before re-check...")
                    time.sleep(30)
                    # Re-check leverage after fills
                    _, new_lev, _, _, _, _ = _get_current_leverage()
                    logger.info(f"Post-fill leverage: {new_lev:.2f}x")
                    if new_lev > args.target:
                        logger.warning(f"Still above target ({new_lev:.2f}x > {args.target:.1f}x) "
                                       f"— will retry next check")
        else:
            logger.info(f"Not in deleverage window yet (>{args.deleverage_window} min to close)")

        if args.once:
            break

        logger.info(f"Sleeping {args.check_interval}s...")
        time.sleep(args.check_interval)


if __name__ == "__main__":
    main()
