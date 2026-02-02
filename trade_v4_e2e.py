#!/usr/bin/env python3
"""V4/V6 End-to-end trading inference script.

Uses the V4 neural daily model to generate trading plans and executes them
via Alpaca using watchers for entries and exits.

The V4 model outputs:
- buy_price: Entry limit price (spawns entry watcher)
- sell_price: Take profit target (spawns exit watcher)
- exit_days: Holding period (spawns time expiry watcher)
- trade_amount: Position size fraction

Watcher architecture:
- Entry watcher: Polls for price to hit buy_price, then executes entry
- Take-profit watcher: Polls for price to hit sell_price, then exits
- Time-expiry watcher: Scheduled task to backout_near_market if position still open

Usage:
    python trade_v4_e2e.py --checkpoint neuraldailyv4/checkpoints/v10_extended/epoch_0020.pt
    python trade_v4_e2e.py --checkpoint ... --dry-run  # Preview without trading

Best checkpoints (90-day backtest Dec 2025):
- V10 epoch 20: 90.30% return, 75.8% win rate, 0.660 Sharpe (recommended)
- V8 epoch 35: 88.87% return, 75.8% win rate, 0.657 Sharpe
- V9 epoch 10: 72.19% return, 56.8% win rate, 0.406 Sortino
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import sleep
from typing import Dict, List, Optional, Set

import pandas as pd
import pytz
from loguru import logger

# Local imports
from alpaca_wrapper import get_account, get_all_positions, close_position_violently
from alpaca.data import StockHistoricalDataClient
from data_curate_daily import download_exchange_latest_data, get_ask, get_bid
from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD
from neuraldailyv4.runtime import DailyTradingRuntimeV4, TradingPlanV4
from src.date_utils import is_nyse_trading_day_now, is_nyse_trading_day_ending
from src.process_utils import (
    backout_near_market,
    ramp_into_position,
    spawn_open_position_at_maxdiff_takeprofit,
    spawn_close_position_at_maxdiff_takeprofit,
    stop_all_entry_watchers,
)
from src.symbol_utils import is_crypto_symbol
from stock.state import get_state_dir, resolve_state_suffix, get_paper_suffix

# Constants
# Best performers from V8/V9 90-day backtest (Dec 2025):
# - AMD: 43% return, 75% TP rate - best single symbol
# - LRCX: 22% return, 60% TP rate - semiconductors
# - AMAT: 14% return, 67% TP rate - semiconductors
# - ETFs (SPY, QQQ, IWM): consistent, low volatility
DEFAULT_SYMBOLS = [
    # ETFs - consistently profitable
    "SPY", "QQQ", "IWM", "XLF", "XLK", "DIA",
    # Mega cap tech
    "AAPL", "GOOGL", "NVDA", "TSLA",
    # Financials
    "V", "MA",
    # Semiconductors - highest returns in backtest
    "AMD", "LRCX", "AMAT", "QCOM",
    # Consumer
    "MCD", "TGT",
    # Healthcare
    "LLY", "PFE",
    # Software
    "ADBE",
    # Note: MRVL, MSFT, META, AMZN, etc. excluded in runtime.DEFAULT_NON_TRADABLE
]

# Backout timing parameters
BACKOUT_START_OFFSET_MINUTES = 30
BACKOUT_SLEEP_SECONDS = 60
BACKOUT_MARKET_CLOSE_BUFFER_MINUTES = 5
BACKOUT_MARKET_CLOSE_FORCE_MINUTES = 2

# Entry watcher parameters
ENTRY_WATCHER_POLL_SECONDS = max(5, int(os.getenv("V4_ENTRY_WATCHER_POLL_SECONDS", "45")))
ENTRY_WATCHER_TOLERANCE_PCT = 0.002  # 0.2% tolerance for entry
EXIT_WATCHER_POLL_SECONDS = max(5, int(os.getenv("V4_EXIT_WATCHER_POLL_SECONDS", "45")))
EXIT_WATCHER_PRICE_TOLERANCE = 0.001  # 0.1% tolerance for take-profit

# Position management
MAX_POSITIONS = 10  # Increased for broader diversification
MIN_TRADE_AMOUNT = 0.05  # Minimum 5% position size to consider
MAX_TRADE_AMOUNT = 0.30  # Maximum 30% position size (now in runtime)

# State management
STATE_SUFFIX = resolve_state_suffix()
PAPER_SUFFIX = get_paper_suffix()
V4_POSITIONS_FILE = get_state_dir() / f"v4_active_positions{PAPER_SUFFIX}{STATE_SUFFIX or ''}.json"
V4_TIME_EXPIRY_DIR = get_state_dir() / f"v4_time_expiry{PAPER_SUFFIX}{STATE_SUFFIX or ''}"
V4_TIME_EXPIRY_DIR.mkdir(parents=True, exist_ok=True)

# Global shutdown flag
_shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global _shutdown_requested
    logger.warning(f"Received signal {signum}, requesting shutdown...")
    _shutdown_requested = True


class ActivePosition:
    """Track an active position opened by V4 model."""

    def __init__(
        self,
        symbol: str,
        entry_price: float,
        sell_price: float,
        exit_timestamp: pd.Timestamp,
        entry_timestamp: pd.Timestamp,
        quantity: float,
        entry_watcher_pid: Optional[int] = None,
        tp_watcher_started: bool = False,
        time_expiry_scheduled: bool = False,
        pre_deadline_exit_started: bool = False,
    ):
        self.symbol = symbol
        self.entry_price = entry_price
        self.sell_price = sell_price
        self.exit_timestamp = exit_timestamp
        self.entry_timestamp = entry_timestamp
        self.quantity = quantity
        self.entry_watcher_pid = entry_watcher_pid
        self.tp_watcher_started = tp_watcher_started
        self.time_expiry_scheduled = time_expiry_scheduled
        self.pre_deadline_exit_started = pre_deadline_exit_started

    def should_force_exit(self, now: pd.Timestamp) -> bool:
        """Check if position has exceeded its hold time."""
        return now >= self.exit_timestamp

    def to_dict(self) -> Dict:
        """Serialize to dictionary for persistence."""
        return {
            "symbol": self.symbol,
            "entry_price": self.entry_price,
            "sell_price": self.sell_price,
            "exit_timestamp": self.exit_timestamp.isoformat(),
            "entry_timestamp": self.entry_timestamp.isoformat(),
            "quantity": self.quantity,
            "entry_watcher_pid": self.entry_watcher_pid,
            "tp_watcher_started": self.tp_watcher_started,
            "time_expiry_scheduled": self.time_expiry_scheduled,
            "pre_deadline_exit_started": self.pre_deadline_exit_started,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ActivePosition":
        """Deserialize from dictionary."""
        return cls(
            symbol=data["symbol"],
            entry_price=data["entry_price"],
            sell_price=data["sell_price"],
            exit_timestamp=pd.Timestamp(data["exit_timestamp"]),
            entry_timestamp=pd.Timestamp(data["entry_timestamp"]),
            quantity=data["quantity"],
            entry_watcher_pid=data.get("entry_watcher_pid"),
            tp_watcher_started=data.get("tp_watcher_started", False),
            time_expiry_scheduled=data.get("time_expiry_scheduled", False),
            pre_deadline_exit_started=data.get("pre_deadline_exit_started", False),
        )

    def __repr__(self) -> str:
        return (
            f"ActivePosition({self.symbol}, entry={self.entry_price:.2f}, "
            f"tp={self.sell_price:.2f}, exit_by={self.exit_timestamp})"
        )


class V4TradingManager:
    """Manages V4 model trading operations with watcher-based execution."""

    def __init__(
        self,
        runtime: DailyTradingRuntimeV4,
        symbols: List[str],
        dry_run: bool = False,
        max_positions: int = MAX_POSITIONS,
    ):
        self.runtime = runtime
        self.symbols = symbols
        self.dry_run = dry_run
        self.max_positions = max_positions

        # Track active positions opened by V4
        self.active_positions: Dict[str, ActivePosition] = {}

        # Track symbols we've attempted entry on today
        self.attempted_entries_today: Set[str] = set()
        self.last_entry_date: Optional[datetime] = None

        # Load persisted state
        self._load_state()

    def _load_state(self):
        """Load persisted position state from disk."""
        if not V4_POSITIONS_FILE.exists():
            return

        try:
            with open(V4_POSITIONS_FILE, "r") as f:
                data = json.load(f)

            for symbol, pos_data in data.get("positions", {}).items():
                try:
                    self.active_positions[symbol] = ActivePosition.from_dict(pos_data)
                    logger.info(f"Restored position: {self.active_positions[symbol]}")
                except Exception as e:
                    logger.warning(f"Failed to restore position {symbol}: {e}")

            logger.info(f"Loaded {len(self.active_positions)} positions from state")
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")

    def _save_state(self):
        """Persist position state to disk."""
        try:
            data = {
                "positions": {sym: pos.to_dict() for sym, pos in self.active_positions.items()},
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            temp_path = V4_POSITIONS_FILE.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=2)
            temp_path.replace(V4_POSITIONS_FILE)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def reset_daily_state(self):
        """Reset daily tracking state."""
        self.attempted_entries_today.clear()
        self.last_entry_date = datetime.now(pytz.UTC).date()
        logger.info("Reset daily entry tracking")

    def get_trading_plans(self) -> List[TradingPlanV4]:
        """Generate trading plans for all symbols."""
        try:
            plans = self.runtime.plan_batch(self.symbols)
            logger.info(f"Generated {len(plans)} trading plans")
            return plans
        except Exception as e:
            logger.error(f"Failed to generate trading plans: {e}")
            return []

    def filter_plans(self, plans: List[TradingPlanV4]) -> List[TradingPlanV4]:
        """Filter plans based on constraints."""
        filtered = []

        for plan in plans:
            # Skip if already attempted today
            if plan.symbol in self.attempted_entries_today:
                logger.debug(f"Skipping {plan.symbol}: already attempted today")
                continue

            # Skip if we already have a position
            if plan.symbol in self.active_positions:
                logger.debug(f"Skipping {plan.symbol}: already have active position")
                continue

            # Check trade amount bounds (max is enforced by runtime's max_position_override)
            if plan.trade_amount < MIN_TRADE_AMOUNT:
                logger.debug(f"Skipping {plan.symbol}: trade_amount {plan.trade_amount:.2%} < {MIN_TRADE_AMOUNT:.2%}")
                continue

            # Model already enforces min_price_gap_pct (0.5%) and naturally outputs 2%+ spreads
            # No additional filter needed - trust the trained model
            filtered.append(plan)

        return filtered

    def calculate_quantity(self, plan: TradingPlanV4) -> Optional[float]:
        """Calculate the number of shares to buy based on account equity."""
        try:
            account = get_account()
            equity = float(account.equity)

            # Position value = equity * trade_amount
            position_value = equity * plan.trade_amount

            # Quantity = position_value / entry_price
            qty = position_value / plan.buy_price

            # Round to appropriate precision
            if plan.buy_price > 100:
                qty = round(qty, 2)
            else:
                qty = round(qty, 4)

            if qty <= 0:
                return None

            logger.info(
                f"{plan.symbol}: equity=${equity:,.0f}, trade_amount={plan.trade_amount:.2%}, "
                f"position=${position_value:,.0f}, qty={qty:.4f}"
            )
            return qty

        except Exception as e:
            logger.error(f"Failed to calculate quantity for {plan.symbol}: {e}")
            return None

    def execute_entry(self, plan: TradingPlanV4) -> bool:
        """Spawn entry watcher for a trading plan.

        Instead of immediately entering, we spawn a watcher that:
        1. Polls the market for the price to approach buy_price
        2. Executes entry when price is within tolerance
        3. After fill, spawns take-profit watcher
        """
        self.attempted_entries_today.add(plan.symbol)

        qty = self.calculate_quantity(plan)
        if qty is None:
            logger.warning(f"Could not calculate quantity for {plan.symbol}")
            return False

        logger.info(
            f"ENTRY PLAN: {plan.symbol} - Buy at ${plan.buy_price:.2f}, "
            f"TP at ${plan.sell_price:.2f}, Exit by {plan.exit_timestamp}, "
            f"Qty={qty:.4f}, Confidence={plan.confidence:.2f}"
        )

        if self.dry_run:
            logger.info(f"[DRY RUN] Would spawn entry watcher for {plan.symbol}")
            return False

        try:
            # Calculate expiry for entry watcher (until exit_timestamp)
            now = datetime.now(timezone.utc)
            exit_dt = plan.exit_timestamp.to_pydatetime()
            if exit_dt.tzinfo is None:
                exit_dt = exit_dt.replace(tzinfo=timezone.utc)
            expiry_minutes = max(60, int((exit_dt - now).total_seconds() / 60))

            # First, clean up any existing entry watchers for this symbol
            stop_all_entry_watchers(plan.symbol, reason="v4_new_plan")

            # Spawn entry watcher to buy at limit price
            logger.info(
                f"Spawning entry watcher for {plan.symbol}: "
                f"buy at ${plan.buy_price:.2f}, qty={qty:.4f}, expiry={expiry_minutes}min"
            )

            spawn_open_position_at_maxdiff_takeprofit(
                plan.symbol,
                "buy",
                float(plan.buy_price),
                float(qty),
                tolerance_pct=ENTRY_WATCHER_TOLERANCE_PCT,
                expiry_minutes=expiry_minutes,
                poll_seconds=ENTRY_WATCHER_POLL_SECONDS,
                entry_strategy="neuraldailyv4",
                force_immediate=False,  # Wait for price to approach limit
            )

            # Track the pending position
            self.active_positions[plan.symbol] = ActivePosition(
                symbol=plan.symbol,
                entry_price=plan.buy_price,
                sell_price=plan.sell_price,
                exit_timestamp=plan.exit_timestamp,
                entry_timestamp=pd.Timestamp.now(tz="UTC"),
                quantity=qty,
                entry_watcher_pid=None,  # Watcher PID not tracked here
                tp_watcher_started=False,
                time_expiry_scheduled=False,
            )

            # Persist state
            self._save_state()

            logger.info(f"Entry watcher spawned for {plan.symbol}")
            return True

        except Exception as e:
            logger.error(f"Failed to spawn entry watcher for {plan.symbol}: {e}")
            return False

    def manage_exits(self):
        """Check and manage exits for active positions.

        For each position:
        1. If not yet filled, check if broker has position
        2. If filled and no TP watcher, spawn take-profit watcher
        3. If exit_timestamp passed, trigger forced exit
        """
        now = pd.Timestamp.now(tz="UTC")

        # Get actual broker positions
        try:
            broker_positions = get_all_positions()
            broker_positions_by_symbol = {p.symbol: p for p in broker_positions}
        except Exception as e:
            logger.error(f"Failed to get broker positions: {e}")
            return

        positions_to_remove = []
        state_changed = False

        for symbol, position in self.active_positions.items():
            try:
                broker_pos = broker_positions_by_symbol.get(symbol)
                has_broker_position = broker_pos is not None and float(broker_pos.qty) > 0

                # If we have a broker position but haven't spawned TP watcher yet
                if has_broker_position and not position.tp_watcher_started:
                    logger.info(
                        f"Position filled for {symbol} - spawning take-profit watcher at ${position.sell_price:.2f}"
                    )
                    if not self.dry_run:
                        spawn_close_position_at_maxdiff_takeprofit(
                            symbol,
                            "sell",  # Sell to exit long position
                            float(position.sell_price),
                            poll_seconds=EXIT_WATCHER_POLL_SECONDS,
                            price_tolerance=EXIT_WATCHER_PRICE_TOLERANCE,
                            entry_strategy="neuraldailyv4",
                        )
                    position.tp_watcher_started = True
                    state_changed = True

                # Pre-deadline check: Start graceful exit 1 hour before deadline
                # This gives time for the position to exit at a good price before forced close
                time_to_deadline = (position.exit_timestamp - now).total_seconds()
                pre_deadline_window = 60 * 60  # 1 hour before deadline
                if 0 < time_to_deadline <= pre_deadline_window:
                    if has_broker_position and not getattr(position, 'pre_deadline_exit_started', False):
                        logger.info(
                            f"PRE-DEADLINE EXIT: {symbol} - Starting graceful exit "
                            f"({time_to_deadline/60:.0f} min until deadline)"
                        )
                        if not self.dry_run:
                            # Start backout process with no delay for pre-deadline
                            backout_near_market(
                                symbol,
                                start_offset_minutes=0,
                                sleep_seconds=60,
                                market_close_buffer_minutes=BACKOUT_MARKET_CLOSE_BUFFER_MINUTES,
                                market_close_force_minutes=BACKOUT_MARKET_CLOSE_FORCE_MINUTES,
                            )
                        position.pre_deadline_exit_started = True
                        state_changed = True

                # Check if we should force exit (time expired)
                if position.should_force_exit(now):
                    if has_broker_position:
                        logger.info(
                            f"FORCED EXIT: {symbol} - Hold time exceeded (exit_timestamp={position.exit_timestamp})"
                        )
                        if not self.dry_run:
                            # Use immediate close for deadline exits (no delay)
                            # close_position_violently uses market order with fallback to limit at midpoint
                            try:
                                result = close_position_violently(broker_pos)
                                if result:
                                    logger.info(f"FORCED EXIT SUCCESS: {symbol} closed immediately")
                                else:
                                    # Fallback to backout_near_market with no delay
                                    logger.warning(f"Immediate close failed for {symbol}, using backout fallback")
                                    backout_near_market(
                                        symbol,
                                        start_offset_minutes=0,  # No delay for deadline exits
                                        sleep_seconds=30,
                                        market_close_buffer_minutes=BACKOUT_MARKET_CLOSE_BUFFER_MINUTES,
                                        market_close_force_minutes=BACKOUT_MARKET_CLOSE_FORCE_MINUTES,
                                    )
                            except Exception as e:
                                logger.error(f"Failed to close {symbol}: {e}")
                                # Still try backout as last resort
                                backout_near_market(
                                    symbol,
                                    start_offset_minutes=0,
                                    sleep_seconds=30,
                                    market_close_buffer_minutes=BACKOUT_MARKET_CLOSE_BUFFER_MINUTES,
                                    market_close_force_minutes=BACKOUT_MARKET_CLOSE_FORCE_MINUTES,
                                )
                    else:
                        logger.info(f"Time expired for {symbol} but no broker position - cleaning up")

                    positions_to_remove.append(symbol)
                    continue

                # If no broker position and watcher should have filled by now, check if it's stale
                if not has_broker_position and position.entry_timestamp:
                    hours_since_entry = (now - position.entry_timestamp).total_seconds() / 3600
                    if hours_since_entry > 24:
                        logger.warning(
                            f"Entry watcher for {symbol} may have expired without fill - cleaning up"
                        )
                        positions_to_remove.append(symbol)

            except Exception as e:
                logger.error(f"Error managing exit for {symbol}: {e}")

        # Remove closed positions
        for symbol in positions_to_remove:
            # Clean up any remaining watchers
            stop_all_entry_watchers(symbol, reason="v4_position_closed")
            del self.active_positions[symbol]
            logger.info(f"Removed {symbol} from active positions")
            state_changed = True

        if state_changed:
            self._save_state()

    def sync_with_broker(self):
        """Sync active positions with actual broker positions."""
        try:
            broker_positions = get_all_positions()
            broker_positions_by_symbol = {p.symbol: p for p in broker_positions}
            state_changed = False

            # Check tracked positions
            to_remove = []
            for symbol, position in self.active_positions.items():
                broker_pos = broker_positions_by_symbol.get(symbol)

                # If we think we have position but broker doesn't, and TP watcher was started
                # it means position was closed (either TP hit or manual)
                if position.tp_watcher_started and broker_pos is None:
                    logger.info(f"Position {symbol} closed (TP hit or manual) - removing from tracking")
                    to_remove.append(symbol)
                    continue

                # If position was never filled and expired, remove it
                if not position.tp_watcher_started and broker_pos is None:
                    now = pd.Timestamp.now(tz="UTC")
                    if position.should_force_exit(now):
                        logger.info(f"Entry never filled for {symbol} and time expired - removing")
                        to_remove.append(symbol)

            for symbol in to_remove:
                stop_all_entry_watchers(symbol, reason="v4_sync_cleanup")
                del self.active_positions[symbol]
                state_changed = True

            if state_changed:
                self._save_state()

        except Exception as e:
            logger.error(f"Failed to sync with broker: {e}")

    def log_status(self):
        """Log current status."""
        logger.info(f"Active positions: {len(self.active_positions)}/{self.max_positions}")
        for symbol, pos in self.active_positions.items():
            now = pd.Timestamp.now(tz="UTC")
            time_remaining = pos.exit_timestamp - now
            status = "filled" if pos.tp_watcher_started else "pending_entry"
            hours_rem = time_remaining.total_seconds() / 3600
            logger.info(
                f"  {symbol}: entry=${pos.entry_price:.2f}, tp=${pos.sell_price:.2f}, "
                f"status={status}, exit in {hours_rem:.1f}h"
            )

    def refresh_data(self):
        """Download latest market data for all symbols."""
        logger.info(f"Refreshing market data for {len(self.symbols)} symbols...")
        try:
            client = StockHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)
            for symbol in self.symbols:
                try:
                    download_exchange_latest_data(client, symbol)
                except Exception as e:
                    logger.warning(f"Failed to refresh data for {symbol}: {e}")
            logger.info("Data refresh complete")
        except Exception as e:
            logger.error(f"Failed to create data client: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="V4/V6 End-to-end Trading")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to V4/V6 checkpoint")
    parser.add_argument("--symbols", type=str, nargs="+", default=None, help="Symbols to trade")
    parser.add_argument("--dry-run", action="store_true", help="Preview without actual trading")
    parser.add_argument("--max-positions", type=int, default=MAX_POSITIONS, help="Maximum concurrent positions")
    parser.add_argument("--max-position-size", type=float, default=0.30, help="Max position size per symbol (default: 0.30 = 30%%)")
    parser.add_argument("--max-exit-days", type=int, default=2, help="Max exit days (default: 2 for better Sharpe)")
    parser.add_argument("--once", action="store_true", help="Run once and exit (don't loop)")
    parser.add_argument("--trade-crypto", action="store_true", help="Enable crypto trading")
    return parser.parse_args()


def get_market_hours():
    """Get today's market hours."""
    et = pytz.timezone("US/Eastern")
    now = datetime.now(et)
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open, market_close


def main():
    args = parse_args()

    # Setup logging
    logger.info(f"V4/V6 Trading Starting - Checkpoint: {args.checkpoint}")
    if args.dry_run:
        logger.warning("DRY RUN MODE - No actual trades will be executed")

    # Load runtime with position size cap and exit day override
    runtime = DailyTradingRuntimeV4(
        Path(args.checkpoint),
        trade_crypto=args.trade_crypto,
        max_position_override=args.max_position_size,
        max_exit_days=args.max_exit_days,
    )
    logger.info(f"Max position size: {args.max_position_size:.0%}, Max exit days: {args.max_exit_days}")

    # Determine symbols
    symbols = args.symbols if args.symbols else DEFAULT_SYMBOLS
    # Filter through runtime's non-tradable list
    symbols = [s for s in symbols if s.upper() not in runtime.non_tradable]
    logger.info(f"Trading symbols: {symbols}")
    logger.info(f"Non-tradable: {runtime.non_tradable}")

    # Create manager
    manager = V4TradingManager(
        runtime=runtime,
        symbols=symbols,
        dry_run=args.dry_run,
        max_positions=args.max_positions,
    )

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    logger.info("Signal handlers registered for graceful shutdown")

    # Refresh market data before starting
    manager.refresh_data()

    last_plan_date = None

    while not _shutdown_requested:
        try:
            now = datetime.now(pytz.timezone("US/Eastern"))
            today = now.date()
            market_open, market_close = get_market_hours()

            # Reset daily state at midnight
            if last_plan_date != today:
                manager.reset_daily_state()
                last_plan_date = today

            # Sync with broker
            manager.sync_with_broker()

            # Manage exits for existing positions
            manager.manage_exits()

            # Generate plans (in dry-run mode, show plans regardless of market hours)
            is_market_hours = is_nyse_trading_day_now() and market_open <= now <= market_close
            should_generate_plans = is_market_hours or args.dry_run

            if should_generate_plans:
                if not is_market_hours:
                    logger.info("Outside market hours - showing plans for preview only")

                # Check if we have room for new positions
                current_positions = len(manager.active_positions)

                if current_positions < manager.max_positions:
                    # Generate and filter plans
                    plans = manager.get_trading_plans()
                    plans = manager.filter_plans(plans)

                    # Sort by confidence (descending)
                    plans.sort(key=lambda p: p.confidence, reverse=True)

                    # Log all plans in dry-run mode
                    if args.dry_run and plans:
                        logger.info(f"\n{'='*60}")
                        logger.info("V4 TRADING PLANS FOR TODAY")
                        logger.info(f"{'='*60}")
                        for i, plan in enumerate(plans, 1):
                            spread = (plan.sell_price - plan.buy_price) / plan.buy_price * 100
                            logger.info(
                                f"{i}. {plan.symbol}: Buy ${plan.buy_price:.2f} -> Sell ${plan.sell_price:.2f} "
                                f"(spread {spread:.1f}%, size {plan.trade_amount:.1%}, "
                                f"exit {plan.exit_days:.1f}d, conf {plan.confidence:.2f})"
                            )
                        logger.info(f"{'='*60}\n")

                    # Execute entries up to max_positions
                    slots_available = manager.max_positions - current_positions
                    for plan in plans[:slots_available]:
                        success = manager.execute_entry(plan)
                        if success:
                            slots_available -= 1
                            if slots_available <= 0:
                                break

            # Log status
            manager.log_status()

            if args.once:
                logger.info("Single run completed (--once flag)")
                break

            # Sleep before next iteration
            sleep(60)

        except Exception as e:
            logger.exception(f"Error in main loop: {e}")
            sleep(60)

    logger.info("V4/V6 Trading shutdown complete")


if __name__ == "__main__":
    main()
