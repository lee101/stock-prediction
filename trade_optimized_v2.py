#!/usr/bin/env python3
"""Optimized Chronos2 Trading Strategy v2.

Key optimizations from backtesting:
1. Winner symbols only (historically positive edge)
2. Top-1 selection by predicted spread
3. Spread filter: 1-5% predicted range
4. 1% stop loss (crucial for profitability!)

Backtest results: +112% annual return with 52% win rate
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytz

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import alpaca_wrapper
from pnlforecast.chronos2_inference_daily import DailyForecastResult, Chronos2DailyInference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Strategy parameters (optimized via backtesting)
WINNER_SYMBOLS = ["AMZN", "SHOP", "MSFT", "COST", "GOOG", "META", "AAPL"]
MIN_SPREAD_PCT = 0.01  # 1% minimum predicted spread
MAX_SPREAD_PCT = 0.05  # 5% maximum predicted spread
STOP_LOSS_PCT = 0.01   # 1% stop loss
TOP_N = 1              # Number of symbols to trade

# Timing
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
ET = pytz.timezone("America/New_York")


def get_current_et() -> datetime:
    """Get current time in Eastern."""
    return datetime.now(ET)


def is_paper_mode() -> bool:
    """Check if running in paper mode."""
    return os.environ.get("PAPER", "1") == "1"


@dataclass
class Position:
    """Track an open position."""
    symbol: str
    qty: float
    entry_price: float
    stop_price: float
    target_price: float
    entry_order_id: Optional[str] = None
    exit_order_id: Optional[str] = None
    stop_order_id: Optional[str] = None


class OptimizedTrader:
    """Optimized trading strategy based on Chronos2 predictions."""

    def __init__(
        self,
        symbols: List[str] = None,
        position_pct: float = 0.95,
        dry_run: bool = False,
    ):
        self.symbols = symbols or WINNER_SYMBOLS
        self.position_pct = position_pct
        self.dry_run = dry_run

        # Initialize inference
        logger.info("Initializing Chronos2 inference...")
        self.inference = Chronos2DailyInference(symbols=self.symbols)

        # State tracking
        self.positions: Dict[str, Position] = {}

        logger.info("OptimizedTrader initialized")
        logger.info("  Symbols: %s", self.symbols)
        logger.info("  Spread filter: %.1f%% - %.1f%%", MIN_SPREAD_PCT*100, MAX_SPREAD_PCT*100)
        logger.info("  Stop loss: %.1f%%", STOP_LOSS_PCT*100)
        logger.info("  Top N: %d", TOP_N)
        logger.info("  Paper mode: %s", is_paper_mode())
        logger.info("  Dry run: %s", dry_run)

    def get_account_info(self) -> Dict:
        """Get account information."""
        account = alpaca_wrapper.get_account()
        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
        }

    def get_current_positions(self) -> Dict[str, Dict]:
        """Get current positions from broker."""
        positions = {}
        for pos in alpaca_wrapper.get_all_positions():
            positions[pos.symbol.replace("/", "")] = {
                "qty": float(pos.qty),
                "avg_entry_price": float(pos.avg_entry_price),
                "current_price": float(pos.current_price),
                "unrealized_pnl": float(pos.unrealized_pl),
            }
        return positions

    def cancel_all_orders(self, symbol: str = None) -> int:
        """Cancel open orders for symbol or all."""
        orders = alpaca_wrapper.get_orders()
        canceled = 0

        for order in orders:
            order_symbol = order.symbol.replace("/", "")
            if symbol is None or order_symbol == symbol:
                try:
                    alpaca_wrapper.cancel_order(order.id)
                    canceled += 1
                except Exception as e:
                    logger.warning("Failed to cancel order %s: %s", order.id, e)

        return canceled

    def place_entry_order(self, symbol: str, qty: float, limit_price: float) -> Optional[str]:
        """Place limit buy order."""
        if self.dry_run:
            logger.info("[DRY RUN] Would buy %.4f %s @ $%.4f", qty, symbol, limit_price)
            return "dry_run_entry"

        try:
            result = alpaca_wrapper.open_order_at_price(
                symbol=symbol,
                qty=qty,
                side="buy",
                price=limit_price,
            )
            if result:
                order_id = getattr(result, "id", str(result))
                logger.info("Entry order: BUY %.4f %s @ $%.4f (ID: %s)", qty, symbol, limit_price, order_id)
                return str(order_id)
        except Exception as e:
            logger.error("Failed to place entry order for %s: %s", symbol, e)

        return None

    def place_exit_order(self, symbol: str, qty: float, limit_price: float) -> Optional[str]:
        """Place limit sell order (take profit)."""
        # Round to whole shares for stocks (GTC requirement)
        original_qty = qty
        qty = max(1, int(qty))
        if qty != original_qty:
            logger.info("Rounded qty from %.4f to %d for GTC", original_qty, qty)

        if self.dry_run:
            logger.info("[DRY RUN] Would sell %.4f %s @ $%.4f", qty, symbol, limit_price)
            return "dry_run_exit"

        try:
            result = alpaca_wrapper.open_order_at_price(
                symbol=symbol,
                qty=qty,
                side="sell",
                price=limit_price,
            )
            if result:
                order_id = getattr(result, "id", str(result))
                logger.info("Exit order: SELL %d %s @ $%.4f (ID: %s)", qty, symbol, limit_price, order_id)
                return str(order_id)
        except Exception as e:
            logger.error("Failed to place exit order for %s: %s", symbol, e)

        return None

    def execute_stop_loss(self, symbol: str) -> bool:
        """Execute stop loss by closing position at market.

        Returns:
            True if successful
        """
        if self.dry_run:
            logger.info("[DRY RUN] Would execute stop loss for %s", symbol)
            return True

        try:
            # Cancel any pending exit orders first
            self.cancel_all_orders(symbol)
            time.sleep(0.5)

            # Close at market
            result = alpaca_wrapper.close_position(symbol)
            logger.warning("STOP LOSS EXECUTED: %s", symbol)
            return result is not None
        except Exception as e:
            logger.error("Failed to execute stop loss for %s: %s", symbol, e)
            return False

    def select_best_trade(self) -> Optional[Tuple[str, DailyForecastResult]]:
        """Select the best symbol to trade based on predicted spread.

        Returns:
            (symbol, forecast) or None if no valid trades
        """
        today = date.today()
        candidates = []

        for symbol in self.symbols:
            try:
                forecast = self.inference.forecast_symbol(symbol, today)
                if forecast is None:
                    continue

                # Calculate predicted spread
                spread = (forecast.sell_price - forecast.buy_price) / forecast.buy_price

                # Apply spread filter
                if MIN_SPREAD_PCT <= spread <= MAX_SPREAD_PCT:
                    candidates.append((symbol, forecast, spread))
                    logger.debug("%s: spread=%.2f%% (valid)", symbol, spread*100)
                else:
                    logger.debug("%s: spread=%.2f%% (filtered)", symbol, spread*100)

            except Exception as e:
                logger.warning("Failed to forecast %s: %s", symbol, e)

        if not candidates:
            logger.info("No valid candidates today")
            return None

        # Sort by spread (highest first)
        candidates.sort(key=lambda x: x[2], reverse=True)

        # Return top 1
        symbol, forecast, spread = candidates[0]
        logger.info("Selected: %s (spread=%.2f%%)", symbol, spread*100)

        return symbol, forecast

    def run_daily_strategy(self) -> bool:
        """Run daily trading strategy.

        Returns:
            True if trade was placed
        """
        logger.info("=" * 60)
        logger.info("Running optimized strategy for %s", date.today())

        # Get account info
        account = self.get_account_info()
        logger.info("Account: Equity=$%.2f, Cash=$%.2f", account["equity"], account["cash"])

        # Check for existing positions
        current_positions = self.get_current_positions()
        if current_positions:
            logger.info("Existing positions: %s", list(current_positions.keys()))
            # Don't open new positions if we have any
            # (could enhance to manage multiple positions)
            return False

        # Cancel any stale orders
        canceled = self.cancel_all_orders()
        if canceled:
            logger.info("Canceled %d stale orders", canceled)
            time.sleep(1)

        # Select best trade
        selection = self.select_best_trade()
        if selection is None:
            logger.info("No trades today")
            return False

        symbol, forecast = selection

        # Calculate position size
        available_cash = account["cash"] * self.position_pct
        qty = available_cash / forecast.buy_price
        qty = int(qty)  # Round to whole shares

        if qty < 1:
            logger.warning("Insufficient cash for 1 share of %s", symbol)
            return False

        # Calculate stop loss price
        stop_price = forecast.buy_price * (1 - STOP_LOSS_PCT)

        logger.info("Trade setup:")
        logger.info("  Symbol: %s", symbol)
        logger.info("  Quantity: %d shares", qty)
        logger.info("  Entry (buy limit): $%.4f", forecast.buy_price)
        logger.info("  Target (sell limit): $%.4f", forecast.sell_price)
        logger.info("  Stop loss: $%.4f (%.1f%%)", stop_price, STOP_LOSS_PCT*100)
        logger.info("  Expected profit: %.2f%%", forecast.expected_pnl*100)

        # Place entry order
        entry_order_id = self.place_entry_order(symbol, qty, forecast.buy_price)
        if not entry_order_id:
            logger.error("Failed to place entry order")
            return False

        # Track position
        self.positions[symbol] = Position(
            symbol=symbol,
            qty=qty,
            entry_price=forecast.buy_price,
            stop_price=stop_price,
            target_price=forecast.sell_price,
            entry_order_id=entry_order_id,
        )

        logger.info("Trade placed successfully")
        return True

    def check_fills_and_manage(self) -> None:
        """Check for order fills and manage positions."""
        if not self.positions:
            return

        current_positions = self.get_current_positions()

        for symbol, pos_info in list(self.positions.items()):
            # Check if entry filled
            if symbol in current_positions:
                actual_pos = current_positions[symbol]

                # Entry filled - place exit orders if not already
                if pos_info.exit_order_id is None:
                    logger.info("Entry filled for %s! Placing exit orders...", symbol)

                    # Place take profit order
                    exit_id = self.place_exit_order(symbol, actual_pos["qty"], pos_info.target_price)
                    if exit_id:
                        pos_info.exit_order_id = exit_id

                    # Place stop loss order
                    # Note: This creates an OCO situation - need to manage manually
                    # For now, we'll monitor price and cancel/replace as needed

                # Check if we should trigger stop manually
                current_price = actual_pos["current_price"]
                if current_price <= pos_info.stop_price:
                    logger.warning("STOP TRIGGERED: %s @ $%.4f (stop: $%.4f)",
                                 symbol, current_price, pos_info.stop_price)
                    if self.execute_stop_loss(symbol):
                        del self.positions[symbol]

            else:
                # Check if exit filled (position closed)
                # Check open orders
                open_orders = alpaca_wrapper.get_orders()
                has_orders = any(o.symbol.replace("/", "") == symbol for o in open_orders)

                if not has_orders:
                    # Position was closed (either take profit or stop hit)
                    logger.info("Position closed for %s", symbol)
                    del self.positions[symbol]

    def run_loop(self, interval_seconds: int = 60) -> None:
        """Main trading loop."""
        logger.info("Starting trading loop (interval: %ds)", interval_seconds)

        last_run_date = None

        while True:
            try:
                now = get_current_et()

                # Run strategy at market open
                if (now.hour == MARKET_OPEN_HOUR and
                    now.minute >= MARKET_OPEN_MINUTE and
                    now.minute < MARKET_OPEN_MINUTE + 5 and
                    last_run_date != now.date()):

                    logger.info("Market open - running strategy")
                    self.run_daily_strategy()
                    last_run_date = now.date()

                # Check and manage positions
                self.check_fills_and_manage()

                # Hourly status
                if now.minute == 0:
                    account = self.get_account_info()
                    positions = self.get_current_positions()
                    logger.info("Status: Equity=$%.2f, Positions=%d, Tracked=%d",
                              account["equity"], len(positions), len(self.positions))

            except KeyboardInterrupt:
                logger.info("Interrupted")
                break
            except Exception as e:
                logger.error("Error in loop: %s", e, exc_info=True)

            time.sleep(interval_seconds)

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.inference.cleanup()
        logger.info("Cleanup complete")


def main():
    parser = argparse.ArgumentParser(description="Optimized Chronos2 Trading Strategy v2")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--interval", type=int, default=60, help="Loop interval seconds")
    parser.add_argument("--position-pct", type=float, default=0.95, help="Position size %")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Optimized Chronos2 Trading Strategy v2")
    logger.info("=" * 60)
    logger.info("Backtest: +112%% annual return")
    logger.info("Parameters: spread=1-5%%, stop=1%%, top-1")
    logger.info("=" * 60)

    trader = OptimizedTrader(
        position_pct=args.position_pct,
        dry_run=args.dry_run,
    )

    try:
        if args.once:
            trader.run_daily_strategy()
        else:
            trader.run_loop(interval_seconds=args.interval)
    finally:
        trader.cleanup()


if __name__ == "__main__":
    main()
