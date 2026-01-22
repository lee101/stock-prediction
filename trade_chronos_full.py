#!/usr/bin/env python3
"""Chronos Full Strategy - Live Trading Script.

This script runs the chronos_full strategy in live/paper trading:
1. Daily inference to predict high/low for each symbol
2. Symbol selection by highest expected PnL
3. Entry order at predicted_low + buffer
4. Exit order at predicted_high - buffer

Usage:
    PAPER=1 python trade_chronos_full.py              # Paper trading
    PAPER=1 python trade_chronos_full.py --once       # Single run
    PAPER=1 python trade_chronos_full.py --dry-run    # Dry run (no orders)

The strategy generated +1742% return in 2025 backtesting.
"""

import argparse
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import pytz

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trade_chronos_full.log"),
    ],
)
logger = logging.getLogger(__name__)

# Timezone
ET = pytz.timezone("America/New_York")
UTC = pytz.UTC

# Trading schedule (Eastern Time)
DAILY_RUN_HOUR = 9  # 9:00 AM ET - just before market open for fresh data
DAILY_RUN_MIN = 25  # 9:25 AM - 5 min before open
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MIN = 30
MARKET_CLOSE_HOUR = 16

# Strategy modes
class ExitMode:
    SAME_DAY_ONLY = "same_day_only"  # Force exit at close (most conservative)
    HOLD_ADAPTIVE = "hold_adaptive"  # Hold overnight, use new day's target (best returns)

MAX_HOLD_DAYS = 10  # Force exit if held longer than this (tuned optimal from simulation)


def is_paper_mode() -> bool:
    """Check if running in paper mode."""
    return os.environ.get("PAPER", "1").lower() in ("1", "true", "yes")


def get_current_et() -> datetime:
    """Get current time in Eastern."""
    return datetime.now(ET)


def is_market_hours(now: datetime = None) -> bool:
    """Check if during market hours."""
    if now is None:
        now = get_current_et()
    # Weekday check
    if now.weekday() >= 5:  # Saturday, Sunday
        return False
    # Hour check
    market_open = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MIN, second=0)
    market_close = now.replace(hour=MARKET_CLOSE_HOUR, minute=0, second=0)
    return market_open <= now <= market_close


def should_run_daily(now: datetime, last_run_date: Optional[date]) -> bool:
    """Check if should run daily inference.

    Runs at 9:25 AM ET (5 min before market open) for freshest data.
    """
    if last_run_date == now.date():
        return False  # Already ran today
    # Check if past the run time
    run_time = now.replace(hour=DAILY_RUN_HOUR, minute=DAILY_RUN_MIN, second=0)
    return now >= run_time


class ChronosFullTrader:
    """Live trader using chronos_full strategy.

    Strategy modes (from rigorous backtesting):

    SAME_DAY_ONLY (conservative, 2658% return):
    - Buy at predicted_low + buffer
    - Exit at predicted_high - buffer if hit same day
    - Force exit at 3:45 PM if target not hit

    HOLD_ADAPTIVE (best returns, 7816% for top3):
    - Buy at predicted_low + buffer
    - Hold overnight if target not hit
    - Each day, update exit target to NEW day's predicted_high
    - Force exit after MAX_HOLD_DAYS as backstop
    """

    def __init__(
        self,
        dry_run: bool = False,
        position_pct: float = 0.95,
        exit_mode: str = ExitMode.HOLD_ADAPTIVE,  # Best returns
        top_n: int = 3,  # Top 3 symbols to consider (best in backtest)
    ):
        self.dry_run = dry_run
        self.position_pct = position_pct
        self.exit_mode = exit_mode
        self.top_n = top_n

        # State - now supports multiple positions for top_n trading
        self.current_strategy = None
        self.current_positions = {}  # symbol -> {qty, buy_price, sell_price, entry_order_id, exit_order_id}
        self.entry_date = None  # Track when we entered for max-hold backstop

        # Initialize inference
        from pnlforecast.inference import ChronosFullInference
        self.inference = ChronosFullInference()

        logger.info(
            "Initialized ChronosFullTrader (paper=%s, dry_run=%s, exit_mode=%s, top_n=%d)",
            is_paper_mode(),
            dry_run,
            exit_mode,
            top_n,
        )

    def get_account_info(self) -> dict:
        """Get account balance and buying power."""
        try:
            import alpaca_wrapper
            status = alpaca_wrapper.get_account_status(force_refresh=True)
            if status.get("success"):
                account = status.get("account")
                return {
                    "equity": float(getattr(account, "equity", 0) or 0),
                    "cash": float(getattr(account, "cash", 0) or 0),
                    "buying_power": float(status.get("buying_power", 0) or 0),
                }
            else:
                logger.error("Account status failed: %s", status.get("error"))
                return {"equity": 0, "cash": 0, "buying_power": 0}
        except Exception as e:
            logger.error("Failed to get account info: %s", e)
            return {"equity": 0, "cash": 0, "buying_power": 0}

    def get_current_positions(self) -> dict:
        """Get current positions."""
        try:
            import alpaca_wrapper
            from src.trading_obj_utils import filter_to_realistic_positions

            positions = filter_to_realistic_positions(alpaca_wrapper.get_all_positions())
            return {
                getattr(p, "symbol", ""): {
                    "qty": float(getattr(p, "qty", 0) or 0),
                    "side": getattr(p, "side", "").lower(),
                    "market_value": float(getattr(p, "market_value", 0) or 0),
                    "avg_entry_price": float(getattr(p, "avg_entry_price", 0) or 0),
                }
                for p in positions
            }
        except Exception as e:
            logger.error("Failed to get positions: %s", e)
            return {}

    def get_open_orders(self, symbol: str = None) -> list:
        """Get open orders, optionally filtered by symbol."""
        try:
            import alpaca_wrapper
            orders = alpaca_wrapper.get_open_orders()
            if symbol:
                # Normalize symbol (Alpaca returns "UNI/USD", we use "UNIUSD")
                norm_symbol = symbol.replace("/", "")
                orders = [
                    o for o in orders
                    if getattr(o, "symbol", "").replace("/", "") == norm_symbol
                ]
            return orders
        except Exception as e:
            logger.error("Failed to get orders: %s", e)
            return []

    def cancel_all_orders(self, symbol: str = None) -> int:
        """Cancel all open orders, optionally for a specific symbol."""
        try:
            import alpaca_wrapper
            orders = self.get_open_orders(symbol)
            count = 0
            for order in orders:
                try:
                    alpaca_wrapper.cancel_order(order)
                    count += 1
                    time.sleep(0.1)
                except Exception as e:
                    logger.warning("Failed to cancel order: %s", e)
            return count
        except Exception as e:
            logger.error("Failed to cancel orders: %s", e)
            return 0

    def close_position(self, symbol: str) -> bool:
        """Close position for a symbol.

        For crypto: uses GTC limit order at bid/ask (persists until filled)
        For stocks: uses market order (only works during market hours)
        """
        try:
            import alpaca_wrapper

            positions = self.get_current_positions()
            if symbol not in positions:
                return True  # Already closed

            pos = positions[symbol]
            qty = abs(pos["qty"])
            side = "sell" if pos["side"] == "long" else "buy"

            if self.dry_run:
                logger.info("[DRY RUN] Would close %s: %s %.4f", symbol, side, qty)
                return True

            is_crypto = symbol.endswith("USD")

            if is_crypto:
                # For crypto, use GTC limit order at bid/ask price
                # This persists until filled (unlike IOC which cancels immediately)
                quote = alpaca_wrapper.latest_data(symbol)
                if quote:
                    if side == "sell":
                        price = float(getattr(quote, "bid_price", 0) or 0)
                    else:
                        price = float(getattr(quote, "ask_price", 0) or 0)

                    if price > 0:
                        result = alpaca_wrapper.open_order_at_price(
                            symbol=symbol,
                            qty=qty,
                            side=side,
                            price=price,
                        )
                        logger.info("Close order for %s: %s @ $%.4f - %s", symbol, side, price, result)
                        return result is not None
                    else:
                        logger.error("No valid price for %s", symbol)
                        return False
                else:
                    logger.error("No quote data for %s", symbol)
                    return False
            else:
                # For stocks, use market order (only works during market hours)
                result = alpaca_wrapper.open_market_order_violently(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    retries=3,
                )
                logger.info("Close order for %s: %s", symbol, result)
                return result is not None

        except Exception as e:
            logger.error("Failed to close position %s: %s", symbol, e)
            return False

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        position_pct: float = None,
    ) -> float:
        """Calculate position size based on available capital.

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            position_pct: Override position percentage (for splitting across top_n)
        """
        account = self.get_account_info()
        pct = position_pct if position_pct is not None else self.position_pct
        available = account["buying_power"] * pct

        qty = available / entry_price

        # Round based on asset type
        if symbol.endswith("USD"):  # Crypto
            qty = round(qty, 6)
        else:  # Stock
            qty = int(qty)

        return max(qty, 0)

    def place_entry_order(
        self,
        symbol: str,
        qty: float,
        limit_price: float,
    ) -> Optional[str]:
        """Place entry limit order.

        Args:
            symbol: Trading symbol
            qty: Quantity to buy
            limit_price: Entry limit price

        Returns:
            Order ID or None if failed
        """
        if self.dry_run:
            logger.info(
                "[DRY RUN] Would place entry: BUY %.4f %s @ $%.4f",
                qty, symbol, limit_price,
            )
            return "dry_run_entry"

        try:
            import alpaca_wrapper

            result = alpaca_wrapper.open_order_at_price_or_all(
                symbol=symbol,
                qty=qty,
                side="buy",
                price=limit_price,
            )

            if result:
                order_id = getattr(result, "id", str(result))
                logger.info(
                    "Placed entry order: BUY %.4f %s @ $%.4f (ID: %s)",
                    qty, symbol, limit_price, order_id,
                )
                return order_id
            else:
                logger.warning("Entry order returned None for %s", symbol)
                return None

        except Exception as e:
            logger.error("Failed to place entry order for %s: %s", symbol, e)
            return None

    def place_exit_order(
        self,
        symbol: str,
        qty: float,
        limit_price: float,
    ) -> Optional[str]:
        """Place exit limit order (take profit).

        Args:
            symbol: Trading symbol
            qty: Quantity to sell
            limit_price: Exit limit price

        Returns:
            Order ID or None if failed
        """
        # Round stock quantities to whole numbers to enable GTC orders
        # (Alpaca requires DAY time-in-force for fractional stock orders, which expire)
        is_crypto = symbol.endswith("USD")
        if not is_crypto:
            original_qty = qty
            qty = max(1, int(qty))  # Round down to whole number, minimum 1
            if qty != original_qty:
                logger.info("Rounded stock qty from %.4f to %d for GTC order", original_qty, qty)

        if self.dry_run:
            logger.info(
                "[DRY RUN] Would place exit: SELL %.4f %s @ $%.4f",
                qty, symbol, limit_price,
            )
            return "dry_run_exit"

        try:
            import alpaca_wrapper

            result = alpaca_wrapper.open_order_at_price(
                symbol=symbol,
                qty=qty,
                side="sell",
                price=limit_price,
            )

            if result:
                order_id = getattr(result, "id", str(result))
                logger.info(
                    "Placed exit order: SELL %.4f %s @ $%.4f (ID: %s)",
                    qty, symbol, limit_price, order_id,
                )
                return order_id
            else:
                logger.warning("Exit order returned None for %s", symbol)
                return None

        except Exception as e:
            logger.error("Failed to place exit order for %s: %s", symbol, e)
            return None

    def check_entry_fill(self, symbol: str) -> Optional[float]:
        """Check if entry order filled.

        Returns:
            Filled quantity or None if not filled
        """
        positions = self.get_current_positions()
        if symbol in positions:
            return positions[symbol]["qty"]
        return None

    def ensure_exit_orders_for_all_positions(self) -> int:
        """Ensure exit orders exist for ALL positions (including ones not tracked).

        This runs on startup to place exit orders for positions from previous sessions.
        Uses today's forecast sell_price, or 5% above current price as fallback.

        Returns:
            Number of exit orders placed
        """
        from pnlforecast.inference import get_tradeable_symbols

        positions = self.get_current_positions()
        if not positions:
            logger.info("No positions to place exit orders for")
            return 0

        # Get today's forecasts
        forecasts_by_symbol = {}
        if self.current_strategy:
            tradeable = get_tradeable_symbols(self.current_strategy, top_n=100)
            forecasts_by_symbol = {sym: forecast for sym, forecast in tradeable}

        exits_placed = 0
        for symbol, pos_info in positions.items():
            qty = pos_info["qty"]
            if qty <= 0:
                continue

            # Check if already has a sell order
            sell_orders = [
                o for o in self.get_open_orders(symbol)
                if getattr(o, "side", "").lower() in ("sell", "ordersid.sell")
                   or "sell" in str(getattr(o, "side", "")).lower()
            ]
            if sell_orders:
                logger.info("Exit order already exists for %s", symbol)
                continue

            # Determine sell price from forecast or fallback
            if symbol in forecasts_by_symbol:
                sell_price = forecasts_by_symbol[symbol].sell_price
                logger.info("Using forecast sell price for %s: $%.4f", symbol, sell_price)
            else:
                # Fallback: 5% above current price
                import alpaca_wrapper
                quote = alpaca_wrapper.latest_data(symbol)
                if quote:
                    current_price = float(getattr(quote, "ask_price", 0) or getattr(quote, "bid_price", 0) or 0)
                    if current_price > 0:
                        sell_price = current_price * 1.05  # 5% above current
                        logger.info("Using 5%% above current price for %s: $%.4f", symbol, sell_price)
                    else:
                        logger.warning("No price data for %s, skipping exit order", symbol)
                        continue
                else:
                    logger.warning("No quote for %s, skipping exit order", symbol)
                    continue

            # Place exit order
            logger.info("Placing exit order for existing position: %s qty=%.4f @ $%.4f", symbol, qty, sell_price)
            exit_order_id = self.place_exit_order(symbol, qty, sell_price)

            if exit_order_id:
                # Track it
                if symbol not in self.current_positions:
                    self.current_positions[symbol] = {
                        "qty": qty,
                        "buy_price": pos_info.get("avg_entry_price", 0),
                        "sell_price": sell_price,
                        "entry_order_id": None,
                        "exit_order_id": exit_order_id,
                        "exit_order_placed_at": time.time(),
                    }
                else:
                    self.current_positions[symbol]["exit_order_id"] = exit_order_id
                    self.current_positions[symbol]["exit_order_placed_at"] = time.time()
                exits_placed += 1

        return exits_placed

    def run_daily_strategy(self) -> bool:
        """Run daily strategy: inference, selection, order placement for TOP N symbols.

        Returns:
            True if strategy executed successfully
        """
        logger.info("=" * 70)
        logger.info("RUNNING DAILY CHRONOS FULL STRATEGY (TOP %d)", self.top_n)
        logger.info("=" * 70)

        # 1. Run inference
        try:
            strategy = self.inference.run_daily_inference()
            self.current_strategy = strategy
        except Exception as e:
            logger.error("Inference failed: %s", e)
            return False

        # 2. Print summary
        from pnlforecast.inference import print_strategy_summary, get_tradeable_symbols
        print_strategy_summary(strategy)

        # 3. Get top N tradeable symbols
        tradeable = get_tradeable_symbols(strategy, top_n=self.top_n)
        if not tradeable:
            logger.warning("No tradeable symbols found")
            return False

        selected_symbols = [sym for sym, _ in tradeable]
        logger.info("Selected top %d symbols: %s", self.top_n, selected_symbols)

        # 4. Cancel ALL existing orders
        canceled = self.cancel_all_orders()
        if canceled > 0:
            logger.info("Canceled %d existing orders", canceled)
            time.sleep(1)

        # 5. Close positions NOT in today's selection
        current_positions = self.get_current_positions()
        for pos_symbol in list(current_positions.keys()):
            if pos_symbol not in selected_symbols:
                logger.info("Closing position not in today's selection: %s", pos_symbol)
                self.close_position(pos_symbol)
                time.sleep(0.5)

        # 6. Place entry orders for each selected symbol
        # Split capital equally among top_n symbols
        per_symbol_pct = self.position_pct / self.top_n
        success_count = 0

        for symbol, forecast in tradeable:
            has_position = symbol in current_positions

            # Calculate position size for this symbol
            qty = self.calculate_position_size(symbol, forecast.buy_price, per_symbol_pct)
            if qty <= 0:
                logger.warning("Calculated quantity is 0 for %s, skipping", symbol)
                continue

            # Track for exit management
            if has_position:
                logger.info("Have position in %s, placing buy order for re-entry after sell", symbol)
                if symbol not in self.current_positions:
                    self.current_positions[symbol] = {
                        "qty": current_positions[symbol]["qty"],
                        "buy_price": forecast.buy_price,
                        "sell_price": forecast.sell_price,
                        "entry_order_id": None,
                        "exit_order_id": None,
                    }
            else:
                logger.info(
                    "Strategy: %s, Qty: %.4f, Buy: $%.4f, Sell: $%.4f, Expected PnL: %.2f%%",
                    symbol, qty, forecast.buy_price, forecast.sell_price,
                    forecast.expected_pnl * 100,
                )

            # ALWAYS place entry order (even if we have position - allows re-entry after sell)
            entry_order_id = self.place_entry_order(symbol, qty, forecast.buy_price)

            if entry_order_id:
                # Store/update state for this symbol
                if symbol not in self.current_positions:
                    self.current_positions[symbol] = {
                        "qty": qty,
                        "buy_price": forecast.buy_price,
                        "sell_price": forecast.sell_price,
                        "entry_order_id": entry_order_id,
                        "exit_order_id": None,
                    }
                else:
                    # Update entry order ID for existing position
                    self.current_positions[symbol]["entry_order_id"] = entry_order_id
                    self.current_positions[symbol]["buy_price"] = forecast.buy_price
                    self.current_positions[symbol]["sell_price"] = forecast.sell_price
                success_count += 1
            else:
                logger.warning("Failed to place entry order for %s", symbol)

            time.sleep(0.5)  # Rate limiting

        logger.info("Strategy executed - %d/%d orders placed", success_count, len(tradeable))
        return success_count > 0

    def check_and_place_exit(self) -> int:
        """Check if entries filled and place exit orders for ALL tracked positions.

        Returns:
            Number of exit orders placed
        """
        if not self.current_positions:
            return 0

        exits_placed = 0
        for symbol, pos_info in list(self.current_positions.items()):
            sell_price = pos_info["sell_price"]

            # Check if we have a filled position
            filled_qty = self.check_entry_fill(symbol)
            if not filled_qty:
                continue

            # Check if we already placed an exit order (tracked internally)
            if pos_info.get("exit_order_id"):
                logger.debug("Exit order already tracked for %s: %s", symbol, pos_info["exit_order_id"])
                continue

            # Also check Alpaca's open orders as backup
            exit_orders = [
                o for o in self.get_open_orders(symbol)
                if getattr(o, "side", "").lower() == "sell"
            ]
            if exit_orders:
                logger.debug("Exit order found in Alpaca for %s", symbol)
                pos_info["exit_order_id"] = str(getattr(exit_orders[0], "id", "unknown"))
                continue

            # Place exit order
            logger.info("Entry filled for %s! Placing exit order at $%.4f", symbol, sell_price)
            exit_order_id = self.place_exit_order(symbol, filled_qty, sell_price)

            if exit_order_id:
                pos_info["exit_order_id"] = exit_order_id
                pos_info["exit_order_placed_at"] = time.time()
                exits_placed += 1

        return exits_placed

    def check_and_refresh_exit_orders(self) -> int:
        """Check if exit orders still exist (may have expired) and re-place if needed.

        This handles the case where DAY orders expire at market close.
        With the fix to round stock quantities to whole numbers, this should be rare.

        Returns:
            Number of exit orders refreshed
        """
        if not self.current_positions:
            return 0

        refreshed = 0
        for symbol, pos_info in list(self.current_positions.items()):
            exit_order_id = pos_info.get("exit_order_id")
            if not exit_order_id or exit_order_id == "dry_run_exit":
                continue

            # Skip if order was just placed (give API time to propagate)
            last_placed = pos_info.get("exit_order_placed_at")
            if last_placed:
                if time.time() - last_placed < 60:  # Skip if placed within last 60 seconds
                    continue

            # Check if the exit order still exists in Alpaca
            exit_orders = [
                o for o in self.get_open_orders(symbol)
                if getattr(o, "side", "").lower() == "sell"
            ]

            if exit_orders:
                # Exit order still exists
                continue

            # Exit order missing! Check if we still have position
            filled_qty = self.check_entry_fill(symbol)
            if not filled_qty:
                # No position either, clean up tracking
                logger.info("No exit order and no position for %s, cleaning up", symbol)
                del self.current_positions[symbol]
                continue

            # Have position but no exit order - re-place it
            sell_price = pos_info.get("sell_price", 0)
            if sell_price <= 0:
                logger.warning("No sell price for %s, cannot refresh exit order", symbol)
                continue

            logger.warning(
                "Exit order expired/missing for %s! Re-placing at $%.4f",
                symbol, sell_price
            )
            new_exit_order_id = self.place_exit_order(symbol, filled_qty, sell_price)
            if new_exit_order_id:
                pos_info["exit_order_id"] = new_exit_order_id
                pos_info["exit_order_placed_at"] = time.time()
                refreshed += 1

        return refreshed

    def should_force_exit(self, now: datetime) -> Tuple[bool, str]:
        """Check if should force exit ALL positions.

        Returns:
            (should_exit, reason) tuple
        """
        if not self.current_positions:
            return False, ""

        # SAME_DAY_ONLY: Force exit 15 min before close
        if self.exit_mode == ExitMode.SAME_DAY_ONLY:
            close_warning = now.replace(hour=15, minute=45, second=0)
            market_close = now.replace(hour=MARKET_CLOSE_HOUR, minute=0, second=0)
            if close_warning <= now < market_close:
                return True, "same_day_only: near market close"

        # HOLD_ADAPTIVE: Force exit after MAX_HOLD_DAYS (backstop)
        if self.exit_mode == ExitMode.HOLD_ADAPTIVE and self.entry_date:
            days_held = (now.date() - self.entry_date).days
            if days_held >= MAX_HOLD_DAYS:
                return True, f"max_hold_days: held {days_held} days (limit: {MAX_HOLD_DAYS})"

        return False, ""

    def force_exit_all_positions(self, reason: str = "") -> int:
        """Force exit ALL current positions at market price.

        Used for:
        - SAME_DAY_ONLY: exit before market close
        - HOLD_ADAPTIVE: backstop after MAX_HOLD_DAYS

        Returns:
            Number of positions closed
        """
        if not self.current_positions:
            return 0

        closed_count = 0
        for symbol in list(self.current_positions.keys()):
            logger.warning("FORCE EXIT: %s - %s", symbol, reason or "sell target not hit")

            # Cancel any pending exit limit orders
            self.cancel_all_orders(symbol)
            time.sleep(0.5)

            # Close at market
            success = self.close_position(symbol)
            if success:
                del self.current_positions[symbol]
                closed_count += 1

        if closed_count > 0:
            self.entry_date = None

        return closed_count

    def update_exit_orders_adaptive(self) -> int:
        """Update ALL exit orders with today's predicted_high (for HOLD_ADAPTIVE mode).

        Each day, we update sell targets based on new Chronos forecasts.

        Returns:
            Number of exit orders updated
        """
        if not self.current_positions:
            return 0

        updated_count = 0

        for symbol, pos_info in list(self.current_positions.items()):
            try:
                # Get fresh forecast for today
                forecast = self.inference.forecast_symbol(symbol, date.today())
                if forecast is None:
                    logger.warning("No forecast for %s today, keeping old exit target", symbol)
                    continue

                new_sell_price = forecast.sell_price
                old_sell_price = pos_info.get("sell_price", 0)

                if abs(new_sell_price - old_sell_price) < 0.001:
                    continue  # No significant change

                logger.info(
                    "ADAPTIVE UPDATE: %s sell target $%.4f -> $%.4f",
                    symbol, old_sell_price, new_sell_price
                )

                # Cancel old exit order
                self.cancel_all_orders(symbol)
                time.sleep(0.5)

                # Update position with new target
                pos_info["sell_price"] = new_sell_price

                # Place new exit order if we have filled position
                filled_qty = self.check_entry_fill(symbol)
                if filled_qty:
                    exit_order_id = self.place_exit_order(symbol, filled_qty, new_sell_price)
                    if exit_order_id:
                        pos_info["exit_order_id"] = exit_order_id
                        pos_info["exit_order_placed_at"] = time.time()
                        updated_count += 1

            except Exception as e:
                logger.error("Failed to update exit order for %s: %s", symbol, e)

        return updated_count

    def run_loop(self, interval_seconds: int = 60, run_now: bool = False) -> None:
        """Run main trading loop.

        Args:
            interval_seconds: Seconds between checks
            run_now: If True, run strategy immediately on startup
        """
        logger.info("Starting trading loop (interval: %ds)", interval_seconds)
        logger.info("Exit mode: %s, Top N: %d", self.exit_mode, self.top_n)

        last_run_date = None
        last_adaptive_update_date = None

        # Force immediate run if requested
        if run_now:
            logger.info("*** RUN NOW: Executing strategy immediately ***")
            success = self.run_daily_strategy()
            if success:
                last_run_date = get_current_et().date()
                self.entry_date = last_run_date
                # Ensure exit orders for all positions (including legacy ones)
                exits = self.ensure_exit_orders_for_all_positions()
                if exits > 0:
                    logger.info("Placed %d exit orders for existing positions", exits)
            else:
                logger.warning("Immediate run failed, will retry in loop")

        while True:
            try:
                now = get_current_et()

                # Daily strategy run (at 9:25 AM for fresh data)
                if should_run_daily(now, last_run_date):
                    # If holding positions in HOLD_ADAPTIVE mode, update exit targets
                    if (self.exit_mode == ExitMode.HOLD_ADAPTIVE and
                        self.current_positions and
                        last_adaptive_update_date != now.date()):
                        updated = self.update_exit_orders_adaptive()
                        logger.info("Updated %d exit orders with new targets", updated)
                        last_adaptive_update_date = now.date()
                    else:
                        # Run new strategy (handles existing positions)
                        success = self.run_daily_strategy()
                        if success:
                            last_run_date = now.date()
                            self.entry_date = now.date()
                            # Ensure exit orders for all positions
                            exits = self.ensure_exit_orders_for_all_positions()
                            if exits > 0:
                                logger.info("Placed %d exit orders for existing positions", exits)
                        else:
                            logger.warning("Daily strategy failed, will retry next loop")

                # Check for entry fills and place exit orders for ALL positions
                if self.current_positions:
                    self.check_and_place_exit()
                    # Also check if any exit orders expired and need refresh
                    refreshed = self.check_and_refresh_exit_orders()
                    if refreshed > 0:
                        logger.info("Refreshed %d expired exit orders", refreshed)

                # Force exit if needed (same_day_only near close OR max hold days)
                should_exit, reason = self.should_force_exit(now)
                if should_exit:
                    logger.info("Force exit triggered: %s", reason)
                    closed = self.force_exit_all_positions(reason)
                    logger.info("Force closed %d positions", closed)

                # Status log
                if now.minute == 0:  # Log every hour
                    account = self.get_account_info()
                    positions = self.get_current_positions()
                    days_held = (now.date() - self.entry_date).days if self.entry_date else 0
                    logger.info(
                        "Status: Equity=$%.2f, Cash=$%.2f, Positions=%d, Tracked=%d, Days Held=%d",
                        account["equity"],
                        account["cash"],
                        len(positions),
                        len(self.current_positions),
                        days_held,
                    )

            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
            except Exception as e:
                logger.error("Error in trading loop: %s", e, exc_info=True)

            time.sleep(interval_seconds)

        logger.info("Trading loop ended")

    def run_once(self) -> bool:
        """Run strategy once (for testing).

        Returns:
            True if successful
        """
        logger.info("Running single strategy execution")
        success = self.run_daily_strategy()

        if success and self.current_positions:
            # Wait a bit and check for fills
            time.sleep(5)
            self.check_and_place_exit()

        return success

    def cleanup(self) -> None:
        """Clean up resources."""
        self.inference.cleanup()
        logger.info("Trader cleaned up")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chronos Full Strategy - Live Trading"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit (for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode (no actual orders)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Loop interval in seconds",
    )
    parser.add_argument(
        "--position-pct",
        type=float,
        default=0.95,
        help="Position size as percent of buying power",
    )
    parser.add_argument(
        "--exit-mode",
        choices=["same_day_only", "hold_adaptive"],
        default="hold_adaptive",
        help="Exit mode: same_day_only (exit at close) or hold_adaptive (best returns, default)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top symbols to consider (default: 3, best in backtest)",
    )
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Run strategy immediately on startup (don't wait for 9:25 AM)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    exit_mode = ExitMode.HOLD_ADAPTIVE if args.exit_mode == "hold_adaptive" else ExitMode.SAME_DAY_ONLY

    logger.info("=" * 70)
    logger.info("CHRONOS FULL STRATEGY - LIVE TRADING")
    logger.info("=" * 70)
    logger.info("Mode: %s", "PAPER" if is_paper_mode() else "LIVE")
    logger.info("Dry Run: %s", args.dry_run)
    logger.info("Exit Mode: %s", args.exit_mode)
    logger.info("Top N Symbols: %d", args.top_n)
    logger.info("Position Size: %.0f%%", args.position_pct * 100)
    logger.info("Max Hold Days: %d (backstop)", MAX_HOLD_DAYS)
    logger.info("=" * 70)

    trader = ChronosFullTrader(
        dry_run=args.dry_run,
        position_pct=args.position_pct,
        exit_mode=exit_mode,
        top_n=args.top_n,
    )

    try:
        if args.once:
            success = trader.run_once()
            logger.info("Single run %s", "succeeded" if success else "failed")
        else:
            trader.run_loop(interval_seconds=args.interval, run_now=args.run_now)
    finally:
        trader.cleanup()


if __name__ == "__main__":
    main()
