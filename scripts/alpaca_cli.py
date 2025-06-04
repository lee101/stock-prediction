from datetime import datetime, timezone
from time import sleep
import traceback
from typing import Optional

import alpaca_trade_api as tradeapi
import math
import pytz
import typer
from alpaca.data import StockHistoricalDataClient

import alpaca_wrapper
from data_curate_daily import download_exchange_latest_data, get_bid, get_ask
from env_real import ALP_KEY_ID, ALP_SECRET_KEY, ALP_ENDPOINT, ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD
from jsonshelve import FlatShelf
from src.fixtures import crypto_symbols
from src.logging_utils import setup_logging
from src.stock_utils import pairs_equal
from src.trading_obj_utils import filter_to_realistic_positions

alpaca_api = tradeapi.REST(
    ALP_KEY_ID,
    ALP_SECRET_KEY,
    ALP_ENDPOINT,
    'v2')

logger = setup_logging("alpaca_cli.log")

# We'll store strategy usage in a persistent shelf
positions_shelf = FlatShelf("positions_shelf.json")


def set_strategy_for_symbol(symbol: str, strategy: str) -> None:
    """Record that a symbol is traded under the given strategy for today's date."""
    day_key = datetime.now().strftime('%Y-%m-%d')
    shelf_key = f"{symbol}-{day_key}"
    positions_shelf[shelf_key] = strategy
    # positions_shelf.commit()


def get_strategy_for_symbol(symbol: str) -> str:
    """Retrieve the strategy for a symbol for today's date, if any."""
    day_key = datetime.now().strftime('%Y-%m-%d')
    # Reload the shelf to avoid race conditions
    positions_shelf.load()
    shelf_key = f"{symbol}-{day_key}"
    return positions_shelf.get(shelf_key, None)


def main(command: str, pair: Optional[str], side: Optional[str] = "buy"):
    """
    cancel_all_orders - cancel all orders

    close_all_positions - close all positions at near market price

    close_position_violently - close position violently

    backout_near_market BTCUSD backout of usd locking to market sell price

    ramp_into_position BTCUSD buy - ramp into a position over time

    show_account - display account summary, positions, and orders

    :param pair: e.g. BTCUSD
    :param command:
    :param side: buy or sell (default: buy)
    :return:
    """
    if command == 'close_all_positions':
        close_all_positions()
    elif command == 'violently_close_all_positions':
        violently_close_all_positions()
    elif command == 'cancel_all_orders':
        alpaca_wrapper.cancel_all_orders()
    elif command == "backout_near_market":
        # loop around until the order is closed at market
        now = datetime.now()
        backout_near_market(pair, start_time=now)
    elif command == "ramp_into_position":
        now = datetime.now()
        ramp_into_position(pair, side, start_time=now)
    elif command == "close_position_at_takeprofit":
        close_position_at_takeprofit(pair, float(side))  # Use side param as target price
    elif command == 'show_account':
        show_account()


client = StockHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)


def backout_near_market(
    pair,
    start_time=None,
    ramp_minutes=15,
    market_after=15,
    sleep_interval=90,
):
    """Back out of an open position by progressively crossing the market.

    The function starts with a limit order slightly favourable to the
    current price and linearly ramps to the opposite side of the spread
    over ``ramp_minutes``.  If the position is still open after
    ``market_after`` minutes, a market order is sent to guarantee the
    exit.

    Args:
        pair: The trading pair symbol, e.g. ``"META"``.
        start_time: ``datetime`` the ramp started. ``None`` means now.
        ramp_minutes: Minutes to complete the limit order ramp.
        market_after: Minutes before switching to a market order.
        sleep_interval: Seconds to wait between iterations.
    """
    if start_time is None:
        start_time = datetime.now()

    retries = 0
    max_retries = 5

    while True:
        try:
            all_positions = alpaca_wrapper.get_all_positions()
            logger.info(f"Retrieved {len(all_positions)} total positions")

            if len(all_positions) == 0:
                logger.info("no positions found, exiting")
                break

            positions = filter_to_realistic_positions(all_positions)
            logger.info(f"After filtering, {len(positions)} positions remain")

            # cancel all orders of pair
            orders = alpaca_wrapper.get_open_orders()
            logger.info(f"Found {len(orders)} open orders")

            for order in orders:
                if hasattr(order, 'symbol') and pairs_equal(order.symbol, pair):
                    logger.info(f"Cancelling order for {pair}")
                    alpaca_wrapper.cancel_order(order)
                    sleep(1)
                    break

            found_position = False
            for position in positions:
                if hasattr(position, 'symbol') and pairs_equal(position.symbol, pair):
                    logger.info(f"Found matching position for {pair}")
                    is_long = hasattr(position, 'side') and position.side == 'long'

                    # Initial and final offsets from market price. Start slightly
                    # favourable, then cross to the other side over ``ramp_minutes``.
                    pct_offset = 0.003 if is_long else -0.003  # 0.3% away
                    pct_final_offset = -0.02 if is_long else 0.02  # 2% past market

                    minutes_since_start = (datetime.now() - start_time).seconds // 60
                    progress = min(minutes_since_start / ramp_minutes, 1.0)
                    if minutes_since_start >= market_after:
                        logger.info("Switching to market order to guarantee close")
                        succeeded = alpaca_wrapper.close_position_violently(position)
                        found_position = True
                        if not succeeded:
                            logger.info("Market order failed, will retry after delay")
                            retries += 1
                            if retries >= max_retries:
                                logger.error("Max retries reached, exiting")
                                return False
                            sleep(60)
                            continue
                        break
                    elif minutes_since_start >= ramp_minutes:
                        # After ramp period, set price well beyond market to guarantee fill
                        pct_above_market = pct_final_offset
                    else:
                        # During ramp period - linear progression from start to final offset
                        pct_above_market = pct_offset + (pct_final_offset - pct_offset) * progress

                    logger.info(f"Position side: {'long' if is_long else 'short'}, "
                              f"pct_above_market: {pct_above_market:.4f}, "
                              f"minutes_since_start: {minutes_since_start}, "
                              f"progress: {progress:.2f}")

                    try:
                        succeeded = alpaca_wrapper.close_position_near_market(position,
                                                                            pct_above_market=pct_above_market)
                        found_position = True
                        if not succeeded:
                            logger.info("failed to close position, will retry after delay")
                            retries += 1
                            if retries >= max_retries:
                                logger.error("Max retries reached, exiting")
                                return False
                            sleep(60)
                            continue
                    except Exception as e:
                        logger.error(f"Error closing position: {e}")
                        retries += 1
                        if retries >= max_retries:
                            logger.error("Max retries reached, exiting")
                            return False
                        sleep(60)
                        continue

            if not found_position:
                logger.info(f"no position found or error closing for {pair}")
                return True

            retries = 0
            sleep(sleep_interval)  # configurable retry interval

        except Exception as e:
            logger.error(f"Error in backout_near_market: {e}")
            retries += 1
            if retries >= max_retries:
                logger.error("Max retries reached, exiting")
                return False
            sleep(60)


def close_all_positions():
    positions = alpaca_wrapper.get_all_positions()

    for position in positions:
        if not hasattr(position, 'symbol'):
            continue

        symbol = position.symbol

        # get latest data then bid/ask
        download_exchange_latest_data(client, symbol)
        bid = get_bid(symbol)
        ask = get_ask(symbol)

        current_price = ask if hasattr(position, 'side') and position.side == 'long' else bid
        # close a long with the ask price
        # close a short with the bid price
        # get bid/ask
        # get current price
        alpaca_wrapper.close_position_at_almost_current_price(
            position, {
                'close_last_price_minute': current_price
            }
        )
        # alpaca_order_stock(position.symbol, position.qty)


def violently_close_all_positions():
    positions = alpaca_wrapper.get_all_positions()
    for position in positions:
        alpaca_wrapper.close_position_violently(position)


def ramp_into_position(pair, side, start_time=None):
    """
    Ramp into a position with different strategies for crypto vs stocks:
    - Crypto: Start slightly worse than market price, ramp to opposite side over 1 hour
    - Stocks: More aggressive pricing starting at market, ramp over 1 hour
    """
    if pair in crypto_symbols and side.lower() == "sell":
        logger.error(f"Cannot short crypto {pair}")
        return False

    if start_time is None:
        start_time = datetime.now()

    retries = 0
    max_retries = 5
    linear_ramp = 60  # 1 hour ramp for both crypto and stocks

    while True:
        try:
            all_positions = alpaca_wrapper.get_all_positions()
            positions = filter_to_realistic_positions(all_positions)

            # First check if we already have the position
            for position in positions:
                if hasattr(position, 'symbol') and pairs_equal(position.symbol, pair):
                    logger.info(f"Position already exists for {pair}")
                    return True

            # Cancel orders with retry logic
            cancel_attempts = 0
            max_cancel_attempts = 3
            orders_cancelled = False

            while cancel_attempts < max_cancel_attempts:
                try:
                    logger.info(f"Attempting to cancel orders for {pair}...")
                    # Get all open orders
                    orders = alpaca_wrapper.get_open_orders()
                    pair_orders = [order for order in orders if
                                   hasattr(order, 'symbol') and pairs_equal(order.symbol, pair)]

                    if not pair_orders:
                        orders_cancelled = True
                        logger.info(f"No existing orders found for {pair}")
                        break

                    # Cancel only orders for this pair
                    for order in pair_orders:
                        alpaca_wrapper.cancel_order(order)
                        sleep(1)  # Small delay between cancellations

                    # Verify cancellations
                    sleep(3)  # Let cancellations propagate
                    orders = alpaca_wrapper.get_open_orders()
                    remaining_orders = [order for order in orders if
                                        hasattr(order, 'symbol') and pairs_equal(order.symbol, pair)]

                    if not remaining_orders:
                        orders_cancelled = True
                        logger.info(f"All orders for {pair} successfully cancelled")
                        break
                    else:
                        logger.info(f"Found {len(remaining_orders)} remaining orders for {pair}, retrying cancellation")

                    cancel_attempts += 1
                    if not orders_cancelled:
                        sleep(5)  # Wait before retry

                except Exception as e:
                    logger.error(f"Error during order cancellation: {e}")
                    cancel_attempts += 1
                    sleep(5)

            if not orders_cancelled:
                logger.error("Failed to cancel orders after maximum attempts")
                retries += 1
                if retries >= max_retries:
                    logger.error("Max retries reached, exiting")
                    return False
                sleep(30)
                continue

            # Get current market prices
            try:
                download_exchange_latest_data(client, pair)
                bid_price = get_bid(pair)
                ask_price = get_ask(pair)

                if bid_price is None or ask_price is None:
                    logger.error(f"Failed to get bid/ask prices for {pair}")
                    retries += 1
                    if retries >= max_retries:
                        return False
                    sleep(30)
                    continue

                minutes_since_start = (datetime.now() - start_time).seconds // 60

                # Calculate the price to place the order
                if pair in crypto_symbols:
                    # For crypto, start slightly worse than market and slowly move to other side
                    offset = 0.0004  # 0.04% initial offset from market
                    if side == "buy":
                        if minutes_since_start >= linear_ramp:
                            order_price = ask_price  # End at ask
                        else:
                            # Start slightly below bid, move to ask
                            progress = minutes_since_start / linear_ramp
                            start_price = bid_price * (1 - offset)  # Start worse than bid
                            price_range = ask_price - start_price
                            order_price = start_price + (price_range * progress)
                    else:  # sell
                        if minutes_since_start >= linear_ramp:
                            order_price = bid_price  # End at bid
                        else:
                            # Start slightly above ask, move to bid
                            progress = minutes_since_start / linear_ramp
                            start_price = ask_price * (1 + offset)  # Start worse than ask
                            price_range = bid_price - start_price
                            order_price = start_price + (price_range * progress)

                    logger.info(f"Crypto order: Starting at {'below bid' if side == 'buy' else 'above ask'}, "
                                f"progress {progress:.2%}, price {order_price:.2f}")
                else:
                    # For stocks, be more aggressive
                    if minutes_since_start >= linear_ramp:
                        order_price = ask_price if side == "buy" else bid_price
                    else:
                        # Start at market and move slightly away
                        progress = minutes_since_start / linear_ramp
                        if side == "buy":
                            price_range = ask_price - bid_price
                            order_price = bid_price + (price_range * progress)
                        else:
                            price_range = ask_price - bid_price
                            order_price = ask_price - (price_range * progress)

                # Calculate position size
                buying_power = alpaca_wrapper.cash
                qty = 0.5 * buying_power / order_price
                qty = math.floor(qty * 1000) / 1000.0  # Round down to 3 decimal places

                if pair not in crypto_symbols:
                    qty = math.floor(qty)  # Round down to whole number for stocks

                if qty <= 0:
                    logger.error(f"Calculated qty {qty} is invalid")
                    return False

                logger.info(f"Attempting to place order: {pair} {side} {qty} @ {order_price}")

                # Place the order with error handling
                succeeded = alpaca_wrapper.open_order_at_price_or_all(pair, qty, side, order_price)
                if not succeeded:
                    logger.info("Failed to open position, will retry after delay")
                    retries += 1
                    if retries >= max_retries:
                        logger.error("Max retries reached, exiting")
                        return False
                    sleep(60)
                    continue

                # Reset retries on successful order placement
                retries = 0

                # Longer sleep for crypto to reduce API calls
                sleep_time = 5 * 60 if pair in crypto_symbols else 2 * 60
                sleep(sleep_time)

            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error during order placement: {e}")
                retries += 1
                if retries >= max_retries:
                    logger.error("Max retries reached, exiting")
                    return False
                sleep(60)
                continue

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error in ramp_into_position main loop: {e}")
            retries += 1
            if retries >= max_retries:
                logger.error("Max retries reached, exiting")
                return False
            sleep(60)


def show_account():
    """Display account summary including positions, orders and market status"""
    # Get market clock using wrapper
    clock = alpaca_wrapper.get_clock()

    # Convert times to NZDT and EDT
    nz_tz = pytz.timezone('Pacific/Auckland')
    edt_tz = pytz.timezone('America/New_York')

    current_time_nz = datetime.now(timezone.utc).astimezone(nz_tz)
    current_time_edt = datetime.now(timezone.utc).astimezone(edt_tz)

    # Print market status and times
    logger.info("\n=== Market Status ===")
    logger.info(f"Market is {'OPEN' if clock.is_open else 'CLOSED'}")
    logger.info(f"Current time (NZDT): {current_time_nz.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    logger.info(f"Current time (EDT): {current_time_edt.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    # Get account info
    logger.info("\n=== Account Summary ===")
    logger.info(f"Equity: ${alpaca_wrapper.equity:,.2f}")
    logger.info(f"Cash: ${alpaca_wrapper.cash:,.2f}")
    logger.info(f"Buying Power: ${alpaca_wrapper.total_buying_power:,.2f}")

    # Get and display positions
    positions = alpaca_wrapper.get_all_positions()
    logger.info("\n=== Open Positions ===")
    if not positions:
        logger.info("No open positions")
    else:
        for pos in positions:
            if hasattr(pos, 'symbol') and hasattr(pos, 'qty') and hasattr(pos, 'current_price'):
                side = "LONG" if hasattr(pos, 'side') and pos.side == 'long' else "SHORT"
                logger.info(f"{pos.symbol}: {side} {pos.qty} shares @ ${float(pos.current_price):,.2f}")

    # Get and display orders
    orders = alpaca_wrapper.get_open_orders()
    logger.info("\n=== Open Orders ===")
    if not orders:
        logger.info("No open orders")
    else:
        for order in orders:
            if hasattr(order, 'symbol') and hasattr(order, 'qty'):
                price_str = f"@ ${float(order.limit_price):,.2f}" if hasattr(order, 'limit_price') else "(market)"
                logger.info(f"{order.symbol}: {order.side.upper()} {order.qty} {price_str}")


def close_position_at_takeprofit(pair: str, takeprofit_price: float, start_time=None):
    """
    Wait for up to 1 hour or 24 hours if symbol is under "highlow" strategy,
    then place a limit order to close that position at takeprofit_price.
    If no position is opened within the hour, or if something fails, exit.
    """
    from datetime import datetime
    from time import sleep

    if start_time is None:
        start_time = datetime.now()

    # Determine wait time by strategy
    strategy = get_strategy_for_symbol(pair)
    if strategy == "highlow":
        max_wait_minutes = 24 * 60
        logger.info(f"{pair} is traded with 'highlow' strategy, using 24-hour wait.")
    else:
        max_wait_minutes = 60  # default

    while True:
        elapsed_minutes = (datetime.now() - start_time).seconds // 60
        if elapsed_minutes >= max_wait_minutes:
            logger.error(f"Timed out waiting for position in {pair} under strategy={strategy}")
            return False

        all_positions = alpaca_wrapper.get_all_positions()
        positions = [p for p in all_positions if hasattr(p, 'symbol') and pairs_equal(p.symbol, pair)]
        if not positions:
            logger.info(f"No position for {pair} yet – waiting. Elapsed: {elapsed_minutes} min")
            sleep(30)
            continue

        # We have at least one matching position
        position = positions[0]
        logger.info(f"Position found for {pair}: side={position.side}, qty={position.qty}")

        # Cancel existing orders for this pair
        orders = alpaca_wrapper.get_open_orders()
        for order in orders:
            if hasattr(order, 'symbol') and pairs_equal(order.symbol, pair):
                logger.info(f"Cancelling order for {pair} before placing takeprofit limit")
                alpaca_wrapper.cancel_order(order)
                sleep(1)

        # Place the takeprofit order
        logger.info(f"Placing limit order to close {pair} at {takeprofit_price}")
        try:
            side = 'sell' if position.side == 'long' else 'buy'
            alpaca_wrapper.open_order_at_price(pair, position.qty, side, takeprofit_price)
            return True
        except Exception as e:
            logger.error(f"Failed to place takeprofit limit order: {e}")
            return False


if __name__ == "__main__":
    typer.run(main)
    # close_all_positions()
