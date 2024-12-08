from datetime import datetime, timezone
import math
from time import sleep
from typing import Optional

import alpaca_trade_api as tradeapi
import typer
from alpaca.data import StockHistoricalDataClient
from src.logging_utils import setup_logging

import alpaca_wrapper
from data_curate_daily import download_exchange_latest_data, get_bid, get_ask
from env_real import ALP_KEY_ID, ALP_SECRET_KEY, ALP_ENDPOINT, ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD
from src.stock_utils import pairs_equal
from src.trading_obj_utils import filter_to_realistic_positions

from src.fixtures import crypto_symbols

import pytz
from alpaca.trading.client import TradingClient


alpaca_api = tradeapi.REST(
    ALP_KEY_ID,
    ALP_SECRET_KEY,
    ALP_ENDPOINT,
    'v2')

logger = setup_logging("alpaca_cli.log")

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
    elif command == 'show_account':
        show_account()



client = StockHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)

def backout_near_market(pair, start_time=None):
    """
    backout at market - linear ramp towards market price within 30min
    For long positions: Sell at progressively lower prices (start above bid, ramp down)
    For short positions: Buy at progressively higher prices (start below ask, ramp up)
    """
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
                    
                    # Initial offset from market (0.015 = 1.5%)
                    pct_offset = 0.010
                    linear_ramp = 30  # 30 minute ramp
                    
                    minutes_since_start = (datetime.now() - start_time).seconds // 60
                    if minutes_since_start >= linear_ramp:
                        # After ramp period, set aggressive price
                        pct_above_market = -pct_offset 
                    else:
                        # During ramp period
                        progress = minutes_since_start / linear_ramp
                        pct_above_market = pct_offset - (2 * pct_offset * progress)

                    logger.info(f"Position side: {'long' if is_long else 'short'}, "
                              f"pct_above_market: {pct_above_market}, "
                              f"minutes_since_start: {minutes_since_start}, "
                              f"progress: {progress if minutes_since_start < linear_ramp else 1.0}")
                    
                    try:
                        succeeded = alpaca_wrapper.close_position_near_market(position, pct_above_market=pct_above_market)
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
            sleep(60*3)  # retry every 3 mins

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
                    pair_orders = [order for order in orders if hasattr(order, 'symbol') and pairs_equal(order.symbol, pair)]
                    
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
                    remaining_orders = [order for order in orders if hasattr(order, 'symbol') and pairs_equal(order.symbol, pair)]
                    
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
                logger.error(f"Error during order placement: {e}")
                retries += 1
                if retries >= max_retries:
                    logger.error("Max retries reached, exiting")
                    return False
                sleep(60)
                continue

        except Exception as e:
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

if __name__ == "__main__":
    typer.run(main)
    # close_all_positions()
