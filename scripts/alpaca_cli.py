from datetime import datetime
import math
from time import sleep
from typing import Optional

import alpaca_trade_api as tradeapi
import typer
from alpaca.data import StockHistoricalDataClient
from loguru import logger

import alpaca_wrapper
from data_curate_daily import download_exchange_latest_data, get_bid, get_ask
from env_real import ALP_KEY_ID, ALP_SECRET_KEY, ALP_ENDPOINT, ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD
from src.trading_obj_utils import filter_to_realistic_positions

from src.fixtures import crypto_symbols

alpaca_api = tradeapi.REST(
    ALP_KEY_ID,
    ALP_SECRET_KEY,
    ALP_ENDPOINT,
    'v2')


def main(command: str, pair: Optional[str], side: Optional[str] = "buy"):
    """
    cancel_all_orders - cancel all orders

    close_all_positions - close all positions at near market price

    close_position_violently - close position violently

    backout_near_market BTCUSD backout of usd locking to market sell price

    ramp_into_position BTCUSD buy - ramp into a position over time

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



client = StockHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)

def backout_near_market(pair, start_time=None):
    """
    backout at market - linear .01pct above to market price within 20min
    """
    retries = 0
    max_retries = 5

    while True:
        try:
            all_positions = alpaca_wrapper.get_all_positions()
            # check if there are any all_positions open
            if len(all_positions) == 0:
                logger.info("no positions found, exiting")
                break
            positions = filter_to_realistic_positions(all_positions)

            # cancel all orders of pair as we are locking to sell at the market
            orders = alpaca_wrapper.get_open_orders()

            for order in orders:
                if hasattr(order, 'symbol') and order.symbol == pair:
                    alpaca_wrapper.cancel_order(order)
                    # Add small delay after canceling to let it propagate
                    sleep(1)
                    break

            found_position = False
            for position in positions:
                if hasattr(position, 'symbol') and position.symbol == pair:
                    pct_above_market = 0.02
                    linear_ramp = 60
                    minutes_since_start = (datetime.now() - start_time).seconds // 60
                    if minutes_since_start >= linear_ramp:
                        pct_above_market = -0.02
                    else:
                        pct_above_market = pct_above_market - (0.04 * minutes_since_start / linear_ramp)

                    logger.info(f"pct_above_market: {pct_above_market}")
                    try:
                        succeeded = alpaca_wrapper.close_position_near_market(position, pct_above_market=pct_above_market)
                        found_position = True
                        if not succeeded:
                            logger.info("failed to close a position, will retry after delay")
                            retries += 1
                            if retries >= max_retries:
                                logger.error("Max retries reached, exiting")
                                return False
                            sleep(60)  # Wait a minute before retrying
                            continue
                    except Exception as e:
                        logger.error(f"Error closing position: {e}")
                        retries += 1
                        if retries >= max_retries:
                            logger.error("Max retries reached, exiting")
                            return False
                        sleep(60)  # Wait a minute before retrying
                        continue

            if not found_position:
                logger.info(f"no position found for {pair}")
                return True

            # Reset retries on successful iteration
            retries = 0
            sleep(60*3)  # retry every 3 mins

        except Exception as e:
            logger.error(f"Error in backout_near_market: {e}")
            retries += 1
            if retries >= max_retries:
                logger.error("Max retries reached, exiting")
                return False
            sleep(60)  # Wait a minute before retrying


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
    - Crypto: Longer ramp (4 hours) with smaller price adjustments to ensure maker orders
    - Stocks: Original 60min ramp with more aggressive pricing
    """
    if pair in crypto_symbols and side.lower() == "sell":
        logger.error(f"Cannot short crypto {pair}")
        return False

    if start_time is None:
        start_time = datetime.now()

    retries = 0
    max_retries = 5
    linear_ramp = 240 if pair in crypto_symbols else 60  # 4 hours for crypto, 1 hour for stocks
    
    while True:
        try:
            all_positions = alpaca_wrapper.get_all_positions()
            positions = filter_to_realistic_positions(all_positions)

            # First check if we already have the position
            for position in positions:
                if hasattr(position, 'symbol') and position.symbol == pair:
                    logger.info(f"Position already exists for {pair}")
                    return True

            # Cancel orders with retry logic
            cancel_attempts = 0
            max_cancel_attempts = 3
            orders_cancelled = False
            
            while cancel_attempts < max_cancel_attempts:
                try:
                    logger.info("Attempting to cancel all orders...")
                    success = alpaca_wrapper.cancel_all_orders()
                    if success:
                        # Add delay to let cancellations propagate
                        sleep(3)
                        
                        # Verify cancellations
                        orders = alpaca_wrapper.get_open_orders()
                        remaining_orders = [order for order in orders if hasattr(order, 'symbol') and order.symbol == pair]
                        
                        if not remaining_orders:
                            orders_cancelled = True
                            logger.info("All relevant orders successfully cancelled")
                            break
                        else:
                            logger.info(f"Found {len(remaining_orders)} remaining orders for {pair}, retrying cancellation")
                            # Try to cancel specific orders
                            for order in remaining_orders:
                                alpaca_wrapper.cancel_order(order)
                                sleep(1)
                    
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

                # Calculate the price to place the order
                if side == "buy":
                    start_price, end_price = bid_price, ask_price
                else:
                    start_price, end_price = ask_price, bid_price

                minutes_since_start = (datetime.now() - start_time).seconds // 60
                
                if minutes_since_start >= linear_ramp:
                    order_price = end_price
                else:
                    price_range = end_price - start_price
                    progress = minutes_since_start / linear_ramp
                    
                    # Less aggressive price adjustment for crypto to ensure maker orders
                    if pair in crypto_symbols:
                        # For crypto, stay closer to the maker side
                        if side == "buy":
                            max_progress = 0.3  # Only move 30% of the way to the ask
                        else:
                            max_progress = 0.3  # Only move 30% of the way to the bid
                        progress = progress * max_progress
                    
                    order_price = start_price + (price_range * progress)

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
                succeeded = alpaca_wrapper.open_order_at_price(pair, qty, side, order_price)
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

if __name__ == "__main__":
    typer.run(main)
    # close_all_positions()
