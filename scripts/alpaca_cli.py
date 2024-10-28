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

    while True:
        all_positions = alpaca_wrapper.get_all_positions()
        # check if there are any all_positions open
        if len(all_positions) == 0:
            logger.info("no positions found, exiting")
            break
        positions = filter_to_realistic_positions(all_positions)

        # cancel all orders of pair as we are locking to sell at the market
        
        orders = alpaca_wrapper.get_open_orders()

        for order in orders:
            if order.symbol == pair:
                alpaca_wrapper.cancel_order(order)

                break
        found_position = False
        for position in positions:
            if position.symbol == pair:
                pct_above_market = 0.02
                linear_ramp  = 60
                minutes_since_start = (datetime.now() - start_time).seconds // 60
                if minutes_since_start >= linear_ramp:
                    pct_above_market = -0.02
                else:
                    pct_above_market = pct_above_market - (0.04 * minutes_since_start / linear_ramp)

                logger.info(f"pct_above_market: {pct_above_market}")
                succeeded = alpaca_wrapper.close_position_near_market(position, pct_above_market=pct_above_market)
                found_position = True
                if not succeeded:
                    ## todo wait untill other time when market is open again to cancel.
                    logger.info("failed to close a position, stopping as we are potentially at market close?")
                    return False
        if not found_position:
            logger.info(f"no position found for {pair}")
            return True

        # cancel all order for produce
        # alpaca_wrapper.cancel_order_at_market(pair)
        sleep(60*3) # retry every 3 mins - leave orders open that long to make sure they have a chance of execution


def close_all_positions():
    positions = alpaca_wrapper.get_all_positions()

    for position in positions:
        symbol = position.symbol

        # get latest data then bid/ask
        download_exchange_latest_data(client, symbol)
        bid = get_bid(symbol)
        ask = get_ask(symbol)


        current_price = ask if position.side == 'long' else bid
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
    Ramp into a position - linear .01pct below to market price within 60min
    """
    if start_time is None:
        start_time = datetime.now()

    while True:
        all_positions = alpaca_wrapper.get_all_positions()
        positions = filter_to_realistic_positions(all_positions)

        # Cancel all orders of pair as we are ramping into the position
        # couldnt find another way so only supports buying one at a time rn
        logger.info("cancelling all orders")
        success = alpaca_wrapper.cancel_all_orders()
        if not success:
            logger.info("failed to cancel all orders, stopping as we are potentially at market close?")


        orders = alpaca_wrapper.get_open_orders()
        # print all order symbols
        for order in orders:
            logger.info(f"order: {order.symbol}")
        for order in orders:
            if order.symbol == pair:
                alpaca_wrapper.cancel_order(order)
                break

        found_position = False
        for position in positions:
            if position.symbol == pair:
                found_position = True
                logger.info(f"Position already exists for {pair}")
                return True

        if not found_position:
            linear_ramp = 60
            minutes_since_start = (datetime.now() - start_time).seconds // 60

            # Get current market prices
            download_exchange_latest_data(client, pair)
            bid_price = get_bid(pair)
            ask_price = get_ask(pair)

            if bid_price is None or ask_price is None:
                logger.error(f"Failed to get bid/ask prices for {pair}")
                return False

            # Calculate the price to place the order
            if side == "buy":
                start_price, end_price = bid_price, ask_price
            else:
                start_price, end_price = ask_price, bid_price

            if minutes_since_start >= linear_ramp:
                order_price = end_price
            else:
                price_range = end_price - start_price
                progress = minutes_since_start / linear_ramp
                order_price = start_price + (price_range * progress)

            # Calculate the qty based on 50% of buying power
            buying_power = alpaca_wrapper.cash
            qty = 0.5 * buying_power / order_price
            qty = math.floor(qty * 1000) / 1000.0  # Round down to 3 decimal places

            # {"code":40310000,"message":"fractional trading is disabled for this account"}
            # round down for now to no dp
            if pair not in crypto_symbols:
                qty = math.floor(qty)


            logger.info(f"qty: {qty}")
            logger.info(f"order_price: {order_price}")
            
            # Place the order
            succeeded = alpaca_wrapper.open_order_at_price(pair, qty, side, order_price)
            if not succeeded:
                logger.info("Failed to open a position, stopping as we are potentially at market close?")
                # return False

        sleep(60 * 2)

if __name__ == "__main__":
    typer.run(main)
    # close_all_positions()
