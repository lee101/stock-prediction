""" fastapi server supporting

adding an order for a stock pair/side/price

polling untill the market is ready to accept the order then making a market order
cancelling all orders
cancelling an order
getting the current orders

"""


from loguru import logger

import datetime
import pytz
import time
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

import threading

from alpaca_wrapper import latest_data
from src.crypto_loop import crypto_order_loop

crypto_symbol_to_orders = {

}
app = FastAPI()

symbols = [
    'BTCUSD',
    'ETHUSD',
    'LTCUSD',
]
def crypto_order_loop():
    while True:
        try:
            # get all data for symbols
            for symbol in symbols:
                very_latest_data = latest_data(symbol)
                order = crypto_symbol_to_orders.get(symbol)
                if order:
                    if order.side == "buy":
                        if float(very_latest_data.ask_price) < order.price:
                            logger.info(f"buying {symbol} at {order.price}")

            # check if market closed
            ask_price = float(very_latest_data.ask_price)
            bid_price = float(very_latest_data.bid_price)
            if bid_price != 0 and ask_price != 0:
                latest_data_dl["close"] = (bid_price + ask_price) / 2.
                spread = ask_price / bid_price
                logger.info(f"{symbol} spread {spread}")
                spreads[symbol] = spread
                bids[symbol] = bid_price
                asks[symbol] = ask_price
            for symbol, orders in crypto_symbol_to_orders.items():
                for order in orders:
                    order.check_order()
        except Exception as e:
            logger.error(e)
        time.sleep(20)

threading.daemon = True
threading.start_new_thread(crypto_order_loop, ())
@route("/api/v1/stock_order", methods=["POST"])
def stock_order():
