""" fastapi server supporting

adding an order for a stock pair/side/price

polling untill the market is ready to accept the order then making a market order
cancelling all orders
cancelling an order
getting the current orders

"""
from datetime import datetime
import json
import time
from threading import Thread

from fastapi import FastAPI
from loguru import logger
from starlette.responses import JSONResponse
from pydantic import BaseModel

from alpaca_wrapper import latest_data, open_market_order_violently
from stc.stock_utils import unmap_symbols

crypto_symbol_to_order = {}
app = FastAPI()

symbols = [
    "BTCUSD",
    "ETHUSD",
    "LTCUSD",
]


def crypto_order_loop():
    while True:
        try:
            # get all data for symbols
            for symbol in symbols:
                very_latest_data = latest_data(symbol)
                order = crypto_symbol_to_order.get(symbol)
                if order:
                    logger.info(f"order {order}")
                    if order['side'] == "buy":
                        if float(very_latest_data.ask_price) < order['price']:
                            logger.info(f"buying {symbol} at {order['price']}")
                            open_market_order_violently(symbol, order['qty'], "buy")
                            crypto_symbol_to_order[symbol] = None
                    elif order['side'] == "sell":
                        if float(very_latest_data.bid_price) > order['price']:
                            logger.info(f"selling {symbol} at {order['price']}")
                            open_market_order_violently(symbol, order['qty'], "sell")
                            crypto_symbol_to_order[symbol] = None
                    else:
                        logger.error(f"unknown side {order['side']}")
                        logger.error(f"order {order}")
        except Exception as e:
            logger.error(e)
        time.sleep(10)


thread_loop = Thread(target=crypto_order_loop).start()


class OrderRequest(BaseModel):
    symbol: str
    side: str
    price: float
    qty: float

@app.post("/api/v1/stock_order")
def stock_order(order: OrderRequest):
    symbol = unmap_symbols(order.symbol)
    crypto_symbol_to_order[symbol] = {
        "symbol": symbol,
        "side": order.side,
        "price": order.price,
        "qty": order.qty,
        "created_at": datetime.now().isoformat(),
    }


@app.get("/api/v1/stock_orders")
def stock_orders():
    return JSONResponse(crypto_symbol_to_order)


@app.get("/api/v1/stock_order/{symbol}")
def stock_order(symbol: str):
    symbol = unmap_symbols(symbol)
    return JSONResponse(crypto_symbol_to_order.get(symbol))


@app.delete("/api/v1/stock_order/{symbol}")
def delete_stock_order(symbol: str):
    symbol = unmap_symbols(symbol)
    crypto_symbol_to_order[symbol] = None


@app.get("/api/v1/stock_order/cancel_all")
def delete_stock_orders():
    for symbol in crypto_symbol_to_order:
        crypto_symbol_to_order[symbol] = None
