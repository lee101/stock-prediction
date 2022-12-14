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
from pathlib import Path
from threading import Thread

from fastapi import FastAPI
from loguru import logger
from starlette.responses import JSONResponse
from pydantic import BaseModel

from alpaca_wrapper import latest_data, open_market_order_violently
from jsonshelve import FlatShelf
from src.binan.binance_wrapper import create_all_in_order, cancel_all_orders
from stc.stock_utils import unmap_symbols


data_dir = Path(__file__).parent.parent / 'data'

dynamic_config_ = data_dir / "dynamic_config"
dynamic_config_.mkdir(exist_ok=True, parents=True)

crypto_symbol_to_order = FlatShelf(str(dynamic_config_ / f"crypto_symbol_to_order.db.json"))

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
    # convert to USDT - assume crypto
    usdt_symbol = symbol[:3] + "USDT"
    # order all on binance
    if order.qty > 0.03 and symbol == "BTCUSD": # going all in on a bitcoin side
        cancel_all_orders()
        # replicate order to binance account for free trading on btc
        create_all_in_order(usdt_symbol, order.side.upper(), order.price)


@app.get("/api/v1/stock_orders")
def stock_orders():
    return JSONResponse(crypto_symbol_to_order.__dict__)


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
