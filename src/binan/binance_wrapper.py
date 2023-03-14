import math

from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
from loguru import logger

from env_real import BINANCE_API_KEY, BINANCE_SECRET
from stc.stock_utils import binance_remap_symbols

client = Client(BINANCE_API_KEY, BINANCE_SECRET)

crypto_symbols = [
    "BTCUSDT",
    "ETHUSDT",
    "LTCUSDT",
]


def create_order(symbol, side, quantity, price=None):
    order = client.create_order(
        symbol=symbol,
        side=side,
        type=Client.ORDER_TYPE_LIMIT,
        timeInForce=Client.TIME_IN_FORCE_GTC,
        quantity=quantity,
        price=price,
    )
    return order


def create_all_in_order(symbol, side, price=None):
    # get balance for SELL SIDE
    balances = get_account_balances()
    for balance in balances:
        if balance["asset"] == symbol[:3]:
            balance_sell = float(balance["free"])
        if balance["asset"] == symbol[3:]:
            balance_buy = float(balance["free"])
    if side == "SELL":
        quantity = balance_sell
    elif side == "BUY":
        quantity = balance_buy / price # both are in btc so not #balance_buy / price
    else:
        raise Exception("Invalid side")
    # round down to 3dp (for btc)
    quantity = math.floor(quantity * 1000) / 1000
    try:
        order = client.create_order(
            symbol=symbol,
            side=side,
            type=Client.ORDER_TYPE_LIMIT,
            timeInForce=Client.TIME_IN_FORCE_GTC,
            quantity=quantity,
            price=price,
        )
        logger.info(f"Created order on binance: {order}")
    except Exception as e:
        logger.error(e)
        logger.error(f"symbol {symbol}")
        logger.error(f"side {side}")
        logger.error(f"quantity {quantity}")
        logger.error(f"price {price}")
        raise e



def open_take_profit_position(position, row, price, qty):
    # entry_price = float(position.avg_entry_price)
    # current_price = row['close_last_price_minute']
    # current_symbol = row['symbol']
    try:
        mapped_symbol = binance_remap_symbols(position.symbol)
        if position.side == "long":
            create_all_in_order(mapped_symbol, "SELL", str(math.ceil(price)))
        else:
            create_all_in_order(mapped_symbol, "BUY", str(math.floor(price)))
    except Exception as e:
        logger.error(e) # can be because theres a sell order already which is still relevant
        # close all positions? perhaps not
        return None
    return True

def close_position_at_current_price(position, row):
    if not row["close_last_price_minute"]:
        logger.info(f"nan price - for {position.symbol} market likely closed")
        return False
    try:
        if position.side == "long":
            create_all_in_order(binance_remap_symbols(position.symbol), "SELL", row["close_last_price_minute"])

        else:
            create_all_in_order(binance_remap_symbols(position.symbol), "BUY", str(math.floor(float(row["close_last_price_minute"]))))
    except Exception as e:
        logger.error(e) # cant convert nan to integer because market is closed for stocks
        # Out of range float values are not JSON compliant
        # could be because theres no minute data /trying to close at when market isn't open (might as well err/do nothing)
        # close all positions? perhaps not
        return None

def cancel_all_orders():
    for symbol in crypto_symbols:
        orders = get_all_orders(symbol)
        for order in orders:
            if order["status"] == "CANCELED" or order["status"] == "FILLED":
                continue
            try:
                client.cancel_order(symbol=order["symbol"], orderId=order["orderId"])
            except Exception as e:
                print(e)
                logger.error(e)


def get_all_orders(symbol):
    orders = client.get_all_orders(symbol=symbol)
    return orders


def get_account_balances():
    balances = client.get_account()["balances"]
    return balances
