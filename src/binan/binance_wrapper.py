from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
from loguru import logger

from env_real import BINANCE_API_KEY, BINANCE_SECRET

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
        quantity = balance_sell # both are in btc so not #balance_buy / price

    else:
        raise Exception("Invalid side")
    # round down to 3dp (for btc)
    quantity = int(quantity * 1000) / 1000
    order = client.create_order(
        symbol=symbol,
        side=side,
        type=Client.ORDER_TYPE_LIMIT,
        timeInForce=Client.TIME_IN_FORCE_GTC,
        quantity=quantity,
        price=price,
    )


def cancel_all_orders():
    for symbol in crypto_symbols:
        orders = get_all_orders(symbol)
        for order in orders:
            if order["status"] == "CANCELLED":
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
