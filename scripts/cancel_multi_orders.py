from time import sleep

from loguru import logger

from alpaca_wrapper import get_open_orders, cancel_order

orders = get_open_orders()

print(orders)


def cancel_multi_orders(orders):
    """
    cancels any old orders
    if there are duplicate orders for a pair
    cancel the older order
    :param orders:
    :return:
    """
    duplicate_symbols = set()
    current_symbols = set()
    for order in orders:
        if order.symbol in current_symbols:
            duplicate_symbols.add(order.symbol)
        current_symbols.add(order.symbol)
    for symbol in duplicate_symbols:
        symbol_orders = [order for order in orders if order.symbol == symbol]
        symbol_orders.sort(key=lambda x: x.created_at)
        for order in symbol_orders[:-1]:
            logger.info(f"canceling dupe order {order.id} for {order.symbol}")
            cancel_order(order)


while True:
    orders = get_open_orders()
    cancel_multi_orders(orders)
    # 5 min sleep
    sleep(5 * 60)
