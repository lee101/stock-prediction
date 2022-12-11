import datetime

import requests
from alpaca.trading import Order


def submit_order(order_data):
    symbol = order_data.symbol
    side = order_data.side
    price = order_data.limit_price
    qty = order_data.qty
    return stock_order(symbol, side, price, qty)


def load_iso_format(dateformat_string):
    return datetime.datetime.strptime(dateformat_string, "%Y-%m-%dT%H:%M:%S.%f")


class FakeOrder:
    def __init__(self):
        self.symbol = None
        self.side = None
        self.limit_price = None
        self.qty = None
        self.created_at = None

    def __repr__(self):
        return f"{self.side} {self.qty} {self.symbol} at {self.limit_price} on {self.created_at}"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if isinstance(other, Order):
            return self.symbol == other.symbol and self.side == other.side and self.limit_price == other.limit_price and self.qty == other.qty
        return False

    def __hash__(self):
        return hash((self.symbol, self.side, self.limit_price, self.qty))


def get_orders():
    response = stock_orders()
    json = response.json()
    orders = []
    for result in json.keys():
        o = FakeOrder()
        json_order = json[result]
        o.symbol = json_order["symbol"]
        o.side = json_order["side"]
        o.limit_price = json_order["price"]
        o.qty = json_order["qty"]
        o.created_at = load_iso_format(json_order["created_at"])
        orders.append(o)

    return orders


def stock_order(symbol, side, price, qty):
    url = "http://localhost:5050/api/v1/stock_order"
    data = {
        "symbol": symbol,
        "side": side,
        "price": price,
        "qty": qty,
    }
    response = requests.post(url, json=data)
    return response


def stock_orders():
    url = "http://localhost:5050/api/v1/stock_orders"
    response = requests.get(url)
    return response


def get_stock_order(symbol):
    url = f"http://localhost:5050/api/v1/stock_order/{symbol}"
    response = requests.get(url)
    return response


def delete_stock_order(symbol):
    url = f"http://localhost:5050/api/v1/stock_order/{symbol}"
    response = requests.delete(url)
    return response


def delete_stock_orders():
    url = f"http://localhost:5050/api/v1/stock_order/cancel_all"
    response = requests.delete(url)
    return response
