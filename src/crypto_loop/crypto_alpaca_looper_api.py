import datetime
from typing import Optional

import requests
from alpaca.trading import Order

from src.logging_utils import setup_logging

logger = setup_logging("crypto_alpaca_looper_api.log")


def submit_order(order_data):
    logger.info(f"Preparing to submit order: {order_data}")
    symbol = order_data.symbol
    side = order_data.side
    price = order_data.limit_price
    qty = order_data.qty
    return stock_order(symbol, side, price, qty)


def load_iso_format(dateformat_string):
    return datetime.datetime.strptime(dateformat_string, "%Y-%m-%dT%H:%M:%S.%f")


class FakeOrder:
    def __init__(self):
        self.symbol: Optional[str] = None
        self.side: Optional[str] = None
        self.limit_price: Optional[str] = None # Alpaca API often uses string for price/qty
        self.qty: Optional[str] = None
        self.created_at: Optional[datetime.datetime] = None # Fixed type hint

    def __repr__(self):
        return f"{self.side} {self.qty} {self.symbol} at {self.limit_price} on {self.created_at}"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if isinstance(other, Order): # Should ideally also compare against FakeOrder if used interchangeably
            return self.symbol == other.symbol and self.side == other.side and self.limit_price == other.limit_price and self.qty == other.qty
        if isinstance(other, FakeOrder):
            return self.symbol == other.symbol and \
                   self.side == other.side and \
                   self.limit_price == other.limit_price and \
                   self.qty == other.qty and \
                   self.created_at == other.created_at # Consider how Nones are compared if that's valid
        return False

    def __hash__(self):
        return hash((self.symbol, self.side, self.limit_price, self.qty, self.created_at))


def get_orders():
    logger.info("Fetching current orders from crypto looper server.")
    response = stock_orders()
    orders = []
    if response is None:
        logger.error("Failed to get response from stock_orders a.k.a crypto_order_loop_server is down?")
        return orders # Return empty list if server call failed

    try:
        response_json = response.json()
        logger.debug(f"Raw orders response: {response_json}")
        server_data = response_json.get('data', {})
        for result_key in server_data.keys():
            o = FakeOrder()
            json_order_data = server_data[result_key]
            o.symbol = json_order_data.get("symbol")
            o.side = json_order_data.get("side")
            o.limit_price = json_order_data.get("price") # Assuming price is string
            o.qty = json_order_data.get("qty") # Assuming qty is string
            created_at_str = json_order_data.get("created_at")
            if created_at_str:
                try:
                    o.created_at = load_iso_format(created_at_str)
                except ValueError as e:
                    logger.error(f"Error parsing created_at string '{created_at_str}': {e}")
            orders.append(o)
        logger.info(f"Successfully fetched and parsed {len(orders)} orders.")
    except requests.exceptions.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response from server: {e}")
        if response: # Check again because it might have been None initially, though less likely here
             logger.error(f"Response text: {response.text}")
    except Exception as e:
        logger.error(f"Error processing orders response: {e}")
    return orders


def stock_order(symbol, side, price, qty):
    url = "http://localhost:5050/api/v1/stock_order"
    data = {
        "symbol": symbol,
        "side": side,
        "price": str(price), # Ensure price is string
        "qty": str(qty),     # Ensure qty is string
    }
    logger.info(f"Submitting stock order to {url} with data: {data}")
    try:
        response = requests.post(url, json=data)
        logger.info(f"Server response status: {response.status_code}, content: {response.text[:500] if response and response.text else 'N/A'}")
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response # Or response.json() if appropriate
    except requests.exceptions.RequestException as e:
        logger.error(f"Error submitting stock order to {url}: {e}")
        return None


def stock_orders():
    url = "http://localhost:5050/api/v1/stock_orders"
    logger.info(f"Fetching stock orders from {url}")
    try:
        response = requests.get(url)
        logger.info(f"Server response status: {response.status_code}, content: {response.text[:500] if response and response.text else 'N/A'}")
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching stock orders from {url}: {e}")
        return None # Or an empty response-like object


def get_stock_order(symbol):
    url = f"http://localhost:5050/api/v1/stock_order/{symbol}"
    logger.info(f"Fetching stock order for {symbol} from {url}")
    try:
        response = requests.get(url)
        logger.info(f"Server response status: {response.status_code}, content: {response.text[:500] if response and response.text else 'N/A'}")
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching stock order for {symbol} from {url}: {e}")
        return None


def delete_stock_order(symbol):
    url = f"http://localhost:5050/api/v1/stock_order/{symbol}"
    logger.info(f"Deleting stock order for {symbol} via {url}")
    try:
        response = requests.delete(url)
        logger.info(f"Server response status: {response.status_code}, content: {response.text[:500] if response and response.text else 'N/A'}")
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"Error deleting stock order for {symbol} via {url}: {e}")
        return None


def delete_stock_orders():
    url = f"http://localhost:5050/api/v1/stock_order/cancel_all"
    logger.info(f"Deleting all stock orders via {url}")
    try:
        response = requests.delete(url)
        logger.info(f"Server response status: {response.status_code}, content: {response.text[:500] if response and response.text else 'N/A'}")
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"Error deleting all stock orders via {url}: {e}")
        return None
