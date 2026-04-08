import datetime
import math
import os
import sys
from dataclasses import dataclass
from typing import Protocol, TypedDict
from urllib.parse import quote, urlparse

import requests

from src.logging_utils import setup_logging, get_log_filename

sys.modules.setdefault("crypto_loop.crypto_alpaca_looper_api", sys.modules[__name__])
sys.modules.setdefault("src.crypto_loop.crypto_alpaca_looper_api", sys.modules[__name__])

# Detect if we're in hourly mode based on TRADE_STATE_SUFFIX env var
_is_hourly = os.getenv("TRADE_STATE_SUFFIX", "") == "hourly"
logger = setup_logging(get_log_filename("crypto_alpaca_looper_api.log", is_hourly=_is_hourly))

_CRYPTO_LOOPER_API_BASE_URL_ENV_VAR = "CRYPTO_LOOPER_API_BASE_URL"
_CRYPTO_LOOPER_API_TIMEOUT_SECONDS_ENV_VAR = "CRYPTO_LOOPER_API_TIMEOUT_SECONDS"
_DEFAULT_CRYPTO_LOOPER_API_BASE_URL = "http://localhost:5050"
_DEFAULT_CRYPTO_LOOPER_API_TIMEOUT_SECONDS = 10.0
_CRYPTO_LOOPER_API_RESPONSE_LOG_PREVIEW_CHARS = 500
_CRYPTO_LOOPER_API_STOCK_ORDER_PATH = "/api/v1/stock_order"
_CRYPTO_LOOPER_API_STOCK_ORDERS_PATH = "/api/v1/stock_orders"
_CRYPTO_LOOPER_API_CANCEL_ALL_ORDERS_PATH = "/api/v1/stock_order/cancel_all"
_CRYPTO_LOOPER_API_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"
_CRYPTO_LOOPER_API_CONFIG_CACHE_KEY: tuple[str | None, str | None] | None = None
_CRYPTO_LOOPER_API_CONFIG_CACHE_VALUE: object | None = None


@dataclass(frozen=True)
class CryptoLooperApiConfig:
    base_url: str
    timeout_seconds: float


class _ResponseLike(Protocol):
    status_code: int
    text: str

    def raise_for_status(self) -> None: ...

    def json(self) -> object: ...


class _CryptoLooperOrderPayload(TypedDict, total=False):
    symbol: str
    side: str
    price: str
    qty: str
    created_at: str


class OrderSubmissionLike(Protocol):
    symbol: object
    side: object
    limit_price: object
    qty: object


def _order_submission_components(
    order_data: object,
) -> tuple[object, object, object, object] | None:
    required_attrs = ("symbol", "side", "limit_price", "qty")
    missing_attrs = [attr for attr in required_attrs if not hasattr(order_data, attr)]
    if missing_attrs:
        logger.error(
            "Invalid order submission payload; missing required attributes: "
            f"{', '.join(missing_attrs)}"
        )
        return None
    return (
        getattr(order_data, "symbol"),
        getattr(order_data, "side"),
        getattr(order_data, "limit_price"),
        getattr(order_data, "qty"),
    )


def submit_order(order_data: OrderSubmissionLike | object) -> requests.Response | None:
    logger.info(f"Preparing to submit order: {order_data}")
    components = _order_submission_components(order_data)
    if components is None:
        return None
    symbol, side, price, qty = components
    return stock_order(symbol, side, price, qty)


def load_iso_format(dateformat_string: str) -> datetime.datetime:
    return datetime.datetime.strptime(dateformat_string, _CRYPTO_LOOPER_API_TIMESTAMP_FORMAT)


def _crypto_looper_api_base_url() -> str:
    base_url = os.getenv(
        _CRYPTO_LOOPER_API_BASE_URL_ENV_VAR,
        _DEFAULT_CRYPTO_LOOPER_API_BASE_URL,
    ).strip()
    if not base_url:
        return _DEFAULT_CRYPTO_LOOPER_API_BASE_URL
    if "://" not in base_url:
        base_url = f"http://{base_url}"
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return _DEFAULT_CRYPTO_LOOPER_API_BASE_URL
    if parsed.username is not None or parsed.password is not None:
        return _DEFAULT_CRYPTO_LOOPER_API_BASE_URL
    return base_url.rstrip("/")


def _crypto_looper_api_url(path: str) -> str:
    return f"{_crypto_looper_api_base_url()}{path}"


def _crypto_looper_api_timeout_seconds() -> float:
    raw_timeout = os.getenv(
        _CRYPTO_LOOPER_API_TIMEOUT_SECONDS_ENV_VAR,
        str(_DEFAULT_CRYPTO_LOOPER_API_TIMEOUT_SECONDS),
    )
    try:
        timeout = float(raw_timeout.strip())
    except (AttributeError, ValueError):
        return _DEFAULT_CRYPTO_LOOPER_API_TIMEOUT_SECONDS
    if timeout <= 0 or not math.isfinite(timeout):
        return _DEFAULT_CRYPTO_LOOPER_API_TIMEOUT_SECONDS
    return timeout


def _crypto_looper_api_config_cache_key() -> tuple[str | None, str | None]:
    return (
        os.getenv(_CRYPTO_LOOPER_API_BASE_URL_ENV_VAR),
        os.getenv(_CRYPTO_LOOPER_API_TIMEOUT_SECONDS_ENV_VAR),
    )


def resolve_crypto_looper_api_config() -> CryptoLooperApiConfig:
    global _CRYPTO_LOOPER_API_CONFIG_CACHE_KEY
    global _CRYPTO_LOOPER_API_CONFIG_CACHE_VALUE

    cache_key = _crypto_looper_api_config_cache_key()
    if (
        _CRYPTO_LOOPER_API_CONFIG_CACHE_KEY == cache_key
        and _CRYPTO_LOOPER_API_CONFIG_CACHE_VALUE is not None
    ):
        return _CRYPTO_LOOPER_API_CONFIG_CACHE_VALUE

    config = CryptoLooperApiConfig(
        base_url=_crypto_looper_api_base_url(),
        timeout_seconds=_crypto_looper_api_timeout_seconds(),
    )
    _CRYPTO_LOOPER_API_CONFIG_CACHE_KEY = cache_key
    _CRYPTO_LOOPER_API_CONFIG_CACHE_VALUE = config
    return config


def _request(
    method,
    path: str,
    *,
    config: CryptoLooperApiConfig | None = None,
    **kwargs,
) -> requests.Response:
    config = resolve_crypto_looper_api_config() if config is None else config
    url = f"{config.base_url}{path}"
    timeout = kwargs.pop("timeout", config.timeout_seconds)
    return method(url, timeout=timeout, **kwargs)


def _stock_order_path(symbol: object | None = None) -> str:
    if symbol is None:
        return _CRYPTO_LOOPER_API_STOCK_ORDER_PATH
    return f"{_CRYPTO_LOOPER_API_STOCK_ORDER_PATH}/{quote(str(symbol), safe='')}"


def _response_log_preview(response: _ResponseLike | None) -> str:
    if not response or not response.text:
        return "N/A"
    return response.text[:_CRYPTO_LOOPER_API_RESPONSE_LOG_PREVIEW_CHARS]


def _parse_order_payload(json_order_data: _CryptoLooperOrderPayload) -> "FakeOrder":
    order = FakeOrder(
        symbol=json_order_data.get("symbol"),
        side=json_order_data.get("side"),
        limit_price=json_order_data.get("price"),
        qty=json_order_data.get("qty"),
    )
    created_at_str = json_order_data.get("created_at")
    if created_at_str:
        try:
            order.created_at = load_iso_format(created_at_str)
        except ValueError as e:
            logger.error(f"Error parsing created_at string '{created_at_str}': {e}")
    return order


def _parse_orders_response_payload(response_json: object) -> list["FakeOrder"]:
    orders: list[FakeOrder] = []
    if not isinstance(response_json, dict):
        logger.error(
            "Unexpected orders response payload type: "
            f"{type(response_json).__name__}; expected dict from stock_orders endpoint."
        )
        return orders

    server_data = response_json.get('data', {})
    if not isinstance(server_data, dict):
        logger.error(
            "Unexpected orders response data type: "
            f"{type(server_data).__name__}; top-level keys={sorted(response_json.keys())!r}"
        )
        return orders

    malformed_entries = 0
    for result_key, json_order_data in server_data.items():
        if not isinstance(json_order_data, dict):
            logger.error(
                f"Skipping malformed order payload for key '{result_key}': "
                f"{type(json_order_data).__name__}"
            )
            malformed_entries += 1
            continue
        orders.append(_parse_order_payload(json_order_data))

    logger.info(
        "Successfully fetched orders from crypto looper server: "
        f"parsed={len(orders)} malformed={malformed_entries} total={len(server_data)}"
    )
    return orders


@dataclass(eq=False, slots=True)
class FakeOrder:
    symbol: str | None = None
    side: str | None = None
    limit_price: str | None = None  # Alpaca API often uses string for price/qty
    qty: str | None = None
    created_at: datetime.datetime | None = None

    def __repr__(self):
        return f"{self.side} {self.qty} {self.symbol} at {self.limit_price} on {self.created_at}"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FakeOrder):
            return self.symbol == other.symbol and \
                   self.side == other.side and \
                   self.limit_price == other.limit_price and \
                   self.qty == other.qty and \
                   self.created_at == other.created_at # Consider how Nones are compared if that's valid
        if all(hasattr(other, attr) for attr in ("symbol", "side", "limit_price", "qty")):
            return (
                self.symbol == getattr(other, "symbol")
                and self.side == getattr(other, "side")
                and self.limit_price == getattr(other, "limit_price")
                and self.qty == getattr(other, "qty")
            )
        return False

    def __hash__(self):
        return hash((self.symbol, self.side, self.limit_price, self.qty, self.created_at))


def get_orders() -> list[FakeOrder]:
    logger.info("Fetching current orders from crypto looper server.")
    response = stock_orders()
    orders: list[FakeOrder] = []
    if response is None:
        logger.error("Failed to get response from stock_orders a.k.a crypto_order_loop_server is down?")
        return orders # Return empty list if server call failed

    try:
        response_json = response.json()
        logger.debug(f"Raw orders response: {response_json}")
        return _parse_orders_response_payload(response_json)
    except requests.exceptions.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response from server: {e}")
        if response: # Check again because it might have been None initially, though less likely here
             logger.error(f"Response text: {response.text}")
    except Exception as e:
        logger.error(f"Error processing orders response: {e}")
    return orders


def stock_order(symbol: object, side: object, price: object, qty: object) -> requests.Response | None:
    path = _stock_order_path()
    config = resolve_crypto_looper_api_config()
    data = {
        "symbol": symbol,
        "side": side,
        "price": str(price), # Ensure price is string
        "qty": str(qty),     # Ensure qty is string
    }
    url = f"{config.base_url}{path}"
    logger.info(f"Submitting stock order to {url} with data: {data}")
    try:
        response = _request(requests.post, path, config=config, json=data)
        logger.info(f"Server response status: {response.status_code}, content: {_response_log_preview(response)}")
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response # Or response.json() if appropriate
    except requests.exceptions.RequestException as e:
        logger.error(f"Error submitting stock order to {url}: {e}")
        return None


def stock_orders() -> requests.Response | None:
    path = _CRYPTO_LOOPER_API_STOCK_ORDERS_PATH
    config = resolve_crypto_looper_api_config()
    url = f"{config.base_url}{path}"
    logger.info(f"Fetching stock orders from {url}")
    try:
        response = _request(requests.get, path, config=config)
        logger.info(f"Server response status: {response.status_code}, content: {_response_log_preview(response)}")
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching stock orders from {url}: {e}")
        return None # Or an empty response-like object


def get_stock_order(symbol: object) -> requests.Response | None:
    path = _stock_order_path(symbol)
    config = resolve_crypto_looper_api_config()
    url = f"{config.base_url}{path}"
    logger.info(f"Fetching stock order for {symbol} from {url}")
    try:
        response = _request(requests.get, path, config=config)
        logger.info(f"Server response status: {response.status_code}, content: {_response_log_preview(response)}")
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching stock order for {symbol} from {url}: {e}")
        return None


def delete_stock_order(symbol: object) -> requests.Response | None:
    path = _stock_order_path(symbol)
    config = resolve_crypto_looper_api_config()
    url = f"{config.base_url}{path}"
    logger.info(f"Deleting stock order for {symbol} via {url}")
    try:
        response = _request(requests.delete, path, config=config)
        logger.info(f"Server response status: {response.status_code}, content: {_response_log_preview(response)}")
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"Error deleting stock order for {symbol} via {url}: {e}")
        return None


def delete_stock_orders() -> requests.Response | None:
    path = _CRYPTO_LOOPER_API_CANCEL_ALL_ORDERS_PATH
    config = resolve_crypto_looper_api_config()
    url = f"{config.base_url}{path}"
    logger.info(f"Deleting all stock orders via {url}")
    try:
        response = _request(requests.delete, path, config=config)
        logger.info(f"Server response status: {response.status_code}, content: {_response_log_preview(response)}")
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"Error deleting all stock orders via {url}: {e}")
        return None
