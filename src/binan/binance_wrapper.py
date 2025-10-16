from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, cast

from binance import Client
from loguru import logger

from env_real import BINANCE_API_KEY, BINANCE_SECRET
from src.stock_utils import binance_remap_symbols

_client: Client | None


def _init_client() -> Client | None:
    try:
        return Client(BINANCE_API_KEY, BINANCE_SECRET)
    except Exception as exc:  # pragma: no cover - connectivity / credential issues
        logger.error("Failed to initialise Binance client: %s", exc)
        logger.info(
            "Maybe you are offline - no connection to Binance; live trading features will be disabled."
        )
        return None


_client = _init_client()


def _require_client() -> Client:
    if _client is None:
        raise RuntimeError("Binance client is not initialised; check credentials and network connectivity.")
    return _client


def _coerce_price(value: float | str | None) -> float:
    if value is None:
        raise ValueError("A price is required for Binance limit orders.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid price {value!r} supplied to Binance order helper.") from exc


def _format_price(value: float) -> str:
    # Binance expects a string; avoid scientific notation.
    return f"{value:.8f}".rstrip("0").rstrip(".") or "0"

crypto_symbols = [
    "BTCUSDT",
    "ETHUSDT",
    "LTCUSDT",
    "PAXGUSDT",
    "UNIUSDT",
]


def create_order(symbol: str, side: str, quantity: float, price: float | str | None = None) -> Dict[str, Any]:
    client = _require_client()
    payload: Dict[str, Any] = {
        "symbol": symbol,
        "side": side,
        "type": Client.ORDER_TYPE_LIMIT,
        "timeInForce": Client.TIME_IN_FORCE_GTC,
        "quantity": quantity,
    }
    if price is not None:
        payload["price"] = _format_price(_coerce_price(price))

    order: Dict[str, Any]
    try:
        order = client.create_order(**payload)
    except Exception as exc:
        logger.error("Failed to create Binance order: %s", exc)
        logger.error("Payload: %s", payload)
        raise
    return order


def create_all_in_order(symbol: str, side: str, price: float | str | None = None) -> Dict[str, Any]:
    balance_sell: float | None = None
    balance_buy: float | None = None
    balances = get_account_balances()
    for balance in balances:
        asset = balance.get("asset")
        free = balance.get("free")
        if free is None:
            continue
        try:
            free_amount = float(free)
        except (TypeError, ValueError):
            logger.warning("Ignoring balance with unparsable free amount: %s", balance)
            continue
        if asset == symbol[:3]:
            balance_sell = free_amount
        if asset == symbol[3:]:
            balance_buy = free_amount

    if balance_sell is None or balance_buy is None:
        raise RuntimeError(f"Cannot determine balances for symbol {symbol}, received: {balances}")

    side_upper = side.upper()
    limit_price = _coerce_price(price) if price is not None else None
    if side_upper == "SELL":
        quantity = balance_sell
    elif side_upper == "BUY":
        if limit_price is None:
            raise ValueError("Price is required for BUY orders.")
        quantity = balance_buy / limit_price
    else:
        raise ValueError(f"Invalid side '{side}'. Expected 'BUY' or 'SELL'.")

    quantity = math.floor(quantity * 1000) / 1000
    if quantity <= 0:
        raise RuntimeError(f"Calculated Binance order quantity {quantity} is not positive for symbol {symbol}.")

    order = create_order(symbol, side_upper, quantity, limit_price)
    logger.info("Created order on Binance: %s", order)
    return order


def open_take_profit_position(position, row, price, qty):
    # entry_price = float(position.avg_entry_price)
    # current_price = row['close_last_price_minute']
    # current_symbol = row['symbol']
    try:
        mapped_symbol = binance_remap_symbols(position.symbol)
        if position.side == "long":
            create_all_in_order(mapped_symbol, "SELL", float(math.ceil(float(price))))
        else:
            create_all_in_order(mapped_symbol, "BUY", float(math.floor(float(price))))
    except Exception as e:
        logger.error(e)  # can be because theres a sell order already which is still relevant
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
            create_all_in_order(binance_remap_symbols(position.symbol), "BUY",
                                float(row["close_last_price_minute"]))
    except Exception as e:
        logger.error(e)  # cant convert nan to integer because market is closed for stocks
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
                _require_client().cancel_order(symbol=order["symbol"], orderId=order["orderId"])
            except Exception as e:
                print(e)
                logger.error(e)


def get_all_orders(symbol: str) -> List[Dict[str, Any]]:
    client = _require_client()
    try:
        raw_orders = client.get_all_orders(symbol=symbol)
    except Exception as e:
        logger.error(e)
        empty: List[Dict[str, Any]] = []
        return empty
    if not isinstance(raw_orders, list):
        logger.error("Unexpected orders payload from Binance: %s", raw_orders)
        return []
    orders: List[Dict[str, Any]] = []
    for entry in raw_orders:
        if isinstance(entry, dict):
            orders.append(entry)
        else:
            logger.debug("Discarding non-dict order entry: %s", entry)
    return orders


def get_account_balances() -> List[Dict[str, Any]]:
    client = _require_client()
    try:
        account = cast(Dict[str, Any], client.get_account())
        balances_obj = cast(Iterable[Dict[str, Any]] | None, account.get("balances", []))
    except Exception as e:
        logger.error(e)
        empty: List[Dict[str, Any]] = []
        return empty

    if balances_obj is None:
        logger.error("Binance account payload missing 'balances' key: %s", account)
        empty: List[Dict[str, Any]] = []
        return empty

    filtered: List[Dict[str, Any]] = []
    for entry in balances_obj:
        if isinstance(entry, dict):
            filtered.append(entry)
        else:
            logger.debug("Discarding non-dict balance entry: %s", entry)
    return filtered
