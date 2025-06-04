import json
import json
import re
import traceback
from time import sleep

import cachetools
import math
import requests.exceptions
from alpaca.data import (
    StockLatestQuoteRequest,
    StockHistoricalDataClient,
    CryptoHistoricalDataClient,
    CryptoLatestQuoteRequest,
)
from alpaca.trading import OrderType, LimitOrderRequest, GetOrdersRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide
from alpaca.trading.requests import MarketOrderRequest
from alpaca_trade_api.rest import APIError
from loguru import logger
from retry import retry

from env_real import ALP_KEY_ID, ALP_SECRET_KEY, ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD, ALP_ENDPOINT
from typing import Iterable, Dict, Any
from src.comparisons import is_buy_side, is_sell_side
from src.crypto_loop import crypto_alpaca_looper_api
from src.fixtures import crypto_symbols
from src.logging_utils import setup_logging
from src.stock_utils import pairs_equal, remap_symbols
from src.trading_obj_utils import filter_to_realistic_positions

logger = setup_logging("stock.log")

alpaca_api = TradingClient(
    ALP_KEY_ID,
    ALP_SECRET_KEY,
    # ALP_ENDPOINT,
    paper=ALP_ENDPOINT != "https://api.alpaca.markets",
)  # todo

data_client = StockHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)

force_open_the_clock = False


@cachetools.cached(cache=cachetools.TTLCache(maxsize=100, ttl=60 * 3)) # 3 mins
def get_clock(retries=3):
    clock = get_clock_internal(retries)
    if not clock.is_open and force_open_the_clock:
        clock.is_open = True
    return clock


def force_open_the_clock_func():
    global force_open_the_clock
    force_open_the_clock = True


def get_clock_internal(retries=3):
    try:
        return alpaca_api.get_clock()
    except Exception as e:
        logger.error(e)
        if retries > 0:
            sleep(.1)
            logger.error("retrying get clock")
            return get_clock_internal(retries - 1)
        raise e


def get_all_positions(retries=3):
    try:
        return alpaca_api.get_all_positions()
    except Exception as e:
        logger.error(e)
        if retries > 0:
            sleep(.1)
            logger.error("retrying get all positions")
            return get_all_positions(retries - 1)
        raise e


def cancel_all_orders(retries=3):
    result = None
    try:
        result = alpaca_api.cancel_orders()
        logger.info("canceled orders")
        logger.info(result)
    except Exception as e:
        logger.error(e)

        if retries > 0:
            sleep(.1)
            logger.error("retrying cancel all orders")
            return cancel_all_orders(retries - 1)
        logger.error("failed to cancel all orders")
        return None
    return result


# alpaca_api.submit_order(short_stock, qty, side, "market", "gtc")
def open_market_order_violently(symbol, qty, side, retries=3):
    result = None
    try:
        result = alpaca_api.submit_order(
            order_data=MarketOrderRequest(
                symbol=remap_symbols(symbol),
                qty=qty,
                side=side,
                type=OrderType.MARKET,
                time_in_force="gtc",
            )
        )
    except Exception as e:
        if retries > 0:
            return open_market_order_violently(symbol, qty, side, retries - 1)
        logger.error(e)
        return None
    print(result)
    return result


def _parse_available_balance(error_str: str) -> float:
    """Extract available balance from an error message."""
    try:
        data = json.loads(error_str)
        return float(data.get("available", 0))
    except Exception:
        pass

    match = re.search(r"available['\"]?:\s*([0-9]*\.?[0-9]+)", error_str)
    if match:
        try:
            return float(match.group(1))
        except Exception:
            pass
    return 0.0


# er_stock:372 - LTCUSD buying 116.104 at 83.755

def has_current_open_position(symbol: str, side: str) -> bool:
    # normalize side out of paranoia
    if side == "long":
        side = "buy"
    if side == "short":
        side = "sell"
    current_positions = []
    for i in range(3):
        try:
            current_positions = get_all_positions()
            break
        except Exception as e:
            traceback.print_exc()
            logger.error(e)
            # sleep(.1)
    current_positions = filter_to_realistic_positions(current_positions)
    for position in current_positions:
        # if market value is significant
        if float(position.market_value) < 4:
            continue
        if pairs_equal(position.symbol, symbol):
            if is_buy_side(position.side) and is_buy_side(side):
                logger.info("position already open")
                return True
            if is_sell_side(position.side) and is_sell_side(side):
                logger.info("position already open")
                return True
    return False


def open_order_at_price(symbol, qty, side, price):
    result = None
    # todo: check if order is already open
    # cancel all other orders on this symbol
    current_open_orders = get_orders()
    for order in current_open_orders:
        if pairs_equal(order.symbol, symbol):
            cancel_order(order)
    # also check that there are not any open positions on this symbol
    has_current_position = has_current_open_position(symbol, side)
    if has_current_position:
        logger.info(f"position {symbol} already open")
        return None
    try:
        price = str(round(price, 2))
        result = alpaca_api.submit_order(
            order_data=LimitOrderRequest(
                symbol=remap_symbols(symbol),
                qty=qty,
                side=side,
                type=OrderType.LIMIT,
                time_in_force="gtc",
                limit_price=price,
            )
        )
    except Exception as e:
        logger.error(e)
        return None
    print(result)
    return result


def open_order_at_price_or_all(symbol, qty, side, price):
    result = None
    # Cancel existing orders for this symbol
    current_open_orders = get_orders()
    for order in current_open_orders:
        if pairs_equal(order.symbol, symbol):
            cancel_order(order)

    # Check for existing position
    has_current_position = has_current_open_position(symbol, side)
    if has_current_position:
        logger.info(f"position {symbol} already open")
        return None

    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Keep price as float for calculations, only convert when submitting order
            price_rounded = round(price, 2)
            result = alpaca_api.submit_order(
                order_data=LimitOrderRequest(
                    symbol=remap_symbols(symbol),
                    qty=qty,
                    side=side,
                    type=OrderType.LIMIT,
                    time_in_force="gtc",
                    limit_price=str(price_rounded),
                )
            )
            return result

        except Exception as e:
            error_str = str(e)
            logger.error(f"Order attempt {retry_count + 1} failed: {error_str}")

            # Check if error indicates insufficient funds
            if "insufficient" in error_str.lower():
                available = _parse_available_balance(error_str)
                if available <= 0:
                    available = cash

                if available > 0:
                    new_qty = math.floor(0.99 * available / price)
                    if new_qty > 0 and new_qty != qty:
                        logger.info(f"Retrying with adjusted quantity: {new_qty}")
                        qty = new_qty
                        retry_count += 1
                        continue

            retry_count += 1
            # if retry_count < max_retries:
            #     time.sleep(2)  # Wait before retry

    logger.error("Max retries reached, order failed")
    return None


def execute_portfolio_orders(orders: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Execute multiple orders sequentially.

    Each order should be a mapping containing ``symbol``, ``qty``, ``side`` and
    ``price`` keys. If an order fails, the error is logged and execution
    continues with the remaining orders.

    Parameters
    ----------
    orders: Iterable[Dict[str, Any]]
        Iterable of order dictionaries.

    Returns
    -------
    Dict[str, Any]
        Mapping of symbol to the result returned by
        :func:`open_order_at_price_or_all` or ``None`` if the order failed.
    """
    results: Dict[str, Any] = {}
    for order in orders:
        symbol = order.get("symbol")
        qty = order.get("qty")
        side = order.get("side")
        price = order.get("price")

        try:
            results[symbol] = open_order_at_price_or_all(symbol, qty, side, price)
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Failed to execute order for {symbol}: {e}")
            results[symbol] = None

    return results


def close_position_violently(position):
    result = None
    try:
        if position.side == "long":
            result = alpaca_api.submit_order(
                order_data=MarketOrderRequest(
                    symbol=remap_symbols(position.symbol),
                    qty=abs(float(position.qty)),
                    side=OrderSide.SELL,
                    type=OrderType.MARKET,
                    time_in_force="gtc",
                )
            )
        else:
            result = alpaca_api.submit_order(
                order_data=MarketOrderRequest(
                    symbol=remap_symbols(position.symbol),
                    qty=abs(float(position.qty)),
                    side=OrderSide.BUY,
                    type=OrderType.MARKET,
                    time_in_force="gtc",
                )
            )
    except Exception as e:
        traceback.print_exc()
        logger.error(e)
        return None
    print(result)
    return result


def close_position_at_current_price(position, row):
    if not row["close_last_price_minute"]:
        logger.info(f"nan price - for {position.symbol} market likely closed")
        return False
    result = None
    try:
        if position.side == "long":
            if position.symbol in crypto_symbols:
                result = crypto_alpaca_looper_api.submit_order(
                    order_data=LimitOrderRequest(
                        symbol=remap_symbols(position.symbol),
                        qty=abs(float(position.qty)),
                        side=OrderSide.SELL,
                        type=OrderType.LIMIT,
                        time_in_force="gtc",
                        limit_price=str(round(float(row["close_last_price_minute"]), 2)),
                    )
                )
            else:
                result = alpaca_api.submit_order(
                    order_data=LimitOrderRequest(
                        symbol=remap_symbols(position.symbol),
                        qty=abs(math.floor(float(position.qty) * 1000) / 1000.0),
                        side="sell",
                        type=OrderType.LIMIT,
                        time_in_force="gtc",
                        limit_price=str(math.ceil(float(row["close_last_price_minute"]))),
                    )
                )
        else:
            if position.symbol in crypto_symbols:
                result = crypto_alpaca_looper_api.submit_order(
                    order_data=LimitOrderRequest(
                        symbol=remap_symbols(position.symbol),
                        qty=abs(math.floor(float(position.qty) * 1000) / 1000.0),
                        side=OrderSide.BUY,
                        type=OrderType.LIMIT,
                        time_in_force="gtc",
                        limit_price=str(math.floor(float(row["close_last_price_minute"]))),
                    )
                )
            else:
                result = alpaca_api.submit_order(
                    order_data=LimitOrderRequest(
                        symbol=remap_symbols(position.symbol),
                        qty=abs(math.floor(float(position.qty) * 1000) / 1000.0),
                        side="buy",
                        type=OrderType.LIMIT,
                        time_in_force="gtc",
                        limit_price=str(math.floor(float(row["close_last_price_minute"]))),
                    )
                )
    except Exception as e:
        logger.error(e)
        traceback.print_exc()
        return None
    print(result)
    return result


def backout_all_non_crypto_positions(positions, predictions):
    for position in positions:
        if position.symbol in crypto_symbols:
            continue
        current_row = None
        for pred in predictions:
            if pairs_equal(pred["symbol"], position.symbol):
                current_row = pred
                break
        logger.info(f"backing out {position.symbol}")
        close_position_at_almost_current_price(position, current_row)
    sleep(60 * 2)

    cancel_all_orders()
    for position in positions:
        if position.symbol in crypto_symbols:
            continue
        current_row = None
        for pred in predictions:
            if pairs_equal(pred["symbol"], position.symbol):
                current_row = pred
                break
        logger.info(f"backing out at market {position.symbol}")

        close_position_at_current_price(position, current_row)
    sleep(60 * 2)

    cancel_all_orders()
    for position in positions:
        if position.symbol in crypto_symbols:
            continue
        # don't violently close here as spreads can be high
        # logger.info(f"violently backing out {position.symbol}")
        # close_position_violently(position)
        current_row = None
        for pred in predictions:
            if pairs_equal(pred["symbol"], position.symbol):
                current_row = pred
                break
        logger.info(f"backing out at market {position.symbol}")

        close_position_at_current_price(position, current_row)


def close_position_at_almost_current_price(position, row):
    result = None
    try:
        if position.side == "long":
            if position.symbol in crypto_symbols:
                result = crypto_alpaca_looper_api.submit_order(
                    order_data=LimitOrderRequest(
                        symbol=remap_symbols(position.symbol),
                        qty=abs(math.floor(float(position.qty) * 1000) / 1000.0),
                        side="sell",
                        type=OrderType.LIMIT,
                        time_in_force="gtc",
                        limit_price=str(round(row["close_last_price_minute"] * 1.0003, 1)),
                    )
                )
            else:
                result = alpaca_api.submit_order(
                    order_data=LimitOrderRequest(
                        symbol=remap_symbols(position.symbol),
                        qty=abs(math.floor(float(position.qty) * 1000) / 1000.0),
                        side="sell",
                        type=OrderType.LIMIT,
                        time_in_force="gtc",
                        limit_price=str(round(row["close_last_price_minute"] * 1.0003, 1)),
                    )
                )
        else:
            if position.symbol in crypto_symbols:
                result = crypto_alpaca_looper_api.submit_order(
                    order_data=LimitOrderRequest(
                        symbol=remap_symbols(position.symbol),
                        qty=abs(math.floor(float(position.qty) * 1000) / 1000.0),
                        side="buy",
                        type=OrderType.LIMIT,
                        time_in_force="gtc",
                        limit_price=str(round(row["close_last_price_minute"] * (1 - 0.0003), 1)),
                    )
                )
            else:
                result = alpaca_api.submit_order(
                    order_data=LimitOrderRequest(
                        symbol=remap_symbols(position.symbol),
                        qty=abs(math.floor(float(position.qty) * 1000) / 1000.0),
                        side="buy",
                        type=OrderType.LIMIT,
                        time_in_force="gtc",
                        limit_price=str(round(row["close_last_price_minute"] * (1 - 0.0003), 1)),
                    )
                )
    except Exception as e:
        logger.error(e)
        return None
    print(result)
    return result


@retry(delay=.1, tries=3)
def get_orders():
    return alpaca_api.get_orders()


def alpaca_order_stock(currentBuySymbol, row, price, margin_multiplier=1.95, side="long", bid=None, ask=None):
    result = None
    # trading at market to add more safety in high spread situations
    side = "buy" if is_buy_side(side) else "sell"
    if side == "buy" and bid:
        price = min(price, bid or price)
    else:
        price = max(price, ask or price)

    # skip crypto for now as its high fee
    # if currentBuySymbol in crypto_symbols and is_buy_side(side):
    #     logger.info(f"Skipping Buying Alpaca crypto order for {currentBuySymbol}")
    #     logger.info(f"TMp measure as fees are too high IMO move to binance")
    #     return False

    # poll untill we have closed all our positions
    # why we would wait here?
    # polls = 0
    # while True:
    #     positions = get_all_positions()
    #     if len(positions) == 0:
    #         break
    #     else:
    #         print('waiting for positions to close')
    #         sleep(.1)
    #         polls += 1
    #         if polls > 5:
    #             print('polling for too long, closing all positions again')
    #             # alpaca_api.close_all_positions() # todo respect manual orders
    #         if polls > 20:
    #             print('polling for too long, exiting, market is probably closed')
    #             break
    # notional_value = total_buying_power * 1.9 # trade with margin
    # notional_value = total_buying_power - 600 # trade with margin
    # non marginable
    if currentBuySymbol in ["BTCUSD", "ETHUSD", "LTCUSD", "PAXGUSD", "UNIUSD"]:

        margin_multiplier = min(margin_multiplier, 1)
        notional_value = cash * margin_multiplier  # todo predict margin/price
    else:
        notional_value = total_buying_power * margin_multiplier  # todo predict margin/price

    # side = 'buy'
    if row["close_predicted_price"] < 0:
        # side = 'sell'
        notional_value = (
                total_buying_power * margin_multiplier
        )  # trade with margin but not too much on the sell side
        # notional_value = total_buying_power - 2000
        # todo dont leave a short open over the weekend perhaps?

    try:
        current_price = float(row["close_last_price_minute"])

        amount_to_trade = notional_value / current_price
        if currentBuySymbol in ["BTCUSD"]:
            if amount_to_trade < 0.001:
                amount_to_trade = 0.001
        elif currentBuySymbol in ["ETHUSD"]:
            if amount_to_trade < 0.01:
                amount_to_trade = 0.01
        elif currentBuySymbol in ["LTCUSD"]:
            if amount_to_trade < 0.1:
                amount_to_trade = 0.1
                # too work out "PAXGUSD", "UNIUSD"
        elif amount_to_trade < 1:
            amount_to_trade = 1

        if currentBuySymbol not in ["BTCUSD", "ETHUSD", "LTCUSD", "PAXGUSD", "UNIUSD"]:
            # fractional orders are okay for crypto.
            amount_to_trade = int(amount_to_trade)
        else:
            amount_to_trade = abs(math.floor(float(amount_to_trade) * 1000) / 1000.0)

        # Cancel existing orders for this symbol
        current_orders = get_orders()
        for order in current_orders:
            if pairs_equal(order.symbol, currentBuySymbol):
                alpaca_api.cancel_order_by_id(order.id)

        # Submit the order
        if currentBuySymbol in crypto_symbols:
            result = crypto_alpaca_looper_api.submit_order(
                order_data=LimitOrderRequest(
                    symbol=remap_symbols(currentBuySymbol),
                    qty=amount_to_trade,
                    side=side,
                    type=OrderType.LIMIT,
                    time_in_force="gtc",
                    limit_price=str(math.floor(price) if is_buy_side(side) else math.ceil(price)),
                )
            )
        else:
            result = alpaca_api.submit_order(
                order_data=LimitOrderRequest(
                    symbol=remap_symbols(currentBuySymbol),
                    qty=amount_to_trade,
                    side=side,
                    type=OrderType.LIMIT,
                    time_in_force="gtc",
                    limit_price=str(math.floor(price) if is_buy_side(side) else math.ceil(price)),
                )
            )
        print(result)
        return True

    except APIError as e:
        logger.error(e)
        return False
    except Exception as e:
        logger.error(e)
        return False


def close_open_orders():
    alpaca_api.cancel_orders()


def re_setup_vars():
    global positions
    global account
    global alpaca_api
    global alpaca_clock
    global total_buying_power
    global equity
    global cash
    global margin_multiplier
    positions = get_all_positions()
    print(positions)
    account = get_account()
    print(account)
    # Figure out how much money we have to work with, accounting for margin
    equity = float(account.equity)
    cash = max(float(account.cash), 0)
    margin_multiplier = float(account.multiplier)
    total_buying_power = margin_multiplier * equity
    print(f"Initial total buying power = {total_buying_power}")
    alpaca_clock = get_clock()
    print(alpaca_clock)
    if not alpaca_clock.is_open:
        print("Market closed")


def open_take_profit_position(position, row, price, qty):
    result = None
    try:
        mapped_symbol = remap_symbols(position.symbol)
        if position.side == "long":
            if position.symbol in crypto_symbols:
                result = crypto_alpaca_looper_api.submit_order(
                    order_data=LimitOrderRequest(
                        symbol=mapped_symbol,
                        qty=abs(math.floor(float(qty) * 1000) / 1000.0),
                        side="sell",
                        type=OrderType.LIMIT,
                        time_in_force="gtc",
                        limit_price=str(math.ceil(price)),
                    )
                )
            else:
                result = alpaca_api.submit_order(
                    order_data=LimitOrderRequest(
                        symbol=mapped_symbol,
                        qty=abs(math.floor(float(qty) * 1000) / 1000.0),
                        side="sell",
                        type=OrderType.LIMIT,
                        time_in_force="gtc",
                        limit_price=str(math.ceil(price)),
                    )
                )
        else:
            if position.symbol in crypto_symbols:
                result = crypto_alpaca_looper_api.submit_order(
                    order_data=LimitOrderRequest(
                        symbol=mapped_symbol,
                        qty=abs(math.floor(float(qty) * 1000) / 1000.0),
                        side="buy",
                        type=OrderType.LIMIT,
                        time_in_force="gtc",
                        limit_price=str(math.floor(price)),
                    )
                )
            else:
                result = alpaca_api.submit_order(
                    order_data=LimitOrderRequest(
                        symbol=mapped_symbol,
                        qty=abs(math.floor(float(qty) * 1000) / 1000.0),
                        side="buy",
                        type=OrderType.LIMIT,
                        time_in_force="gtc",
                        limit_price=str(math.floor(price)),
                    )
                )
    except Exception as e:
        logger.error(e)
        traceback.print_exc()
        return None
    return result


def cancel_order(order):
    try:
        alpaca_api.cancel_order_by_id(order.id)
    except Exception as e:
        logger.error(e)
        # traceback
        traceback.print_exc()


def get_open_orders():
    # try:
    #     crypto_orders = crypto_alpaca_looper_api.get_orders()
    # except Exception as e:
    #     logger.error(e)
    #     crypto_orders = []
    #     traceback.print_exc()

    try:
        return get_orders()  # + crypto_orders
    except Exception as e:
        logger.error(e)
        traceback.print_exc()
        return []


crypto_client = CryptoHistoricalDataClient()


def latest_data(symbol):
    if symbol in crypto_symbols:
        symbol = remap_symbols(symbol)
        response = crypto_client.get_crypto_latest_quote(
            CryptoLatestQuoteRequest(symbol_or_symbols=[symbol])
        )
        return response[symbol]

    multisymbol_request_params = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
    latest_multisymbol_quotes = data_client.get_stock_latest_quote(multisymbol_request_params)

    return latest_multisymbol_quotes[symbol]


@retry(delay=.1, tries=3)
def get_account():
    return alpaca_api.get_account()


equity = 30000
cash = 30000
total_buying_power = 20000

try:
    positions = get_all_positions()
    print(positions)
    account = get_account()
    print(account)
    # Figure out how much money we have to work with, accounting for margin
    equity = float(account.equity)
    cash = max(float(account.cash), 0)
    margin_multiplier = float(account.multiplier)
    total_buying_power = margin_multiplier * equity
    print(f"Initial total buying power = {total_buying_power}")
    alpaca_clock = get_clock()
    print(alpaca_clock)
    if not alpaca_clock.is_open:
        print("Market closed")
except requests.exceptions.ConnectionError as e:
    logger.error("offline/connection error", e)
except APIError as e:
    logger.error("alpaca error", e)
except Exception as e:
    logger.error("exception", e)
    traceback.print_exc()


def close_position_near_market(position, pct_above_market=0.0):
    """Place a limit order at ``pct_above_market`` relative to the quote."""
    bids = {}
    asks = {}
    symbol = position.symbol
    very_latest_data = latest_data(position.symbol)
    # check if market closed
    ask_price = float(very_latest_data.ask_price)
    bid_price = float(very_latest_data.bid_price)
    if bid_price != 0 and ask_price != 0:
        bids[symbol] = bid_price
        asks[symbol] = ask_price

    ask_price = asks.get(position.symbol)
    bid_price = bids.get(position.symbol)

    if not ask_price or not bid_price:
        logger.error(f"error getting ask/bid price for {position.symbol}")
        return False

    if position.side == "long":
        # For long positions, reference the bid price when selling
        price = bid_price
    else:
        # For short positions, reference the ask price when buying back
        price = ask_price

    result = None
    try:
        if position.side == "long":
            sell_price = price * (1 + pct_above_market)
            sell_price = str(round(sell_price, 2))
            logger.info(f"selling {position.symbol} at {sell_price}")
            result = alpaca_api.submit_order(
                order_data=LimitOrderRequest(
                    symbol=remap_symbols(position.symbol),
                    qty=abs(float(position.qty)),
                    side=OrderSide.SELL,
                    type=OrderType.LIMIT,
                    time_in_force="gtc",
                    limit_price=sell_price,
                )
            )
        else:
            buy_price = price * (1 + pct_above_market)
            buy_price = str(round(buy_price, 2))
            logger.info(f"buying {position.symbol} at {buy_price}")
            result = alpaca_api.submit_order(
                order_data=LimitOrderRequest(
                    symbol=remap_symbols(position.symbol),
                    qty=abs(float(position.qty)),
                    side=OrderSide.BUY,
                    type=OrderType.LIMIT,
                    time_in_force="gtc",
                    limit_price=buy_price,
                )
            )

    except Exception as e:
        logger.error(e)
        traceback.print_exc()
        return False

    return result


def get_executed_orders(alpaca_api):
    """
    Gets all historical orders that were executed.

    Args:
        alpaca_api: The Alpaca trading client instance

    Returns:
        List of executed orders
    """
    try:
        # Get all orders with status=filled filter
        orders = alpaca_api.get_orders(
            filter=GetOrdersRequest(
                status="filled"
            )
        )
        return orders

    except Exception as e:
        logger.error(f"Error getting executed orders: {e}")
        traceback.print_exc()
        return []


def get_account_activities(
        alpaca_api,
        activity_types=None,
        date=None,
        direction='desc',
        page_size=100,
        page_token=None
):
    """
    Retrieve account activities (trades, dividends, etc.) from the Alpaca API.
    Pagination is handled via page_token. The activity_types argument can be any of:
    'FILL', 'DIV', 'TRANS', 'MISC', etc.

    Args:
        alpaca_api: The Alpaca trading client instance.
        activity_types: List of activity type strings (e.g. ['FILL', 'DIV']).
        date: (Optional) The date for which you'd like to see activities.
        direction: 'asc' or 'desc' for sorting.
        page_size: The number of records to return per page (up to 100 if date is not set).
        page_token: Used for pagination.

    Returns:
        A list of account activity records, or an empty list on error.
    """
    query_params = {}
    if activity_types:
        # Convert single str to list if needed
        if isinstance(activity_types, str):
            activity_types = [activity_types]
        query_params["activity_types"] = ",".join(activity_types)

    if date:
        query_params["date"] = date
    if direction:
        query_params["direction"] = direction
    if page_size:
        query_params["page_size"] = str(page_size)
    if page_token:
        query_params["page_token"] = page_token

    try:
        # Directly use the TradingClient's underlying request method to access this endpoint
        response = alpaca_api._request("GET", "/account/activities", data=query_params)
        return response
    except Exception as e:
        logger.error(f"Error retrieving account activities: {e}")
        return []
