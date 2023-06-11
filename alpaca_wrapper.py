import math
import traceback
from time import sleep

import requests.exceptions
from alpaca.data import (
    StockLatestQuoteRequest,
    StockHistoricalDataClient,
    CryptoHistoricalDataClient,
    CryptoLatestQuoteRequest,
)
from alpaca.trading import OrderType, LimitOrderRequest
from alpaca_trade_api.rest import APIError
from loguru import logger
from env_real import ALP_KEY_ID, ALP_SECRET_KEY, ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD, ALP_ENDPOINT
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide

from src.crypto_loop import crypto_alpaca_looper_api
from src.fixtures import crypto_symbols
from stc.stock_utils import remap_symbols

alpaca_api = TradingClient(
    ALP_KEY_ID,
    ALP_SECRET_KEY,
    # ALP_ENDPOINT,
    paper=ALP_ENDPOINT != "https://api.alpaca.markets",
)  # todo

data_client = StockHistoricalDataClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD)

equity = 30000
cash = 30000
total_buying_power = 20000

try:
    positions = alpaca_api.get_all_positions()
    print(positions)
    account = alpaca_api.get_account()
    print(account)
    # Figure out how much money we have to work with, accounting for margin
    equity = float(account.equity)
    cash = max(float(account.cash), 0)
    margin_multiplier = float(account.multiplier)
    total_buying_power = margin_multiplier * equity
    print(f"Initial total buying power = {total_buying_power}")
    alpaca_clock = alpaca_api.get_clock()
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


def get_all_positions():
    return alpaca_api.get_all_positions()


def cancel_all_orders():
    result = alpaca_api.cancel_orders()
    print(result)


# alpaca_api.submit_order(short_stock, qty, side, "market", "gtc")
def open_market_order_violently(symbol, qty, side):
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
        logger.error(e)
        return None
    print(result)


# er_stock:372 - LTCUSD buying 116.104 at 83.755

def has_current_open_position(symbol: str, side: str) -> bool:
    # normalize side out of paranoia
    if side == "long":
        side = "buy"
    if side == "short":
        side = "sell"
    current_positions = alpaca_api.get_all_positions()
    for position in current_positions:
        # if market value is significant
        if float(position.market_value) < 4:
            continue
        if position.symbol == symbol:
            if position.side == "long" and side == "buy":
                logger.info("position already open")
                return True
            if position.side == "short" and side == "sell":
                logger.info("position already open")
                return True
    return False


def open_order_at_price(symbol, qty, side, price):
    # todo: check if order is already open
    # cancel all other orders on this symbol
    current_open_orders = alpaca_api.get_orders()
    for order in current_open_orders:
        if order.symbol == symbol:
            cancel_order(order)
    # also check that there are not any open positions on this symbol
    has_current_position = has_current_open_position(symbol, side)
    if has_current_position:
        logger.info(f"position {symbol} already open")
        return
    try:
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


def close_position_violently(position):
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
        # close all positions? perhaps not
        return None
    print(result)


def close_position_at_current_price(position, row):
    if not row["close_last_price_minute"]:
        logger.info(f"nan price - for {position.symbol} market likely closed")
        return False
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
                        limit_price=row["close_last_price_minute"],
                    )
                )
            else:
                result = alpaca_api.submit_order(
                    order_data=LimitOrderRequest(
                        symbol=remap_symbols(position.symbol),
                        qty=abs(math.floor(float(position.qty) * 1000) / 1000.0),  # qty rounded down to 3dp
                        side="sell",
                        type=OrderType.LIMIT,
                        time_in_force="gtc",
                        limit_price=str(math.ceil(float(row["close_last_price_minute"]))),
                        # rounded up to whole number as theres an error limit price increment must be \u003e 1
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
        logger.error(e)  # cant convert nan to integer because market is closed for stocks
        traceback.print_exc()
        # Out of range float values are not JSON compliant
        # could be because theres no minute data /trying to close at when market isn't open (might as well err/do nothing)
        # close all positions? perhaps not
        return None
    print(result)


def backout_all_non_crypto_positions(positions, predictions):
    for position in positions:
        if position.symbol in crypto_symbols:
            continue
        current_row = None
        for pred in predictions:
            if pred["symbol"] == position.symbol:
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
            if pred["symbol"] == position.symbol:
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
            if pred["symbol"] == position.symbol:
                current_row = pred
                break
        logger.info(f"backing out at market {position.symbol}")

        close_position_at_current_price(position, current_row)


def close_position_at_almost_current_price(position, row):
    try:
        if position.side == "long":
            if position.symbol in crypto_symbols:
                result = crypto_alpaca_looper_api.submit_order(
                    order_data=LimitOrderRequest(
                        symbol=remap_symbols(position.symbol),
                        qty=abs(math.floor(float(position.qty) * 1000) / 1000.0),
                        # down to 3dp rounding up sometimes makes it cost too much when closing positions
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
                        # down to 3dp rounding up sometimes makes it cost too much when closing positions
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
        # close all positions? perhaps not
        return None
    print(result)


def alpaca_order_stock(currentBuySymbol, row, price, margin_multiplier=1.95, side="long", bid=None, ask=None):
    # trading at market to add more safety in high spread situations
    side = "buy" if side == "long" else "sell"
    if side == "buy" and bid:
        price = min(price, bid)
    else:
        price = max(price, ask)

    # poll untill we have closed all our positions
    # why we would wait here?
    # polls = 0
    # while True:
    #     positions = alpaca_api.get_all_positions()
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
    if currentBuySymbol in ["BTCUSD", "ETHUSD", "LTCUSD"]:
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
        elif amount_to_trade < 1:
            amount_to_trade = 1

        if currentBuySymbol not in ["BTCUSD", "ETHUSD", "LTCUSD"]:
            # fractional orders are okay for crypto.
            amount_to_trade = int(amount_to_trade)
        else:
            amount_to_trade = abs(math.floor(float(amount_to_trade) * 1000) / 1000.0)

        if side == "sell":
            # price_to_trade_at = max(current_price, row['high_last_price_minute'])
            #
            # take_profit_price = price_to_trade_at - abs(price_to_trade_at * (3*float(row['close_predicted_price_minute'])))
            logger.info(f"{currentBuySymbol} shorting {amount_to_trade} at {current_price}")
            if currentBuySymbol in crypto_symbols:
                # todo sure we can't sell?
                logger.info(f"cant short crypto {currentBuySymbol} - {amount_to_trade} for {price}")
                return False
            result = alpaca_api.submit_order(
                order_data=LimitOrderRequest(
                    symbol=remap_symbols(currentBuySymbol),
                    qty=amount_to_trade,
                    side=side,
                    type=OrderType.LIMIT,
                    time_in_force="gtc",
                    limit_price=str(math.ceil(price)),  # .001 sell margin
                    # take_profit={
                    #     "limit_price": take_profit_price
                    # }
                )
            )
            print(result)

        else:
            # price_to_trade_at = min(current_price, row['low_last_price_minute'])
            #
            # take_profit_price = current_price + abs(current_price * (3*float(row['close_predicted_price_minute']))) # todo takeprofit doesn't really work
            # we could use a limit with limit price but then couldn't do a notional order
            logger.info(
                f"{currentBuySymbol} buying {amount_to_trade} at {str(math.floor(price))}: current price {current_price}")
            # todo if crypto use loop
            # stop trying to trade too much - cancel current orders on same symbol
            current_orders = alpaca_api.get_orders() # also cancel binance orders?
            # cancel all orders on this symbol
            for order in current_orders:
                if order.symbol == currentBuySymbol:
                    alpaca_api.cancel_order_by_id(order.id)
            if currentBuySymbol in crypto_symbols:
                result = crypto_alpaca_looper_api.submit_order(
                    order_data=LimitOrderRequest(
                        symbol=remap_symbols(currentBuySymbol),
                        qty=amount_to_trade,
                        side=side,
                        type=OrderType.LIMIT,
                        time_in_force="gtc",
                        limit_price=str(math.floor(price)),
                        # aggressive rounding because btc gave errors for now "limit price increment must be \u003e 1"
                        # notional=notional_value,
                        # take_profit={
                        #     "limit_price": take_profit_price
                        # }
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
                        limit_price=str(math.floor(price)),
                        # aggressive rounding because btc gave errors for now "limit price increment must be \u003e 1"
                        # notional=notional_value,
                        # take_profit={
                        #     "limit_price": take_profit_price
                        # }
                    )
                )
            print(result)

    except APIError as e:  # insufficient buying power if market closed
        logger.error(e)
        return False
    return True


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
    positions = alpaca_api.get_all_positions()
    print(positions)
    account = alpaca_api.get_account()
    print(account)
    # Figure out how much money we have to work with, accounting for margin
    equity = float(account.equity)
    cash = max(float(account.cash), 0)
    margin_multiplier = float(account.multiplier)
    total_buying_power = margin_multiplier * equity
    print(f"Initial total buying power = {total_buying_power}")
    alpaca_clock = alpaca_api.get_clock()
    print(alpaca_clock)
    if not alpaca_clock.is_open:
        print("Market closed")


def open_take_profit_position(position, row, price, qty):
    # entry_price = float(position.avg_entry_price)
    # current_price = row['close_last_price_minute']
    # current_symbol = row['symbol']
    try:
        mapped_symbol = remap_symbols(position.symbol)
        if position.side == "long":
            if position.symbol in crypto_symbols:
                result = crypto_alpaca_looper_api.submit_order(
                    order_data=LimitOrderRequest(
                        symbol=mapped_symbol,
                        qty=abs(math.floor(float(qty) * 1000) / 1000.0),  # todo? round 3 didnt work?
                        side="sell",
                        type=OrderType.LIMIT,
                        time_in_force="gtc",
                        limit_price=str(math.ceil(price)),  # str(entry_price * (1 + .004),)
                    )
                )
            else:
                result = alpaca_api.submit_order(
                    order_data=LimitOrderRequest(
                        symbol=mapped_symbol,
                        qty=abs(math.floor(float(qty) * 1000) / 1000.0),  # todo? round 3 didnt work?
                        side="sell",
                        type=OrderType.LIMIT,
                        time_in_force="gtc",
                        limit_price=str(math.ceil(price)),  # str(entry_price * (1 + .004),)
                    )
                )
        else:
            if position.symbol in crypto_symbols:
                result = crypto_alpaca_looper_api.submit_order(order_data=LimitOrderRequest(
                    symbol=mapped_symbol,
                    qty=abs(math.floor(float(qty) * 1000) / 1000.0),
                    side="buy",
                    type=OrderType.LIMIT,
                    time_in_force="gtc",
                    limit_price=str(math.floor(price)),
                ))

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
        logger.error(e)  # can be because theres a sell order already which is still relevant
        # close all positions? perhaps not
        return None
    print(result)
    return True


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
        return alpaca_api.get_orders()  # + crypto_orders
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
