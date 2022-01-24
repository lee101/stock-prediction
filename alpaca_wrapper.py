import traceback
from time import time, sleep

import requests.exceptions
from alpaca_trade_api.rest import APIError
from loguru import logger
from env_real import ALP_KEY_ID, ALP_SECRET_KEY, ALP_ENDPOINT
import alpaca_trade_api as tradeapi
alpaca_api = tradeapi.REST(
    ALP_KEY_ID,
    ALP_SECRET_KEY,
    ALP_ENDPOINT,
    'v2')

equity = 30000
total_buying_power = 20000

try:
    positions = alpaca_api.list_positions()
    print(positions)
    account = alpaca_api.get_account()
    print(account)
    # Figure out how much money we have to work with, accounting for margin
    equity = float(account.equity)
    margin_multiplier = float(account.multiplier)
    total_buying_power = margin_multiplier * equity
    print(f'Initial total buying power = {total_buying_power}')
    alpaca_clock = alpaca_api.get_clock()
    print(alpaca_clock)
    if not alpaca_clock.is_open:
        print('Market closed')
except requests.exceptions.ConnectionError as e:
    logger.error('offline/connection error', e)


def list_positions():
    return alpaca_api.list_positions()

def cancel_all_orders():
    result = alpaca_api.cancel_all_orders()
    print(result)

# alpaca_api.submit_order(short_stock, qty, side, "market", "gtc")
def close_position_violently(position):
    try:
        if position.side == 'long':
            result = alpaca_api.submit_order(
                position.symbol,
                abs(float(position.qty)),
                'sell',
                'market',
                'gtc')

        else:
            result = alpaca_api.submit_order(
                position.symbol,
                abs(float(position.qty)),
                'buy',
                'market',
                'gtc')
    except Exception as e:
        traceback.print_exc()

        logger.error(e)
        # close all positions? perhaps not
        return None
    print(result)


def close_position_at_current_price(position, row):
    try:
        if position.side == 'long':
            result = alpaca_api.submit_order(
                position.symbol,
                abs(float(position.qty)),
                'sell',
                'limit',
                'gtc',
                limit_price=row['close_last_price_minute'],
            )

        else:
            result = alpaca_api.submit_order(
                position.symbol,
                abs(float(position.qty)),
                'buy',
                'limit',
                'gtc',
                limit_price=row['close_last_price_minute'],
            )
    except Exception as e:
        logger.error(e)
        # close all positions? perhaps not
        return None
    print(result)

def buy_stock(currentBuySymbol, row, price, margin_multiplier=1.95, side='long'):
    side = 'buy' if side == 'long' else 'sell'

    # poll untill we have closed all our positions
    # why we would wait here?
    # polls = 0
    # while True:
    #     positions = alpaca_api.list_positions()
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
    # notional_value = abs(float(account.cash)) * 1.9 # trade with margin
    # notional_value = total_buying_power - 600 # trade with margin
    notional_value = abs(float(account.cash)) * margin_multiplier # todo predict margin/price

    # side = 'buy'
    if row['close_predicted_price'] < 0:
        # side = 'sell'
        notional_value = abs(float(account.cash)) * margin_multiplier  # trade with margin but not too much on the sell side
        # notional_value = total_buying_power - 2000
        # todo dont leave a short open over the weekend perhaps?


    try:
        current_price = row['close_last_price_minute']
        amount_to_trade = int(notional_value / current_price)
        if amount_to_trade > 0:
            amount_to_trade = 1

        if side == 'short':
            price_to_trade_at = max(current_price, row['high_last_price_minute'])

            take_profit_price = price_to_trade_at - abs(price_to_trade_at * (3*float(row['close_predicted_price_minute'])))
            result = alpaca_api.submit_order(
                currentBuySymbol,
                amount_to_trade,
                side,
                'limit',
                'gtc',
                limit_price=price,  # .001 sell margin
                # take_profit={
                #     "limit_price": take_profit_price
                # }
            )
        else:
            price_to_trade_at = min(current_price, row['low_last_price_minute'])

            take_profit_price = current_price + abs(current_price * (3*float(row['close_predicted_price_minute']))) # todo takeprofit doesn't really work
            # we could use a limit with limit price but then couldn't do a notional order
            result = alpaca_api.submit_order(
                currentBuySymbol,
                amount_to_trade,
                side,
                'limit',
                'gtc',
                limit_price=price,
                # notional=notional_value,
                # take_profit={
                #     "limit_price": take_profit_price
                # }
            )
        print(result)

    except APIError as e: # insufficient buying power if market closed
        logger.error(e)
        return False
    return True


def close_open_orders():
    alpaca_api.cancel_all_orders()


def re_setup_vars():
    global positions
    global account
    global alpaca_api
    global alpaca_clock
    global total_buying_power
    global equity
    global margin_multiplier
    positions = alpaca_api.list_positions()
    print(positions)
    account = alpaca_api.get_account()
    print(account)
    # Figure out how much money we have to work with, accounting for margin
    equity = float(account.equity)
    margin_multiplier = float(account.multiplier)
    total_buying_power = margin_multiplier * equity
    print(f'Initial total buying power = {total_buying_power}')
    alpaca_clock = alpaca_api.get_clock()
    print(alpaca_clock)
    if not alpaca_clock.is_open:
        print('Market closed')


def open_take_profit_position(position, row, price):
    # entry_price = float(position.avg_entry_price)
    # current_price = row['close_last_price_minute']
    # current_symbol = row['symbol']
    try:
        if position.side == 'long':
            result = alpaca_api.submit_order(
                position.symbol,
                abs(float(position.qty)),
                'sell',
                'limit',
                'gtc',
                limit_price=str(price)#str(entry_price * (1 + .004),)
            )

        else:
            result = alpaca_api.submit_order(
                position.symbol,
                abs(float(position.qty)),
                'buy',
                'limit',
                'gtc',
                limit_price=str(price)
            )
    except Exception as e:
        logger.error(e)
        # close all positions? perhaps not
        return None
    print(result)
    return True


def cancel_order(order):
    try:
        alpaca_api.cancel_order(order.id)
    except Exception as e:
        logger.error(e)


def get_open_orders():
    try:

        return alpaca_api.list_orders()
    except Exception as e:
        logger.error(e)
        return []
