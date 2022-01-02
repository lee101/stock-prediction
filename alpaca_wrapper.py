from time import time, sleep

from alpaca_trade_api.rest import APIError
from loguru import logger
from env import ALP_KEY_ID, ALP_SECRET_KEY, ALP_ENDPOINT
import alpaca_trade_api as tradeapi
alpaca_api = tradeapi.REST(
    ALP_KEY_ID,
    ALP_SECRET_KEY,
    ALP_ENDPOINT,
    'v2')
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

def list_positions():
    return alpaca_api.list_positions()

def cancel_all_orders():
    result = alpaca_api.cancel_all_orders()
    print(result)

# alpaca_api.submit_order(short_stock, qty, side, "market", "day")
def close_position(position):
    try:
        if position.side == 'long':
            result = alpaca_api.submit_order(
                position.symbol,
                abs(float(position.qty)),
                'sell',
                'market',
                'day')

        else:
            result = alpaca_api.submit_order(
                position.symbol,
                abs(float(position.qty)),
                'buy',
                'market',
                'day')
    except Exception as e:
        logger.error(e)
        # close all positions? perhaps not
        return None
    print(result)


def buy_stock(currentBuySymbol, row):
    # poll untill we have closed all our positions
    polls = 0
    while True:
        positions = alpaca_api.list_positions()
        if len(positions) == 0:
            break
        else:
            print('waiting for positions to close')
            sleep(1)
            polls += 1
            if polls > 10:
                print('polling for too long, closing all positions again')
                # alpaca_api.close_all_positions() # todo respect manual orders
            if polls > 20:
                print('polling for too long, exiting, market is probably closed')
                break
    # notional_value = abs(float(account.cash)) * 1.9 # trade with margin
    notional_value = total_buying_power - 300 # trade with margin
    side = 'buy'
    if row['close_predicted_price'] < 0:
        side = 'sell'
        # notional_value = abs(float(account.cash)) * 1.2  # trade with margin but not too much on the sell side
        # todo dont leave a short open over the weekend perhaps?
    try:
        current_price = row['close_last_price']
        amount_to_trade = int(notional_value / current_price)
        if side == 'sell':
            take_profit_price = current_price - abs(current_price * float(row['close_predicted_price']))
            result = alpaca_api.submit_order(
                currentBuySymbol,
                amount_to_trade,
                side,
                'limit',
                'day',
                limit_price=current_price,
                take_profit={
                    "limit_price": take_profit_price
                }
            )
        else:
            take_profit_price = current_price + abs(current_price * float(row['close_predicted_price']))
            # we could use a limit with limit price but then couldnt do a notional order
            result = alpaca_api.submit_order(
                currentBuySymbol,
                amount_to_trade,
                side,
                'limit',
                'day',
                limit_price=current_price,
                # notional=notional_value,
                take_profit={
                    "limit_price": take_profit_price
                }
            )
        print(result)

    except APIError as e: # insufficient buying power if market closed
        logger.error(e)
    return None


def close_open_orders():
    alpaca_api.cancel_all_orders()
