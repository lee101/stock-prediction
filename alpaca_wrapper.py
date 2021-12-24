from time import time, sleep

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
    if position.side == 'long':
        result = alpaca_api.submit_order(
            position.symbol,
            position.qty,
            'sell',
            'market',
            'day')

    else:
        result = alpaca_api.submit_order(
            position.symbol,
            position.qty,
            'buy',
            'market',
            'day')
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

    notional_value = abs(account.cash) * 1.9 # trade with margin
    result = alpaca_api.submit_order(
        currentBuySymbol,
        None,
        'buy',
        'market',
        'day',
        notional=notional_value,
    )
    print(result)
    return None
