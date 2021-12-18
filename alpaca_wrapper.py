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

alpaca_api.submit_order(short_stock, qty, side, "market", "day")
