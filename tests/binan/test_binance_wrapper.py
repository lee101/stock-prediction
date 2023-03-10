from src.binan.binance_wrapper import get_account_balances, get_all_orders, cancel_all_orders, create_order, \
    create_all_in_order
from src.crypto_loop.crypto_alpaca_looper_api import get_orders


def test_get_account():
    balances = get_account_balances()
    assert len(balances) > 0
    print(balances) # {'asset': 'BTC', 'free': '0.02332178', 'locked': '0.00000000'}

def test_get_all_orders():
    orders = get_all_orders('BTCUSDT')
    # assert len(orders) == 0

def test_get_orders():
    get_orders()


def test_cancel_all_orders():
    cancel_all_orders()

# def test_add_order():
#     # order = create_order('BTCUSDT', 'SELL', 0.001, 18000) # buying usdt with btc
#     order = create_order('BTCUSDT', 'BUY', 0.01, 15000) # buying btc with usdt
#     # assert order['status'] == 'NEW'
#     orders = get_all_orders('BTCUSDT') # race cond?
#     assert len(orders) == 1
#     print(orders)
#     cancel_all_orders()
#     orders = get_all_orders('BTCUSDT')
#     assert len(orders) == 0

# def test_create_all_in_order():
#     # order = create_all_in_order('BTCUSDT', 'SELL', 18000) # buying usdt with btc
#     # order = create_all_in_order('BTCUSDT', 'BUY', 17000) # buying btc with usdt
#     # assert order['status'] == 'NEW'
#     orders = get_all_orders('BTCUSDT')
#     # check for non canceled/filled orders
#     assert len(orders) == 0
