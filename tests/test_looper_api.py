import math

from alpaca.trading import LimitOrderRequest

from src.crypto_loop import crypto_alpaca_looper_api
from stc.stock_utils import remap_symbols


def test_submit_order():
    """ test that we can submit an order, warning dont do this in live mode """
    price = 17176.675000000003
    result = crypto_alpaca_looper_api.submit_order(
        order_data=LimitOrderRequest(
            symbol=remap_symbols("BTCUSD"),
            qty=.0000001,
            side="buy",
            type="limit",
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
