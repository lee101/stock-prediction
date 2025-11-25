import math

import pandas as pd

from hourlycrypto.trade_stock_crypto_hourly import TradingPlan, _adjust_for_maker_liquidity


def _plan(buy: float, sell: float) -> TradingPlan:
    return TradingPlan(
        timestamp=pd.Timestamp.utcnow(),
        buy_price=buy,
        sell_price=sell,
        buy_amount=0.5,
        sell_amount=0.5,
    )


def test_buy_price_clamped_to_bid_for_maker_entry():
    plan = _plan(6.30, 6.40)
    adjusted = _adjust_for_maker_liquidity(plan, bid=6.24, ask=6.26, midpoint=6.25)
    assert math.isclose(adjusted.buy_price, 6.24, rel_tol=1e-6)


def test_sell_price_sets_to_ask():
    plan = _plan(6.10, 6.36)
    adjusted = _adjust_for_maker_liquidity(plan, bid=6.20, ask=6.30, midpoint=6.25)
    assert math.isclose(adjusted.sell_price, 6.30, rel_tol=1e-6)


def test_sell_price_raised_above_bid_when_too_low():
    plan = _plan(6.00, 6.05)
    adjusted = _adjust_for_maker_liquidity(plan, bid=6.10, ask=6.20, midpoint=6.15)
    assert math.isclose(adjusted.sell_price, 6.20, rel_tol=1e-6)


def test_no_quote_returns_original_plan():
    plan = _plan(1.23, 1.45)
    adjusted = _adjust_for_maker_liquidity(plan, bid=None, ask=None, midpoint=None)
    assert adjusted.buy_price == plan.buy_price
    assert adjusted.sell_price == plan.sell_price


def test_midpoint_used_when_bid_ask_missing():
    plan = _plan(10.0, 11.0)
    adjusted = _adjust_for_maker_liquidity(plan, bid=None, ask=None, midpoint=10.5)
    assert math.isclose(adjusted.buy_price, 10.0 if 10.0 < 10.5 else 10.5, rel_tol=1e-6)
    assert math.isclose(adjusted.sell_price, 10.5, rel_tol=1e-6)
