import math

from binanceneural.execution import (
    SymbolRules,
    available_quote_budget,
    compute_order_quantities,
    quantize_price,
    quote_asset_for_symbol,
    reserve_quote_budget,
)


def test_quantize_price_buy_sell():
    tick = 0.05
    buy = quantize_price(100.123, tick_size=tick, side="buy")
    sell = quantize_price(100.123, tick_size=tick, side="sell")
    assert math.isclose(buy, 100.10)
    assert math.isclose(sell, 100.15)


def test_compute_order_quantities_respects_min_notional():
    rules = SymbolRules(min_notional=10.0, min_qty=0.001, step_size=0.001, tick_size=0.01)
    sizing = compute_order_quantities(
        symbol="SOLUSD",
        buy_amount=10.0,
        sell_amount=10.0,
        buy_price=100.0,
        sell_price=110.0,
        quote_free=50.0,
        base_free=1.0,
        allocation_usdt=20.0,
        rules=rules,
    )
    # buy_notional = 20 * 0.1 = 2 < min_notional -> zeroed
    assert sizing.buy_qty == 0.0
    # sell_notional = 1 * 0.1 * 110 = 11 >= min_notional -> keep
    assert sizing.sell_qty > 0


def test_quote_budget_is_shared_by_quote_asset():
    budgets = {}
    assert quote_asset_for_symbol("BTCUSD") == "USDT"
    assert available_quote_budget(budgets, symbol="BTCUSD", observed_quote_free=100.0) == 100.0
    assert reserve_quote_budget(budgets, symbol="BTCUSD", reserved_notional=60.0) == 40.0
    assert available_quote_budget(budgets, symbol="ETHUSD", observed_quote_free=999.0) == 40.0

    assert quote_asset_for_symbol("DOGEFDUSD") == "FDUSD"
    assert available_quote_budget(budgets, symbol="DOGEFDUSD", observed_quote_free=25.0) == 25.0
