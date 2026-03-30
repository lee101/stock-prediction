import math

from binanceneural.execution import (
    SymbolRules,
    available_quote_budget,
    compute_order_quantities,
    quantize_down,
    quantize_price,
    quantize_qty,
    quantize_up,
    quote_asset_for_symbol,
    reserve_quote_budget,
    split_binance_symbol,
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


def test_quantize_down_basic():
    assert math.isclose(quantize_down(836.5626, 0.1), 836.5)
    assert math.isclose(quantize_down(163.9359, 0.1), 163.9)
    assert math.isclose(quantize_down(3.1908, 0.001), 3.190)


def test_quantize_down_step_one():
    assert quantize_down(836.5626, 1.0) == 836.0
    assert quantize_down(0.99, 1.0) == 0.0


def test_quantize_down_none_step_passthrough():
    assert quantize_down(12.345, None) == 12.345
    assert quantize_down(12.345, 0.0) == 12.345
    assert quantize_down(12.345, -1.0) == 12.345


def test_quantize_up_basic():
    assert math.isclose(quantize_up(100.01, 0.05), 100.05)
    assert math.isclose(quantize_up(100.00, 0.05), 100.00)


def test_quantize_qty_uses_step_size():
    assert math.isclose(quantize_qty(28530.521935, step_size=1.0), 28530.0)
    assert math.isclose(quantize_qty(7430.767198, step_size=0.01), 7430.76)
    assert quantize_qty(0.5, step_size=1.0) == 0.0


def test_quantize_price_none_tick_passthrough():
    assert quantize_price(100.123, tick_size=None, side="buy") == 100.123
    assert quantize_price(100.123, tick_size=0.0, side="sell") == 100.123


def test_split_binance_symbol_common_pairs():
    assert split_binance_symbol("BTCUSDT") == ("BTC", "USDT")
    assert split_binance_symbol("ETHFDUSD") == ("ETH", "FDUSD")
    assert split_binance_symbol("SOLUSDT") == ("SOL", "USDT")
    assert split_binance_symbol("DOGEUSDT") == ("DOGE", "USDT")


def test_quantize_price_buy_rounds_down_sell_rounds_up():
    assert math.isclose(quantize_price(0.2644, tick_size=0.001, side="buy"), 0.264)
    assert math.isclose(quantize_price(0.2644, tick_size=0.001, side="sell"), 0.265)
    assert math.isclose(quantize_price(82000.123, tick_size=0.01, side="buy"), 82000.12)
    assert math.isclose(quantize_price(82000.123, tick_size=0.01, side="sell"), 82000.13)
