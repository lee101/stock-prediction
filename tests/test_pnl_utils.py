from src.pnl_utils import compute_trade_pnl


def test_long_realized_unrealized_no_fees():
    fills = [
        {"symbol": "AAA", "side": "buy", "qty": 10, "price": 100},
        {"symbol": "AAA", "side": "sell", "qty": 4, "price": 110},
    ]
    current_prices = {"AAA": 105}
    fee_rates = {"AAA": 0.0}
    summary = compute_trade_pnl(
        fills,
        current_prices=current_prices,
        fee_rate_by_symbol=fee_rates,
    )

    assert round(summary.realized, 6) == 40.0
    assert round(summary.unrealized, 6) == 30.0
    assert round(summary.net, 6) == 70.0


def test_short_realized_unrealized_no_fees():
    fills = [
        {"symbol": "BBB", "side": "sell", "qty": 5, "price": 200},
        {"symbol": "BBB", "side": "buy", "qty": 2, "price": 180},
    ]
    current_prices = {"BBB": 190}
    fee_rates = {"BBB": 0.0}
    summary = compute_trade_pnl(
        fills,
        current_prices=current_prices,
        fee_rate_by_symbol=fee_rates,
    )

    assert round(summary.realized, 6) == 40.0
    assert round(summary.unrealized, 6) == 30.0
    assert round(summary.net, 6) == 70.0


def test_flip_long_to_short_with_fees():
    fills = [
        {"symbol": "CCC", "side": "buy", "qty": 5, "price": 100},
        {"symbol": "CCC", "side": "sell", "qty": 10, "price": 110},
    ]
    current_prices = {"CCC": 105}
    fee_rates = {"CCC": 0.001}  # 10 bps
    summary = compute_trade_pnl(
        fills,
        current_prices=current_prices,
        fee_rate_by_symbol=fee_rates,
    )

    # Realized: (110-100)*5 = 50
    # Unrealized: short 5 @110, price 105 -> (110-105)*5 = 25
    assert round(summary.realized, 6) == 50.0
    assert round(summary.unrealized, 6) == 25.0

    # Fees: (5*100 + 10*110) * 0.001 = 1.6
    assert round(summary.fees, 6) == 1.6
    assert round(summary.net, 6) == round(75.0 - 1.6, 6)
