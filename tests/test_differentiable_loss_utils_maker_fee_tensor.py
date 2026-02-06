import torch

from differentiable_loss_utils import simulate_hourly_trades


def test_simulate_hourly_trades_accepts_tensor_maker_fee() -> None:
    highs = torch.ones((2, 3), dtype=torch.float32)
    lows = torch.ones((2, 3), dtype=torch.float32)
    closes = torch.ones((2, 3), dtype=torch.float32)
    buy_prices = torch.ones((2, 3), dtype=torch.float32)
    sell_prices = torch.ones((2, 3), dtype=torch.float32)
    trade_intensity = torch.ones((2, 3), dtype=torch.float32)

    maker_fee = torch.tensor([0.0, 0.01], dtype=torch.float32)
    result = simulate_hourly_trades(
        highs=highs,
        lows=lows,
        closes=closes,
        buy_prices=buy_prices,
        sell_prices=sell_prices,
        trade_intensity=trade_intensity,
        maker_fee=maker_fee,
        initial_cash=1.0,
        can_long=True,
        can_short=False,
    )
    assert result.portfolio_values.shape == (2, 3)
    # Higher fee should reduce portfolio value when we buy (sell is disabled without inventory).
    assert float(result.portfolio_values[1, -1]) < float(result.portfolio_values[0, -1])

    fee_val = 0.001
    scalar = simulate_hourly_trades(
        highs=highs,
        lows=lows,
        closes=closes,
        buy_prices=buy_prices,
        sell_prices=sell_prices,
        trade_intensity=trade_intensity,
        maker_fee=fee_val,
        initial_cash=1.0,
        can_long=True,
        can_short=False,
    )
    vector = simulate_hourly_trades(
        highs=highs,
        lows=lows,
        closes=closes,
        buy_prices=buy_prices,
        sell_prices=sell_prices,
        trade_intensity=trade_intensity,
        maker_fee=torch.full((2,), fee_val, dtype=torch.float32),
        initial_cash=1.0,
        can_long=True,
        can_short=False,
    )
    assert torch.allclose(scalar.portfolio_values, vector.portfolio_values, rtol=1e-6, atol=1e-6)

