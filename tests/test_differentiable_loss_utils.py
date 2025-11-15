import torch

from differentiable_loss_utils import (
    DEFAULT_MAKER_FEE_RATE,
    approx_buy_fill_probability,
    approx_sell_fill_probability,
    combined_sortino_pnl_loss,
    simulate_hourly_trades,
)


def test_fill_probability_is_monotonic() -> None:
    close = torch.tensor([10.0])
    low = torch.tensor([9.5])
    high = torch.tensor([10.5])
    buy_lo = approx_buy_fill_probability(torch.tensor([9.4]), low, close)
    buy_hi = approx_buy_fill_probability(torch.tensor([9.9]), low, close)
    sell_lo = approx_sell_fill_probability(torch.tensor([10.6]), high, close)
    sell_hi = approx_sell_fill_probability(torch.tensor([10.2]), high, close)
    assert torch.all(buy_hi > buy_lo)
    assert torch.all(sell_hi > sell_lo)


def test_simulation_respects_cash_and_inventory_limits() -> None:
    highs = torch.tensor([1.05, 1.30])
    lows = torch.tensor([0.90, 1.00])
    closes = torch.tensor([1.00, 1.20])
    buy_prices = torch.tensor([0.95, 1.10])
    sell_prices = torch.tensor([1.10, 1.25])
    buy_amounts = torch.tensor([10.0, 0.0])
    sell_amounts = torch.tensor([0.0, 10.0])

    result = simulate_hourly_trades(
        highs=highs,
        lows=lows,
        closes=closes,
        buy_prices=buy_prices,
        sell_prices=sell_prices,
        buy_amounts=buy_amounts,
        sell_amounts=sell_amounts,
        initial_cash=1.0,
        maker_fee=DEFAULT_MAKER_FEE_RATE,
    )

    # First hour should buy nearly entire cash balance (no leverage)
    expected_max_buy = 1.0 / (buy_prices[0] * (1 + DEFAULT_MAKER_FEE_RATE))
    assert torch.isclose(result.executed_buys[..., 0], torch.tensor(expected_max_buy), rtol=1e-3, atol=1e-3)

    # Second hour can only sell what was bought
    assert torch.isclose(result.executed_sells[..., 1], result.executed_buys[..., 0], rtol=1e-3, atol=1e-3)
    assert result.inventory.abs() < 1e-4
    assert result.cash > 1.0  # Profitable trade


def test_combined_sortino_prefers_positive_returns() -> None:
    good_returns = torch.full((24,), 0.001)
    bad_returns = torch.full((24,), -0.001)
    good_loss = combined_sortino_pnl_loss(good_returns)
    bad_loss = combined_sortino_pnl_loss(bad_returns)
    assert good_loss < bad_loss
