import torch

from src.forecast_math import absolute_prices_to_pct_returns


def test_absolute_to_pct_returns_matches_manual():
    abs_prices = [101.0, 102.0, 100.0]
    pct = absolute_prices_to_pct_returns(abs_prices, last_price=100.0)
    expected = torch.tensor([(101 - 100) / 100, (102 - 101) / 101, (100 - 102) / 102], dtype=torch.float32)
    assert torch.allclose(pct, expected, atol=1e-7)


def test_absolute_to_pct_handles_zero_last_price():
    abs_prices = [0.0, 5.0]
    pct = absolute_prices_to_pct_returns(abs_prices, last_price=0.0)
    assert pct.tolist() == [0.0, 0.0]
