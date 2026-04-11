"""Tests for position-target rebalancing simulator."""
import torch
import pytest
from differentiable_loss_utils import simulate_rebalance


def test_zero_allocation_stays_cash():
    closes = torch.tensor([100.0, 101.0, 102.0, 100.0])
    alloc = torch.zeros(4)
    result = simulate_rebalance(closes=closes, allocation=alloc, initial_cash=1000.0)
    assert result.portfolio_values[-1].item() == pytest.approx(1000.0, abs=0.01)
    assert result.inventory.item() == pytest.approx(0.0, abs=1e-6)


def test_full_allocation_tracks_asset():
    closes = torch.tensor([100.0, 110.0, 121.0])
    alloc = torch.ones(3)
    result = simulate_rebalance(closes=closes, allocation=alloc, maker_fee=0.0, initial_cash=1000.0)
    # Should track asset price minus rebalancing friction
    # First bar: buy 10 units at 100, fully invested
    # Price goes to 110: equity = 10 * 110 = 1100, rebalance to 10 units (no change)
    # Price goes to 121: equity = 10 * 121 = 1210
    assert result.portfolio_values[-1].item() == pytest.approx(1210.0, abs=1.0)


def test_fees_reduce_returns():
    closes = torch.tensor([100.0, 100.0, 100.0])
    alloc = torch.ones(3)
    result_nofee = simulate_rebalance(closes=closes, allocation=alloc, maker_fee=0.0, initial_cash=1000.0)
    result_fee = simulate_rebalance(closes=closes, allocation=alloc, maker_fee=0.01, initial_cash=1000.0)
    # With fees, buying costs more
    assert result_fee.portfolio_values[-1].item() < result_nofee.portfolio_values[-1].item()


def test_decision_lag():
    closes = torch.tensor([100.0, 105.0, 110.0, 100.0, 95.0])
    alloc = torch.ones(5)
    result_lag0 = simulate_rebalance(closes=closes, allocation=alloc, maker_fee=0.0, decision_lag_bars=0)
    result_lag2 = simulate_rebalance(closes=closes, allocation=alloc, maker_fee=0.0, decision_lag_bars=2)
    # With lag=2: alloc from t=0,1,2 applied to bars t=2,3,4
    # Lag reduces number of steps
    assert result_lag0.returns.shape[-1] == 5
    assert result_lag2.returns.shape[-1] == 3


def test_allocation_switching():
    closes = torch.tensor([100.0, 110.0, 105.0, 115.0, 100.0])
    # Go full in, then out, then in, then out
    alloc = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0])
    result = simulate_rebalance(closes=closes, allocation=alloc, maker_fee=0.0, initial_cash=1000.0)
    # Should have traded and ended in cash
    assert result.inventory.item() == pytest.approx(0.0, abs=1e-4)
    assert result.portfolio_values[-1].item() > 0


def test_batch_dimension():
    closes = torch.randn(4, 20).abs() * 100 + 50
    alloc = torch.rand(4, 20)
    result = simulate_rebalance(closes=closes, allocation=alloc)
    assert result.returns.shape == (4, 20)
    assert result.portfolio_values.shape == (4, 20)


def test_gradients_flow():
    closes = torch.tensor([100.0, 105.0, 102.0, 108.0, 103.0])
    alloc_logit = torch.randn(5, requires_grad=True)
    alloc = torch.sigmoid(alloc_logit)
    result = simulate_rebalance(closes=closes, allocation=alloc, maker_fee=0.001)
    loss = -result.returns.mean()
    loss.backward()
    assert alloc_logit.grad is not None
    assert not torch.all(alloc_logit.grad == 0)


def test_uses_opens_for_execution():
    closes = torch.tensor([100.0, 110.0, 120.0])
    opens = torch.tensor([100.0, 105.0, 115.0])
    alloc = torch.ones(3)
    result = simulate_rebalance(closes=closes, opens=opens, allocation=alloc, maker_fee=0.0, initial_cash=1000.0)
    # First bar: buy at open=100, 10 units, close=100, value=1000
    # Second bar: rebalance at open=105, equity ~= 10*105=1050, target=1050/105=10, no rebal needed
    # Value at close=110: 10*110 = 1100
    # Third bar: rebalance at open=115, equity ~= 10*115=1150, target=1150/115=10, no rebal
    # Value at close=120: 10*120 = 1200
    assert result.portfolio_values[-1].item() == pytest.approx(1200.0, abs=1.0)


def test_max_leverage_caps_allocation():
    closes = torch.tensor([100.0, 100.0, 100.0])
    alloc = torch.tensor([2.0, 2.0, 2.0])  # Wants 2x leverage
    result_1x = simulate_rebalance(closes=closes, allocation=alloc, max_leverage=1.0, maker_fee=0.0, initial_cash=1000.0)
    result_2x = simulate_rebalance(closes=closes, allocation=alloc, max_leverage=2.0, maker_fee=0.0, initial_cash=1000.0)
    # At 1x: allocation capped to 1.0, at 2x: allocation kept at 2.0
    assert result_1x.inventory.item() < result_2x.inventory.item()
