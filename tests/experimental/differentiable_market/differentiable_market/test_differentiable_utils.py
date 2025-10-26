from __future__ import annotations

import math

import torch

from differentiable_market.differentiable_utils import (
    TradeMemoryState,
    augment_market_features,
    haar_wavelet_pyramid,
    risk_budget_mismatch,
    soft_drawdown,
    taylor_time_encoding,
    trade_memory_update,
)


def test_taylor_time_encoding_gradients() -> None:
    steps = torch.linspace(0, 31, steps=32, requires_grad=True)
    encoding = taylor_time_encoding(steps, order=3, scale=16.0)
    assert encoding.shape == (32, 3)
    loss = encoding.mean()
    loss.backward()
    assert steps.grad is not None
    assert torch.all(torch.isfinite(steps.grad))


def test_haar_wavelet_levels() -> None:
    series = torch.randn(2, 3, 64, requires_grad=True)
    approx, details = haar_wavelet_pyramid(series, levels=2)
    assert len(details) == 2
    assert approx.shape == (2, 3, 16)
    assert details[0].shape == (2, 3, 32)
    assert details[1].shape == (2, 3, 16)

    objective = approx.pow(2).mean() + sum(detail.abs().mean() for detail in details)
    objective.backward()
    assert series.grad is not None
    assert torch.all(torch.isfinite(series.grad))


def test_soft_drawdown_behaviour() -> None:
    returns = torch.tensor([[0.1, -0.2, 0.05, -0.1]], requires_grad=True)
    wealth, drawdown = soft_drawdown(returns, smoothing=20.0)
    assert wealth.shape == returns.shape
    assert drawdown.shape == returns.shape
    assert drawdown.max() <= 1.0 + 1e-5
    loss = (wealth + drawdown).sum()
    loss.backward()
    assert returns.grad is not None


def test_risk_budget_mismatch_zero_for_equal_weights() -> None:
    weights = torch.tensor([0.25, 0.25, 0.25, 0.25], requires_grad=True)
    cov = torch.eye(4) * 0.5
    target = torch.ones(4)
    penalty = risk_budget_mismatch(weights, cov, target)
    assert math.isclose(penalty.detach().item(), 0.0, abs_tol=1e-6)
    penalty.backward()
    assert weights.grad is not None


def test_trade_memory_update_signals() -> None:
    pnl = torch.tensor([0.1, -0.2, -0.3, 0.5], requires_grad=True)
    state: TradeMemoryState | None = None
    regrets = []
    leverages = []
    for value in pnl:
        state, regret, leverage = trade_memory_update(state, value)
        regrets.append(regret)
        leverages.append(leverage)
    assert state is not None
    assert state.steps.shape == ()
    total = torch.stack(regrets).sum() + torch.stack(leverages).sum()
    total.backward()
    assert pnl.grad is not None
    assert torch.all(torch.isfinite(pnl.grad))


def test_augment_market_features_shapes_and_gradients() -> None:
    base_feat = torch.randn(32, 3, 4, requires_grad=True)
    returns = torch.randn(32, 3, requires_grad=True)
    augmented = augment_market_features(
        base_feat,
        returns,
        use_taylor=True,
        taylor_order=2,
        taylor_scale=16.0,
        use_wavelet=True,
        wavelet_levels=1,
    )
    assert augmented.shape[-1] == 8
    loss = augmented.sum()
    loss.backward()
    assert base_feat.grad is not None
    assert torch.all(torch.isfinite(base_feat.grad))
    assert returns.grad is not None
    assert torch.all(torch.isfinite(returns.grad))
