"""Tests for Unit P4-3: constrained-MDP losses + Lagrangian multiplier."""
from __future__ import annotations

import math

import pytest
import torch

from fp4.losses import (
    cvar_loss,
    entropy_bonus,
    ppo_clipped_surrogate,
    smooth_pnl_loss,
    value_loss_mse,
)
from fp4.lagrangian import Lagrangian


# ---------------------------------------------------------------------------
# PPO building blocks
# ---------------------------------------------------------------------------

def test_ppo_surrogate_zero_when_ratio_is_one_and_adv_zero():
    logp = torch.zeros(32)
    adv = torch.zeros(32)
    loss = ppo_clipped_surrogate(logp, logp.clone(), adv, clip_eps=0.2)
    assert torch.isfinite(loss)
    # Zero advantages → zero surrogate regardless of ratio.
    assert loss.abs().item() < 1e-6


def test_ppo_surrogate_differentiable():
    logp_new = torch.zeros(16, requires_grad=True)
    logp_old = torch.zeros(16)
    adv = torch.randn(16)
    loss = ppo_clipped_surrogate(logp_new, logp_old, adv)
    loss.backward()
    assert logp_new.grad is not None
    assert torch.isfinite(logp_new.grad).all()


def test_value_loss_mse_matches_torch():
    v = torch.randn(64, requires_grad=True)
    r = torch.randn(64)
    loss = value_loss_mse(v, r)
    expected = torch.nn.functional.mse_loss(v, r)
    assert torch.allclose(loss, expected)


def test_entropy_bonus_std_tensor_closed_form():
    std = torch.full((8, 3), 1.0)
    ent = entropy_bonus(std)
    # H of N(0,I_3) = 0.5 * log((2*pi*e)^3) = 1.5 * log(2*pi*e)
    expected = 1.5 * math.log(2.0 * math.pi * math.e)
    assert abs(ent.item() - expected) < 1e-5


def test_entropy_bonus_distribution_object():
    mean = torch.zeros(4, 2)
    std = torch.ones(4, 2)
    dist = torch.distributions.Normal(mean, std)
    ent = entropy_bonus(dist)
    expected = 0.5 * math.log(2.0 * math.pi * math.e)  # Normal.entropy() is per-element
    assert abs(ent.item() - expected) < 1e-5


# ---------------------------------------------------------------------------
# CVaR
# ---------------------------------------------------------------------------

def test_cvar_uniform_analytic():
    # For U[0,1], CVaR_alpha = alpha/2 (mean of lower alpha-tail).
    torch.manual_seed(0)
    n = 200_000
    x = torch.rand(n)
    alpha = 0.05
    cv = cvar_loss(x, alpha=alpha).item()
    analytic = alpha / 2.0  # 0.025
    assert abs(cv - analytic) < 5e-3, f"cvar={cv}, expected≈{analytic}"


def test_cvar_differentiable_and_gpu_safe():
    x = torch.randn(1024, requires_grad=True)
    cv = cvar_loss(x, alpha=0.1)
    cv.backward()
    assert x.grad is not None
    # Exactly floor(0.1*1024)=102 elements should have non-zero grad.
    nonzero = (x.grad != 0).sum().item()
    assert nonzero == 102


def test_cvar_bad_alpha():
    with pytest.raises(ValueError):
        cvar_loss(torch.zeros(10), alpha=0.0)
    with pytest.raises(ValueError):
        cvar_loss(torch.zeros(10), alpha=1.5)


# ---------------------------------------------------------------------------
# Smooth PnL
# ---------------------------------------------------------------------------

def test_smooth_pnl_zero_on_linear_trajectory():
    B, T = 4, 32
    slopes = torch.tensor([0.001, 0.0, -0.002, 0.005]).view(B, 1)
    log_eq = slopes * torch.arange(T, dtype=torch.float32).view(1, T)
    loss = smooth_pnl_loss(log_eq)
    assert loss.item() < 1e-12


def test_smooth_pnl_positive_on_curved_trajectory():
    log_eq = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])  # spike → nonzero curvature
    loss = smooth_pnl_loss(log_eq)
    assert loss.item() > 0.0


def test_smooth_pnl_differentiable():
    log_eq = torch.randn(2, 10, requires_grad=True)
    loss = smooth_pnl_loss(log_eq)
    loss.backward()
    assert log_eq.grad is not None
    assert torch.isfinite(log_eq.grad).all()


# ---------------------------------------------------------------------------
# Lagrangian
# ---------------------------------------------------------------------------

def test_lagrangian_apply_returns_weighted_slack():
    lag = Lagrangian(["dd", "lev"], init_lambda=0.5, lr_lambda=1e-2,
                     slow_update_every=1, target_d={"dd": 0.1, "lev": 0.2})
    costs = {"dd": torch.tensor(0.3), "lev": torch.tensor(0.5)}
    out = lag.apply(costs)
    # 0.5*(0.3-0.1) + 0.5*(0.5-0.2) = 0.1 + 0.15 = 0.25
    assert abs(out.item() - 0.25) < 1e-6


def test_lagrangian_apply_differentiable_wrt_costs():
    lag = Lagrangian(["c"], init_lambda=2.0, lr_lambda=1e-2, slow_update_every=1,
                     target_d={"c": 0.0})
    cost = torch.tensor(0.5, requires_grad=True)
    out = lag.apply({"c": cost})
    out.backward()
    assert cost.grad is not None
    assert abs(cost.grad.item() - 2.0) < 1e-6  # d/dcost (lam * (cost - d)) = lam


def test_lagrangian_converges_when_feasible():
    """λ should converge to 0 when cost is consistently below target."""
    lag = Lagrangian(["c"], init_lambda=1.0, lr_lambda=0.1, slow_update_every=1,
                     target_d={"c": 0.5})
    for _ in range(200):
        lag.step({"c": 0.2})  # well below target
    assert lag.lambdas["c"] == 0.0  # clamped at 0


def test_lagrangian_grows_when_infeasible():
    """λ should grow when cost is consistently above target."""
    lag = Lagrangian(["c"], init_lambda=0.0, lr_lambda=0.1, slow_update_every=1,
                     target_d={"c": 0.5})
    for _ in range(200):
        lag.step({"c": 0.8})  # above target
    # Expected growth ≈ 200 * 0.1 * 0.3 = 6.0
    assert lag.lambdas["c"] > 5.0
    assert lag.lambdas["c"] < 7.0


def test_lagrangian_slow_update_every():
    lag = Lagrangian(["c"], init_lambda=0.0, lr_lambda=0.1, slow_update_every=5,
                     target_d={"c": 0.0})
    for _ in range(4):
        lag.step({"c": 1.0})
    assert lag.lambdas["c"] == 0.0  # no update yet
    lag.step({"c": 1.0})  # 5th call → fires
    assert abs(lag.lambdas["c"] - 0.1) < 1e-9


def test_lagrangian_state_dict_roundtrip():
    lag = Lagrangian(["a", "b"], init_lambda=0.0, lr_lambda=0.05,
                     slow_update_every=2, target_d={"a": 0.1, "b": -0.2})
    for _ in range(10):
        lag.step({"a": 0.3, "b": 0.0})
    state = lag.state_dict()

    lag2 = Lagrangian(["a", "b"], init_lambda=99.0, lr_lambda=0.01,
                      slow_update_every=1, target_d={"a": 0.0, "b": 0.0})
    lag2.load_state_dict(state)
    assert lag2.lambdas == lag.lambdas
    assert lag2.target_d == lag.target_d
    assert lag2.lr_lambda == lag.lr_lambda
    assert lag2.slow_update_every == lag.slow_update_every


def test_lagrangian_empty_names_rejected():
    with pytest.raises(ValueError):
        Lagrangian([], init_lambda=0.0, lr_lambda=0.1, slow_update_every=1,
                   target_d={})


def test_lagrangian_missing_constraint_in_apply_raises():
    lag = Lagrangian(["c"], init_lambda=0.0, lr_lambda=0.1, slow_update_every=1,
                     target_d={"c": 0.0})
    with pytest.raises(KeyError):
        lag.apply({})
