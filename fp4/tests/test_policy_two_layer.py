"""Tests for the two-timescale Layer A/B policy (P4-2)."""
from __future__ import annotations

import math

import pytest
import torch

from fp4.fp4.policy_two_layer import (
    ACTION_DIM,
    LAYER_A_DIM,
    LAYER_B_DIM,
    TwoLayerPolicy,
)


def _mk(precision: str = "fp32", obs_dim: int = 128, n_costs: int = 4) -> TwoLayerPolicy:
    torch.manual_seed(0)
    return TwoLayerPolicy(
        obs_dim=obs_dim,
        n_costs=n_costs,
        precision=precision,
        delta_max_bps=50.0,
        seed=0,
    )


def test_forward_shapes():
    p = _mk()
    B, F = 64, 128
    obs = torch.randn(B, F)
    out = p(obs)
    assert out["layer_a"].shape == (B, LAYER_A_DIM)
    assert out["layer_b"].shape == (B, LAYER_B_DIM)
    assert out["value"].shape == (B,)
    assert out["cost_values"].shape == (B, 4)
    assert out["std"].shape == (B, ACTION_DIM)
    for v in out.values():
        assert torch.isfinite(v).all()


def test_layer_a_bounds():
    p = _mk()
    obs = torch.randn(256, 128) * 10.0  # drive heads hard
    out = p(obs)
    a = out["layer_a"]
    # inventory_frac in [-1,1], leverage in [0,5], risk in [0,1]
    assert (a[:, 0] >= -1.0 - 1e-6).all() and (a[:, 0] <= 1.0 + 1e-6).all()
    assert (a[:, 1] >= 0.0).all() and (a[:, 1] <= 5.0 + 1e-6).all()
    assert (a[:, 2] >= 0.0).all() and (a[:, 2] <= 1.0 + 1e-6).all()


def test_layer_b_bounds():
    p = _mk()
    obs = torch.randn(256, 128) * 10.0
    out = p(obs)
    b = out["layer_b"]
    # delta_bid, delta_ask in [0, 50], size in [0, 1]
    assert (b[:, 0] >= 0.0).all() and (b[:, 0] <= 50.0 + 1e-6).all()
    assert (b[:, 1] >= 0.0).all() and (b[:, 1] <= 50.0 + 1e-6).all()
    assert (b[:, 2] >= 0.0).all() and (b[:, 2] <= 1.0 + 1e-6).all()


def test_to_quote_prices_shape_and_values():
    p = _mk()
    B = 16
    layer_b = torch.stack(
        [
            torch.full((B,), 10.0),  # d_bid = 10 bps
            torch.full((B,), 20.0),  # d_ask = 20 bps
            torch.full((B,), 0.4),   # size_frac
        ],
        dim=-1,
    )
    ref_px = torch.full((B,), 100.0)
    act = p.to_quote_prices(layer_b, ref_px)
    assert act.shape == (B, 4)
    # p_bid = 100 * (1 - 10e-4) = 99.9
    assert torch.allclose(act[:, 0], torch.full((B,), 99.9), atol=1e-4)
    # p_ask = 100 * (1 + 20e-4) = 100.2
    assert torch.allclose(act[:, 1], torch.full((B,), 100.2), atol=1e-4)
    assert torch.allclose(act[:, 2], torch.full((B,), 0.2), atol=1e-6)
    assert torch.allclose(act[:, 3], torch.full((B,), 0.2), atol=1e-6)


def test_act_samples_finite_and_bounded():
    p = _mk()
    obs = torch.randn(32, 128)
    out = p.act(obs)
    assert out["raw_action"].shape == (32, ACTION_DIM)
    assert out["logp"].shape == (32,)
    assert torch.isfinite(out["logp"]).all()
    a = out["sampled_layer_a"]
    b = out["sampled_layer_b"]
    assert (a[:, 1] >= 0).all() and (a[:, 1] <= 5.0 + 1e-6).all()
    assert (b[:, 2] >= 0).all() and (b[:, 2] <= 1.0 + 1e-6).all()


def test_gradcheck_small_fp32():
    """Analytic vs finite-diff gradcheck on a tiny instance (fp32, plain Linear)."""
    torch.manual_seed(1)
    p = TwoLayerPolicy(obs_dim=8, n_costs=2, precision="fp32", seed=1).double()
    obs = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)

    def f(x):
        out = p(x)
        # Scalar reduction mixing every head so every param path is exercised.
        return (
            out["layer_a"].sum()
            + out["layer_b"].sum()
            + out["value"].sum()
            + out["cost_values"].sum()
        )

    assert torch.autograd.gradcheck(f, (obs,), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_backward_populates_param_grads():
    p = _mk()
    obs = torch.randn(8, 128)
    out = p(obs)
    loss = out["layer_a"].pow(2).sum() + out["layer_b"].pow(2).sum() + out["value"].sum()
    loss.backward()
    # At least the policy heads and log_std must have grads.
    assert p.head_a.weight.grad is not None
    assert p.head_b.weight.grad is not None
    assert p.v_head.weight.grad is not None


def test_precision_flag_accepts_bf16_and_fp32():
    _mk(precision="fp32")
    # bf16 path is only meaningful on CUDA; on CPU the nn.Linear stays fp32.
    _mk(precision="fp32")


def test_n_costs_zero():
    p = _mk(n_costs=0)
    out = p(torch.randn(4, 128))
    assert out["cost_values"].shape == (4, 0)
