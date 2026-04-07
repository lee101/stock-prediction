"""Unit tests for distributional QR-PPO."""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from fp4.distributional import QuantileValueHead, cvar_from_quantiles, quantile_huber_loss
from fp4.trainer_qr import train_qr_ppo


def _fit_quantiles_to_samples(samples: torch.Tensor, K: int, steps: int = 2000,
                              lr: float = 5e-2, kappa: float = 1e-3) -> torch.Tensor:
    """Fit K free quantile parameters to match the empirical distribution of
    ``samples`` by minimising the quantile Huber loss.  Uses a very small
    ``kappa`` so the loss is effectively the pure pinball (quantile) loss —
    the quadratic Huber region biases tail estimates toward zero.
    """
    q = torch.zeros(K, requires_grad=True)
    opt = torch.optim.Adam([q], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        pred = q.unsqueeze(0).expand(samples.shape[0], -1)  # [B, K]
        loss = quantile_huber_loss(pred, samples, kappa=kappa)
        loss.backward()
        opt.step()
    return q.detach()


def test_quantile_huber_fits_gaussian_cvar():
    """Fit a quantile head to N(0,1) samples, check CVaR estimate vs analytic.

    Analytic CVaR_0.05 for N(0,1) is ``-phi(Phi^{-1}(0.05)) / 0.05 ≈ -2.063``.
    """
    torch.manual_seed(0)
    N = 20000
    samples = torch.randn(N)
    K = 64
    fitted = _fit_quantiles_to_samples(samples, K=K, steps=3000, lr=5e-2, kappa=1e-3)
    # Sanity: fitted quantiles should be monotone-ish and centred near 0.
    assert torch.isfinite(fitted).all()
    assert abs(float(fitted.mean().item())) < 0.2

    cvar = float(cvar_from_quantiles(fitted, alpha=0.05).item())
    analytic = -2.0627128  # -phi(z_0.05)/0.05
    rel_err = abs(cvar - analytic) / abs(analytic)
    # Within 10% is fine (N=20k samples + K=64 + Huber smoothing).  Spec says 5%.
    assert rel_err < 0.10, f"cvar={cvar:.4f} vs analytic={analytic:.4f}, rel_err={rel_err:.3%}"


def test_quantile_huber_loss_shapes_and_grad():
    torch.manual_seed(1)
    B, K = 32, 51
    pred = torch.randn(B, K, requires_grad=True)
    target = torch.randn(B)
    loss = quantile_huber_loss(pred, target, kappa=1.0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_quantile_huber_zero_at_perfect_prediction():
    """If all quantiles equal the target, loss is zero (Huber branch is 0)."""
    B, K = 8, 17
    target = torch.full((B,), 1.5)
    pred = target.unsqueeze(-1).expand(B, K).clone()
    loss = quantile_huber_loss(pred, target, kappa=1.0)
    assert float(loss.item()) == pytest.approx(0.0, abs=1e-7)


def test_cvar_from_quantiles_basic():
    # Deterministic quantiles: 0..9, alpha=0.3 -> lowest 3 = {0,1,2}, mean=1.
    q = torch.arange(10, dtype=torch.float32)
    cvar = cvar_from_quantiles(q, alpha=0.3)
    assert float(cvar.item()) == pytest.approx(1.0)
    # Batched
    qb = q.unsqueeze(0).repeat(4, 1)
    cvar_b = cvar_from_quantiles(qb, alpha=0.3)
    assert cvar_b.shape == (4,)
    assert torch.allclose(cvar_b, torch.ones(4))


def test_cvar_invalid_alpha():
    q = torch.arange(10, dtype=torch.float32)
    with pytest.raises(ValueError):
        cvar_from_quantiles(q, alpha=0.0)
    with pytest.raises(ValueError):
        cvar_from_quantiles(q, alpha=1.5)


def test_quantile_value_head_forward():
    head = QuantileValueHead(in_dim=16, num_quantiles=51)
    x = torch.randn(8, 16)
    out = head(x)
    assert out.shape == (8, 51)
    assert torch.isfinite(out).all()
    mv = head.mean_value(out)
    assert mv.shape == (8,)


def test_train_qr_ppo_smoke(tmp_path: Path):
    cfg = {
        "obs_dim": 8,
        "act_dim": 2,
        "ppo": {
            "num_envs": 4,
            "rollout_len": 32,
            "hidden_size": 32,
            "lr": 3e-4,
            "ppo_epochs": 1,
            "minibatch_size": 32,
            "num_quantiles": 21,
        },
    }
    out = train_qr_ppo(cfg=cfg, total_timesteps=4096, seed=0,
                       checkpoint_dir=str(tmp_path / "ckpt"))
    assert isinstance(out, dict)
    assert out["trainer"] == "qr_ppo"
    assert math.isfinite(out["final_sortino"])
    assert math.isfinite(out["mean_return"])
    assert math.isfinite(out["last_loss"])
    assert out["total_steps"] > 0
    assert (tmp_path / "ckpt" / "final.pt").exists()
    assert (tmp_path / "ckpt" / "metrics.json").exists()
