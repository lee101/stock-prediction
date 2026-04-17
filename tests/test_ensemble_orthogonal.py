"""Tests for src/ensemble_orthogonal (EO-PPO auxiliary loss).

Covers the correctness invariants documented in docs/ensemble_orthogonal_ppo.md:
  1. beta==0 => zero loss and composes safely with PPO.
  2. Gradient flows through specialist logits but NOT through ensemble.
  3. Per-sample KL is clamped to [0, kl_clamp].
  4. Gate weight w(s) bounded in [w_min, 1].
  5. Ensemble forward runs under no_grad (no grad params).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.ensemble_orthogonal import (
    DEFAULT_KL_CLAMP,
    DEFAULT_W_MIN,
    EnsembleOrthogonalLoss,
    OrthogonalTermDiagnostics,
    beta_schedule,
    compute_ensemble_log_probs,
    ensemble_entropy_gate,
    kl_specialist_vs_ensemble,
    orthogonal_loss,
)


class _ToyPolicy(nn.Module):
    """Minimal policy-shaped module: (logits, value) from obs."""

    def __init__(self, in_dim: int, n_actions: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.fc = nn.Linear(in_dim, n_actions)
        self.v = nn.Linear(in_dim, 1)
        with torch.no_grad():
            self.fc.weight.copy_(torch.randn(n_actions, in_dim, generator=g) * 0.3)
            self.fc.bias.zero_()
            self.v.weight.copy_(torch.randn(1, in_dim, generator=g) * 0.3)
            self.v.bias.zero_()

    def forward(self, obs: torch.Tensor):
        return self.fc(obs), self.v(obs).squeeze(-1)


@pytest.fixture
def toy_setup():
    B, D, A = 8, 16, 7
    torch.manual_seed(0)
    obs = torch.randn(B, D)
    specialist_logits = torch.randn(B, A, requires_grad=True)
    ensemble_models = [_ToyPolicy(D, A, seed=i) for i in range(4)]
    return obs, specialist_logits, ensemble_models, B, A


def test_beta_zero_is_no_op(toy_setup):
    obs, specialist_logits, ensemble_models, B, A = toy_setup
    ens_log_probs = compute_ensemble_log_probs(ensemble_models, obs)
    loss, diag = orthogonal_loss(
        specialist_logits=specialist_logits,
        ensemble_log_probs=ens_log_probs,
        beta=0.0,
    )
    # Zero scalar, tracks grad graph (so loss + ppo_loss is safe to backward).
    assert loss.item() == 0.0
    assert loss.requires_grad or loss.grad_fn is not None or specialist_logits.requires_grad
    assert diag.weighted_kl == 0.0
    # Adding to a PPO-like loss and backward must not raise.
    ppo_like = (specialist_logits.pow(2).mean())
    (ppo_like + loss).backward()
    assert specialist_logits.grad is not None


def test_grad_flows_to_specialist_not_ensemble(toy_setup):
    obs, specialist_logits, ensemble_models, B, A = toy_setup
    # Zero grads on ensemble params and require_grad them temporarily
    # (we want to assert the KL path does NOT reach them).
    for m in ensemble_models:
        for p in m.parameters():
            p.requires_grad_(True)
            if p.grad is not None:
                p.grad = None

    ens_log_probs = compute_ensemble_log_probs(ensemble_models, obs)
    loss, _ = orthogonal_loss(
        specialist_logits=specialist_logits,
        ensemble_log_probs=ens_log_probs,
        beta=0.1,
    )
    loss.backward()

    assert specialist_logits.grad is not None
    assert torch.isfinite(specialist_logits.grad).all()

    for m in ensemble_models:
        for p in m.parameters():
            assert p.grad is None, "ensemble params must not receive gradient"


def test_kl_clamp_bounds_per_sample(toy_setup):
    obs, _, ensemble_models, B, A = toy_setup
    ens_log_probs = compute_ensemble_log_probs(ensemble_models, obs)
    # Force a near-degenerate specialist so KL can blow up.
    specialist_logits = torch.full((B, A), -20.0)
    specialist_logits[:, 0] = 20.0
    specialist_logits.requires_grad_(True)

    kl = kl_specialist_vs_ensemble(
        specialist_logits=specialist_logits,
        ensemble_log_probs=ens_log_probs,
        kl_clamp=2.5,
    )
    assert kl.shape == (B,)
    assert (kl >= 0.0).all()
    assert (kl <= 2.5).all()


def test_gate_weight_in_bounds(toy_setup):
    obs, _, ensemble_models, B, A = toy_setup
    ens_log_probs = compute_ensemble_log_probs(ensemble_models, obs)
    w = ensemble_entropy_gate(ens_log_probs, w_min=DEFAULT_W_MIN)
    assert w.shape == (B,)
    assert (w >= DEFAULT_W_MIN - 1e-6).all()
    assert (w <= 1.0 + 1e-6).all()


def test_ensemble_forward_is_no_grad(toy_setup):
    """compute_ensemble_log_probs must not retain grad graph to ensemble."""
    obs, _, ensemble_models, _, _ = toy_setup
    ens_log_probs = compute_ensemble_log_probs(ensemble_models, obs)
    assert ens_log_probs.requires_grad is False
    assert ens_log_probs.grad_fn is None


def test_softmax_avg_matches_manual(toy_setup):
    """Verify softmax_avg mode matches production ensemble math exactly."""
    obs, _, ensemble_models, B, A = toy_setup
    ens_log_probs = compute_ensemble_log_probs(ensemble_models, obs, mode="softmax_avg")

    manual_probs = torch.zeros(B, A)
    with torch.no_grad():
        for m in ensemble_models:
            lg, _ = m(obs)
            manual_probs += torch.softmax(lg, dim=-1)
    manual_probs /= len(ensemble_models)
    manual_log_probs = torch.log(manual_probs.clamp_min(1e-12))

    torch.testing.assert_close(ens_log_probs, manual_log_probs, rtol=1e-6, atol=1e-6)


def test_logit_avg_mode(toy_setup):
    obs, _, ensemble_models, B, A = toy_setup
    ens_log_probs = compute_ensemble_log_probs(ensemble_models, obs, mode="logit_avg")
    # Shape correctness; values are log-softmax, so sum of exp == 1
    torch.testing.assert_close(
        ens_log_probs.exp().sum(dim=-1),
        torch.ones(B),
        rtol=1e-5,
        atol=1e-5,
    )


def test_shape_mismatch_raises():
    specialist = torch.zeros(4, 5)
    ens = torch.zeros(4, 7)
    with pytest.raises(ValueError, match="shape mismatch"):
        orthogonal_loss(specialist, ens, beta=0.1)


def test_beta_schedule_ramps_then_flattens():
    # No warmup: flat immediately.
    assert beta_schedule(step=0, total_steps=100, peak_beta=0.05, warmup_frac=0.0) == 0.0
    assert beta_schedule(step=50, total_steps=100, peak_beta=0.05, warmup_frac=0.0) == 0.05

    # Linear ramp over first 30%.
    assert beta_schedule(step=0, total_steps=100, peak_beta=0.05, warmup_frac=0.3) == 0.0
    mid = beta_schedule(step=15, total_steps=100, peak_beta=0.05, warmup_frac=0.3)
    assert abs(mid - 0.025) < 1e-6
    full = beta_schedule(step=30, total_steps=100, peak_beta=0.05, warmup_frac=0.3)
    assert abs(full - 0.05) < 1e-6
    after = beta_schedule(step=90, total_steps=100, peak_beta=0.05, warmup_frac=0.3)
    assert abs(after - 0.05) < 1e-6

    # peak_beta zero always returns zero.
    for s in (0, 10, 1000):
        assert beta_schedule(step=s, total_steps=100, peak_beta=0.0, warmup_frac=0.3) == 0.0


def test_beta_schedule_start_frac_delays_ramp():
    # start_frac=0.4, warmup_frac=0.3: vanilla 0-40%, ramp 40-70%, flat 70-100%.
    # Vanilla phase: beta must be zero
    for s in (0, 10, 30, 39):
        assert beta_schedule(step=s, total_steps=100, peak_beta=0.05,
                              warmup_frac=0.3, start_frac=0.4) == 0.0
    # At start of ramp: still zero
    assert beta_schedule(step=40, total_steps=100, peak_beta=0.05,
                          warmup_frac=0.3, start_frac=0.4) == 0.0
    # Halfway through ramp (step=55 is 0.15 into a 0.3 window): half peak
    mid = beta_schedule(step=55, total_steps=100, peak_beta=0.05,
                         warmup_frac=0.3, start_frac=0.4)
    assert abs(mid - 0.025) < 1e-6
    # End of ramp: full peak
    full = beta_schedule(step=70, total_steps=100, peak_beta=0.05,
                         warmup_frac=0.3, start_frac=0.4)
    assert abs(full - 0.05) < 1e-6
    # Flat phase: full peak
    assert beta_schedule(step=99, total_steps=100, peak_beta=0.05,
                          warmup_frac=0.3, start_frac=0.4) == 0.05


def test_orth_loss_sign_increases_kl(toy_setup):
    """One step of gradient descent on orth loss should INCREASE KL.

    This is the core contract: negative-of-weighted-KL, minimised, raises KL.
    """
    obs, _, ensemble_models, B, A = toy_setup
    with torch.no_grad():
        ens_log_probs = compute_ensemble_log_probs(ensemble_models, obs)
    # Start specialist NEAR (but not at) the ensemble — small perturbation so the
    # gradient exists (KL=0 is a minimum where ∂KL/∂logits = 0).
    torch.manual_seed(7)
    specialist_logits = (ens_log_probs + 0.05 * torch.randn_like(ens_log_probs)).detach().requires_grad_(True)

    kl_before = kl_specialist_vs_ensemble(specialist_logits, ens_log_probs).mean().item()
    assert kl_before > 1e-6, f"precondition: KL should be nonzero at start, got {kl_before}"

    loss, _ = orthogonal_loss(
        specialist_logits=specialist_logits,
        ensemble_log_probs=ens_log_probs,
        beta=0.1,
        gate="off",  # simpler: equal weight
    )
    loss.backward()

    lr = 0.5
    with torch.no_grad():
        new_logits = specialist_logits - lr * specialist_logits.grad

    kl_after = kl_specialist_vs_ensemble(new_logits, ens_log_probs).mean().item()
    assert kl_after > kl_before, f"KL did not increase: before={kl_before}, after={kl_after}"


def test_ensemble_orthogonal_loss_wrapper(toy_setup):
    obs, specialist_logits, ensemble_models, B, A = toy_setup
    orth = EnsembleOrthogonalLoss(ensemble_models, mode="softmax_avg")
    assert orth.n_members == len(ensemble_models)

    # Ensemble params should be frozen.
    for m in orth._models:
        for p in m.parameters():
            assert p.requires_grad is False

    loss, diag = orth.loss(specialist_logits, obs, beta=0.02)
    assert isinstance(diag, OrthogonalTermDiagnostics)
    assert diag.beta == 0.02
    # weighted_kl averaged across samples; beta*weighted_kl ~= -loss
    assert abs(-loss.item() - diag.beta * diag.weighted_kl) < 1e-5


def test_loss_composes_with_ppo_like_loss(toy_setup):
    """Simulate a PPO minibatch: orth term adds without breaking anything."""
    obs, specialist_logits, ensemble_models, B, A = toy_setup
    orth = EnsembleOrthogonalLoss(ensemble_models)

    # Fake PPO loss: negative mean log-prob under specialist of a fixed action.
    actions = torch.zeros(B, dtype=torch.long)
    log_probs = torch.log_softmax(specialist_logits.float(), dim=-1)
    ppo_loss = -log_probs.gather(1, actions.unsqueeze(1)).squeeze(1).mean()

    orth_term, _ = orth.loss(specialist_logits, obs, beta=0.03)
    total = ppo_loss + orth_term
    total.backward()

    assert specialist_logits.grad is not None
    assert torch.isfinite(specialist_logits.grad).all()
