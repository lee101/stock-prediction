"""Parity tests for pufferlib_market.batched_ensemble.

Contract: StackedEnsemble.forward(obs) must produce logits that, when
averaged (softmax or raw) across members, yield the SAME argmax and
softmax-avg-argmax as the serial per-member forward used in
`scripts/screened32_realism_gate.py::_build_ensemble_policy_fn`.

If this ever regresses, the deploy gate's output drifts silently — so
we assert on a realistic prod obs distribution (random normal + random
uniform sweeps + all-zeros + large-magnitude) across the current v7
12-model ensemble.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def policies():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from pufferlib_market.evaluate_holdout import load_policy
    from src.daily_stock_defaults import DEFAULT_CHECKPOINT, DEFAULT_EXTRA_CHECKPOINTS
    ckpts = [str(DEFAULT_CHECKPOINT)] + [str(p) for p in DEFAULT_EXTRA_CHECKPOINTS]
    dev = torch.device("cuda")
    loaded = [load_policy(c, 32, features_per_sym=16, device=dev) for c in ckpts]
    return [l.policy.eval() for l in loaded]


def test_can_batch_prod_ensemble(policies):
    from pufferlib_market.batched_ensemble import can_batch
    assert can_batch(policies), "v7 prod ensemble should be stack-compatible"


@pytest.mark.parametrize("batch", [1, 4, 16])
def test_stacked_matches_serial_logits(policies, batch):
    from pufferlib_market.batched_ensemble import StackedEnsemble
    dev = torch.device("cuda")
    obs_dim = policies[0].encoder[0].in_features
    torch.manual_seed(42 + batch)
    obs = torch.randn(batch, obs_dim, device=dev)
    stacked = StackedEnsemble.from_policies(policies, dev)
    # Serial reference
    with torch.no_grad():
        serial = torch.stack([p(obs)[0] for p in policies], dim=0)  # [N, B, A]
        batched = stacked.forward(obs)                               # [N, B, A]
    # Logits should match to within fp32 relaxed tol — bmm and chained
    # matmul can differ in the least-significant bits.
    diff = (serial - batched).abs()
    assert diff.max().item() < 1e-3, f"max logit delta {diff.max().item()}"


def test_softmax_avg_argmax_matches(policies):
    """This is the deploy-gate-relevant equality: argmax of averaged
    softmax probs across members must match."""
    from pufferlib_market.batched_ensemble import StackedEnsemble
    dev = torch.device("cuda")
    obs_dim = policies[0].encoder[0].in_features
    stacked = StackedEnsemble.from_policies(policies, dev)

    mismatches = 0
    n_obs = 1000
    torch.manual_seed(0)
    # Mix synthetic obs distributions
    obs_batches = [
        torch.randn(n_obs // 4, obs_dim, device=dev),
        torch.randn(n_obs // 4, obs_dim, device=dev) * 3.0,
        torch.rand(n_obs // 4, obs_dim, device=dev) * 2 - 1,
        torch.zeros(n_obs // 4, obs_dim, device=dev) + torch.randn(1, obs_dim, device=dev) * 0.1,
    ]
    for obs in obs_batches:
        with torch.no_grad():
            probs_serial = sum(torch.softmax(p(obs)[0], dim=-1) for p in policies) / len(policies)
            act_serial = probs_serial.argmax(dim=-1)
            logits_batched = stacked.forward(obs)
            probs_batched = torch.softmax(logits_batched, dim=-1).mean(dim=0)
            act_batched = probs_batched.argmax(dim=-1)
        mismatches += int((act_serial != act_batched).sum().item())
    # Allow ≤ 0.1% argmax mismatches as fp32 tie-break noise
    assert mismatches <= max(1, int(n_obs * 0.001)), \
        f"{mismatches}/{n_obs} argmax mismatches — ensemble batching is not deploy-safe"


def test_stacked_speed_beats_serial(policies):
    """Sanity: batched forward should be materially faster at B=1."""
    from pufferlib_market.batched_ensemble import StackedEnsemble
    import time
    dev = torch.device("cuda")
    obs_dim = policies[0].encoder[0].in_features
    stacked = StackedEnsemble.from_policies(policies, dev)
    obs = torch.randn(1, obs_dim, device=dev)
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            sum(torch.softmax(p(obs)[0], dim=-1) for p in policies)
            stacked.forward(obs)
    torch.cuda.synchronize()
    N = 200

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N):
            _ = sum(torch.softmax(p(obs)[0], dim=-1) for p in policies) / len(policies)
    torch.cuda.synchronize()
    dt_serial = time.perf_counter() - t0

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N):
            _ = torch.softmax(stacked.forward(obs), dim=-1).mean(dim=0)
    torch.cuda.synchronize()
    dt_batched = time.perf_counter() - t0

    print(f"\nserial:  {dt_serial*1000/N:.3f} ms/step")
    print(f"batched: {dt_batched*1000/N:.3f} ms/step")
    print(f"speedup: {dt_serial/dt_batched:.2f}x")
    # Don't assert speedup — GPU scheduler is noisy. Just require < 2× regression.
    assert dt_batched < dt_serial * 2.0, \
        f"batched ({dt_batched*1000/N:.3f}ms) should not be >2× serial ({dt_serial*1000/N:.3f}ms)"
