"""Tests for cuda_graph_full.build_real_full_step (P6-5).

Captures the full PPO inner loop — including a real gpu_trading_env rollout
— into a single CUDA graph and verifies replay produces finite outputs and
mutates policy parameters.
"""
from __future__ import annotations

import math

import pytest
import torch


def _cuda_ready() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    try:
        _ = torch.zeros(1, device="cuda")
    except Exception as e:
        return False, f"CUDA init failed: {type(e).__name__}: {e}"
    try:
        import gpu_trading_env as gte
        if gte._load_ext() is None:
            return False, f"gpu_trading_env ext unavailable: {gte._EXT_ERR}"
    except Exception as e:
        return False, f"gpu_trading_env import failed: {type(e).__name__}: {e}"
    return True, ""


_ok, _why = _cuda_ready()
real_env_required = pytest.mark.skipif(not _ok, reason=_why or "real env required")


@real_env_required
def test_real_full_step_capture_replay_finite():
    from fp4.policy import ActorCritic
    from fp4.env_adapter import make_env
    from fp4.cuda_graph_full import build_real_full_step, capture_full_step

    device = torch.device("cuda")
    torch.manual_seed(0)

    handle = make_env(
        cfg={"env": "gpu_trading_env"},
        num_envs=64,
        seed=0,
    )
    assert handle.backend_name == "gpu_trading_env", \
        f"expected gpu_trading_env, got {handle.backend_name}"

    policy = ActorCritic(
        obs_dim=handle.obs_dim, act_dim=handle.action_dim, hidden=32, seed=0,
    ).to(device)
    optim = torch.optim.SGD(policy.parameters(), lr=1e-3)

    step_fn, _state = build_real_full_step(
        handle, policy, optim,
        rollout_len=8, device=device,
    )
    captured = capture_full_step(step_fn)

    pre = next(iter(policy.parameters())).detach().clone()
    for _ in range(100):
        captured.replay()
    torch.cuda.synchronize()
    post = next(iter(policy.parameters())).detach()

    loss = captured.outputs["loss"].item()
    mean_r = captured.outputs["mean_reward"].item()
    assert math.isfinite(loss), f"loss not finite: {loss}"
    assert math.isfinite(mean_r), f"mean_reward not finite: {mean_r}"
    assert not torch.equal(pre, post), \
        "optimizer.step() inside captured real-env graph had no effect"
