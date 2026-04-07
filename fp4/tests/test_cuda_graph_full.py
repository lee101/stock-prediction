"""Tests for fp4.cuda_graph_full — full PPO-step CUDA graph capture."""
from __future__ import annotations

import math

import pytest
import torch

def _cuda_ready() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    try:
        _ = torch.zeros(1, device="cuda")
        return True, ""
    except Exception as e:
        return False, f"CUDA init failed: {type(e).__name__}: {e}"


_ok, _why = _cuda_ready()
cuda_required = pytest.mark.skipif(not _ok, reason=_why or "CUDA required")


@cuda_required
def test_capture_and_replay_finite():
    from fp4.policy import ActorCritic
    from fp4.cuda_graph_full import build_synthetic_full_step, capture_full_step

    device = torch.device("cuda")
    torch.manual_seed(0)
    policy = ActorCritic(obs_dim=8, act_dim=2, hidden=32, seed=0).to(device)
    optim = torch.optim.SGD(policy.parameters(), lr=1e-3)

    step_fn, state = build_synthetic_full_step(
        policy, optim,
        num_envs=16, rollout_len=8, obs_dim=8, act_dim=2, device=device,
    )
    captured = capture_full_step(step_fn)

    for _ in range(100):
        captured.replay()
    torch.cuda.synchronize()

    loss = captured.outputs["loss"].item()
    mean_r = captured.outputs["mean_reward"].item()
    assert math.isfinite(loss), f"loss not finite: {loss}"
    assert math.isfinite(mean_r), f"mean_reward not finite: {mean_r}"

    # Sanity: a parameter changed at least once during the 100 replays.
    pre = next(iter(policy.parameters())).detach().clone()
    captured.replay()
    torch.cuda.synchronize()
    post = next(iter(policy.parameters())).detach()
    assert not torch.equal(pre, post), "optimizer.step() inside graph had no effect"


@cuda_required
def test_trainer_full_graph_branch():
    """The opt-in `cfg['full_graph_capture']` branch in train_ppo runs to
    completion and reports `full_graph_used=True` plus a positive sps."""
    from fp4.trainer import train_ppo

    metrics = train_ppo(
        cfg={
            "obs_dim": 8,
            "act_dim": 2,
            "ppo": {"num_envs": 16, "rollout_len": 8, "hidden_size": 32,
                    "minibatch_size": 64, "ppo_epochs": 1},
            "full_graph_capture": True,
        },
        total_timesteps=16 * 8 * 20,  # 20 graph replays
        seed=0,
        checkpoint_dir="/tmp/fp4_full_graph_test_ckpt",
    )
    assert metrics["full_graph_used"] is True
    assert metrics["steps_per_sec"] > 0
    assert math.isfinite(metrics["last_loss"])
