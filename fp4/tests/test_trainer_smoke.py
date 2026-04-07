"""Smoke test for fp4.trainer.train_ppo on the internal stub vec env."""
from __future__ import annotations

import math
import tempfile

import torch

from fp4.trainer import train_ppo, compute_gae, sortino_ratio


def test_compute_gae_shapes_and_finite():
    T, N = 8, 4
    rewards = torch.randn(T, N)
    values = torch.randn(T, N)
    dones = torch.zeros(T, N)
    last_v = torch.zeros(N)
    adv, ret = compute_gae(rewards, values, dones, last_v, gamma=0.99, lam=0.95)
    assert adv.shape == (T, N)
    assert ret.shape == (T, N)
    assert torch.isfinite(adv).all()
    assert torch.isfinite(ret).all()


def test_sortino_basic():
    s = sortino_ratio(torch.tensor([0.1, 0.2, -0.05, 0.15, -0.02]))
    assert math.isfinite(s)


def test_train_ppo_smoke_4096_steps():
    cfg = {
        "env": "stub",
        "obs_dim": 8,
        "act_dim": 3,
        "ppo": {
            "num_envs": 8,
            "rollout_len": 32,
            "hidden_size": 32,
            "ppo_epochs": 1,
            "minibatch_size": 64,
            "lr": 3e-4,
            "ent_coef": 0.01,
        },
        "episode_len": 16,
    }
    with tempfile.TemporaryDirectory() as tmp:
        metrics = train_ppo(cfg=cfg, total_timesteps=4096, seed=0, checkpoint_dir=tmp)
    assert isinstance(metrics, dict)
    for key in ("final_sortino", "final_p10", "mean_return", "steps_per_sec"):
        assert key in metrics, f"missing key {key}"
        assert math.isfinite(metrics[key]), f"non-finite {key}={metrics[key]}"
    assert metrics["total_steps"] >= 4096
    assert metrics["n_episodes"] > 0
