"""Smoke test for fp4.trainer_sac.train_sac — 4096 env steps, no NaN."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import torch

from fp4.trainer_sac import train_sac
from fp4.replay import GPUReplayBuffer


def test_replay_buffer_add_sample_gpu_resident():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rb = GPUReplayBuffer(capacity=64, obs_dim=4, act_dim=2, device=device)
    for _ in range(10):
        obs = torch.randn(8, 4, device=device)
        act = torch.randn(8, 2, device=device)
        rew = torch.randn(8, device=device)
        next_obs = torch.randn(8, 4, device=device)
        done = torch.zeros(8, device=device)
        rb.add(obs, act, rew, next_obs, done)
    # Wrap-around exercised (10*8 > 64).
    assert len(rb) == 64
    batch = rb.sample(16)
    assert batch["obs"].shape == (16, 4)
    assert batch["act"].shape == (16, 2)
    assert batch["rew"].shape == (16,)
    assert batch["obs"].device.type == device.type
    assert torch.isfinite(batch["obs"]).all()


def test_train_sac_smoke_no_nan():
    cfg = {
        "obs_dim": 8,
        "act_dim": 3,
        "episode_len": 32,
        "sac": {
            "num_envs": 8,
            "hidden_size": 32,
            "lr": 3e-4,
            "batch_size": 64,
            "replay_capacity": 4096,
            "warmup_steps": 128,
            "updates_per_step": 1,
        },
    }
    with tempfile.TemporaryDirectory() as td:
        metrics = train_sac(cfg, total_timesteps=4096, seed=0, checkpoint_dir=td)
        ckpt = Path(td) / "final.pt"
        assert ckpt.exists()
    assert "final_sortino" in metrics
    assert math.isfinite(metrics["final_sortino"])
    assert math.isfinite(metrics["mean_return"])
    assert metrics["total_steps"] >= 4096
    assert metrics["algorithm"] == "sac"
