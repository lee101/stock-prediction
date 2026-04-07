"""Smoke + perf tests for `fp4.vec_env.GPUVecEnv` and `fp4.cuda_graph`."""
from __future__ import annotations

import time

import pytest
import torch

from fp4.vec_env import GPUVecEnv, SyntheticOHLCEnv
from fp4.cuda_graph import capture_step


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_synthetic_env_shapes_and_dtypes():
    env = SyntheticOHLCEnv(num_envs=8, obs_dim=12, act_dim=2, device=_device(), seed=0)
    obs = env.reset()
    assert obs.shape == (8, 12)
    assert obs.device.type == _device()
    action = torch.zeros(8, 2, device=_device())
    out = env.step(action)
    for k in ("obs", "reward", "done", "info_pnl"):
        assert k in out
        assert out[k].device.type == _device()
    assert out["obs"].shape == (8, 12)
    assert out["reward"].shape == (8,)
    assert out["done"].dtype == torch.bool


def test_gpu_vec_env_factory_falls_back():
    env = GPUVecEnv(num_envs=4, obs_dim=8, act_dim=1, device=_device(), seed=1)
    assert hasattr(env, "reset") and hasattr(env, "step")
    obs = env.reset()
    assert obs.shape[0] == 4


def test_determinism_same_seed():
    e1 = SyntheticOHLCEnv(num_envs=4, obs_dim=8, device=_device(), seed=42)
    e2 = SyntheticOHLCEnv(num_envs=4, obs_dim=8, device=_device(), seed=42)
    o1, o2 = e1.reset(), e2.reset()
    assert torch.allclose(o1, o2)
    a = torch.zeros(4, 1, device=_device())
    for _ in range(10):
        r1 = e1.step(a)["reward"]
        r2 = e2.step(a)["reward"]
        assert torch.allclose(r1, r2)


def test_episode_auto_reset():
    env = SyntheticOHLCEnv(num_envs=2, obs_dim=4, device=_device(), seed=0, episode_len=3)
    env.reset()
    a = torch.zeros(2, 1, device=_device())
    dones = []
    for _ in range(6):
        dones.append(env.step(a)["done"].clone())
    # Should have triggered done at step 3 and step 6.
    flat = torch.stack(dones).any(dim=-1)
    assert flat[2].item() is True or flat[2].item() == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for SPS gate")
def test_sps_smoke_gpu():
    """≥ 50k env-steps/sec on GPU with a moderate batch."""
    N = 4096
    env = SyntheticOHLCEnv(num_envs=N, obs_dim=16, act_dim=1, device="cuda", seed=0)
    env.reset()
    action = torch.zeros(N, 1, device="cuda")
    # Warmup
    for _ in range(20):
        env.step(action)
    torch.cuda.synchronize()
    iters = 200
    t0 = time.perf_counter()
    for _ in range(iters):
        env.step(action)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    sps = (iters * N) / dt
    assert sps > 50_000, f"SPS={sps:.0f} below 50k floor"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture")
def test_cuda_graph_capture_replay():
    N, D = 256, 8
    env = SyntheticOHLCEnv(num_envs=N, obs_dim=D, act_dim=1, device="cuda", seed=0)
    env.reset()
    # Pre-allocate persistent output tensors.
    out_obs = torch.zeros(N, D, device="cuda")
    out_reward = torch.zeros(N, device="cuda")

    def step_fn(inputs):
        out = env.step(inputs["action"])
        out_obs.copy_(out["obs"])
        out_reward.copy_(out["reward"])
        return {"obs": out_obs, "reward": out_reward}

    captured = capture_step(step_fn, {"action": torch.zeros(N, 1, device="cuda")}, warmup=3)
    captured.copy_inputs(action=torch.ones(N, 1, device="cuda") * 0.5)
    out = captured.replay()
    torch.cuda.synchronize()
    assert out["obs"].shape == (N, D)
    assert torch.isfinite(out["reward"]).all()
