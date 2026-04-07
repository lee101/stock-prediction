"""Tests for fp4.env_adapter — the canonical env factory used by trainers."""
from __future__ import annotations

import math

import pytest
import torch

from fp4.env_adapter import EnvHandle, make_env


def _is_finite(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


def test_stub_backend_cpu_ok():
    h = make_env({"env": "stub"}, num_envs=4, obs_dim=8, act_dim=3,
                 device=torch.device("cpu"), seed=0)
    assert isinstance(h, EnvHandle)
    assert h.backend_name == "stub"
    assert h.obs_dim == 8
    assert h.action_dim == 3
    obs = h.reset()
    assert obs.shape == (4, 8)
    a = torch.zeros(4, 3)
    obs2, rew, done, cost = h.step(a)
    assert obs2.shape == (4, 8)
    assert rew.shape == (4,)
    assert done.shape == (4,)
    assert cost is None


def test_stub_smoke_rollout_finite():
    h = make_env({"env": "stub", "episode_len": 32}, num_envs=2, obs_dim=6, act_dim=2,
                 device=torch.device("cpu"), seed=1)
    obs = h.reset()
    total = torch.zeros(h.num_envs)
    for _ in range(256):
        a = torch.randn(h.num_envs, h.action_dim) * 0.1
        obs, rew, done, _cost = h.step(a)
        total = total + rew
        assert _is_finite(rew), "stub reward should stay finite"
        assert _is_finite(obs), "stub obs should stay finite"
    assert _is_finite(total)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gpu_trading_env_backend_or_skip():
    try:
        import gpu_trading_env as gte
    except Exception as exc:
        pytest.skip(f"gpu_trading_env import failed: {exc}")
    if gte._load_ext() is None:
        pytest.skip(f"gpu_trading_env ext not built: {gte._EXT_ERR}")
    try:
        h = make_env({"env": "gpu_trading_env"}, num_envs=8, seed=0)
    except RuntimeError as exc:
        msg = str(exc)
        if "out of memory" in msg.lower() or "OutOfMemory" in msg:
            pytest.skip(f"GPU OOM under contention: {msg}")
        raise
    assert h.backend_name == "gpu_trading_env"
    assert h.action_dim == 4
    obs = h.reset()
    assert obs.shape[0] == 8
    assert obs.shape[-1] == h.obs_dim
    # 256-step smoke rollout with random ActorCritic-style scalar actions —
    # the adapter shim should map them onto the 4-wide env action.
    total = torch.zeros(h.num_envs, device=obs.device)
    for _ in range(256):
        a = torch.randn(h.num_envs, 1, device=obs.device) * 0.5
        obs, rew, done, cost = h.step(a)
        assert _is_finite(rew), "gpu_trading_env reward must stay finite"
        assert _is_finite(obs), "gpu_trading_env obs must stay finite"
        total = total + rew
    assert _is_finite(total)


def test_auto_falls_back_when_no_cuda():
    if torch.cuda.is_available():
        pytest.skip("auto-fallback path is only meaningful on CPU-only boxes")
    h = make_env({}, num_envs=2, obs_dim=4, act_dim=2,
                 device=torch.device("cpu"), seed=0)
    assert h.backend_name == "stub"
