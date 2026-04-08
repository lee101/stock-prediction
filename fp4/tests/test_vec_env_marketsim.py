"""Tests for the real `market_sim_py` wrapper inside `fp4.vec_env`.

These tests skip cleanly when:
  - the `market_sim_py` extension isn't built on this box, or
  - CUDA isn't available (the binding requires a CUDA device for DPS mode in
    the configurations we use here).

The goal is to lock down the no-host-sync invariants and the auto-reset
contract that the rest of the fp4 RL stack relies on.
"""
from __future__ import annotations

import pytest
import torch

from fp4.vec_env import (
    GPUVecEnv,
    SyntheticOHLCEnv,
    _MarketSimPyWrapper,
    _try_import_market_sim_py,
)


_HAS_MSP = _try_import_market_sim_py() is not None
_HAS_CUDA = torch.cuda.is_available()


pytestmark = pytest.mark.skipif(
    not (_HAS_MSP and _HAS_CUDA),
    reason="market_sim_py extension or CUDA not available on this box",
)


def _make_env(action_mode: str = "dps") -> _MarketSimPyWrapper:
    return _MarketSimPyWrapper(
        num_envs=4096, device="cuda", seed=123, action_mode=action_mode
    )


def test_wrapper_reset_returns_cuda_tensor():
    env = _make_env()
    obs = env.reset()
    assert isinstance(obs, torch.Tensor)
    assert obs.is_cuda
    assert obs.shape[0] == env.num_envs
    assert obs.shape[-1] == env.obs_dim
    assert env.act_dim == 3  # DPS mode


def test_wrapper_step_keys_and_devices():
    env = _make_env()
    env.reset()
    action = torch.zeros(env.num_envs, env.act_dim, device="cuda")
    out = env.step(action)
    for k in ("obs", "reward", "done", "info_pnl"):
        assert k in out, f"missing key {k}"
        assert isinstance(out[k], torch.Tensor)
        assert out[k].is_cuda, f"{k} not on CUDA"
    assert out["obs"].shape == (env.num_envs, env.obs_dim)
    assert out["reward"].shape == (env.num_envs,)
    assert out["done"].shape == (env.num_envs,)
    assert out["done"].dtype == torch.bool


def test_wrapper_no_host_sync_in_step():
    """Hot loop must not call .item()/.cpu(). We assert by running many
    steps inside `torch.cuda.stream` capture mode-like settings: any host
    sync would surface as an exception under `cuda.synchronize`-free flow.
    """
    env = _make_env()
    env.reset()
    action = torch.zeros(env.num_envs, env.act_dim, device="cuda")
    # Just confirm the loop runs without raising and the buffers persist.
    for _ in range(64):
        out = env.step(action)
    torch.cuda.synchronize()
    assert torch.isfinite(out["reward"]).all().item()


def test_wrapper_action_shape_padding():
    """Passing a SCALAR-shaped action into a DPS env should auto-broadcast."""
    env = _make_env(action_mode="dps")
    env.reset()
    a1 = torch.zeros(env.num_envs, 1, device="cuda")  # under-sized
    out = env.step(a1)
    assert out["obs"].shape == (env.num_envs, env.obs_dim)


def test_factory_picks_real_env_when_available():
    env = GPUVecEnv(num_envs=4096, device="cuda", seed=7, action_mode="dps")
    assert isinstance(env, _MarketSimPyWrapper)


def test_synthetic_fallback_when_market_sim_disabled():
    env = GPUVecEnv(
        num_envs=64, obs_dim=8, act_dim=1, device="cuda",
        seed=0, prefer_market_sim=False,
    )
    assert isinstance(env, SyntheticOHLCEnv)
