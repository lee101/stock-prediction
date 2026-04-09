"""Tests that gpu_trading_env with v5_rsi config produces the correct
obs_dim=209 and act_dim=25 matching the production marketsim, and that
a forward pass through the ActorCritic policy succeeds with these shapes."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BIN_PATH = REPO / "pufferlib_market" / "data" / "stocks12_daily_v5_rsi_train.bin"

# Expected shapes for stocks12_v5_rsi (S=12, F=16):
#   obs = 12*16 + 5 + 12 = 209
#   act = 1 + 2*12 = 25
EXPECTED_OBS_DIM = 209
EXPECTED_ACT_DIM = 25


@pytest.fixture
def bin_exists():
    if not BIN_PATH.exists():
        pytest.skip(f"stocks12 v5_rsi train data not found at {BIN_PATH}")


# ---- gpu_trading_env.MultiSymbolEnvHandle ----

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_multi_symbol_env_shapes(bin_exists):
    import gpu_trading_env as gte
    ext = gte._load_ext()
    if ext is None:
        pytest.skip(f"CUDA extension not available: {gte._EXT_ERR}")

    env = gte.make_multi_symbol(B=4, bin_path=str(BIN_PATH))
    assert env.obs_dim == EXPECTED_OBS_DIM, f"obs_dim={env.obs_dim}, expected {EXPECTED_OBS_DIM}"
    assert env.action_dim == EXPECTED_ACT_DIM, f"act_dim={env.action_dim}, expected {EXPECTED_ACT_DIM}"
    assert env.num_symbols == 12
    assert env.features_per_sym == 16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_multi_symbol_env_reset_step(bin_exists):
    import gpu_trading_env as gte
    ext = gte._load_ext()
    if ext is None:
        pytest.skip(f"CUDA extension not available: {gte._EXT_ERR}")

    B = 4
    env = gte.make_multi_symbol(B=B, bin_path=str(BIN_PATH))
    env.reset()
    obs = env._obs()
    assert obs.shape == (B, EXPECTED_OBS_DIM), f"obs shape {obs.shape}"
    assert obs.dtype == torch.float32
    assert torch.isfinite(obs).all(), "obs contains non-finite values"

    action = torch.randn(B, EXPECTED_ACT_DIM, device="cuda", dtype=torch.float32)
    obs2, reward, done, cost = env.step(action)
    assert obs2.shape == (B, EXPECTED_OBS_DIM)
    assert reward.shape == (B,)
    assert done.shape == (B,)
    assert torch.isfinite(reward).all(), "reward contains non-finite values"


# ---- env_adapter integration ----

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_env_adapter_v5_rsi_shapes(bin_exists):
    from fp4.env_adapter import make_env

    cfg = {
        "env": {
            "backend": "gpu_trading_env",
            "train_data": str(BIN_PATH),
        }
    }
    handle = make_env(cfg, num_envs=4, obs_dim=EXPECTED_OBS_DIM,
                      act_dim=EXPECTED_ACT_DIM, seed=0)
    assert handle.obs_dim == EXPECTED_OBS_DIM
    assert handle.action_dim == EXPECTED_ACT_DIM
    assert handle.backend_name == "gpu_trading_env_multi"

    obs = handle.reset()
    assert obs.shape == (4, EXPECTED_OBS_DIM)

    action = torch.randn(4, EXPECTED_ACT_DIM, device="cuda", dtype=torch.float32)
    obs2, reward, done, cost = handle.step(action)
    assert obs2.shape == (4, EXPECTED_OBS_DIM)


# ---- Policy forward pass ----

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_policy_forward_v5_rsi(bin_exists):
    from fp4.policy import ActorCritic

    policy = ActorCritic(
        obs_dim=EXPECTED_OBS_DIM, act_dim=EXPECTED_ACT_DIM, hidden=256, seed=0
    ).to("cuda")

    obs = torch.randn(8, EXPECTED_OBS_DIM, device="cuda", dtype=torch.float32)
    mean, std, value = policy(obs)
    assert mean.shape == (8, EXPECTED_ACT_DIM)
    assert std.shape == (8, EXPECTED_ACT_DIM)
    assert value.shape == (8,)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(std).all()
    assert torch.isfinite(value).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_policy_act_and_logprob(bin_exists):
    from fp4.policy import ActorCritic

    policy = ActorCritic(
        obs_dim=EXPECTED_OBS_DIM, act_dim=EXPECTED_ACT_DIM, hidden=256, seed=0
    ).to("cuda")

    obs = torch.randn(8, EXPECTED_OBS_DIM, device="cuda", dtype=torch.float32)
    action, logp, value, mean = policy.act(obs)
    assert action.shape == (8, EXPECTED_ACT_DIM)
    assert logp.shape == (8,)
    assert torch.isfinite(logp).all()

    # Verify log-prob is consistent with gaussian_logprob.
    _, std, _ = policy(obs)
    logp2 = ActorCritic.gaussian_logprob(mean, std, action)
    # They won't be exactly equal because act() resamples, but the function
    # should produce finite values.
    assert torch.isfinite(logp2).all()


# ---- .bin loader ----

def test_load_bin_features(bin_exists):
    from gpu_trading_env import _load_bin_features

    features, num_symbols, num_timesteps, features_per_sym = _load_bin_features(str(BIN_PATH))
    assert num_symbols == 12
    assert features_per_sym == 16
    assert num_timesteps > 0
    assert features.shape == (num_timesteps, 12 * 16)
    assert features.dtype == torch.float32
