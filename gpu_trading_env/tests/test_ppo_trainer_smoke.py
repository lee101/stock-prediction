"""Smoke test for the PortfolioBracketEnv + PPO trainer.

Goal: confirm the full loop (env.step → policy.forward → GAE → PPO update)
runs end-to-end without NaN / crashes on small synthetic data, and that
the optimizer actually moves weights.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

cuda_available = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(
    not cuda_available, reason="gpu_trading_env requires CUDA"
)

import gpu_trading_env
from gpu_trading_env.ppo_trainer import (
    PortfolioBracketActor, PPOConfig, train, collect_rollout, compute_gae, ppo_update,
)


def _require_ext():
    ext = gpu_trading_env._load_ext()
    if ext is None:
        pytest.skip(f"gpu_trading_env CUDA ext not built: {gpu_trading_env._EXT_ERR}")


def _synth_prices(T: int, S: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    daily_ret = rng.normal(0.0005, 0.02, size=(T, S)).astype(np.float32)
    close = 100.0 * np.exp(np.cumsum(daily_ret, axis=0))
    open_ = close * (1.0 + rng.normal(0.0, 0.005, size=(T, S))).astype(np.float32)
    band = np.abs(rng.normal(0.0, 0.01, size=(T, S))).astype(np.float32)
    high = np.maximum(open_, close) * (1.0 + band)
    low = np.minimum(open_, close) * (1.0 - band)
    vol = np.full((T, S), 1e6, dtype=np.float32)
    return torch.from_numpy(np.stack([open_, high, low, close, vol], axis=-1))


def test_actor_forward_shapes_and_finite():
    _require_ext()
    B, T, S = 32, 64, 4
    prices = _synth_prices(T, S, seed=0)
    env = gpu_trading_env.make_portfolio_bracket(B=B, prices=prices)
    obs = env.obs()
    policy = PortfolioBracketActor(obs_dim=obs.shape[-1], num_symbols=S).to("cuda")
    mu, value = policy(obs)
    assert mu.shape == (B, S, 4)
    assert value.shape == (B,)
    assert torch.isfinite(mu).all() and torch.isfinite(value).all()
    # log_prob round-trip
    action = torch.zeros_like(mu)
    lp = policy.dist_log_prob(mu, action)
    assert lp.shape == (B,)
    assert torch.isfinite(lp).all()


def test_rollout_collect_runs_clean():
    _require_ext()
    B, T, S = 64, 80, 4
    prices = _synth_prices(T, S, seed=1)
    env = gpu_trading_env.make_portfolio_bracket(
        B=B, prices=prices, params={"episode_len": 30},
    )
    obs = env.obs()
    policy = PortfolioBracketActor(obs_dim=obs.shape[-1], num_symbols=S).to("cuda")
    cfg = PPOConfig(rollout_steps=16, bf16=True)
    buf, last_value = collect_rollout(env, policy, cfg)
    assert buf.obs.shape == (16, B, obs.shape[-1])
    assert buf.actions.shape == (16, B, S, 4)
    assert torch.isfinite(buf.rewards).all()
    assert torch.isfinite(buf.values).all()
    assert torch.isfinite(buf.log_probs).all()
    assert torch.isfinite(last_value).all()


def test_ppo_step_moves_weights_no_nan():
    _require_ext()
    B, T, S = 64, 80, 4
    prices = _synth_prices(T, S, seed=2)
    env = gpu_trading_env.make_portfolio_bracket(
        B=B, prices=prices, params={"episode_len": 40},
    )
    obs = env.obs()
    policy = PortfolioBracketActor(obs_dim=obs.shape[-1], num_symbols=S).to("cuda")
    cfg = PPOConfig(rollout_steps=16, epochs=2, minibatches=2, bf16=True)
    optim = torch.optim.Adam(policy.parameters(), lr=3e-4, eps=1e-5)

    # Snapshot every parameter — at least one of (encoder, actor, critic)
    # must have moved. With tiny initial rewards the actor head signal is
    # weak (advantages collapse near zero), but vf_loss should still drive
    # encoder + critic weights every update.
    snap = {n: p.detach().clone() for n, p in policy.named_parameters()}

    buf, last_v = collect_rollout(env, policy, cfg)
    adv, ret = compute_gae(buf, last_v, cfg)
    stats = ppo_update(buf, adv, ret, policy, optim, cfg)

    moved = [n for n, p in policy.named_parameters()
             if not torch.allclose(snap[n], p.detach())]
    assert moved, "PPO step did not move ANY weights"
    for p in policy.parameters():
        assert torch.isfinite(p).all(), "param diverged to NaN/Inf"
    for v in stats.values():
        assert np.isfinite(v), f"stat NaN: {stats}"


def test_train_loop_completes_short():
    """Sanity: 5 PPO iters on a tiny env. Just make sure nothing explodes
    and the average reward is finite at the end.
    """
    _require_ext()
    B, T, S = 128, 128, 4
    prices = _synth_prices(T, S, seed=3)
    env = gpu_trading_env.make_portfolio_bracket(
        B=B, prices=prices, params={"episode_len": 60},
    )
    cfg = PPOConfig(rollout_steps=16, epochs=2, minibatches=2,
                    bf16=True, log_every=10)
    out = train(env, cfg=cfg, iters=5)
    history = out["history"]
    assert len(history["mean_reward"]) == 5
    for m in history["mean_reward"]:
        assert np.isfinite(m), f"mean reward NaN at iter: {history}"
    for k in ("pg_loss", "vf_loss", "entropy", "approx_kl"):
        for v in history[k]:
            assert np.isfinite(v), f"{k} NaN: {history}"
