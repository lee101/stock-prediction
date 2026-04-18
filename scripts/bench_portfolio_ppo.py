#!/usr/bin/env python3
"""Quick PPO training benchmark on PortfolioBracketEnv with synthetic prices.

Measures:
  - throughput: env-steps/sec, iters/sec, wallclock per iter
  - learning: mean reward + episode equity over time

Run: python scripts/bench_portfolio_ppo.py
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import gpu_trading_env
from gpu_trading_env.ppo_trainer import (
    PortfolioBracketActor, PPOConfig, train,
)


def synth_prices(T: int, S: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    daily_ret = rng.normal(0.0005, 0.02, size=(T, S)).astype(np.float32)
    close = 100.0 * np.exp(np.cumsum(daily_ret, axis=0))
    open_ = close * (1.0 + rng.normal(0.0, 0.005, size=(T, S))).astype(np.float32)
    band = np.abs(rng.normal(0.0, 0.01, size=(T, S))).astype(np.float32)
    high = np.maximum(open_, close) * (1.0 + band)
    low = np.minimum(open_, close) * (1.0 - band)
    vol = np.full((T, S), 1e6, dtype=np.float32)
    return torch.from_numpy(np.stack([open_, high, low, close, vol], axis=-1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=2048)
    p.add_argument("--T", type=int, default=512)
    p.add_argument("--S", type=int, default=8)
    p.add_argument("--rollout-steps", type=int, default=64)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.001)
    p.add_argument("--reward-scale", type=float, default=10.0)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    args = p.parse_args()

    print(f"[bench] PortfolioBracketEnv PPO smoke run")
    print(f"  B={args.B} T={args.T} S={args.S} "
          f"rollout={args.rollout_steps} iters={args.iters} bf16={args.bf16}")

    if not torch.cuda.is_available():
        raise RuntimeError("Need CUDA")

    prices = synth_prices(args.T, args.S, seed=0)
    env = gpu_trading_env.make_portfolio_bracket(
        B=args.B, prices=prices,
        params={"episode_len": args.T - 2, "fee_bps": 0.278,
                "max_leverage": 2.0},
    )
    obs_dim = env.obs().shape[-1]
    print(f"  obs_dim={obs_dim} action=[B, {args.S}, 4] env={type(env).__name__}")

    policy = PortfolioBracketActor(
        obs_dim=obs_dim, num_symbols=args.S,
        hidden_dim=256, num_layers=3,
    ).to("cuda")
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  policy params: {n_params:,}")

    cfg = PPOConfig(
        rollout_steps=args.rollout_steps,
        epochs=args.epochs, minibatches=4,
        lr=args.lr, ent_coef=args.ent_coef,
        bf16=args.bf16, reward_scale=args.reward_scale,
        log_every=10,
    )
    torch.cuda.synchronize()
    t0 = time.time()
    out = train(env, policy=policy, cfg=cfg, iters=args.iters)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    hist = out["history"]
    total_env_steps = args.iters * args.rollout_steps * args.B
    print(f"\n[done] {elapsed:.1f}s "
          f"= {args.iters/elapsed:.2f} iter/s "
          f"= {total_env_steps/elapsed/1e6:.1f} M env-steps/s")

    n = max(1, len(hist["mean_reward"]) // 4)
    early = float(np.mean(hist["mean_reward"][:n]))
    late = float(np.mean(hist["mean_reward"][-n:]))
    print(f"[reward] early-quartile mean: {early:+.6f}  "
          f"late-quartile mean: {late:+.6f}  delta: {late-early:+.6f}")
    print(f"[final] equity mean: {env.state['equity'].mean().item():,.2f}  "
          f"(start {10_000.0})")
    print(f"[final] equity p10: {env.state['equity'].quantile(0.1).item():,.2f}  "
          f"p90: {env.state['equity'].quantile(0.9).item():,.2f}")


if __name__ == "__main__":
    main()
