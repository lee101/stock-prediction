#!/usr/bin/env python3
"""Extended stock training: 30+ symbols, shorts, 200M steps.

Usage:
    python -m experiments.extended_stock_corr_200M.train_extended \
        --data-path pufferlib_market/data/extended_stocks.bin \
        --checkpoint-dir experiments/extended_stock_corr_200M/checkpoints
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pufferlib_market.environment import TradingEnvConfig, TradingEnv
from pufferlib_market.train import ResidualTradingPolicy
from pufferlib_market import binding
from experiments.extended_stock_corr_200M.config import TRAINING_CONFIG, ALL_SYMBOLS


def add_obs_noise(obs: np.ndarray, std: float) -> np.ndarray:
    """Add Gaussian noise to observations for robustness."""
    if std > 0:
        return obs + np.random.randn(*obs.shape).astype(np.float32) * std
    return obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, default="experiments/extended_stock_corr_200M/checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    cfg = TRAINING_CONFIG
    device = torch.device(args.device)

    # Environment config
    config = TradingEnvConfig(
        data_path=args.data_path,
        max_steps=cfg["max_steps"],
        fee_rate=0.001,
        max_leverage=2.0,
        periods_per_year=8760.0,
        num_symbols=len(ALL_SYMBOLS),
        reward_scale=10.0,
        reward_clip=5.0,
        cash_penalty=cfg["cash_penalty"],
        drawdown_penalty=cfg["drawdown_penalty"],
        downside_penalty=cfg["downside_penalty"],
        trade_penalty=cfg["trade_penalty"],
        smoothness_penalty=cfg["smoothness_penalty"],
    )

    # Vector env
    N = cfg["num_envs"]
    T = cfg["rollout_len"]

    vec_handle = binding.vec_create(N, config.to_dict())
    obs_size, num_actions = binding.vec_obs_action_sizes(vec_handle)
    print(f"obs_size={obs_size}, num_actions={num_actions}, num_symbols={len(ALL_SYMBOLS)}")

    # Shared buffers
    obs_buf = np.zeros((N, obs_size), dtype=np.float32)
    act_buf = np.zeros((N,), dtype=np.int32)
    rew_buf = np.zeros((N,), dtype=np.float32)
    term_buf = np.zeros((N,), dtype=np.uint8)
    binding.vec_set_buffers(vec_handle, obs_buf, act_buf, rew_buf, term_buf)

    # Reset
    binding.vec_reset(vec_handle)

    # Policy
    policy = ResidualTradingPolicy(
        obs_size=obs_size,
        num_actions=num_actions,
        hidden=cfg["hidden_size"],
        num_blocks=cfg["num_blocks"],
    ).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg["lr"], eps=1e-5)
    start_update = 1
    global_step = 0
    best_sortino = -float("inf")

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        policy.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_update = ckpt.get("update", 0) + 1
        global_step = ckpt.get("global_step", 0)
        best_sortino = ckpt.get("best_sortino", -float("inf"))
        print(f"Resumed from {args.resume}, update={start_update}, step={global_step}")

    # Rollout buffers
    buf_obs = torch.zeros((T, N, obs_size), dtype=torch.float32)
    buf_act = torch.zeros((T, N), dtype=torch.int64)
    buf_logprob = torch.zeros((T, N), dtype=torch.float32)
    buf_reward = torch.zeros((T, N), dtype=torch.float32)
    buf_done = torch.zeros((T, N), dtype=torch.float32)
    buf_value = torch.zeros((T, N), dtype=torch.float32)

    num_updates = cfg["total_timesteps"] // (T * N)
    start_time = time.time()

    print(f"\nExtended Stock Training: {len(ALL_SYMBOLS)} symbols, {cfg['total_timesteps']:,} steps")
    print(f"  arch: ResidualMLP {cfg['hidden_size']}x{cfg['num_blocks']}")
    print(f"  downside_penalty={cfg['downside_penalty']}, smoothness_penalty={cfg['smoothness_penalty']}")
    print(f"  obs_noise_std={cfg['obs_noise_std']}")
    print()

    for update in range(start_update, num_updates + 1):
        # LR annealing
        frac = 1.0 - (update - 1) / num_updates
        lr_now = frac * cfg["lr"]
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        # Collect rollout
        policy.eval()
        for step in range(T):
            # Add noise for robustness
            noisy_obs = add_obs_noise(obs_buf, cfg["obs_noise_std"])
            obs_tensor = torch.from_numpy(noisy_obs).to(device)
            buf_obs[step] = obs_tensor

            with torch.no_grad():
                logits, value = policy(obs_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)

            buf_act[step] = action.cpu()
            buf_logprob[step] = logprob.cpu()
            buf_value[step] = value.cpu()

            act_buf[:] = action.cpu().numpy().astype(np.int32)
            binding.vec_step(vec_handle)

            buf_reward[step] = torch.from_numpy(rew_buf.copy())
            buf_done[step] = torch.from_numpy(term_buf.copy().astype(np.float32))
            global_step += N

        # GAE
        with torch.no_grad():
            next_obs = torch.from_numpy(obs_buf.copy()).to(device)
            next_value = policy.get_value(next_obs).cpu()

        advantages = torch.zeros_like(buf_reward)
        last_gae = torch.zeros(N)
        gamma, gae_lambda = cfg["gamma"], cfg["gae_lambda"]

        for t in reversed(range(T)):
            next_val = next_value if t == T - 1 else buf_value[t + 1]
            not_done = 1.0 - buf_done[t]
            delta = buf_reward[t] + gamma * next_val * not_done - buf_value[t]
            last_gae = delta + gamma * gae_lambda * not_done * last_gae
            advantages[t] = last_gae

        returns = advantages + buf_value

        # PPO update
        policy.train()
        b_obs = buf_obs.reshape(-1, obs_size).to(device)
        b_act = buf_act.reshape(-1).to(device)
        b_logprob = buf_logprob.reshape(-1).to(device)
        b_advantages = advantages.reshape(-1).to(device)
        b_returns = returns.reshape(-1).to(device)

        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        batch_size = T * N
        mb_size = cfg["minibatch_size"]

        for _ in range(cfg["ppo_epochs"]):
            indices = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, mb_size):
                end = min(start + mb_size, batch_size)
                mb_idx = indices[start:end]

                logits, new_value = policy(b_obs[mb_idx])
                dist = torch.distributions.Categorical(logits=logits)
                new_logprob = dist.log_prob(b_act[mb_idx])
                entropy = dist.entropy()

                log_ratio = new_logprob - b_logprob[mb_idx]
                ratio = log_ratio.exp()
                mb_adv = b_advantages[mb_idx]

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - cfg["clip_eps"], 1 + cfg["clip_eps"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((new_value - b_returns[mb_idx]) ** 2).mean()
                ent_loss = entropy.mean()

                loss = pg_loss + cfg["vf_coef"] * v_loss - cfg["ent_coef"] * ent_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

        # Logging
        log_info = binding.vec_log(vec_handle)
        elapsed = time.time() - start_time
        sps = global_step / elapsed

        if log_info and "sortino" in log_info:
            sortino = log_info["sortino"]
            ret = log_info.get("total_return", 0)
            trades = log_info.get("num_trades", 0)
            wr = log_info.get("win_rate", 0)

            if sortino > best_sortino:
                best_sortino = sortino
                ckpt_path = Path(args.checkpoint_dir) / "best_sortino.pt"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "update": update,
                    "global_step": global_step,
                    "best_sortino": best_sortino,
                    "config": cfg,
                }, ckpt_path)

            if update % 10 == 0:
                print(
                    f"[{update:5d}/{num_updates}] step={global_step:10,d} sps={sps:.0f} "
                    f"ret={ret:+.4f} sortino={sortino:.2f} trades={trades:.0f} wr={wr:.2f} "
                    f"best_sortino={best_sortino:.2f}"
                )

        # Periodic checkpoint
        if update % 500 == 0:
            ckpt_path = Path(args.checkpoint_dir) / f"update_{update:06d}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update": update,
                "global_step": global_step,
                "best_sortino": best_sortino,
                "config": cfg,
            }, ckpt_path)

    # Final
    ckpt_path = Path(args.checkpoint_dir) / "final.pt"
    torch.save({
        "model": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": num_updates,
        "global_step": global_step,
        "best_sortino": best_sortino,
        "config": cfg,
    }, ckpt_path)

    print(f"\nTraining complete. Best sortino: {best_sortino:.2f}")
    binding.vec_close(vec_handle)


if __name__ == "__main__":
    main()
