"""Binance PufferLib PPO training script for autoresearch optimization."""
from __future__ import annotations

import argparse
import logging
import math
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from .prepare import (
    TIME_BUDGET,
    BinanceTaskConfig,
    evaluate_checkpoint,
    print_metrics,
    resolve_task_config,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


class RunningObsNorm(nn.Module):
    def __init__(self, size, eps=1e-5, clip=10.0):
        super().__init__()
        self.eps = eps
        self.clip = clip
        self.register_buffer('running_mean', torch.zeros(size))
        self.register_buffer('running_var', torch.ones(size))
        self.register_buffer('count', torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        delta = batch_mean - self.running_mean
        tot_count = self.count + batch_count
        self.running_mean.add_(delta * batch_count / tot_count)
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        self.running_var.copy_((m_a + m_b + delta ** 2 * self.count * batch_count / tot_count) / tot_count)
        self.count.copy_(tot_count)

    def forward(self, x):
        return ((x - self.running_mean) / (self.running_var.sqrt() + self.eps)).clamp(-self.clip, self.clip)


class TradingPolicy(nn.Module):
    def __init__(self, obs_size: int, num_actions: int, hidden: int = 1024):
        super().__init__()
        self.obs_size = obs_size
        self.num_actions = num_actions
        self.obs_norm = RunningObsNorm(obs_size)
        self.encoder = nn.Sequential(
            nn.Linear(obs_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, num_actions))
        self.critic = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, x):
        x = self.obs_norm(x)
        h = self.encoder(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def get_action_and_value(self, x, action=None):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def get_value(self, x):
        x = self.obs_norm(x)
        h = self.encoder(x)
        return self.critic(h).squeeze(-1)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--hidden-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2.0e-3)
    p.add_argument("--anneal-lr", action="store_true", default=True)
    p.add_argument("--ent-coef", type=float, default=0.08)
    p.add_argument("--ent-coef-end", type=float, default=0.02)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--clip-vf", action="store_true", default=True)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.97)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--clip-eps-end", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--num-envs", type=int, default=2048)
    p.add_argument("--rollout-steps", type=int, default=256)
    p.add_argument("--ppo-epochs", type=int, default=3)
    p.add_argument("--minibatch-size", type=int, default=32768)
    p.add_argument("--warmup-updates", type=int, default=8)
    p.add_argument("--weight-decay", type=float, default=0.005)
    p.add_argument("--reward-scale", type=float, default=10.0)
    p.add_argument("--reward-clip", type=float, default=5.0)
    p.add_argument("--cash-penalty", type=float, default=0.01)
    p.add_argument("--max-steps", type=int, default=720)
    p.add_argument("--fee-rate", type=float, default=0.001)
    p.add_argument("--max-leverage", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-interval", type=int, default=20)
    return p.parse_args(argv)


def _save_ckpt(policy, args, num_updates, total_timesteps, best_return, path):
    torch.save({
        "model": policy.state_dict(),
        "hidden_size": args.hidden_size,
        "update": num_updates,
        "global_step": total_timesteps,
        "best_return": best_return,
    }, str(path))


def train(args=None):
    if args is None:
        args = parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from pufferlib_market import binding

    config = resolve_task_config()
    data_path = str(config.data_path.resolve())

    with open(data_path, "rb") as f:
        header = f.read(64)
    _, _, num_symbols, num_timesteps, _, _ = struct.unpack("<4sIIIII", header[:24])

    binding.shared(data_path=data_path)

    obs_size = num_symbols * 16 + 5 + num_symbols
    num_actions = 1 + 2 * num_symbols
    N = args.num_envs
    T = args.rollout_steps

    obs_buf = np.zeros((N, obs_size), dtype=np.float32)
    act_buf = np.zeros((N,), dtype=np.int32)
    rew_buf = np.zeros((N,), dtype=np.float32)
    term_buf = np.zeros((N,), dtype=np.uint8)
    trunc_buf = np.zeros((N,), dtype=np.uint8)

    vec_handle = binding.vec_init(
        obs_buf, act_buf, rew_buf, term_buf, trunc_buf,
        N, args.seed,
        max_steps=args.max_steps,
        fee_rate=args.fee_rate,
        max_leverage=args.max_leverage,
        reward_scale=args.reward_scale,
        reward_clip=args.reward_clip,
        cash_penalty=args.cash_penalty,
    )
    binding.vec_reset(vec_handle, args.seed)

    policy = TradingPolicy(obs_size, num_actions, hidden=args.hidden_size).to(device)
    optimizer = optim.AdamW(policy.parameters(), lr=args.lr, eps=1e-5, weight_decay=args.weight_decay)

    params = sum(p.numel() for p in policy.parameters())
    logger.info(f"policy: {params/1e6:.2f}M params, obs={obs_size}, act={num_actions}, envs={N}, T={T}")

    batch_size = N * T
    best_return = -float("inf")
    total_timesteps = 0
    num_updates = 0
    peak_vram = 0.0
    checkpoint_path = config.checkpoint_dir / "autoresearch_best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    saved_checkpoints = []

    buf_obs = torch.zeros((T, N, obs_size), dtype=torch.float32)
    buf_act = torch.zeros((T, N), dtype=torch.long)
    buf_logprob = torch.zeros((T, N), dtype=torch.float32)
    buf_reward = torch.zeros((T, N), dtype=torch.float32)
    buf_done = torch.zeros((T, N), dtype=torch.float32)
    buf_value = torch.zeros((T, N), dtype=torch.float32)

    train_deadline = TIME_BUDGET - 12

    while True:
        elapsed = time.time() - t0
        if elapsed > train_deadline:
            break

        progress = min(elapsed / train_deadline, 1.0)

        if num_updates < args.warmup_updates:
            warmup_frac = (num_updates + 1) / args.warmup_updates
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * warmup_frac
        elif args.anneal_lr:
            # Cosine annealing with min_lr = 5% of peak
            min_lr_frac = 0.05
            cosine_frac = min_lr_frac + (1.0 - min_lr_frac) * 0.5 * (1.0 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * cosine_frac

        current_clip_eps = args.clip_eps + (args.clip_eps_end - args.clip_eps) * progress
        current_ent_coef = args.ent_coef + (args.ent_coef_end - args.ent_coef) * progress

        policy.eval()
        for step in range(T):
            obs_t = torch.from_numpy(obs_buf).to(device, non_blocking=True)
            buf_obs[step] = obs_t

            policy.obs_norm.update(obs_t)

            with torch.no_grad():
                action, logprob, _, value = policy.get_action_and_value(obs_t)

            buf_act[step] = action.cpu()
            buf_logprob[step] = logprob.cpu()
            buf_value[step] = value.cpu()

            act_buf[:] = action.cpu().numpy().astype(np.int32)
            binding.vec_step(vec_handle)

            buf_reward[step] = torch.from_numpy(rew_buf)
            buf_done[step] = torch.from_numpy(term_buf.astype(np.float32))

        total_timesteps += batch_size

        with torch.no_grad():
            next_obs = torch.from_numpy(obs_buf).to(device, non_blocking=True)
            policy.obs_norm.update(next_obs)
            next_value = policy.get_value(next_obs).cpu()

        advantages = torch.zeros_like(buf_reward)
        last_gae = torch.zeros(N)
        for t in reversed(range(T)):
            if t == T - 1:
                nv = next_value
            else:
                nv = buf_value[t + 1]
            not_done = 1.0 - buf_done[t]
            delta = buf_reward[t] + args.gamma * nv * not_done - buf_value[t]
            last_gae = delta + args.gamma * args.gae_lambda * not_done * last_gae
            advantages[t] = last_gae

        returns = advantages + buf_value

        policy.train()
        b_obs = buf_obs.reshape(-1, obs_size).to(device)
        b_act = buf_act.reshape(-1).to(device)
        b_logprob = buf_logprob.reshape(-1).to(device)
        b_old_value = buf_value.reshape(-1).to(device)
        b_advantages = advantages.reshape(-1).to(device)
        b_returns = returns.reshape(-1).to(device)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        for _ in range(args.ppo_epochs):
            indices = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_idx = indices[start:end]

                _, new_logprob, entropy, new_value = policy.get_action_and_value(b_obs[mb_idx], b_act[mb_idx])
                ratio = (new_logprob - b_logprob[mb_idx]).exp()
                mb_adv = b_advantages[mb_idx]

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - current_clip_eps, 1 + current_clip_eps)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                if args.clip_vf:
                    v_clipped = b_old_value[mb_idx] + torch.clamp(
                        new_value - b_old_value[mb_idx], -current_clip_eps, current_clip_eps
                    )
                    v_loss = 0.5 * torch.max(
                        (new_value - b_returns[mb_idx]) ** 2,
                        (v_clipped - b_returns[mb_idx]) ** 2,
                    ).mean()
                else:
                    v_loss = 0.5 * ((new_value - b_returns[mb_idx]) ** 2).mean()
                ent_loss = entropy.mean()
                loss = pg_loss + args.vf_coef * v_loss - current_ent_coef * ent_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

        num_updates += 1
        mean_reward = buf_reward.mean().item()

        if torch.cuda.is_available():
            peak_vram = max(peak_vram, torch.cuda.max_memory_allocated() / 1e6)

        if mean_reward > best_return:
            best_return = mean_reward
            _save_ckpt(policy, args, num_updates, total_timesteps, best_return, checkpoint_path)

        if num_updates % args.save_interval == 0:
            ckpt_path = config.checkpoint_dir / f"autoresearch_u{num_updates:04d}.pt"
            _save_ckpt(policy, args, num_updates, total_timesteps, best_return, ckpt_path)
            saved_checkpoints.append(ckpt_path)

        if num_updates % 10 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            logger.info(f"upd={num_updates} steps={total_timesteps} rew={mean_reward:.4f} best={best_return:.4f} lr={lr_now:.2e}")

    binding.vec_close(vec_handle)
    training_seconds = time.time() - t0

    final_path = config.checkpoint_dir / "autoresearch_final.pt"
    _save_ckpt(policy, args, num_updates, total_timesteps, best_return, final_path)

    eval_checkpoints = [final_path]
    if checkpoint_path.exists() and checkpoint_path != final_path:
        eval_checkpoints.append(checkpoint_path)
    for cp in reversed(saved_checkpoints):
        if cp.exists() and cp not in eval_checkpoints:
            eval_checkpoints.append(cp)

    logger.info(f"training done: {num_updates} updates, {total_timesteps} steps in {training_seconds:.1f}s")
    logger.info(f"evaluating {len(eval_checkpoints)} checkpoints (latest first)...")

    best_metrics = {"robust_score": -999.0}
    best_ckpt_name = "none"

    for ckpt_path in eval_checkpoints:
        if time.time() - t0 > TIME_BUDGET - 5:
            logger.info("eval time limit, stopping")
            break
        try:
            metrics = evaluate_checkpoint(ckpt_path, TradingPolicy, config, device=device)
            rs = metrics.get("robust_score", 0.0)
            logger.info(f"  {ckpt_path.name}: robust_score={rs:.6f}")
            if rs > best_metrics["robust_score"]:
                best_metrics = metrics
                best_ckpt_name = ckpt_path.name
        except Exception as e:
            logger.warning(f"  {ckpt_path.name}: eval failed: {e}")

    if best_metrics["robust_score"] < -100:
        best_metrics = {"robust_score": 0.0}

    logger.info(f"best checkpoint: {best_ckpt_name} robust_score={best_metrics.get('robust_score', 0):.6f}")

    total_seconds = time.time() - t0
    print_metrics(best_metrics, training_seconds, total_seconds, peak_vram, total_timesteps, num_updates)
    return 0


def main():
    return train()


if __name__ == "__main__":
    sys.exit(main())
