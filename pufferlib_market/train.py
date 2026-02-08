"""
PPO training for the C trading environment using PufferLib.

Designed to stay under 8GB GPU memory:
  - Small MLP policy (~500K params)
  - 64 parallel envs
  - 256-step rollouts
  - Batch size 2048
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# PufferLib imports
import pufferlib
import pufferlib.vector

# Local
from pufferlib_market.environment import TradingEnvConfig, TradingEnv


# ─── Policy Network ───────────────────────────────────────────────────

class TradingPolicy(nn.Module):
    """
    Small MLP policy + value head for trading.
    ~500K params to stay well under 8GB GPU.
    """

    def __init__(self, obs_size: int, num_actions: int, hidden: int = 256):
        super().__init__()
        self.obs_size = obs_size
        self.num_actions = num_actions

        # Shared feature extractor
        self.encoder = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Policy head (actor)
        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_actions),
        )

        # Value head (critic)
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

        # Orthogonal init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Smaller init for policy output
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, x):
        h = self.encoder(x)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

    def get_action_and_value(self, x, action=None):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def get_value(self, x):
        h = self.encoder(x)
        return self.critic(h).squeeze(-1)


# ─── PPO Training Loop ────────────────────────────────────────────────

def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """Generalized Advantage Estimation."""
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae

    returns = advantages + values
    return advantages, returns


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    # ── Load shared data ──
    data_path = str(Path(args.data_path).resolve())
    import pufferlib_market.binding as binding
    binding.shared(data_path=data_path)
    print(f"Loaded market data from {data_path}")

    # ── Read binary header to get num_symbols ──
    import struct
    with open(data_path, "rb") as f:
        header = f.read(64)
    _, _, num_symbols, num_timesteps, _, _ = struct.unpack("<4sIIIII", header[:24])
    print(f"  {num_symbols} symbols, {num_timesteps} timesteps")

    # ── Env config ──
    config = TradingEnvConfig(
        data_path=data_path,
        max_steps=args.max_steps,
        fee_rate=args.fee_rate,
        max_leverage=args.max_leverage,
        num_symbols=num_symbols,
    )

    obs_size = num_symbols * 16 + 5 + num_symbols
    num_actions = 1 + 2 * num_symbols
    print(f"  obs_size={obs_size}, num_actions={num_actions}")

    # ── Create vectorised envs ──
    # PufferLib vec_init handles batched numpy buffers
    num_envs = args.num_envs
    obs_buf = np.zeros((num_envs, obs_size), dtype=np.float32)
    act_buf = np.zeros((num_envs,), dtype=np.int32)
    rew_buf = np.zeros((num_envs,), dtype=np.float32)
    term_buf = np.zeros((num_envs,), dtype=np.uint8)
    trunc_buf = np.zeros((num_envs,), dtype=np.uint8)

    vec_handle = binding.vec_init(
        obs_buf, act_buf, rew_buf, term_buf, trunc_buf,
        num_envs, args.seed,
        max_steps=config.max_steps,
        fee_rate=config.fee_rate,
        max_leverage=config.max_leverage,
    )
    binding.vec_reset(vec_handle, args.seed)
    print(f"  Created {num_envs} parallel envs")

    # ── Policy ──
    policy = TradingPolicy(obs_size, num_actions, hidden=args.hidden_size).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"  Policy: {total_params:,} params ({total_params * 4 / 1e6:.1f} MB)")

    # ── Estimate GPU memory ──
    rollout_mem = num_envs * args.rollout_len * (obs_size * 4 + 4 * 4) / 1e6
    print(f"  Estimated rollout buffer: {rollout_mem:.1f} MB")

    # ── Rollout buffers ──
    T = args.rollout_len
    N = num_envs
    buf_obs = torch.zeros((T, N, obs_size), dtype=torch.float32)
    buf_act = torch.zeros((T, N), dtype=torch.long)
    buf_logprob = torch.zeros((T, N), dtype=torch.float32)
    buf_reward = torch.zeros((T, N), dtype=torch.float32)
    buf_done = torch.zeros((T, N), dtype=torch.float32)
    buf_value = torch.zeros((T, N), dtype=torch.float32)

    # ── Training loop ──
    global_step = 0
    num_updates = args.total_timesteps // (T * N)
    start_time = time.time()
    best_return = -float("inf")

    print(f"\nTraining: {num_updates} updates, {args.total_timesteps:,} total steps")
    print(f"  rollout_len={T}, num_envs={N}, batch_size={T * N}")
    print(f"  PPO epochs={args.ppo_epochs}, minibatch_size={args.minibatch_size}")
    print()

    for update in range(1, num_updates + 1):
        # ── Learning rate annealing ──
        if args.anneal_lr:
            frac = 1.0 - (update - 1) / num_updates
            lr_now = frac * args.lr
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

        # ── Collect rollout ──
        policy.eval()
        for step in range(T):
            obs_tensor = torch.from_numpy(obs_buf.copy()).to(device)
            buf_obs[step] = obs_tensor

            with torch.no_grad():
                action, logprob, _, value = policy.get_action_and_value(obs_tensor)

            buf_act[step] = action.cpu()
            buf_logprob[step] = logprob.cpu()
            buf_value[step] = value.cpu()

            # Write actions to C buffer and step
            act_buf[:] = action.cpu().numpy().astype(np.int32)
            binding.vec_step(vec_handle)

            buf_reward[step] = torch.from_numpy(rew_buf.copy())
            buf_done[step] = torch.from_numpy(term_buf.copy().astype(np.float32))

            global_step += N

        # ── Compute advantages ──
        with torch.no_grad():
            next_obs = torch.from_numpy(obs_buf.copy()).to(device)
            next_value = policy.get_value(next_obs).cpu()

        # Flatten envs for GAE
        advantages = torch.zeros_like(buf_reward)
        last_gae = torch.zeros(N)

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_value
            else:
                next_val = buf_value[t + 1]
            not_done = 1.0 - buf_done[t]
            delta = buf_reward[t] + args.gamma * next_val * not_done - buf_value[t]
            last_gae = delta + args.gamma * args.gae_lambda * not_done * last_gae
            advantages[t] = last_gae

        returns = advantages + buf_value

        # ── PPO update ──
        policy.train()
        b_obs = buf_obs.reshape(-1, obs_size).to(device)
        b_act = buf_act.reshape(-1).to(device)
        b_logprob = buf_logprob.reshape(-1).to(device)
        b_advantages = advantages.reshape(-1).to(device)
        b_returns = returns.reshape(-1).to(device)

        # Normalise advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        total_pg_loss = 0
        total_v_loss = 0
        total_entropy = 0
        num_mb = 0

        batch_size = T * N
        mb_size = args.minibatch_size

        for epoch in range(args.ppo_epochs):
            indices = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, mb_size):
                end = min(start + mb_size, batch_size)
                mb_idx = indices[start:end]

                _, new_logprob, entropy, new_value = policy.get_action_and_value(
                    b_obs[mb_idx], b_act[mb_idx]
                )

                # PPO clipped objective
                log_ratio = new_logprob - b_logprob[mb_idx]
                ratio = log_ratio.exp()
                mb_adv = b_advantages[mb_idx]

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped)
                v_loss = 0.5 * ((new_value - b_returns[mb_idx]) ** 2).mean()

                # Entropy bonus
                ent_loss = entropy.mean()

                loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * ent_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

                total_pg_loss += pg_loss.item()
                total_v_loss += v_loss.item()
                total_entropy += ent_loss.item()
                num_mb += 1

        # ── Logging ──
        log_info = binding.vec_log(vec_handle)
        elapsed = time.time() - start_time
        sps = global_step / elapsed

        avg_pg = total_pg_loss / max(num_mb, 1)
        avg_vl = total_v_loss / max(num_mb, 1)
        avg_ent = total_entropy / max(num_mb, 1)

        if log_info and "total_return" in log_info:
            ep_return = log_info["total_return"]
            ep_sortino = log_info.get("sortino", 0)
            ep_trades = log_info.get("num_trades", 0)
            ep_wr = log_info.get("win_rate", 0)
            n = log_info.get("n", 0)

            if ep_return > best_return:
                best_return = ep_return
                ckpt_path = Path(args.checkpoint_dir) / "best.pt"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "update": update,
                    "global_step": global_step,
                    "best_return": best_return,
                }, ckpt_path)

            print(
                f"[{update:4d}/{num_updates}] "
                f"step={global_step:8,d}  sps={sps:.0f}  "
                f"ret={ep_return:+.4f}  sortino={ep_sortino:.2f}  "
                f"trades={ep_trades:.0f}  wr={ep_wr:.2f}  "
                f"pg={avg_pg:.4f}  vl={avg_vl:.4f}  ent={avg_ent:.3f}  "
                f"n={n:.0f}"
            )
        else:
            print(
                f"[{update:4d}/{num_updates}] "
                f"step={global_step:8,d}  sps={sps:.0f}  "
                f"pg={avg_pg:.4f}  vl={avg_vl:.4f}  ent={avg_ent:.3f}"
            )

        # ── Periodic checkpoint ──
        if update % args.save_every == 0:
            ckpt_path = Path(args.checkpoint_dir) / f"update_{update:06d}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update": update,
                "global_step": global_step,
                "best_return": best_return,
            }, ckpt_path)

    # ── Final save ──
    ckpt_path = Path(args.checkpoint_dir) / "final.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": num_updates,
        "global_step": global_step,
        "best_return": best_return,
    }, ckpt_path)

    binding.vec_close(vec_handle)
    print(f"\nTraining complete. Best return: {best_return:.4f}")
    print(f"Checkpoints saved to {args.checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="PPO training for C trading env")
    # Environment
    parser.add_argument("--data-path", default="pufferlib_market/data/market_data.bin")
    parser.add_argument("--max-steps", type=int, default=720, help="Episode length (hours)")
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--max-leverage", type=float, default=1.0)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    # Policy
    parser.add_argument("--hidden-size", type=int, default=256)

    # PPO
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--rollout-len", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--anneal-lr", action="store_true", help="Linear LR annealing")

    # Output
    parser.add_argument("--checkpoint-dir", default="pufferlib_market/checkpoints")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
