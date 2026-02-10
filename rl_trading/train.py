from __future__ import annotations

import argparse
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_trading.config import EnvConfig, TrainConfig
from rl_trading.data_loader import load_market_data
from rl_trading.env import TradingEnv
from rl_trading.policy import TradingPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fee-rate", type=float, default=0.0)
    parser.add_argument("--max-hold-bars", type=int, default=6)
    parser.add_argument("--episode-length", type=int, default=168)
    parser.add_argument("--initial-cash", type=float, default=10000.0)
    parser.add_argument("--checkpoint-dir", default="rl_trading/checkpoints")
    parser.add_argument("--anneal-lr", action="store_true", default=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    env_config = EnvConfig(
        initial_cash=args.initial_cash,
        fee_rate=args.fee_rate,
        max_hold_bars=args.max_hold_bars,
        episode_length=args.episode_length,
        num_envs=args.num_envs,
    )

    print(f"Loading market data...")
    t0 = time.time()
    market_data = load_market_data(
        env_config.symbols, env_config.data_root, env_config.validation_days,
    )
    print(f"Loaded {market_data['n_bars']} bars x {market_data['n_symbols']} syms "
          f"x {market_data['n_features']} feats in {time.time()-t0:.1f}s")
    print(f"Train: 24..{market_data['train_end']}, Val: {market_data['val_start']}..{market_data['n_bars']}")

    env = TradingEnv(
        num_envs=args.num_envs,
        env_config=env_config,
        market_data=market_data,
        seed=args.seed,
    )

    policy = TradingPolicy(env, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate, eps=1e-5)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy: {n_params:,} params")

    obs_shape = env.single_observation_space.shape
    n_actions = env.single_action_space.n
    num_envs = args.num_envs
    num_steps = args.num_steps
    batch_size = num_envs * num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_updates = args.total_timesteps // batch_size

    obs_buf = torch.zeros((num_steps, num_envs) + obs_shape, device=device)
    actions_buf = torch.zeros((num_steps, num_envs), dtype=torch.long, device=device)
    logprobs_buf = torch.zeros((num_steps, num_envs), device=device)
    rewards_buf = torch.zeros((num_steps, num_envs), device=device)
    dones_buf = torch.zeros((num_steps, num_envs), device=device)
    values_buf = torch.zeros((num_steps, num_envs), device=device)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    global_step = 0
    best_pnl = -float("inf")
    ep_pnls = deque(maxlen=100)
    ep_wrs = deque(maxlen=100)
    ep_dds = deque(maxlen=100)

    obs_np, _ = env.reset()
    next_obs = torch.from_numpy(obs_np).float().to(device)
    next_done = torch.zeros(num_envs, device=device)

    start_time = time.time()
    print(f"Training for {args.total_timesteps:,} steps ({num_updates} updates)...")

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(num_steps):
            global_step += num_envs
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                logits, value = policy(next_obs)
                value = value.flatten()
                probs = torch.distributions.Categorical(logits=logits)
                action = probs.sample()
                logprob = probs.log_prob(action)

            values_buf[step] = value
            actions_buf[step] = action
            logprobs_buf[step] = logprob

            obs_np, rew_np, term_np, trunc_np, infos = env.step(action.cpu().numpy().astype(np.float32))
            next_obs = torch.from_numpy(obs_np).float().to(device)
            rewards_buf[step] = torch.from_numpy(rew_np).float().to(device)
            next_done = torch.from_numpy(term_np.astype(np.float32)).to(device)

            for info in infos:
                if info and "episode_pnl_pct" in info:
                    ep_pnls.append(info["episode_pnl_pct"])
                    ep_wrs.append(info.get("win_rate", 0))
                    ep_dds.append(info.get("max_drawdown", 0))

        with torch.no_grad():
            _, next_value = policy(next_obs)
            next_value = next_value.flatten()

        advantages = torch.zeros_like(rewards_buf)
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones_buf[t + 1]
                nextvalues = values_buf[t + 1]
            delta = rewards_buf[t] + args.gamma * nextvalues * nextnonterminal - values_buf[t]
            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values_buf

        b_obs = obs_buf.reshape((-1,) + obs_shape)
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                new_logits, new_value = policy(b_obs[mb_inds])
                new_value = new_value.flatten()
                probs = torch.distributions.Categorical(logits=new_logits)
                new_logprob = probs.log_prob(b_actions[mb_inds])
                entropy = probs.entropy()

                logratio = new_logprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

        if update % 10 == 0 or update == 1:
            elapsed = time.time() - start_time
            sps = global_step / elapsed
            mean_pnl = np.mean(ep_pnls) if ep_pnls else 0
            mean_wr = np.mean(ep_wrs) if ep_wrs else 0
            mean_dd = np.mean(ep_dds) if ep_dds else 0
            lr = optimizer.param_groups[0]["lr"]
            cf = np.mean(clipfracs) if clipfracs else 0
            print(f"u={update:>5} step={global_step:>10,} sps={sps:>8,.0f} "
                  f"pnl={mean_pnl:+.4f} wr={mean_wr:.2f} dd={mean_dd:.4f} "
                  f"pg={pg_loss.item():.3f} v={v_loss.item():.3f} ent={entropy_loss.item():.3f} "
                  f"cf={cf:.3f} lr={lr:.2e}")

            if mean_pnl > best_pnl and len(ep_pnls) >= 3:
                best_pnl = mean_pnl
                torch.save(policy.state_dict(), os.path.join(args.checkpoint_dir, "best.pt"))

        if update % 100 == 0:
            torch.save(policy.state_dict(), os.path.join(args.checkpoint_dir, f"step_{global_step}.pt"))

    torch.save(policy.state_dict(), os.path.join(args.checkpoint_dir, "final.pt"))
    env.close()
    print(f"Done. Best PnL: {best_pnl:+.4f}. Models in {args.checkpoint_dir}/")


if __name__ == "__main__":
    main()
