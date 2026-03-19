#!/usr/bin/env python3
"""
Train portfolio RL agent with architecture search.
Tests multiple configs and picks best based on validation Sortino.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).parent.parent))

from pufferlib_market.portfolio_env import PortfolioEnv


class PortfolioPolicy(nn.Module):
    """Actor-Critic policy for portfolio allocation."""

    def __init__(
        self,
        obs_dim: int,
        num_symbols: int,
        discrete_bins: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        self.num_symbols = num_symbols
        self.discrete_bins = discrete_bins

        # Shared encoder
        layers = [nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
        self.encoder = nn.Sequential(*layers)

        # Per-symbol actor heads (each outputs discrete_bins logits)
        self.actor_heads = nn.ModuleList([
            nn.Linear(hidden_dim, discrete_bins) for _ in range(num_symbols)
        ])

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, obs):
        h = self.encoder(obs)
        value = self.critic(h).squeeze(-1)

        # Per-symbol logits
        logits = [head(h) for head in self.actor_heads]
        return logits, value

    def get_action(self, obs, deterministic=False):
        logits, value = self.forward(obs)

        actions = []
        log_probs = []
        for logit in logits:
            dist = Categorical(logits=logit)
            if deterministic:
                action = logit.argmax(dim=-1)
            else:
                action = dist.sample()
            actions.append(action)
            log_probs.append(dist.log_prob(action))

        actions = torch.stack(actions, dim=-1)  # [batch, num_symbols]
        log_prob = torch.stack(log_probs, dim=-1).sum(dim=-1)  # [batch]

        return actions, log_prob, value

    def evaluate_actions(self, obs, actions):
        logits, value = self.forward(obs)

        log_probs = []
        entropies = []
        for i, logit in enumerate(logits):
            dist = Categorical(logits=logit)
            log_probs.append(dist.log_prob(actions[:, i]))
            entropies.append(dist.entropy())

        log_prob = torch.stack(log_probs, dim=-1).sum(dim=-1)
        entropy = torch.stack(entropies, dim=-1).mean(dim=-1)

        return log_prob, value, entropy


class PPOTrainer:
    """PPO trainer for portfolio environment."""

    def __init__(
        self,
        env: PortfolioEnv,
        policy: PortfolioPolicy,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.env = env
        self.policy = policy.to(device)
        self.device = device

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

    def collect_rollout(self, num_steps: int = 2048):
        """Collect experience from environment."""
        obs_list, actions_list, rewards_list = [], [], []
        log_probs_list, values_list, dones_list = [], [], []

        obs, _ = self.env.reset()

        for _ in range(num_steps):
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                actions, log_prob, value = self.policy.get_action(obs_t)

            action = actions.squeeze(0).cpu().numpy()
            next_obs, reward, term, trunc, info = self.env.step(action)
            done = term or trunc

            obs_list.append(obs)
            actions_list.append(action)
            rewards_list.append(reward)
            log_probs_list.append(log_prob.item())
            values_list.append(value.item())
            dones_list.append(done)

            obs = next_obs
            if done:
                obs, _ = self.env.reset()

        return {
            'obs': np.array(obs_list),
            'actions': np.array(actions_list),
            'rewards': np.array(rewards_list),
            'log_probs': np.array(log_probs_list),
            'values': np.array(values_list),
            'dones': np.array(dones_list),
        }

    def compute_gae(self, rewards, values, dones):
        """Compute GAE advantages."""
        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, rollout, num_epochs: int = 4, batch_size: int = 64):
        """PPO update."""
        obs = torch.from_numpy(rollout['obs']).float().to(self.device)
        actions = torch.from_numpy(rollout['actions']).long().to(self.device)
        old_log_probs = torch.from_numpy(rollout['log_probs']).float().to(self.device)

        advantages, returns = self.compute_gae(
            rollout['rewards'], rollout['values'], rollout['dones']
        )
        advantages = torch.from_numpy(advantages).float().to(self.device)
        returns = torch.from_numpy(returns).float().to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_pg_loss, total_vf_loss, total_ent = 0, 0, 0
        num_updates = 0

        for _ in range(num_epochs):
            indices = np.random.permutation(len(obs))
            for start in range(0, len(obs), batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                log_prob, value, entropy = self.policy.evaluate_actions(
                    obs[batch_idx], actions[batch_idx]
                )

                # Policy loss
                ratio = torch.exp(log_prob - old_log_probs[batch_idx])
                pg_loss1 = -advantages[batch_idx] * ratio
                pg_loss2 = -advantages[batch_idx] * torch.clamp(
                    ratio, 1 - self.clip_eps, 1 + self.clip_eps
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                vf_loss = ((value - returns[batch_idx]) ** 2).mean()

                # Entropy
                ent_loss = -entropy.mean()

                loss = pg_loss + self.vf_coef * vf_loss + self.ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_ent += entropy.mean().item()
                num_updates += 1

        return {
            'pg_loss': total_pg_loss / num_updates,
            'vf_loss': total_vf_loss / num_updates,
            'entropy': total_ent / num_updates,
        }

    def evaluate(self, num_episodes: int = 10) -> dict:
        """Evaluate policy."""
        returns, sortinos = [], []

        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            episode_return = 0
            ep_returns = []

            while True:
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    actions, _, _ = self.policy.get_action(obs_t, deterministic=True)
                action = actions.squeeze(0).cpu().numpy()

                obs, reward, term, trunc, info = self.env.step(action)
                episode_return += reward
                ep_returns.append(info.get('return', 0))

                if term or trunc:
                    break

            returns.append(info.get('total_return', episode_return / 100))
            sortinos.append(info.get('sortino', 0))

        return {
            'mean_return': np.mean(returns),
            'mean_sortino': np.mean(sortinos),
            'std_return': np.std(returns),
        }


def train_config(
    data_path: str,
    hidden_dim: int,
    num_layers: int,
    discrete_bins: int,
    lr: float,
    total_steps: int,
    checkpoint_dir: str,
):
    """Train with specific config."""
    env = PortfolioEnv(data_path, discrete_bins=discrete_bins, max_steps=720)
    obs_dim = env.observation_space.shape[0]

    policy = PortfolioPolicy(
        obs_dim=obs_dim,
        num_symbols=env.num_symbols,
        discrete_bins=discrete_bins,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )

    trainer = PPOTrainer(env, policy, lr=lr)

    best_sortino = -np.inf
    step = 0
    rollout_steps = 2048

    while step < total_steps:
        rollout = trainer.collect_rollout(rollout_steps)
        losses = trainer.update(rollout)
        step += rollout_steps

        if step % (rollout_steps * 10) == 0:
            eval_results = trainer.evaluate(num_episodes=5)
            print(f"[{step:,}] ret={eval_results['mean_return']:.2%} "
                  f"sortino={eval_results['mean_sortino']:.1f} "
                  f"pg={losses['pg_loss']:.4f} ent={losses['entropy']:.3f}", flush=True)

            if eval_results['mean_sortino'] > best_sortino:
                best_sortino = eval_results['mean_sortino']
                Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                torch.save({
                    'policy': policy.state_dict(),
                    'config': {
                        'hidden_dim': hidden_dim,
                        'num_layers': num_layers,
                        'discrete_bins': discrete_bins,
                        'num_symbols': env.num_symbols,
                        'obs_dim': obs_dim,
                    },
                    'sortino': best_sortino,
                    'step': step,
                }, f"{checkpoint_dir}/best.pt")

    return best_sortino


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="pufferlib_market/data/stocks10_data.bin")
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--search", action="store_true", help="Architecture search")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--discrete-bins", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint-dir", default="experiments/portfolio_rl")
    args = parser.parse_args()

    if args.search:
        # Architecture search
        configs = [
            {'hidden_dim': 128, 'num_layers': 2, 'discrete_bins': 5, 'lr': 3e-4},
            {'hidden_dim': 256, 'num_layers': 3, 'discrete_bins': 5, 'lr': 3e-4},
            {'hidden_dim': 256, 'num_layers': 3, 'discrete_bins': 10, 'lr': 3e-4},
            {'hidden_dim': 512, 'num_layers': 4, 'discrete_bins': 5, 'lr': 1e-4},
        ]

        results = []
        for i, cfg in enumerate(configs):
            print(f"\n=== Config {i+1}/{len(configs)}: {cfg} ===", flush=True)
            ckpt_dir = f"{args.checkpoint_dir}/config_{i}"
            sortino = train_config(
                args.data_path,
                cfg['hidden_dim'],
                cfg['num_layers'],
                cfg['discrete_bins'],
                cfg['lr'],
                args.total_steps,
                ckpt_dir,
            )
            results.append({'config': cfg, 'sortino': sortino, 'checkpoint': ckpt_dir})
            print(f"Config {i+1} final sortino: {sortino:.2f}")

        # Print best
        best = max(results, key=lambda x: x['sortino'])
        print(f"\n=== Best config: {best['config']} ===")
        print(f"Sortino: {best['sortino']:.2f}")
        print(f"Checkpoint: {best['checkpoint']}")

    else:
        # Single training run
        train_config(
            args.data_path,
            args.hidden_dim,
            args.num_layers,
            args.discrete_bins,
            args.lr,
            args.total_steps,
            args.checkpoint_dir,
        )


if __name__ == "__main__":
    main()
