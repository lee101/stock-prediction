from __future__ import annotations

import argparse
import configparser
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

import gymnasium as gym
from pufferlib import vector

from .market_env import MarketEnvConfig, make_market_env


@dataclass
class PPOConfig:
    # Vectorisation
    num_envs: int = 1024
    num_workers: int = 4
    rollout_len: int = 128
    minibatches: int = 8
    update_iters: int = 2

    # Optimisation
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    grad_clip: float = 0.5

    # Runtime
    max_updates: int = 1000
    device: str = "auto"
    precision: str = "bf16"
    torch_compile: bool = True
    cuda_graph: bool = False

    # Architecture
    hidden_size: int = 256
    recurrent_hidden: int = 0


def _load_config(path: Optional[str]) -> Tuple[PPOConfig, MarketEnvConfig]:
    parser = configparser.ConfigParser()
    if path:
        parser.read(path)

    def _maybe(section: str, key: str, fallback):
        try:
            return parser[section][key]
        except KeyError:
            return fallback

    ppo_cfg = PPOConfig(
        num_envs=int(_maybe("vec", "num_envs", 1024)),
        num_workers=int(_maybe("vec", "num_workers", 4)),
        rollout_len=int(_maybe("train", "rollout_len", 128)),
        minibatches=int(_maybe("train", "minibatches", 8)),
        update_iters=int(_maybe("train", "update_iters", 2)),
        learning_rate=float(_maybe("train", "learning_rate", 3e-4)),
        gamma=float(_maybe("train", "gamma", 0.99)),
        gae_lambda=float(_maybe("train", "gae_lambda", 0.95)),
        clip_coef=float(_maybe("train", "clip_coef", 0.2)),
        entropy_coef=float(_maybe("train", "entropy_coef", 0.01)),
        value_coef=float(_maybe("train", "vf_coef", 0.5)),
        grad_clip=float(_maybe("train", "grad_clip", 0.5)),
        max_updates=int(_maybe("train", "max_updates", 1000)),
        device=_maybe("train", "device", "auto"),
        precision=_maybe("train", "mixed_precision", "bf16"),
        torch_compile=_maybe("train", "torch_compile", "true").lower() == "true",
        cuda_graph=_maybe("train", "cuda_graph", "false").lower() == "true",
        hidden_size=int(_maybe("policy", "hidden_size", 256)),
        recurrent_hidden=int(_maybe("policy", "rnn_hidden_size", 0)),
    )

    env_cfg = MarketEnvConfig(
        data_dir=_maybe("env", "data_dir", "trainingdata"),
        tickers=None,
        context_len=int(_maybe("env", "context_len", 128)),
        episode_len=int(_maybe("env", "episode_len", 256)),
        horizon=int(_maybe("env", "horizon", 1)),
        fee_bps=float(_maybe("env", "fee_bps", 0.5)),
        slippage_bps=float(_maybe("env", "slippage_bps", 1.5)),
        leverage_limit=float(_maybe("env", "leverage_limit", 1.5)),
        device=ppo_cfg.device,
        precision=ppo_cfg.precision,
    )

    return ppo_cfg, env_cfg


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _resolve_dtype(name: str) -> torch.dtype:
    lower = name.lower()
    if lower in {"bf16", "bfloat16"}:
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if lower in {"fp16", "float16"}:
        return torch.float16 if torch.cuda.is_available() else torch.float32
    return torch.float32


class TanhDiagNormal:
    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor):
        self.mean = mean
        self.log_std = log_std
        self.std = torch.exp(log_std)
        self.base = Normal(mean, self.std)

    def sample(self):
        z = self.base.rsample()
        action = torch.tanh(z)
        log_prob = self.base.log_prob(z) - torch.log1p(-action.pow(2) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True)

    def log_prob(self, action: torch.Tensor):
        z = torch.atanh(torch.clamp(action, -0.999999, 0.999999))
        log_prob = self.base.log_prob(z) - torch.log1p(-action.pow(2) + 1e-6)
        return log_prob.sum(-1, keepdim=True)

    def entropy(self):
        # Use base entropy adjusted by squashing correction.
        return self.base.entropy().sum(-1, keepdim=True)


class MarketPolicy(nn.Module):
    def __init__(self, obs_space: gym.Space, action_dim: int, hidden_size: int = 256):
        super().__init__()
        obs_dim = int(np.prod(obs_space.shape))
        self.obs_dim = obs_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.actor_mean = nn.Linear(hidden_size, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(hidden_size, 1)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.zeros_(self.critic.bias)

    def _forward(self, obs: torch.Tensor) -> Tuple[TanhDiagNormal, torch.Tensor]:
        x = obs.view(obs.shape[0], -1)
        hidden = self.net(x)
        mean = self.actor_mean(hidden)
        dist = TanhDiagNormal(mean, self.actor_log_std.expand_as(mean))
        value = self.critic(hidden)
        return dist, value

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        dist, value = self._forward(obs)
        if deterministic:
            action = torch.tanh(dist.mean)
            logprob = dist.log_prob(action)
        else:
            action, logprob = dist.sample()
        return action, logprob, value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        dist, value = self._forward(obs)
        logprob = dist.log_prob(actions)
        entropy = dist.entropy()
        return logprob, entropy, value


class RolloutBuffer:
    def __init__(self, rollout_len: int, num_envs: int, obs_shape: Tuple[int, ...], action_dim: int, device: torch.device):
        self.rollout_len = rollout_len
        self.num_envs = num_envs
        self.device = device
        self.obs = torch.zeros((rollout_len, num_envs) + obs_shape, device=device, dtype=torch.float32)
        self.actions = torch.zeros(rollout_len, num_envs, action_dim, device=device, dtype=torch.float32)
        self.logprobs = torch.zeros(rollout_len, num_envs, 1, device=device, dtype=torch.float32)
        self.rewards = torch.zeros(rollout_len, num_envs, 1, device=device, dtype=torch.float32)
        self.dones = torch.zeros(rollout_len, num_envs, 1, device=device, dtype=torch.float32)
        self.values = torch.zeros(rollout_len, num_envs, 1, device=device, dtype=torch.float32)
        self._step = 0

    def add(self, obs, actions, logprobs, rewards, dones, values):
        idx = self._step
        self.obs[idx].copy_(obs)
        self.actions[idx].copy_(actions)
        self.logprobs[idx].copy_(logprobs)
        self.rewards[idx].copy_(rewards)
        self.dones[idx].copy_(dones)
        self.values[idx].copy_(values)
        self._step += 1

    def reset(self):
        self._step = 0

    def compute_returns_and_advantages(self, last_values, gamma, gae_lambda):
        rollout = self.rollout_len
        advantages = torch.zeros_like(self.rewards, device=self.device)
        last_adv = torch.zeros(self.num_envs, 1, device=self.device)
        for step in reversed(range(rollout)):
            if step == rollout - 1:
                next_values = last_values
                next_non_terminal = 1.0 - self.dones[step]
            else:
                next_values = self.values[step + 1]
                next_non_terminal = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            last_adv = delta + gamma * gae_lambda * next_non_terminal * last_adv
            advantages[step] = last_adv
        returns = advantages + self.values
        return returns, advantages


def _prepare_actions(actions: torch.Tensor, action_space) -> np.ndarray:
    actions = actions.detach().cpu().numpy()
    low = action_space.low
    high = action_space.high
    return np.clip(actions, low, high).astype(np.float32)


def _reset_done_envs(vec_env, obs_batch, done_mask):
    for env_id, done in enumerate(done_mask):
        if done:
            new_obs, info = vec_env.envs[env_id].reset()
            obs_batch[env_id] = new_obs


def train(config_path: Optional[str] = None):
    ppo_cfg, env_cfg = _load_config(config_path)
    device = _resolve_device(ppo_cfg.device)
    compute_dtype = _resolve_dtype(ppo_cfg.precision)

    env_kwargs = dict(
        data_dir=env_cfg.data_dir,
        tickers=env_cfg.tickers,
        context_len=env_cfg.context_len,
        episode_len=env_cfg.episode_len,
        horizon=env_cfg.horizon,
        fee_bps=env_cfg.fee_bps,
        slippage_bps=env_cfg.slippage_bps,
        leverage_limit=env_cfg.leverage_limit,
        device=env_cfg.device,
        precision=env_cfg.precision,
    )

    backend = vector.Multiprocessing if ppo_cfg.num_workers > 0 else vector.Serial
    make_kwargs = {
        "backend": backend,
        "num_envs": ppo_cfg.num_envs,
    }
    if ppo_cfg.num_workers > 0:
        make_kwargs["num_workers"] = ppo_cfg.num_workers
    vec_env = vector.make(
        lambda buf=None: make_market_env(buf=buf, **env_kwargs),
        **make_kwargs,
    )

    single_space = vec_env.single_observation_space
    action_space = vec_env.single_action_space
    policy = MarketPolicy(single_space, action_space.shape[0], hidden_size=ppo_cfg.hidden_size).to(device)

    if ppo_cfg.torch_compile and hasattr(torch, "compile"):
        policy = torch.compile(policy, mode="max-autotune", fullgraph=False)

    optimizer = optim.AdamW(policy.parameters(), lr=ppo_cfg.learning_rate, fused=torch.cuda.is_available())
    scaler = torch.cuda.amp.GradScaler(enabled=(compute_dtype in {torch.float16, torch.bfloat16}))

    obs, info = vec_env.reset()
    obs_tensor = torch.as_tensor(obs, device=device, dtype=torch.float32)

    buffer = RolloutBuffer(
        rollout_len=ppo_cfg.rollout_len,
        num_envs=ppo_cfg.num_envs,
        obs_shape=single_space.shape,
        action_dim=action_space.shape[0],
        device=device,
    )

    total_steps = 0
    for update in range(ppo_cfg.max_updates):
        buffer.reset()
        for step in range(ppo_cfg.rollout_len):
            with torch.autocast(device_type=device.type, dtype=compute_dtype, enabled=scaler.is_enabled()):
                actions, logprobs, values = policy.act(obs_tensor)

            actions_np = _prepare_actions(actions, action_space)
            next_obs, reward, terminated, truncated, info = vec_env.step(actions_np)
            done_mask = np.logical_or(terminated, truncated).astype(np.float32)

            reward_tensor = torch.as_tensor(reward, device=device, dtype=torch.float32).unsqueeze(-1)
            done_tensor = torch.as_tensor(done_mask, device=device, dtype=torch.float32).unsqueeze(-1)

            buffer.add(
                obs_tensor.detach(),
                actions.detach(),
                logprobs.detach(),
                reward_tensor,
                done_tensor,
                values.detach(),
            )

            obs = next_obs
            if done_mask.any():
                _reset_done_envs(vec_env, obs, done_mask.astype(bool))
            obs_tensor = torch.as_tensor(obs, device=device, dtype=torch.float32)
            total_steps += ppo_cfg.num_envs

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=compute_dtype, enabled=scaler.is_enabled()):
            _, _, last_values = policy.act(obs_tensor, deterministic=True)

        returns, advantages = buffer.compute_returns_and_advantages(last_values.detach(), ppo_cfg.gamma, ppo_cfg.gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_flat = buffer.obs.view(-1, *single_space.shape)
        actions_flat = buffer.actions.view(-1, action_space.shape[0])
        logprob_flat = buffer.logprobs.view(-1, 1)
        returns_flat = returns.view(-1, 1)
        adv_flat = advantages.view(-1, 1)

        num_samples = obs_flat.shape[0]
        batch_size = num_samples // ppo_cfg.minibatches
        indices = torch.arange(num_samples, device=device)

        for epoch in range(ppo_cfg.update_iters):
            shuffled = indices[torch.randperm(num_samples, device=device)]
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                mb_idx = shuffled[start:end]
                mb_obs = obs_flat[mb_idx]
                mb_actions = actions_flat[mb_idx]
                mb_old_log = logprob_flat[mb_idx]
                mb_adv = adv_flat[mb_idx]
                mb_returns = returns_flat[mb_idx]

                with torch.autocast(device_type=device.type, dtype=compute_dtype, enabled=scaler.is_enabled()):
                    new_logprob, entropy, values = policy.evaluate(mb_obs, mb_actions)
                    ratio = torch.exp(new_logprob - mb_old_log)
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1.0 - ppo_cfg.clip_coef, 1.0 + ppo_cfg.clip_coef) * mb_adv
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = 0.5 * (mb_returns - values).pow(2).mean()
                    entropy_loss = entropy.mean()
                    loss = policy_loss + ppo_cfg.value_coef * value_loss - ppo_cfg.entropy_coef * entropy_loss

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                if ppo_cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), ppo_cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()

        mean_return = returns.mean().item()
        mean_adv = advantages.mean().item()
        print(
            f"[update {update:04d}] steps={total_steps:,} "
            f"return={mean_return:.4f} advantage={mean_adv:.4f}"
        )

    vec_env.close()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PufferLib 3.0 PPO trainer for the differentiable market environment.")
    parser.add_argument("--config", type=str, default=None, help="Path to rl.ini override.")
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
