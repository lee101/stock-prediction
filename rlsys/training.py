"""PPO trainer optimized for trading environments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast

from .buffers import RolloutBuffer
from .config import MarketConfig, TrainingConfig
from .llm_guidance import StrategyLLMGuidance
from .market_environment import MarketEnvironment
from .policy import ActorCriticPolicy
from .utils import ObservationNormalizer, EpisodeMetrics, get_device, seed_everything, update_dict


@dataclass(slots=True)
class TrainingState:
    global_step: int = 0
    episode: int = 0


class PPOTrainer:
    def __init__(
        self,
        env: MarketEnvironment,
        policy: ActorCriticPolicy,
        training_config: TrainingConfig,
        market_config: Optional[MarketConfig] = None,
        guidance: Optional[StrategyLLMGuidance] = None,
    ) -> None:
        self.env = env
        self.policy = policy
        self.config = training_config
        self.market_config = market_config or MarketConfig()
        self.guidance = guidance

        seed_everything(self.config.seed)
        self.device = get_device(self.config.device)
        self.policy.to(self.device)

        observation_dim = env.observation_space.shape[0]
        self.buffer = RolloutBuffer(self.config.rollout_steps, observation_dim, self.device)

        adamw_kwargs = {
            "lr": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
        }
        if torch.cuda.is_available():
            adamw_kwargs["fused"] = True
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), **adamw_kwargs)
        self.scaler = GradScaler(enabled=self.config.use_amp and torch.cuda.is_available())
        self.state = TrainingState()
        self._last_obs = np.zeros(observation_dim, dtype=np.float32)
        self._last_done = True
        self.total_updates = max(1, math.ceil(self.config.total_timesteps / self.config.rollout_steps))
        self.updates_completed = 0
        self._normalizer = (
            ObservationNormalizer(observation_dim, device=self.device)
            if self.config.normalize_observations
            else None
        )

    def collect_rollout(self) -> EpisodeMetrics:
        self.buffer.reset()
        obs, _ = self.env.reset()
        episodes: list[EpisodeMetrics] = []
        last_done = False
        for step in range(self.config.rollout_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            obs_tensor = self._normalize_observation(obs_tensor, update=True)
            with torch.no_grad():
                action, log_prob, value = self.policy.act(obs_tensor.unsqueeze(0))
            action_np = action.squeeze(0).cpu().numpy()
            next_obs, reward, done, truncated, info = self.env.step(action_np)

            reward_tensor = torch.tensor(reward, dtype=torch.float32, device=self.device)
            done_tensor = torch.tensor(float(done), dtype=torch.float32, device=self.device)
            self.buffer.add(
                observation=obs_tensor,
                action=action.squeeze(0),
                log_prob=log_prob.squeeze(0),
                reward=reward_tensor,
                done=done_tensor,
                value=value.squeeze(),
            )

            self.state.global_step += 1
            obs = next_obs
            last_done = bool(done or truncated)

            if done or truncated:
                episode_metric = EpisodeMetrics(
                    reward=float(info.get("episode_reward", reward)),
                    length=int(info.get("episode_length", step + 1)),
                    max_drawdown=float(info.get("episode_max_drawdown", 0.0)),
                    sharpe_ratio=float(info.get("episode_sharpe", 0.0)),
                    turnover=float(info.get("episode_turnover", 0.0)),
                    sortino_ratio=float(info.get("episode_sortino", 0.0)),
                )
                episodes.append(episode_metric)
                self.state.episode += 1
                if step < self.config.rollout_steps - 1:
                    obs, _ = self.env.reset()
            if self.state.global_step >= self.config.total_timesteps:
                break
        if episodes:
            avg_reward = float(np.mean([e.reward for e in episodes]))
            avg_length = int(np.mean([e.length for e in episodes]))
            avg_sharpe = float(np.mean([e.sharpe_ratio for e in episodes]))
            avg_turnover = float(np.mean([e.turnover for e in episodes]))
            avg_sortino = float(np.mean([e.sortino_ratio for e in episodes]))
            worst_drawdown = float(min(e.max_drawdown for e in episodes))
            episode_metrics = EpisodeMetrics(
                reward=avg_reward,
                length=avg_length,
                max_drawdown=worst_drawdown,
                sharpe_ratio=avg_sharpe,
                turnover=avg_turnover,
                sortino_ratio=avg_sortino,
            )
        else:
            episode_metrics = EpisodeMetrics(
                reward=0.0,
                length=0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                turnover=0.0,
                sortino_ratio=0.0,
            )
        last_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        last_tensor = self._normalize_observation(last_tensor, update=False)
        self._last_obs = last_tensor.detach().cpu().numpy()
        self._last_done = last_done
        return episode_metrics

    def _update_policy(self) -> Dict[str, float]:
        last_obs = torch.tensor(self._last_obs, dtype=torch.float32, device=self.device)
        if self._last_done:
            last_value = torch.zeros(1, device=self.device)
        else:
            with torch.no_grad():
                _, _, last_value = self.policy.act(last_obs.unsqueeze(0))
        self.buffer.compute_returns_and_advantages(last_value, self.config.gamma, self.config.gae_lambda)

        metrics: Dict[str, float] = {}
        updates = 0
        for _ in range(self.config.num_epochs):
            for batch in self.buffer.get(self.config.minibatch_size):
                loss_dict = self._ppo_loss(batch)
                for key, value in loss_dict.items():
                    metrics[key] = metrics.get(key, 0.0) + float(value)
                updates += 1
        if updates:
            metrics = {key: value / updates for key, value in metrics.items()}
        return metrics

    def _ppo_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        observations = batch["observations"]
        actions = batch["actions"]
        old_log_probs = batch["log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        old_values = batch["values"]

        with autocast(enabled=self.scaler.is_enabled()):
            log_probs, entropy, values = self.policy.evaluate_actions(observations, actions)
            ratio = torch.exp(log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

            value_pred_clipped = old_values + torch.clamp(
                values - old_values, -self.config.clip_range_value, self.config.clip_range_value
            )
            value_losses = (values - returns).pow(2)
            value_losses_clipped = (value_pred_clipped - returns).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

            entropy_loss = -entropy.mean()
            loss = policy_loss + self.config.vf_coef * value_loss + self.config.ent_coef * entropy_loss

        self.optimizer.zero_grad(set_to_none=True)
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

        approx_kl = torch.mean(old_log_probs - log_probs).item()
        clip_frac = (torch.abs(ratio - 1.0) > self.config.clip_range).float().mean().item()
        metrics = {
            "loss_policy": float(policy_loss.detach().cpu()),
            "loss_value": float(value_loss.detach().cpu()),
            "loss_entropy": float(entropy_loss.detach().cpu()),
            "loss_total": float(loss.detach().cpu()),
            "approx_kl": approx_kl,
            "clip_fraction": clip_frac,
        }
        return metrics

    def train(self) -> Iterable[Dict[str, float]]:
        while self.state.global_step < self.config.total_timesteps:
            episode_metrics = self.collect_rollout()
            metrics = self._update_policy()
            iteration_logs: Dict[str, float] = {}
            update_dict(iteration_logs, metrics)
            update_dict(iteration_logs, episode_metrics.as_dict())

            if self.config.target_kl and metrics.get("approx_kl", 0.0) > 1.5 * self.config.target_kl:
                self.updates_completed += 1
                current_lr = self._apply_lr_schedule()
                iteration_logs["learning_rate"] = current_lr
                yield iteration_logs
                break

            if self.guidance is not None:
                guidance_result = self.guidance.summarize(iteration_logs)
                update_dict(
                    iteration_logs,
                    {"guidance_tokens": float(len(guidance_result.response.split()))},
                )
            self.updates_completed += 1
            current_lr = self._apply_lr_schedule()
            iteration_logs["learning_rate"] = current_lr
            yield iteration_logs

    def evaluate(self, num_episodes: int = 5) -> Dict[str, float]:
        returns = []
        sharpe_ratios = []
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            total_reward = 0.0
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                obs_tensor = self._normalize_observation(obs_tensor, update=False)
                with torch.no_grad():
                    action, _, _ = self.policy.act(obs_tensor.unsqueeze(0))
                obs, reward, done, truncated, info = self.env.step(action.squeeze(0).cpu().numpy())
                total_reward += reward
                if truncated:
                    break
            returns.append(total_reward)
            sharpe_ratios.append(info.get("episode_sharpe", 0.0))
        return {
            "eval_return_mean": float(np.mean(returns)),
            "eval_return_std": float(np.std(returns)),
            "eval_sharpe_mean": float(np.mean(sharpe_ratios)),
        }

    def _apply_lr_schedule(self) -> float:
        if self.config.lr_schedule == "none":
            return float(self.optimizer.param_groups[0]["lr"])
        progress_remaining = max(0.0, 1.0 - self.updates_completed / self.total_updates)
        if self.config.lr_schedule == "linear":
            lr = self.config.learning_rate * progress_remaining
        else:
            lr = self.config.learning_rate * 0.5 * (
                1.0 + math.cos(math.pi * (1.0 - progress_remaining))
            )
        lr = float(max(lr, 1e-8))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def _normalize_observation(self, observation: torch.Tensor, update: bool) -> torch.Tensor:
        if self._normalizer is None:
            return observation
        if update:
            self._normalizer.update(observation)
        return self._normalizer.normalize(observation)
