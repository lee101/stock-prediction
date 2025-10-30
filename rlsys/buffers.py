"""Buffers for storing on-policy rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass(slots=True)
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor


class RolloutBuffer:
    def __init__(self, size: int, observation_dim: int, device: torch.device) -> None:
        self.capacity = size
        self.observation_dim = observation_dim
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.observations = torch.zeros((self.capacity, self.observation_dim), device=self.device)
        self.actions = torch.zeros((self.capacity, 1), device=self.device)
        self.log_probs = torch.zeros(self.capacity, device=self.device)
        self.rewards = torch.zeros(self.capacity, device=self.device)
        self.dones = torch.zeros(self.capacity, device=self.device)
        self.values = torch.zeros(self.capacity, device=self.device)
        self.advantages = torch.zeros(self.capacity, device=self.device)
        self.returns = torch.zeros(self.capacity, device=self.device)
        self.ptr = 0

    def add(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        if self.ptr >= self.capacity:
            raise RuntimeError("Rollout buffer overflow")
        self.observations[self.ptr].copy_(observation)
        self.actions[self.ptr].copy_(action)
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.ptr += 1

    def compute_returns_and_advantages(self, last_value: torch.Tensor, gamma: float, gae_lambda: float) -> None:
        last_value = last_value.squeeze(-1)
        last_gae = torch.zeros(1, device=self.device)
        for step in reversed(range(self.ptr)):
            mask = 1.0 - self.dones[step]
            next_value = last_value if step == self.ptr - 1 else self.values[step + 1]
            delta = self.rewards[step] + gamma * next_value * mask - self.values[step]
            last_gae = delta + gamma * gae_lambda * mask * last_gae
            self.advantages[step] = last_gae
        self.returns = self.advantages + self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std(unbiased=False) + 1e-8)

    def get(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if self.ptr == 0:
            raise RuntimeError("Rollout buffer is empty")
        indices = torch.randperm(self.ptr, device=self.device)
        for start in range(0, self.ptr, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            yield {
                "observations": self.observations[batch_idx],
                "actions": self.actions[batch_idx],
                "log_probs": self.log_probs[batch_idx],
                "advantages": self.advantages[batch_idx],
                "returns": self.returns[batch_idx],
                "values": self.values[batch_idx],
            }
