from __future__ import annotations

from typing import Iterator, Tuple

import torch


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    bootstrap_value: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generalised Advantage Estimation.

    Args:
        rewards: [T, 1] tensor of rewards.
        dones: [T, 1] tensor of episode terminations (1 for done).
        values: [T, 1] tensor of value estimates.
        bootstrap_value: [1] tensor containing the value prediction for the next state.
    """
    T = rewards.size(0)
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(1, device=rewards.device, dtype=rewards.dtype)
    next_value = bootstrap_value

    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_adv = delta + gamma * gae_lambda * mask * last_adv
        advantages[t] = last_adv
        next_value = values[t]

    returns = advantages + values
    return advantages, returns


def minibatch_iterator(
    *tensors: torch.Tensor,
    batch_size: int | None = None,
    shuffle: bool = True,
) -> Iterator[Tuple[torch.Tensor, ...]]:
    """
    Yield mini-batches from the provided tensors while preserving alignment.
    """
    if not tensors:
        return
    N = tensors[0].size(0)
    if batch_size is None or batch_size >= N:
        yield tensors
        return

    indices = torch.randperm(N) if shuffle else torch.arange(N)
    for start in range(0, N, batch_size):
        batch_idx = indices[start:start + batch_size]
        yield tuple(t[batch_idx] for t in tensors)
