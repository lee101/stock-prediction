"""Utility helpers for the RL system."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, MutableMapping, Optional

import numpy as np
import torch


@dataclass(slots=True)
class EpisodeMetrics:
    """Container for aggregated episode statistics."""

    reward: float
    length: int
    max_drawdown: float
    sharpe_ratio: float
    turnover: float
    sortino_ratio: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "episode_reward": self.reward,
            "episode_length": float(self.length),
            "episode_max_drawdown": self.max_drawdown,
            "episode_sharpe": self.sharpe_ratio,
            "episode_turnover": self.turnover,
            "episode_sortino": self.sortino_ratio,
        }


def compute_sharpe_ratio(returns: Iterable[float], eps: float = 1e-8) -> float:
    arr = np.asarray(tuple(returns), dtype=np.float64)
    if arr.size < 2:
        return 0.0
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    if std < eps:
        return 0.0
    return math.sqrt(252.0) * mean / (std + eps)


def compute_sortino_ratio(returns: Iterable[float], eps: float = 1e-8, target: float = 0.0) -> float:
    arr = np.asarray(tuple(returns), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    downside = np.clip(target - arr, a_min=0.0, a_max=None)
    downside_std = float(np.sqrt(np.mean(np.square(downside))))
    if downside_std < eps:
        return 0.0
    mean_excess = float(arr.mean() - target)
    return math.sqrt(252.0) * mean_excess / (downside_std + eps)


def compute_max_drawdown(equity_curve: Iterable[float]) -> float:
    arr = np.asarray(tuple(equity_curve), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(arr)
    drawdowns = (arr - peaks) / peaks
    return float(drawdowns.min(initial=0.0))


def seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred: Optional[str] = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


def update_dict(target: MutableMapping[str, float], updates: Mapping[str, float]) -> None:
    for key, value in updates.items():
        target[key] = float(value)


def discounted_returns(rewards: Iterable[float], gamma: float) -> np.ndarray:
    returns = []
    running = 0.0
    for reward in reversed(tuple(rewards)):
        running = reward + gamma * running
        returns.append(running)
    return np.asarray(list(reversed(returns)), dtype=np.float64)


class ObservationNormalizer:
    """Tracks running statistics to normalize observations online."""

    def __init__(self, size: int, device: Optional[torch.device] = None, eps: float = 1e-6) -> None:
        self.mean = torch.zeros(size, device=device)
        self.m2 = torch.zeros(size, device=device)
        self.count = 0
        self.eps = eps

    def update(self, value: torch.Tensor) -> None:
        tensor = value.detach()
        if tensor.dim() != 1 or tensor.shape != self.mean.shape:
            raise ValueError("ObservationNormalizer expects 1D tensors matching initialized shape")
        self.count += 1
        delta = tensor - self.mean
        self.mean += delta / self.count
        delta2 = tensor - self.mean
        self.m2 += delta * delta2

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        if self.count < 2:
            return value
        variance = self.m2 / (self.count - 1)
        std = torch.sqrt(variance + self.eps)
        return (value - self.mean) / std
