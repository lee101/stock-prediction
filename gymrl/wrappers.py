"""
Lightweight wrappers for gymrl environments.

Currently provides an observation normalizer that applies online mean/std
normalization to observations without peeking into the future.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
except ImportError as exc:  # pragma: no cover - explicit optional dependency
    raise ImportError(
        "gymnasium is required for gymrl.wrappers. Install via `uv pip install gymnasium`."
    ) from exc


@dataclass
class NormalizerConfig:
    per_episode: bool = False
    epsilon: float = 1e-6
    clip: Optional[float] = 10.0


class ObservationNormalizer(gym.Wrapper):
    """
    Online observation normalizer using Welford's algorithm.

    - Maintains running mean/std over emitted observations.
    - If `per_episode=True`, stats reset on `reset()`.
    - Avoids lookahead by updating only on observations actually seen by the agent.
    """

    def __init__(self, env: gym.Env, config: Optional[NormalizerConfig] = None):
        super().__init__(env)
        self.config = config or NormalizerConfig()
        self._count = 0
        self._mean: Optional[np.ndarray] = None
        self._M2: Optional[np.ndarray] = None

    def _update(self, x: np.ndarray) -> None:
        x = x.astype(np.float32, copy=False)
        if self._mean is None:
            self._mean = np.zeros_like(x)
            self._M2 = np.zeros_like(x)
            self._count = 0
        self._count += 1
        delta = x - self._mean
        self._mean = self._mean + delta / self._count
        delta2 = x - self._mean
        self._M2 = self._M2 + delta * delta2

    def _normalise(self, x: np.ndarray) -> np.ndarray:
        if self._mean is None or self._M2 is None or self._count < 2:
            return x.astype(np.float32, copy=False)
        var = np.maximum(self._M2 / (self._count - 1), 0.0)
        std = np.sqrt(var) + self.config.epsilon
        z = (x - self._mean) / std
        if self.config.clip is not None:
            z = np.clip(z, -self.config.clip, self.config.clip)
        return z.astype(np.float32, copy=False)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.config.per_episode:
            self._count = 0
            self._mean = None
            self._M2 = None
        obs, info = self.env.reset(seed=seed, options=options)
        self._update(obs)
        return self._normalise(obs), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, float]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._update(obs)
        return self._normalise(obs), reward, terminated, truncated, info


__all__ = ["ObservationNormalizer", "NormalizerConfig"]

