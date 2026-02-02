from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RunningMoments:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        if not np.isfinite(value):
            return
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return float(self.m2 / (self.count - 1))

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance))


@dataclass
class DrawdownTracker:
    peak: Optional[float] = None
    drawdown: float = 0.0

    def update(self, equity: float) -> float:
        if not np.isfinite(equity):
            return self.drawdown
        if self.peak is None or equity > self.peak:
            self.peak = equity
        if self.peak is None or self.peak == 0.0:
            self.drawdown = 0.0
            return self.drawdown
        self.drawdown = float((equity - self.peak) / self.peak)
        return self.drawdown


@dataclass
class RewardState:
    moments: RunningMoments
    drawdown: DrawdownTracker


def risk_adjusted_reward(
    *,
    step_return: float,
    state: RewardState,
    equity: Optional[float] = None,
    drawdown_penalty: float = 0.0,
    volatility_penalty: float = 0.0,
) -> float:
    """Compute a risk-adjusted reward from a single return step.

    The state is updated in-place with the new return/equity.
    """
    state.moments.update(step_return)
    current_drawdown = 0.0
    if equity is not None:
        current_drawdown = min(0.0, state.drawdown.update(equity))
    vol = state.moments.std
    reward = float(step_return)
    if drawdown_penalty:
        reward -= drawdown_penalty * abs(current_drawdown)
    if volatility_penalty:
        reward -= volatility_penalty * vol
    return reward


def sharpe_like_reward(
    *,
    step_return: float,
    state: RunningMoments,
    eps: float = 1e-8,
    clip: Optional[float] = None,
) -> float:
    """Reward scaled by running volatility (online Sharpe proxy)."""
    state.update(step_return)
    denom = state.std + eps
    reward = float(step_return / denom) if denom > 0 else 0.0
    if clip is not None:
        reward = float(np.clip(reward, -clip, clip))
    return reward


__all__ = [
    "DrawdownTracker",
    "RewardState",
    "RunningMoments",
    "risk_adjusted_reward",
    "sharpe_like_reward",
]
