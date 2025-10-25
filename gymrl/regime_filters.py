"""
Helpers for applying simple regime guards to the GymRL allocator.

These guards are lightweight heuristics derived from hold-out analysis. They
modulate leverage, turnover penalties, and loss-shutdown behaviour when recent
returns or expected trade activity indicate unfavourable regimes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Mapping, Optional


@dataclass(slots=True)
class RegimeGuardDecision:
    """Adjustment signals emitted by :class:`RegimeGuard.pre_step`."""

    leverage_scale: Optional[float] = None
    turnover_penalty: Optional[float] = None
    loss_probe_weight: Optional[float] = None
    flags: Mapping[str, bool] = field(default_factory=dict)


class RegimeGuard:
    """
    Stateful guard that tracks recent returns/turnover and emits overrides.

    Parameters mirror the proposal captured in ``evaltests/gymrl_regime_filter_plan.md``.
    The guard is intentionally conservative: it only reduces risk (lower leverage,
    higher turnover penalty, stricter loss shutdown) and never scales them up.
    """

    def __init__(
        self,
        *,
        drawdown_threshold: Optional[float],
        leverage_scale: float,
        negative_return_window: int,
        negative_return_threshold: float,
        negative_return_turnover_penalty: Optional[float],
        turnover_threshold: Optional[float],
        turnover_probe_weight: Optional[float],
        base_turnover_penalty: float,
        base_loss_probe_weight: Optional[float],
        loss_shutdown_enabled: bool,
    ) -> None:
        window = max(1, int(negative_return_window))
        self._returns: Deque[float] = deque(maxlen=window)
        self._drawdown_threshold = drawdown_threshold if drawdown_threshold is None else max(0.0, float(drawdown_threshold))
        self._leverage_scale = max(0.0, float(leverage_scale))
        self._negative_return_threshold = float(negative_return_threshold)
        self._negative_return_turnover_penalty = negative_return_turnover_penalty
        self._turnover_threshold = turnover_threshold if turnover_threshold is None else max(0.0, float(turnover_threshold))
        self._turnover_probe_weight = turnover_probe_weight if turnover_probe_weight is None else max(0.0, float(turnover_probe_weight))
        self._base_turnover_penalty = float(base_turnover_penalty)
        self._base_loss_probe_weight = base_loss_probe_weight if base_loss_probe_weight is None else max(0.0, float(base_loss_probe_weight))
        self._loss_shutdown_enabled = bool(loss_shutdown_enabled)
        self._last_flags: Dict[str, bool] = {"drawdown": False, "negative_return": False, "turnover": False}
        self._last_cumulative_return = 0.0

    def reset(self) -> None:
        """Clear trailing statistics."""
        self._returns.clear()
        self._last_flags = {"drawdown": False, "negative_return": False, "turnover": False}
        self._last_cumulative_return = 0.0

    def trailing_cumulative_return(self) -> float:
        """Return the compounded return over the recent window."""
        cumulative = 1.0
        for value in self._returns:
            cumulative *= max(1e-8, 1.0 + float(value))
        result = cumulative - 1.0
        self._last_cumulative_return = result
        return result

    @property
    def last_flags(self) -> Mapping[str, bool]:
        """Expose the most recent guard activation flags."""
        return dict(self._last_flags)

    @property
    def last_cumulative_return(self) -> float:
        """Expose the most recently computed cumulative return."""
        return float(self._last_cumulative_return)

    def pre_step(self, *, drawdown: float, anticipated_turnover: float) -> RegimeGuardDecision:
        """
        Evaluate guard conditions before executing the next action.

        Returns a :class:`RegimeGuardDecision` describing overrides that should
        be applied for the upcoming transition.
        """

        trailing_return = self.trailing_cumulative_return()
        flags = {"drawdown": False, "negative_return": False, "turnover": False}

        leverage_scale: Optional[float] = None
        if self._drawdown_threshold is not None and drawdown >= self._drawdown_threshold:
            leverage_scale = min(1.0, self._leverage_scale)
            flags["drawdown"] = True

        turnover_penalty: Optional[float] = None
        if self._returns and trailing_return <= self._negative_return_threshold and self._negative_return_turnover_penalty is not None:
            turnover_penalty = max(0.0, float(self._negative_return_turnover_penalty))
            flags["negative_return"] = True

        loss_probe_weight: Optional[float] = None
        if self._turnover_threshold is not None and anticipated_turnover >= self._turnover_threshold and self._loss_shutdown_enabled:
            if self._turnover_probe_weight is not None:
                loss_probe_weight = self._turnover_probe_weight
                flags["turnover"] = True
        if loss_probe_weight is None and self._loss_shutdown_enabled:
            loss_probe_weight = self._base_loss_probe_weight

        if turnover_penalty is None:
            turnover_penalty = self._base_turnover_penalty

        self._last_flags = flags
        decision = RegimeGuardDecision(
            leverage_scale=leverage_scale,
            turnover_penalty=turnover_penalty,
            loss_probe_weight=loss_probe_weight if self._loss_shutdown_enabled else None,
            flags=flags,
        )
        return decision

    def post_step(self, *, net_return: float) -> None:
        """Record the realised net return for future guard decisions."""
        self._returns.append(float(net_return))
