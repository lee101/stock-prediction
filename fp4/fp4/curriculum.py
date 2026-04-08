"""Leverage curriculum for fp4 RL trainers (Phase 6, Unit P6-2).

Slowly widens the allowed leverage cap as training progresses, but only when
the current cap is producing a positive Sortino ratio. Before `ramp_steps`
the cap is held at `start`. After that, each query advances the cap toward
`target`; once `target` is reached, it further advances toward `cap` (the
hard ceiling). Negative/zero Sortino holds the cap where it is — the policy
must first earn the right to more leverage.

The scheduler is purely functional (a Python float state) so it's safe to
share between CUDA-graph trainers: callers pass it as an int + float and
use the returned cap to clamp the env_adapter / policy action.
"""
from __future__ import annotations


class LeverageCurriculum:
    """Sortino-gated leverage cap scheduler.

    Parameters
    ----------
    start:
        Initial cap (used before `ramp_steps`).
    target:
        First milestone — the cap the curriculum ramps to after `ramp_steps`.
    ramp_steps:
        Number of training steps to hold `start` before ramping.
    cap:
        Hard ceiling the curriculum will not exceed.
    step_size:
        Fractional advance per successful `current_cap` call. Default 0.05
        of the remaining distance to the next milestone.
    """

    def __init__(
        self,
        start: float = 1.0,
        target: float = 2.0,
        ramp_steps: int = 500_000,
        cap: float = 5.0,
        step_size: float = 0.05,
    ) -> None:
        if start <= 0.0:
            raise ValueError(f"LeverageCurriculum: start must be > 0, got {start}")
        if target < start:
            raise ValueError(
                f"LeverageCurriculum: target ({target}) must be >= start ({start})"
            )
        if cap < target:
            raise ValueError(
                f"LeverageCurriculum: cap ({cap}) must be >= target ({target})"
            )
        if ramp_steps < 0:
            raise ValueError(
                f"LeverageCurriculum: ramp_steps must be >= 0, got {ramp_steps}"
            )
        if not (0.0 < step_size <= 1.0):
            raise ValueError(
                f"LeverageCurriculum: step_size must be in (0, 1], got {step_size}"
            )
        self.start = float(start)
        self.target = float(target)
        self.ramp_steps = int(ramp_steps)
        self.cap = float(cap)
        self.step_size = float(step_size)
        self._current = float(start)

    def current_cap(self, step: int, last_sortino: float) -> float:
        """Return the cap for the current step.

        Advances the internal cap if `step >= ramp_steps` and `last_sortino > 0`.
        Otherwise holds. Never exceeds `cap`.
        """
        if step < self.ramp_steps:
            return self._current
        if not (last_sortino > 0.0):
            return self._current
        if self._current < self.target:
            nxt = self._current + self.step_size * (self.target - self._current)
            if nxt > self.target:
                nxt = self.target
            self._current = nxt
        elif self._current < self.cap:
            nxt = self._current + self.step_size * (self.cap - self._current)
            if nxt > self.cap:
                nxt = self.cap
            self._current = nxt
        # Hard clamp (defensive).
        if self._current > self.cap:
            self._current = self.cap
        return self._current

    # --- checkpointing -------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "start": self.start,
            "target": self.target,
            "ramp_steps": self.ramp_steps,
            "cap": self.cap,
            "step_size": self.step_size,
            "current": self._current,
        }

    def load_state_dict(self, state: dict) -> None:
        self.start = float(state["start"])
        self.target = float(state["target"])
        self.ramp_steps = int(state["ramp_steps"])
        self.cap = float(state["cap"])
        self.step_size = float(state["step_size"])
        self._current = float(state["current"])
