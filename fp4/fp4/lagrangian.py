"""Lagrangian multiplier for constrained-MDP RL (Phase 4, Unit P4-3).

Implements the primal-dual pattern used by CPO / PPO-Lagrangian / SAC-Lag:

- The policy minimises `policy_loss + Σ λ_i * (E[cost_i] - d_i)`.
- The multipliers `λ_i >= 0` are ascended at a slow rate:
  `λ_i ← max(0, λ_i + lr_lambda * (E[cost_i] - d_i))`.

Each constraint has a target `d_i` (the allowed mean cost). When the running
cost equals the target the gradient vanishes and `λ_i` stops moving. When
the constraint is violated (`mean_i > d_i`) `λ_i` grows; when satisfied
(`mean_i < d_i`) `λ_i` shrinks and is clamped at 0.
"""
from __future__ import annotations

from typing import Dict, Iterable, List

import torch


class Lagrangian:
    """Slow-update Lagrangian multipliers for a set of named constraints."""

    def __init__(
        self,
        constraint_names: Iterable[str],
        init_lambda: float = 0.0,
        lr_lambda: float = 1e-2,
        slow_update_every: int = 1,
        target_d: Dict[str, float] | None = None,
    ) -> None:
        names: List[str] = list(constraint_names)
        if not names:
            raise ValueError("Lagrangian: constraint_names must be non-empty")
        if len(set(names)) != len(names):
            raise ValueError(f"Lagrangian: duplicate names in {names}")
        if lr_lambda <= 0.0:
            raise ValueError(f"Lagrangian: lr_lambda must be > 0, got {lr_lambda}")
        if slow_update_every < 1:
            raise ValueError(
                f"Lagrangian: slow_update_every must be >= 1, got {slow_update_every}"
            )
        self.names: List[str] = names
        self.lr_lambda: float = float(lr_lambda)
        self.slow_update_every: int = int(slow_update_every)
        self.lambdas: Dict[str, float] = {n: float(init_lambda) for n in names}
        self.target_d: Dict[str, float] = {
            n: float((target_d or {}).get(n, 0.0)) for n in names
        }
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Policy-side (per-update, differentiable) contribution
    # ------------------------------------------------------------------

    def apply(self, losses_per_constraint: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return `Σ λ_i * (E[cost_i] - d_i)` as a differentiable tensor.

        `losses_per_constraint[name]` must be a scalar tensor (the mean cost
        for that constraint on the current minibatch). Multipliers are held
        constant here (stop-grad — they're updated separately via `step`).
        """
        total: torch.Tensor | None = None
        for name in self.names:
            if name not in losses_per_constraint:
                raise KeyError(f"Lagrangian.apply: missing constraint '{name}'")
            cost = losses_per_constraint[name]
            lam = float(self.lambdas[name])
            d = float(self.target_d[name])
            term = lam * (cost - d)
            total = term if total is None else total + term
        assert total is not None  # non-empty by construction
        return total

    # ------------------------------------------------------------------
    # Dual-side (slow) update
    # ------------------------------------------------------------------

    def step(self, constraint_means: Dict[str, float]) -> Dict[str, float]:
        """Primal-dual ascent on each `λ_i`, called once per policy update.

        `constraint_means[name]` is a plain Python float holding the running
        mean cost for that constraint since the previous `step`. Returns the
        updated `{name: lambda}` dict for logging.
        """
        self._step_count += 1
        if self._step_count % self.slow_update_every != 0:
            return dict(self.lambdas)
        for name in self.names:
            if name not in constraint_means:
                raise KeyError(f"Lagrangian.step: missing constraint '{name}'")
            mean_i = float(constraint_means[name])
            d = self.target_d[name]
            new_lam = self.lambdas[name] + self.lr_lambda * (mean_i - d)
            if new_lam < 0.0:
                new_lam = 0.0
            self.lambdas[name] = new_lam
        return dict(self.lambdas)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, object]:
        return {
            "names": list(self.names),
            "lr_lambda": self.lr_lambda,
            "slow_update_every": self.slow_update_every,
            "lambdas": dict(self.lambdas),
            "target_d": dict(self.target_d),
            "step_count": self._step_count,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        names = list(state["names"])  # type: ignore[arg-type]
        if names != self.names:
            # Allow reordering but require the same set.
            if set(names) != set(self.names):
                raise ValueError(
                    f"Lagrangian.load_state_dict: name mismatch "
                    f"(have {self.names}, got {names})"
                )
            self.names = names
        self.lr_lambda = float(state["lr_lambda"])  # type: ignore[arg-type]
        self.slow_update_every = int(state["slow_update_every"])  # type: ignore[arg-type]
        self.lambdas = {k: float(v) for k, v in dict(state["lambdas"]).items()}  # type: ignore[arg-type]
        self.target_d = {k: float(v) for k, v in dict(state["target_d"]).items()}  # type: ignore[arg-type]
        self._step_count = int(state["step_count"])  # type: ignore[arg-type]
