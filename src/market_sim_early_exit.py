from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class EarlyExitDecision:
    should_stop: bool
    progress_fraction: float
    total_return: float
    max_drawdown: float
    reason: str = ""


def evaluate_drawdown_vs_profit_early_exit(
    equity_values: Sequence[float] | np.ndarray,
    *,
    total_steps: int,
    label: str,
    min_total_steps: int = 20,
    progress_fraction: float = 0.5,
) -> EarlyExitDecision:
    """Stop long simulations once drawdown overtakes profit after halfway.

    This is intentionally conservative for short synthetic unit-test windows:
    the rule is ignored unless the full simulation has at least ``min_total_steps``
    observations.
    """

    values = np.asarray(equity_values, dtype=np.float64)
    total_steps = int(total_steps)
    if values.size < 2 or total_steps < max(2, int(min_total_steps)):
        return EarlyExitDecision(False, 0.0, 0.0, 0.0)

    progress = float(values.size) / float(max(total_steps, 1))
    if progress < float(progress_fraction):
        return EarlyExitDecision(False, progress, 0.0, 0.0)

    initial_equity = float(values[0])
    if not np.isfinite(initial_equity) or initial_equity <= 0.0:
        return EarlyExitDecision(False, progress, 0.0, 0.0)

    running_max = np.maximum.accumulate(values)
    drawdowns = np.where(running_max > 0.0, (running_max - values) / running_max, 0.0)
    max_drawdown = float(np.max(drawdowns)) if drawdowns.size else 0.0
    total_return = float((values[-1] - initial_equity) / initial_equity)

    if max_drawdown <= total_return:
        return EarlyExitDecision(False, progress, total_return, max_drawdown)

    reason = (
        f"[{label}] early stopping at {progress:.1%}: "
        f"max drawdown {max_drawdown:.2%} exceeds profit {total_return:.2%}."
    )
    return EarlyExitDecision(True, progress, total_return, max_drawdown, reason)


def print_early_exit(decision: EarlyExitDecision) -> None:
    if decision.should_stop and decision.reason:
        print(decision.reason, flush=True)


__all__ = [
    "EarlyExitDecision",
    "evaluate_drawdown_vs_profit_early_exit",
    "print_early_exit",
]
