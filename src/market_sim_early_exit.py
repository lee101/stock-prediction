from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.metrics_utils import annualized_sortino, compute_step_returns


@dataclass(frozen=True)
class EarlyExitDecision:
    should_stop: bool
    progress_fraction: float
    total_return: float
    max_drawdown: float
    reason: str = ""
    sortino: float = 0.0


def _coerce_equity_values(equity_values: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(equity_values, dtype=np.float64)
    if values.ndim != 1:
        values = values.reshape(-1)
    return values


def _compute_return_drawdown_sortino(
    values: np.ndarray,
    *,
    periods_per_year: float,
) -> tuple[float, float, float]:
    initial_equity = float(values[0])
    if not np.isfinite(initial_equity) or initial_equity <= 0.0:
        return 0.0, 0.0, 0.0
    running_max = np.maximum.accumulate(values)
    drawdowns = np.where(running_max > 0.0, (running_max - values) / running_max, 0.0)
    max_drawdown = float(np.max(drawdowns)) if drawdowns.size else 0.0
    total_return = float((values[-1] - initial_equity) / initial_equity)
    step_returns = compute_step_returns(values)
    sortino = float(annualized_sortino(step_returns, periods_per_year=float(periods_per_year)))
    return total_return, max_drawdown, sortino


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

    values = _coerce_equity_values(equity_values)
    total_steps = int(total_steps)
    if values.size < 2 or total_steps < max(2, int(min_total_steps)):
        return EarlyExitDecision(False, 0.0, 0.0, 0.0)

    progress = float(values.size) / float(max(total_steps, 1))
    if progress < float(progress_fraction):
        return EarlyExitDecision(False, progress, 0.0, 0.0)

    initial_equity = float(values[0])
    if not np.isfinite(initial_equity) or initial_equity <= 0.0:
        return EarlyExitDecision(False, progress, 0.0, 0.0)

    total_return, max_drawdown, sortino = _compute_return_drawdown_sortino(values, periods_per_year=252.0)

    if max_drawdown <= total_return:
        return EarlyExitDecision(False, progress, total_return, max_drawdown, sortino=sortino)

    reason = (
        f"[{label}] early stopping at {progress:.1%}: "
        f"max drawdown {max_drawdown:.2%} exceeds profit {total_return:.2%}."
    )
    return EarlyExitDecision(True, progress, total_return, max_drawdown, reason, sortino=sortino)


def evaluate_baseline_comparability_early_exit(
    equity_values: Sequence[float] | np.ndarray,
    *,
    total_steps: int,
    label: str,
    periods_per_year: float,
    baseline_total_return: float | None = None,
    baseline_sortino: float | None = None,
    baseline_max_drawdown: float | None = None,
    min_total_steps: int = 40,
    stage1_progress: float = 0.30,
    stage2_progress: float = 0.50,
    stage3_progress: float = 0.75,
    return_tolerance: float = 0.02,
    sortino_tolerance: float = 0.50,
    max_drawdown_tolerance: float = 0.02,
) -> EarlyExitDecision:
    """Stop long evaluations that are clearly failing baseline comparability gates.

    The rule is intentionally conservative and staged:
    - Stage 1: drawdown is already materially worse while returns remain weak.
    - Stage 2: risk-adjusted quality is far behind baseline and drawdown is still worse.
    - Stage 3: both return and Sortino remain materially below baseline deep into the run.
    """

    values = _coerce_equity_values(equity_values)
    total_steps = int(total_steps)
    if values.size < 2 or total_steps < max(2, int(min_total_steps)):
        return EarlyExitDecision(False, 0.0, 0.0, 0.0)

    progress = float(values.size) / float(max(total_steps, 1))
    if progress < min(float(stage1_progress), float(stage2_progress), float(stage3_progress)):
        return EarlyExitDecision(False, progress, 0.0, 0.0)

    total_return, max_drawdown, sortino = _compute_return_drawdown_sortino(
        values,
        periods_per_year=float(periods_per_year),
    )

    has_baseline_return = baseline_total_return is not None and np.isfinite(float(baseline_total_return))
    has_baseline_sortino = baseline_sortino is not None and np.isfinite(float(baseline_sortino))
    has_baseline_drawdown = baseline_max_drawdown is not None and np.isfinite(float(baseline_max_drawdown))
    if not any((has_baseline_return, has_baseline_sortino, has_baseline_drawdown)):
        return EarlyExitDecision(False, progress, total_return, max_drawdown, sortino=sortino)

    baseline_return_value = float(baseline_total_return or 0.0)
    baseline_sortino_value = float(baseline_sortino or 0.0)
    baseline_drawdown_value = float(baseline_max_drawdown or 0.0)

    if progress >= float(stage1_progress) and has_baseline_drawdown:
        dd_limit = max(0.0, baseline_drawdown_value + float(max_drawdown_tolerance))
        return_floor = max(0.0, baseline_return_value - float(return_tolerance)) if has_baseline_return else 0.0
        if max_drawdown > dd_limit and total_return < return_floor:
            reason = (
                f"[{label}] early stopping at {progress:.1%}: "
                f"max drawdown {max_drawdown:.2%} exceeds baseline gate {dd_limit:.2%} "
                f"while return {total_return:.2%} remains below {return_floor:.2%}."
            )
            return EarlyExitDecision(True, progress, total_return, max_drawdown, reason, sortino=sortino)

    if progress >= float(stage2_progress) and has_baseline_sortino and has_baseline_drawdown:
        dd_limit = max(0.0, baseline_drawdown_value + float(max_drawdown_tolerance))
        sortino_floor = baseline_sortino_value - float(sortino_tolerance)
        if sortino < sortino_floor and max_drawdown > dd_limit:
            reason = (
                f"[{label}] early stopping at {progress:.1%}: "
                f"sortino {sortino:.3f} is below baseline gate {sortino_floor:.3f} "
                f"and max drawdown {max_drawdown:.2%} exceeds {dd_limit:.2%}."
            )
            return EarlyExitDecision(True, progress, total_return, max_drawdown, reason, sortino=sortino)

    if progress >= float(stage3_progress) and has_baseline_return and has_baseline_sortino:
        return_floor = baseline_return_value - float(return_tolerance)
        sortino_floor = baseline_sortino_value - float(sortino_tolerance)
        if total_return < return_floor and sortino < sortino_floor:
            reason = (
                f"[{label}] early stopping at {progress:.1%}: "
                f"return {total_return:.2%} is below baseline gate {return_floor:.2%} "
                f"and sortino {sortino:.3f} is below {sortino_floor:.3f}."
            )
            return EarlyExitDecision(True, progress, total_return, max_drawdown, reason, sortino=sortino)

    return EarlyExitDecision(False, progress, total_return, max_drawdown, sortino=sortino)


def print_early_exit(decision: EarlyExitDecision) -> None:
    if decision.should_stop and decision.reason:
        print(decision.reason, flush=True)


__all__ = [
    "EarlyExitDecision",
    "evaluate_baseline_comparability_early_exit",
    "evaluate_drawdown_vs_profit_early_exit",
    "print_early_exit",
]
