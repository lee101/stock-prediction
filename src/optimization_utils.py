"""
Optimization utilities for trading strategy parameter tuning.

Uses scipy's differential_evolution for efficient global optimization
of entry/exit multipliers and other strategy hyperparameters.
"""

from typing import Callable, Optional, Tuple

import torch
from loss_utils import calculate_trading_profit_torch_with_entry_buysell
from scipy.optimize import differential_evolution


def optimize_entry_exit_multipliers(
    close_actual: torch.Tensor,
    positions: torch.Tensor,
    high_actual: torch.Tensor,
    high_pred: torch.Tensor,
    low_actual: torch.Tensor,
    low_pred: torch.Tensor,
    *,
    trading_fee: Optional[float] = None,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((-0.03, 0.03), (-0.03, 0.03)),
    maxiter: int = 50,
    popsize: int = 10,
    atol: float = 1e-5,
    seed: Optional[int] = 42,
) -> Tuple[float, float, float]:
    """
    Optimize high/low multipliers for entry/exit targets using differential evolution.

    Args:
        close_actual: Actual close price movements/returns
        positions: Position sizes (+1 for long, -1 for short, 0 for neutral)
        high_actual: Actual high price movements
        high_pred: Predicted high exit targets
        low_actual: Actual low price movements
        low_pred: Predicted low entry targets
        bounds: Search bounds for (high_multiplier, low_multiplier)
        maxiter: Max iterations for differential evolution
        popsize: Population size per iteration
        atol: Absolute tolerance for convergence
        seed: Random seed for reproducibility

    Returns:
        (best_high_multiplier, best_low_multiplier, best_profit)
    """

    def objective(multipliers):
        high_mult, low_mult = multipliers
        profit = calculate_trading_profit_torch_with_entry_buysell(
            None,
            None,
            close_actual,
            positions,
            high_actual,
            high_pred + float(high_mult),
            low_actual,
            low_pred + float(low_mult),
            trading_fee=trading_fee,
        ).item()
        return -profit  # minimize negative = maximize profit

    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=maxiter,
        popsize=popsize,
        atol=atol,
        seed=seed,
        workers=1,
    )

    return float(result.x[0]), float(result.x[1]), float(-result.fun)


def optimize_entry_exit_multipliers_with_callback(
    profit_calculator: Callable[[float, float], float],
    *,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((-0.03, 0.03), (-0.03, 0.03)),
    maxiter: int = 50,
    popsize: int = 10,
    atol: float = 1e-5,
    seed: Optional[int] = 42,
) -> Tuple[float, float, float]:
    """
    Generic optimizer that accepts a custom profit calculator callback.

    Args:
        profit_calculator: Function (high_mult, low_mult) -> profit
        bounds: Search bounds for (high_multiplier, low_multiplier)
        maxiter: Max iterations
        popsize: Population size
        atol: Convergence tolerance
        seed: Random seed

    Returns:
        (best_high_multiplier, best_low_multiplier, best_profit)
    """

    def objective(multipliers):
        high_mult, low_mult = multipliers
        profit = profit_calculator(float(high_mult), float(low_mult))
        return -profit

    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=maxiter,
        popsize=popsize,
        atol=atol,
        seed=seed,
        workers=1,
    )

    return float(result.x[0]), float(result.x[1]), float(-result.fun)


def optimize_single_parameter(
    profit_calculator: Callable[[float], float],
    *,
    bounds: Tuple[float, float] = (-0.05, 0.05),
    maxiter: int = 30,
    popsize: int = 8,
    atol: float = 1e-5,
    seed: Optional[int] = 42,
) -> Tuple[float, float]:
    """
    Optimize a single scalar parameter.

    Args:
        profit_calculator: Function (param) -> profit
        bounds: Search bounds (min, max)
        maxiter: Max iterations
        popsize: Population size
        atol: Convergence tolerance
        seed: Random seed

    Returns:
        (best_parameter, best_profit)
    """

    def objective(params):
        param = params[0]
        profit = profit_calculator(float(param))
        return -profit

    result = differential_evolution(
        objective,
        bounds=[bounds],
        maxiter=maxiter,
        popsize=popsize,
        atol=atol,
        seed=seed,
        workers=1,
    )

    return float(result.x[0]), float(-result.fun)
