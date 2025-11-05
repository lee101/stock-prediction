"""
Optimization utilities for trading strategy parameter tuning.

Uses scipy's differential_evolution for efficient global optimization
of entry/exit multipliers and other strategy hyperparameters.
"""

from typing import Callable, Optional, Tuple

import torch
from loss_utils import (
    calculate_profit_torch_with_entry_buysell_profit_values,
    calculate_trading_profit_torch_with_entry_buysell,
)
from scipy.optimize import differential_evolution


class _EntryExitObjective:
    """Picklable objective function for multiprocessing"""

    def __init__(self, close_actual, positions, high_actual, high_pred, low_actual, low_pred, close_at_eod, trading_fee):
        self.close_actual = close_actual
        self.positions = positions
        self.high_actual = high_actual
        self.high_pred = high_pred
        self.low_actual = low_actual
        self.low_pred = low_pred
        self.close_at_eod = close_at_eod
        self.trading_fee = trading_fee

    def __call__(self, multipliers):
        high_mult, low_mult = multipliers
        profit = calculate_trading_profit_torch_with_entry_buysell(
            None,
            None,
            self.close_actual,
            self.positions,
            self.high_actual,
            self.high_pred + float(high_mult),
            self.low_actual,
            self.low_pred + float(low_mult),
            close_at_eod=self.close_at_eod,
            trading_fee=self.trading_fee,
        ).item()
        return -profit


class _AlwaysOnObjective:
    """Picklable objective for AlwaysOn strategy with separate buy/sell"""

    def __init__(
        self,
        close_actual,
        buy_indicator,
        sell_indicator,
        high_actual,
        high_pred,
        low_actual,
        low_pred,
        close_at_eod,
        trading_fee,
        is_crypto,
    ):
        self.close_actual = close_actual
        self.buy_indicator = buy_indicator
        self.sell_indicator = sell_indicator
        self.high_actual = high_actual
        self.high_pred = high_pred
        self.low_actual = low_actual
        self.low_pred = low_pred
        self.close_at_eod = close_at_eod
        self.trading_fee = trading_fee
        self.is_crypto = is_crypto

    def __call__(self, multipliers):
        high_mult, low_mult = multipliers
        buy_returns = calculate_profit_torch_with_entry_buysell_profit_values(
            self.close_actual,
            self.high_actual,
            self.high_pred + float(high_mult),
            self.low_actual,
            self.low_pred + float(low_mult),
            self.buy_indicator,
            close_at_eod=self.close_at_eod,
            trading_fee=self.trading_fee,
        )
        if self.is_crypto:
            return -float(buy_returns.sum().item())
        else:
            sell_returns = calculate_profit_torch_with_entry_buysell_profit_values(
                self.close_actual,
                self.high_actual,
                self.high_pred + float(high_mult),
                self.low_actual,
                self.low_pred + float(low_mult),
                self.sell_indicator,
                close_at_eod=self.close_at_eod,
                trading_fee=self.trading_fee,
            )
            return -float(buy_returns.sum().item() + sell_returns.sum().item())


def optimize_entry_exit_multipliers(
    close_actual: torch.Tensor,
    positions: torch.Tensor,
    high_actual: torch.Tensor,
    high_pred: torch.Tensor,
    low_actual: torch.Tensor,
    low_pred: torch.Tensor,
    *,
    close_at_eod: bool = False,
    trading_fee: Optional[float] = None,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((-0.03, 0.03), (-0.03, 0.03)),
    maxiter: int = 50,
    popsize: int = 10,
    atol: float = 1e-5,
    seed: Optional[int] = 42,
    workers: int = 1,
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
        close_at_eod: If True, force positions to close at end-of-day close price
        bounds: Search bounds for (high_multiplier, low_multiplier)
        maxiter: Max iterations for differential evolution
        popsize: Population size per iteration
        atol: Absolute tolerance for convergence
        seed: Random seed for reproducibility
        workers: Number of parallel workers (-1 = all CPUs, 1 = sequential)

    Returns:
        (best_high_multiplier, best_low_multiplier, best_profit)
    """

    objective = _EntryExitObjective(close_actual, positions, high_actual, high_pred, low_actual, low_pred, close_at_eod, trading_fee)

    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=maxiter,
        popsize=popsize,
        atol=atol,
        seed=seed,
        workers=workers,
        updating="deferred" if workers != 1 else "immediate",
    )

    return float(result.x[0]), float(result.x[1]), float(-result.fun)


def optimize_always_on_multipliers(
    close_actual: torch.Tensor,
    buy_indicator: torch.Tensor,
    sell_indicator: torch.Tensor,
    high_actual: torch.Tensor,
    high_pred: torch.Tensor,
    low_actual: torch.Tensor,
    low_pred: torch.Tensor,
    *,
    close_at_eod: bool = False,
    trading_fee: Optional[float] = None,
    is_crypto: bool = False,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((-0.03, 0.03), (-0.03, 0.03)),
    maxiter: int = 30,
    popsize: int = 8,
    atol: float = 1e-5,
    seed: Optional[int] = 42,
    workers: int = -1,
) -> Tuple[float, float, float]:
    """
    Optimize AlwaysOn strategy with separate buy/sell indicators (parallelizable).

    Args:
        close_actual, high_actual, low_actual: Market data
        buy_indicator, sell_indicator: Position indicators
        high_pred, low_pred: Predicted targets
        close_at_eod: Close positions at end of day
        trading_fee: Trading fee per trade
        is_crypto: True for crypto (buy only), False for stocks (buy+sell)
        bounds, maxiter, popsize, atol, seed: Optimizer params
        workers: Parallel workers (-1 = all CPUs)

    Returns:
        (best_high_mult, best_low_mult, best_profit)
    """
    objective = _AlwaysOnObjective(
        close_actual,
        buy_indicator,
        sell_indicator,
        high_actual,
        high_pred,
        low_actual,
        low_pred,
        close_at_eod,
        trading_fee,
        is_crypto,
    )

    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=maxiter,
        popsize=popsize,
        atol=atol,
        seed=seed,
        workers=workers,
        updating="deferred" if workers != 1 else "immediate",
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
    workers: int = 1,
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
        workers: Number of parallel workers (-1 = all CPUs, 1 = sequential)

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
        workers=workers,
        updating="deferred" if workers != 1 else "immediate",
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
    workers: int = 1,
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
        workers: Number of parallel workers (-1 = all CPUs, 1 = sequential)

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
        workers=workers,
        updating="deferred" if workers != 1 else "immediate",
    )

    return float(result.x[0]), float(-result.fun)
