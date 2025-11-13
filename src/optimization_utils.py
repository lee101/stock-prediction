"""
Optimization utilities for trading strategy parameter tuning.

Uses scipy.optimize.direct (Dividing Rectangles) by default for 1.5x speedup.
Falls back to differential_evolution if direct fails or is disabled.

Environment Variables:
    MARKETSIM_USE_DIRECT_OPTIMIZER: Use DIRECT optimizer (default: 1)
    MARKETSIM_FAST_OPTIMIZE: Fast optimize mode for rapid iteration (default: 0)
        - Fast mode (1): maxfun=100, ~6x speedup, ~28% quality loss
        - Normal mode (0): maxfun=500, best balance (default)
    MARKETSIM_FAST_SIMULATE: Fast simulate mode for backtesting (default: 0)
        - Reduces num_simulations to 35 for 2x speedup

Examples:
    # Fast optimize + fast simulate for development (12x speedup combined)
    export MARKETSIM_FAST_OPTIMIZE=1
    export MARKETSIM_FAST_SIMULATE=1

    # Production mode (default)
    export MARKETSIM_FAST_OPTIMIZE=0
    export MARKETSIM_FAST_SIMULATE=0
"""

from typing import Callable, Optional, Sequence, Tuple
import logging
import os
from dataclasses import dataclass

import torch
from loss_utils import (
    TRADING_FEE,
    calculate_profit_torch_with_entry_buysell_profit_values,
    calculate_trading_profit_torch_with_entry_buysell,
)
from scipy.optimize import direct, differential_evolution

# Use faster 'direct' optimizer by default (1.5x faster, better results)
# Set MARKETSIM_USE_DIRECT_OPTIMIZER=0 to use differential_evolution
_USE_DIRECT = os.getenv("MARKETSIM_USE_DIRECT_OPTIMIZER", "1") in {"1", "true", "yes", "on"}

# Fast optimize mode: 6x speedup with ~28% quality loss (good for development/testing)
# Set MARKETSIM_FAST_OPTIMIZE=1 for rapid iteration (maxfun=100 instead of 500)
# Similar to MARKETSIM_FAST_SIMULATE but for the optimizer itself
_FAST_MODE = os.getenv("MARKETSIM_FAST_OPTIMIZE", "0") in {"1", "true", "yes", "on"}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _EntryExitOptimizationContext:
    close_actual: torch.Tensor
    high_actual: torch.Tensor
    low_actual: torch.Tensor
    long_multiplier: torch.Tensor
    short_multiplier: torch.Tensor
    abs_positions: torch.Tensor
    raw_high_pred: torch.Tensor
    raw_low_pred: torch.Tensor


def _prepare_entry_exit_context(
    close_actual: torch.Tensor,
    positions: torch.Tensor,
    high_actual: torch.Tensor,
    high_pred: torch.Tensor,
    low_actual: torch.Tensor,
    low_pred: torch.Tensor,
) -> _EntryExitOptimizationContext:
    close_actual = close_actual.view(-1)
    positions = positions.view(-1)
    high_actual = high_actual.view(-1)
    high_pred = high_pred.view(-1)
    low_actual = low_actual.view(-1)
    low_pred = low_pred.view(-1)

    return _EntryExitOptimizationContext(
        close_actual=close_actual,
        high_actual=high_actual,
        low_actual=low_actual,
        long_multiplier=torch.clamp(positions, 0, 10),
        short_multiplier=torch.clamp(positions, -10, 0),
        abs_positions=torch.abs(positions),
        raw_high_pred=high_pred,
        raw_low_pred=low_pred,
    )


def _evaluate_entry_exit_profit(
    ctx: _EntryExitOptimizationContext,
    *,
    high_mult: float,
    low_mult: float,
    close_at_eod: bool,
    trading_fee: Optional[float],
) -> torch.Tensor:
    fee = float(trading_fee if trading_fee is not None else TRADING_FEE)

    high_pred = torch.clamp(ctx.raw_high_pred + high_mult, 0.0, 10.0)
    low_pred = torch.clamp(ctx.raw_low_pred + low_mult, -1.0, 0.0)

    long_entry_mask_bool = low_pred > ctx.low_actual
    short_entry_mask_bool = high_pred < ctx.high_actual
    reached_high_mask_bool = high_pred <= ctx.high_actual
    reached_low_mask_bool = low_pred >= ctx.low_actual

    dtype = ctx.close_actual.dtype
    long_entry = long_entry_mask_bool.to(dtype)
    short_entry = short_entry_mask_bool.to(dtype)
    reached_high = reached_high_mask_bool.to(dtype)
    reached_low = reached_low_mask_bool.to(dtype)

    low_to_close = ctx.close_actual - low_pred
    high_to_close = ctx.close_actual - high_pred

    if close_at_eod:
        bought_profits = ctx.long_multiplier * low_to_close * long_entry
        sold_profits = ctx.short_multiplier * high_to_close * short_entry
        long_fee = torch.abs(ctx.long_multiplier) * fee * long_entry
        short_fee = torch.abs(ctx.short_multiplier) * fee * short_entry
        fee_cost = long_fee + short_fee
        return bought_profits + sold_profits - fee_cost

    bought_profits = ctx.long_multiplier * low_to_close * long_entry
    sold_profits = ctx.short_multiplier * high_to_close * short_entry

    low_to_high = high_pred - low_pred
    hit_high_points = low_to_high * reached_high * ctx.long_multiplier * long_entry
    missed_high_points = bought_profits * (1.0 - reached_high)
    bought_adjusted = hit_high_points + missed_high_points

    high_to_low = low_pred - high_pred
    hit_low_points = high_to_low * reached_low * ctx.short_multiplier * short_entry
    missed_low_points = sold_profits * (1.0 - reached_low)
    adjusted_profits = hit_low_points + missed_low_points

    long_fee = torch.abs(ctx.long_multiplier) * fee * long_entry
    short_fee = torch.abs(ctx.short_multiplier) * fee * short_entry
    fee_cost = long_fee + short_fee

    return bought_adjusted + adjusted_profits - fee_cost


class _EntryExitObjective:
    """Picklable objective function for multiprocessing"""

    def __init__(self, close_actual, positions, high_actual, high_pred, low_actual, low_pred, close_at_eod, trading_fee):
        self.context = _prepare_entry_exit_context(
            close_actual,
            positions,
            high_actual,
            high_pred,
            low_actual,
            low_pred,
        )
        self.close_at_eod = close_at_eod
        self.trading_fee = trading_fee

    def __call__(self, multipliers):
        high_mult, low_mult = multipliers
        profit_tensor = _evaluate_entry_exit_profit(
            self.context,
            high_mult=float(high_mult),
            low_mult=float(low_mult),
            close_at_eod=self.close_at_eod,
            trading_fee=self.trading_fee,
        )
        return -float(profit_tensor.sum().item())


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
    Optimize high/low multipliers for entry/exit targets.

    Uses scipy.optimize.direct by default (1.5x faster), falls back to differential_evolution.

    Args:
        close_actual: Actual close price movements/returns
        positions: Position sizes (+1 for long, -1 for short, 0 for neutral)
        high_actual: Actual high price movements
        high_pred: Predicted high exit targets
        low_actual: Actual low price movements
        low_pred: Predicted low entry targets
        close_at_eod: If True, force positions to close at end-of-day close price
        bounds: Search bounds for (high_multiplier, low_multiplier)
        maxiter: Max iterations (DE only)
        popsize: Population size (DE only)
        atol: Absolute tolerance for convergence (DE only)
        seed: Random seed for reproducibility
        workers: Number of parallel workers (DE only, -1 = all CPUs, 1 = sequential)

    Returns:
        (best_high_multiplier, best_low_multiplier, best_profit)
    """

    objective = _EntryExitObjective(close_actual, positions, high_actual, high_pred, low_actual, low_pred, close_at_eod, trading_fee)

    if _USE_DIRECT:
        try:
            # DIRECT is 1.5x faster and finds better solutions
            # Fast mode: 100 evals (6x faster), Normal mode: 500 evals (default)
            maxfun = 100 if _FAST_MODE else (maxiter * popsize)
            result = direct(
                objective,
                bounds=bounds,
                maxfun=maxfun,
            )
            return float(result.x[0]), float(result.x[1]), float(-result.fun)
        except Exception as e:
            # Fallback to DE if direct fails
            import logging
            logging.getLogger(__name__).debug(f"DIRECT optimizer failed for entry_exit, falling back to DE: {e}")

    # Fallback or explicit DE mode
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
    Optimize AlwaysOn strategy with separate buy/sell indicators.

    Uses scipy.optimize.direct by default (1.5x faster), falls back to differential_evolution.

    Args:
        close_actual, high_actual, low_actual: Market data
        buy_indicator, sell_indicator: Position indicators
        high_pred, low_pred: Predicted targets
        close_at_eod: Close positions at end of day
        trading_fee: Trading fee per trade
        is_crypto: True for crypto (buy only), False for stocks (buy+sell)
        bounds, maxiter, popsize, atol, seed: Optimizer params
        workers: Parallel workers (DE only, -1 = all CPUs)

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

    if _USE_DIRECT:
        try:
            # DIRECT is 1.5x faster and finds better solutions
            # Fast mode: 100 evals (6x faster), Normal mode: 240 evals (default)
            maxfun = 100 if _FAST_MODE else (maxiter * popsize)
            result = direct(
                objective,
                bounds=bounds,
                maxfun=maxfun,
            )
            return float(result.x[0]), float(result.x[1]), float(-result.fun)
        except Exception as e:
            # Fallback to DE if direct fails
            import logging
            logging.getLogger(__name__).debug(f"DIRECT optimizer failed for always_on, falling back to DE: {e}")

    # Fallback or explicit DE mode
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


def run_bounded_optimizer(
    objective: Callable[[Sequence[float]], float],
    bounds: Sequence[Tuple[float, float]],
    *,
    maxiter: int = 50,
    popsize: int = 10,
    atol: float = 1e-5,
    seed: Optional[int] = 42,
    workers: int = 1,
):
    """Minimize a bounded objective using DIRECT (preferred) or differential evolution."""

    if not bounds:
        raise ValueError("bounds must be non-empty")

    maxfun = 100 if _FAST_MODE else maxiter * max(popsize, 1)
    if _USE_DIRECT:
        try:
            result = direct(
                objective,
                bounds=bounds,
                maxfun=maxfun,
            )
            best_params = tuple(float(x) for x in result.x)
            return best_params, float(result.fun)
        except Exception as exc:  # pragma: no cover - fallback path
            logger.debug("DIRECT optimizer failed, falling back to differential evolution: %s", exc)

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
    best_params = tuple(float(x) for x in result.x)
    return best_params, float(result.fun)
