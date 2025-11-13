"""
Fast optimization utilities using Nevergrad for better performance.
Drop-in replacement for optimization_utils.py with ~1.6x speedup.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Optional, Sequence, Tuple

import torch
from loss_utils import (
    TRADING_FEE,
    calculate_profit_torch_with_entry_buysell_profit_values,
    calculate_trading_profit_torch_with_entry_buysell,
)

logger = logging.getLogger(__name__)


def _parse_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


def _parse_steps(env_name: str, default: Sequence[int]) -> Tuple[int, ...]:
    raw = os.getenv(env_name)
    if not raw:
        return tuple(default)
    parts: list[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            value = int(chunk)
        except ValueError:
            continue
        if value >= 3:
            parts.append(value)
    return tuple(parts) if parts else tuple(default)


_USE_TORCH_GRID = _parse_flag("MARKETSIM_USE_TORCH_GRID", "1")
_ENTRY_EXIT_GRID_STEPS = _parse_steps("MARKETSIM_TORCH_GRID_STEPS", (33, 17, 9))
_ALWAYSON_GRID_STEPS = _parse_steps("MARKETSIM_TORCH_GRID_STEPS", (33, 17, 9))
_GRID_SHRINK = float(os.getenv("MARKETSIM_TORCH_GRID_SHRINK", "0.6"))
_GRID_SHRINK = min(max(_GRID_SHRINK, 0.05), 0.95)
_MIN_WINDOW = float(os.getenv("MARKETSIM_TORCH_GRID_MIN_WINDOW", "1e-4"))

try:
    import nevergrad as ng

    HAS_NEVERGRAD = True
except ImportError:  # pragma: no cover - optional dep absent on CI
    HAS_NEVERGRAD = False
    # Fallback to scipy
    from scipy.optimize import differential_evolution

if _USE_TORCH_GRID:
    if HAS_NEVERGRAD:
        ENTRY_EXIT_OPTIMIZER_BACKEND = "torch-grid+nevergrad"
    else:
        ENTRY_EXIT_OPTIMIZER_BACKEND = "torch-grid"
else:
    ENTRY_EXIT_OPTIMIZER_BACKEND = "nevergrad" if HAS_NEVERGRAD else "scipy-direct"


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
    Optimize using Nevergrad (1.6x faster) or scipy as fallback.
    """

    grid_result = None
    if _USE_TORCH_GRID:
        try:
            grid_result = _grid_search_entry_exit_multipliers(
                close_actual,
                positions,
                high_actual,
                high_pred,
                low_actual,
                low_pred,
                close_at_eod=close_at_eod,
                trading_fee=trading_fee,
                bounds=bounds,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Torch grid search failed, falling back to Nevergrad/scipy: %s", exc)
            grid_result = None
        else:
            if grid_result is not None:
                return grid_result

    if not HAS_NEVERGRAD:
        # Fallback to scipy implementation
        from src.optimization_utils import optimize_entry_exit_multipliers as _scipy_opt

        return _scipy_opt(
            close_actual,
            positions,
            high_actual,
            high_pred,
            low_actual,
            low_pred,
            close_at_eod=close_at_eod,
            trading_fee=trading_fee,
            bounds=bounds,
            maxiter=maxiter,
            popsize=popsize,
            atol=atol,
            seed=seed,
            workers=workers,
        )

    # Nevergrad implementation
    def objective(params):
        h_mult, l_mult = params
        profit_tensor = calculate_profit_torch_with_entry_buysell_profit_values(
            close_actual,
            high_actual,
            high_pred + float(h_mult),
            low_actual,
            low_pred + float(l_mult),
            positions,
            close_at_eod=close_at_eod,
            trading_fee=trading_fee,
        )
        return -float(profit_tensor.sum().item())  # minimize negative = maximize profit

    parametrization = ng.p.Array(shape=(2,), lower=bounds[0][0], upper=bounds[0][1])
    budget = maxiter * popsize
    optimizer = ng.optimizers.TwoPointsDE(parametrization=parametrization, budget=budget)

    if seed is not None:
        ng.optimizers.registry.RandomSearchMaker(sampler='RandomSearch').no_parallelization.seed(seed)

    for _ in range(budget):
        x = optimizer.ask()
        loss = objective(x.value)
        optimizer.tell(x, loss)

    recommendation = optimizer.provide_recommendation()
    best_params = recommendation.value
    best_loss = objective(best_params)

    return float(best_params[0]), float(best_params[1]), float(-best_loss)


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
    Optimize AlwaysOn strategy using Nevergrad or scipy fallback.
    """

    grid_result = None
    if _USE_TORCH_GRID:
        try:
            grid_result = _grid_search_always_on_multipliers(
                close_actual,
                buy_indicator,
                sell_indicator,
                high_actual,
                high_pred,
                low_actual,
                low_pred,
                close_at_eod=close_at_eod,
                trading_fee=trading_fee,
                is_crypto=is_crypto,
                bounds=bounds,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Torch grid search (always-on) failed, falling back: %s", exc)
            grid_result = None
        else:
            if grid_result is not None:
                return grid_result

    if not HAS_NEVERGRAD:
        from src.optimization_utils import optimize_always_on_multipliers as _scipy_opt

        return _scipy_opt(
            close_actual,
            buy_indicator,
            sell_indicator,
            high_actual,
            high_pred,
            low_actual,
            low_pred,
            close_at_eod=close_at_eod,
            trading_fee=trading_fee,
            is_crypto=is_crypto,
            bounds=bounds,
            maxiter=maxiter,
            popsize=popsize,
            atol=atol,
            seed=seed,
            workers=workers,
        )

    # Nevergrad implementation
    def objective(params):
        h_mult, l_mult = params
        buy_returns = calculate_profit_torch_with_entry_buysell_profit_values(
            close_actual, high_actual, high_pred + float(h_mult),
            low_actual, low_pred + float(l_mult), buy_indicator,
            close_at_eod=close_at_eod, trading_fee=trading_fee,
        )
        if is_crypto:
            return -float(buy_returns.sum().item())
        else:
            sell_returns = calculate_profit_torch_with_entry_buysell_profit_values(
                close_actual, high_actual, high_pred + float(h_mult),
                low_actual, low_pred + float(l_mult), sell_indicator,
                close_at_eod=close_at_eod, trading_fee=trading_fee,
            )
            return -float(buy_returns.sum().item() + sell_returns.sum().item())

    parametrization = ng.p.Array(shape=(2,), lower=bounds[0][0], upper=bounds[0][1])
    budget = maxiter * popsize
    optimizer = ng.optimizers.TwoPointsDE(parametrization=parametrization, budget=budget)

    if seed is not None:
        ng.optimizers.registry.RandomSearchMaker(sampler="RandomSearch").no_parallelization.seed(seed)

    for _ in range(budget):
        x = optimizer.ask()
        loss = objective(x.value)
        optimizer.tell(x, loss)

    recommendation = optimizer.provide_recommendation()
    best_params = recommendation.value
    best_loss = objective(best_params)

    return float(best_params[0]), float(best_params[1]), float(-best_loss)


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
    Generic optimizer with custom callback, using Nevergrad or scipy fallback.
    """

    if not HAS_NEVERGRAD:
        from src.optimization_utils import optimize_entry_exit_multipliers_with_callback as _scipy_opt
        return _scipy_opt(
            profit_calculator, bounds=bounds, maxiter=maxiter, popsize=popsize,
            atol=atol, seed=seed, workers=workers
        )

    def objective(params):
        return -profit_calculator(float(params[0]), float(params[1]))

    parametrization = ng.p.Array(shape=(2,), lower=bounds[0][0], upper=bounds[0][1])
    budget = maxiter * popsize
    optimizer = ng.optimizers.TwoPointsDE(parametrization=parametrization, budget=budget)

    if seed is not None:
        ng.optimizers.registry.RandomSearchMaker(sampler="RandomSearch").no_parallelization.seed(seed)

    for _ in range(budget):
        x = optimizer.ask()
        loss = objective(x.value)
        optimizer.tell(x, loss)

    recommendation = optimizer.provide_recommendation()
    best_params = recommendation.value
    best_loss = objective(best_params)

    return float(best_params[0]), float(best_params[1]), float(-best_loss)


def _grid_search_entry_exit_multipliers(
    close_actual: torch.Tensor,
    positions: torch.Tensor,
    high_actual: torch.Tensor,
    high_pred: torch.Tensor,
    low_actual: torch.Tensor,
    low_pred: torch.Tensor,
    *,
    close_at_eod: bool,
    trading_fee: Optional[float],
    bounds: Tuple[Tuple[float, float], Tuple[float, float]],
):
    if close_actual.numel() == 0:
        return 0.0, 0.0, 0.0

    device = close_actual.device
    dtype = close_actual.dtype

    def evaluator(high_offsets: torch.Tensor, low_offsets: torch.Tensor) -> torch.Tensor:
        return _entry_exit_profit_grid(
            close_actual,
            positions,
            high_actual,
            high_pred,
            low_actual,
            low_pred,
            high_offsets,
            low_offsets,
            close_at_eod=close_at_eod,
            trading_fee=trading_fee,
        )

    result = _multi_stage_grid_search(bounds, _ENTRY_EXIT_GRID_STEPS, evaluator, device=device, dtype=dtype)
    if result is None:
        return None

    best_h, best_l, _ = result
    final_profit = calculate_trading_profit_torch_with_entry_buysell(
        None,
        None,
        close_actual,
        positions,
        high_actual,
        high_pred + best_h,
        low_actual,
        low_pred + best_l,
        close_at_eod=close_at_eod,
        trading_fee=trading_fee,
    ).item()
    return best_h, best_l, float(final_profit)


def _grid_search_always_on_multipliers(
    close_actual: torch.Tensor,
    buy_indicator: torch.Tensor,
    sell_indicator: torch.Tensor,
    high_actual: torch.Tensor,
    high_pred: torch.Tensor,
    low_actual: torch.Tensor,
    low_pred: torch.Tensor,
    *,
    close_at_eod: bool,
    trading_fee: Optional[float],
    is_crypto: bool,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]],
):
    if close_actual.numel() == 0:
        return 0.0, 0.0, 0.0

    device = close_actual.device
    dtype = close_actual.dtype

    def evaluator(high_offsets: torch.Tensor, low_offsets: torch.Tensor) -> torch.Tensor:
        buy_profit = _entry_exit_profit_grid(
            close_actual,
            buy_indicator,
            high_actual,
            high_pred,
            low_actual,
            low_pred,
            high_offsets,
            low_offsets,
            close_at_eod=close_at_eod,
            trading_fee=trading_fee,
        )
        if is_crypto:
            return buy_profit
        sell_profit = _entry_exit_profit_grid(
            close_actual,
            sell_indicator,
            high_actual,
            high_pred,
            low_actual,
            low_pred,
            high_offsets,
            low_offsets,
            close_at_eod=close_at_eod,
            trading_fee=trading_fee,
        )
        return buy_profit + sell_profit

    result = _multi_stage_grid_search(bounds, _ALWAYSON_GRID_STEPS, evaluator, device=device, dtype=dtype)
    if result is None:
        return None

    best_h, best_l, _ = result

    buy_returns = calculate_profit_torch_with_entry_buysell_profit_values(
        close_actual,
        high_actual,
        high_pred + best_h,
        low_actual,
        low_pred + best_l,
        buy_indicator,
        close_at_eod=close_at_eod,
        trading_fee=trading_fee,
    )
    if is_crypto:
        sell_returns = torch.zeros_like(buy_returns)
    else:
        sell_returns = calculate_profit_torch_with_entry_buysell_profit_values(
            close_actual,
            high_actual,
            high_pred + best_h,
            low_actual,
            low_pred + best_l,
            sell_indicator,
            close_at_eod=close_at_eod,
            trading_fee=trading_fee,
        )

    total_profit = float((buy_returns + sell_returns).sum().item())
    return best_h, best_l, total_profit


def _multi_stage_grid_search(
    bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    steps: Tuple[int, ...],
    evaluator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[Tuple[float, float, float]]:
    if not steps:
        return None

    lo_h, hi_h = float(bounds[0][0]), float(bounds[0][1])
    lo_l, hi_l = float(bounds[1][0]), float(bounds[1][1])
    best: Optional[Tuple[float, float, float]] = None

    for stage_steps in steps:
        if stage_steps < 2 or hi_h - lo_h < _MIN_WINDOW * 0.5 or hi_l - lo_l < _MIN_WINDOW * 0.5:
            continue
        high_candidates = torch.linspace(lo_h, hi_h, stage_steps, device=device, dtype=dtype)
        low_candidates = torch.linspace(lo_l, hi_l, stage_steps, device=device, dtype=dtype)
        if high_candidates.numel() == 0 or low_candidates.numel() == 0:
            continue
        profits = evaluator(high_candidates, low_candidates)
        if profits.numel() == 0:
            break
        profits = torch.nan_to_num(profits, nan=-1e12, posinf=1e9, neginf=-1e12)
        flat_profits = profits.view(-1)
        max_idx = int(torch.argmax(flat_profits).item())
        stage_profit = float(flat_profits[max_idx].item())
        idx_h = max_idx // low_candidates.numel()
        idx_l = max_idx % low_candidates.numel()
        best_h = float(high_candidates[idx_h].item())
        best_l = float(low_candidates[idx_l].item())
        if best is None or stage_profit > best[2]:
            best = (best_h, best_l, stage_profit)

        half_window_h = max((hi_h - lo_h) * _GRID_SHRINK * 0.5, _MIN_WINDOW)
        half_window_l = max((hi_l - lo_l) * _GRID_SHRINK * 0.5, _MIN_WINDOW)
        lo_h = max(bounds[0][0], best_h - half_window_h)
        hi_h = min(bounds[0][1], best_h + half_window_h)
        lo_l = max(bounds[1][0], best_l - half_window_l)
        hi_l = min(bounds[1][1], best_l + half_window_l)

    return best


def _entry_exit_profit_grid(
    close_actual: torch.Tensor,
    positions: torch.Tensor,
    high_actual: torch.Tensor,
    high_pred: torch.Tensor,
    low_actual: torch.Tensor,
    low_pred: torch.Tensor,
    high_offsets: torch.Tensor,
    low_offsets: torch.Tensor,
    *,
    close_at_eod: bool,
    trading_fee: Optional[float],
) -> torch.Tensor:
    H = int(high_offsets.numel())
    L = int(low_offsets.numel())
    if H == 0 or L == 0:
        return torch.zeros(H, L, device=close_actual.device, dtype=close_actual.dtype)

    if trading_fee is None:
        trading_fee = TRADING_FEE

    with torch.no_grad():
        close = close_actual.view(1, 1, -1)
        high_act = high_actual.view(1, 1, -1)
        low_act = low_actual.view(1, 1, -1)
        pos = positions.to(close.dtype).view(1, 1, -1)

        base_high_pred = high_pred.view(1, 1, -1)
        base_low_pred = low_pred.view(1, 1, -1)

        high_pred_grid = base_high_pred + high_offsets.view(H, 1, 1)
        high_pred_grid = torch.clamp(high_pred_grid, 0.0, 10.0).expand(H, L, -1)

        low_pred_grid = base_low_pred + low_offsets.view(1, L, 1)
        low_pred_grid = torch.clamp(low_pred_grid, -1.0, 0.0).expand(H, L, -1)

        close = close.expand_as(high_pred_grid)
        high_act = high_act.expand_as(high_pred_grid)
        low_act = low_act.expand_as(high_pred_grid)
        pos = pos.expand_as(high_pred_grid)

        pred_low_to_close = close - low_pred_grid
        pred_high_to_close = close - high_pred_grid
        pred_low_to_high = high_pred_grid - low_pred_grid
        pred_high_to_low = low_pred_grid - high_pred_grid

        long_mask = (low_pred_grid > low_act).to(close.dtype)
        short_mask = (high_pred_grid < high_act).to(close.dtype)
        long_positions = torch.clamp(pos, 0.0, 10.0)
        short_positions = torch.clamp(pos, -10.0, 0.0)

        if close_at_eod:
            bought_profits = long_positions * pred_low_to_close * long_mask
            sold_profits = short_positions * pred_high_to_close * short_mask
            hit_points = ((long_mask > 0) | (short_mask > 0)).to(close.dtype)
            profit_values = bought_profits + sold_profits - (pos.abs() * trading_fee) * hit_points
        else:
            bought_profits = long_positions * pred_low_to_close * long_mask
            sold_profits = short_positions * pred_high_to_close * short_mask

            hit_high_points = (
                pred_low_to_high
                * (high_pred_grid <= high_act).to(close.dtype)
                * long_positions
                * long_mask
            )
            missed_high_points = bought_profits * (high_pred_grid > high_act).to(close.dtype)
            bought_adjusted = hit_high_points + missed_high_points

            hit_low_points = (
                pred_high_to_low
                * (low_pred_grid >= low_act).to(close.dtype)
                * short_positions
                * (high_pred_grid < high_act).to(close.dtype)
            )
            missed_low_points = sold_profits * (low_pred_grid < low_act).to(close.dtype)
            adjusted_profits = hit_low_points + missed_low_points

            hit_trading_points = ((high_pred_grid < high_act) & (low_pred_grid > low_act)).to(close.dtype)
            profit_values = (
                bought_adjusted
                + adjusted_profits
                - (pos.abs() * trading_fee) * hit_trading_points
            )

        return profit_values.sum(dim=-1)
