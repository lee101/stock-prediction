"""Reusable helpers for MaxDiff multiplier optimization."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, Sequence, Tuple

import torch

from loss_utils import calculate_profit_torch_with_entry_buysell_profit_values

try:  # Prefer fast optimizer when available
    from src import optimization_utils_fast as _opt_module  # type: ignore
except ImportError:  # pragma: no cover - fallback path
    from src import optimization_utils as _opt_module  # type: ignore

if hasattr(_opt_module, "ENTRY_EXIT_OPTIMIZER_BACKEND"):
    ENTRY_EXIT_OPTIMIZER_BACKEND = getattr(_opt_module, "ENTRY_EXIT_OPTIMIZER_BACKEND")
else:
    if getattr(_opt_module, "__name__", "").endswith("optimization_utils_fast"):
        ENTRY_EXIT_OPTIMIZER_BACKEND = "torch-grid+nevergrad"
    else:
        ENTRY_EXIT_OPTIMIZER_BACKEND = "scipy-direct"

optimize_entry_exit_multipliers = _opt_module.optimize_entry_exit_multipliers
optimize_always_on_multipliers = _opt_module.optimize_always_on_multipliers


@dataclass
class EntryExitOptimizationResult:
    base_profit: torch.Tensor
    final_profit: torch.Tensor
    best_high_multiplier: float
    best_low_multiplier: float
    best_close_at_eod: bool
    timings: Dict[str, float] = field(default_factory=dict)


@dataclass
class AlwaysOnOptimizationResult:
    buy_returns: torch.Tensor
    sell_returns: torch.Tensor
    best_high_multiplier: float
    best_low_multiplier: float
    best_close_at_eod: bool
    timings: Dict[str, float] = field(default_factory=dict)


def optimize_maxdiff_entry_exit(
    close_actual: torch.Tensor,
    maxdiff_trades: torch.Tensor,
    high_actual: torch.Tensor,
    high_pred: torch.Tensor,
    low_actual: torch.Tensor,
    low_pred: torch.Tensor,
    *,
    close_at_eod_candidates: Sequence[bool],
    trading_fee: float,
    optim_kwargs: Dict[str, object] | None = None,
    log_timings: bool = False,
) -> EntryExitOptimizationResult:
    """Optimize MaxDiff multipliers over the provided candidates."""

    optim_kwargs = dict(optim_kwargs or {})
    timings: Dict[str, float] = {}
    best_total_profit = float("-inf")
    best_close_at_eod = False
    best_high_multiplier = 0.0
    best_low_multiplier = 0.0
    best_base_profit: torch.Tensor | None = None
    best_final_profit: torch.Tensor | None = None

    for candidate in close_at_eod_candidates:
        stage_start = time.perf_counter()
        base_profit = calculate_profit_torch_with_entry_buysell_profit_values(
            close_actual,
            high_actual,
            high_pred,
            low_actual,
            low_pred,
            maxdiff_trades,
            close_at_eod=candidate,
            trading_fee=trading_fee,
        )

        high_mult, low_mult, _ = optimize_entry_exit_multipliers(
            close_actual,
            maxdiff_trades,
            high_actual,
            high_pred,
            low_actual,
            low_pred,
            close_at_eod=candidate,
            trading_fee=trading_fee,
            **optim_kwargs,
        )

        final_profit = calculate_profit_torch_with_entry_buysell_profit_values(
            close_actual,
            high_actual,
            high_pred + high_mult,
            low_actual,
            low_pred + low_mult,
            maxdiff_trades,
            close_at_eod=candidate,
            trading_fee=trading_fee,
        )
        total_profit = float(final_profit.sum().item())

        if total_profit > best_total_profit:
            best_total_profit = total_profit
            best_close_at_eod = candidate
            best_high_multiplier = high_mult
            best_low_multiplier = low_mult
            best_base_profit = base_profit.detach().clone()
            best_final_profit = final_profit.detach().clone()

        if log_timings:
            timings.setdefault(f"opt_close_{candidate}", 0.0)
            timings[f"opt_close_{candidate}"] += time.perf_counter() - stage_start

    if best_base_profit is None or best_final_profit is None:
        zeros = torch.zeros_like(close_actual)
        return EntryExitOptimizationResult(zeros, zeros, 0.0, 0.0, False, timings)

    return EntryExitOptimizationResult(
        best_base_profit,
        best_final_profit,
        best_high_multiplier,
        best_low_multiplier,
        best_close_at_eod,
        timings,
    )


def optimize_maxdiff_always_on(
    close_actual: torch.Tensor,
    buy_indicator: torch.Tensor,
    sell_indicator: torch.Tensor,
    high_actual: torch.Tensor,
    high_pred: torch.Tensor,
    low_actual: torch.Tensor,
    low_pred: torch.Tensor,
    *,
    is_crypto: bool,
    close_at_eod_candidates: Sequence[bool],
    trading_fee: float,
    optim_kwargs: Dict[str, object] | None = None,
    log_timings: bool = False,
) -> AlwaysOnOptimizationResult:
    """Optimize AlwaysOn multipliers, returning buy/sell return tensors."""

    optim_kwargs = dict(optim_kwargs or {})
    timings: Dict[str, float] = {}
    best_total_profit = float("-inf")
    best_close_at_eod = False
    best_high_multiplier = 0.0
    best_low_multiplier = 0.0
    best_buy_returns: torch.Tensor | None = None
    best_sell_returns: torch.Tensor | None = None

    for candidate in close_at_eod_candidates:
        stage_start = time.perf_counter()
        high_mult, low_mult, _ = optimize_always_on_multipliers(
            close_actual,
            buy_indicator,
            sell_indicator,
            high_actual,
            high_pred,
            low_actual,
            low_pred,
            close_at_eod=candidate,
            trading_fee=trading_fee,
            is_crypto=is_crypto,
            **optim_kwargs,
        )

        buy_returns = calculate_profit_torch_with_entry_buysell_profit_values(
            close_actual,
            high_actual,
            high_pred + high_mult,
            low_actual,
            low_pred + low_mult,
            buy_indicator,
            close_at_eod=candidate,
            trading_fee=trading_fee,
        )
        if is_crypto:
            sell_returns = torch.zeros_like(buy_returns)
        else:
            sell_returns = calculate_profit_torch_with_entry_buysell_profit_values(
                close_actual,
                high_actual,
                high_pred + high_mult,
                low_actual,
                low_pred + low_mult,
                sell_indicator,
                close_at_eod=candidate,
                trading_fee=trading_fee,
            )

        total_profit = float((buy_returns + sell_returns).sum().item())
        if total_profit > best_total_profit:
            best_total_profit = total_profit
            best_close_at_eod = candidate
            best_high_multiplier = high_mult
            best_low_multiplier = low_mult
            best_buy_returns = buy_returns.detach().clone()
            best_sell_returns = sell_returns.detach().clone()

        if log_timings:
            timings.setdefault(f"opt_close_{candidate}", 0.0)
            timings[f"opt_close_{candidate}"] += time.perf_counter() - stage_start

    if best_buy_returns is None or best_sell_returns is None:
        zeros = torch.zeros_like(close_actual)
        return AlwaysOnOptimizationResult(zeros, zeros, 0.0, 0.0, False, timings)

    return AlwaysOnOptimizationResult(
        best_buy_returns,
        best_sell_returns,
        best_high_multiplier,
        best_low_multiplier,
        best_close_at_eod,
        timings,
    )
