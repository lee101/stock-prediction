from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import math

from .planner import WidePlannerConfig, build_wide_plan
from .types import DaySimulationResult, FillResult, WideCandidate, WideOrder


def _monthly_from_total(total_return: float, window_days: int, trading_days_per_month: float = 21.0) -> float:
    if window_days <= 0:
        return 0.0
    try:
        return math.expm1(math.log1p(float(total_return)) * (trading_days_per_month / float(window_days)))
    except Exception:
        return 0.0


@dataclass(frozen=True)
class ReplaySummary:
    start_equity: float
    end_equity: float
    total_pnl: float
    total_return: float
    monthly_return: float
    max_drawdown: float
    trade_count: int
    filled_count: int
    day_results: tuple[DaySimulationResult, ...]


def _work_steal_order_key(order: WideOrder) -> tuple[float, float, float]:
    candidate = order.candidate
    return (candidate.entry_gap_pct, -candidate.forecasted_pnl, -candidate.score)


def simulate_day(
    candidates: Sequence[WideCandidate],
    *,
    account_equity: float,
    config: WidePlannerConfig | None = None,
    day_index: int = 0,
) -> DaySimulationResult:
    planner = config or WidePlannerConfig()
    orders = build_wide_plan(candidates, account_equity=account_equity, config=planner)
    max_gross_notional = account_equity * planner.max_leverage
    fee_rate = planner.fee_bps / 10_000.0
    fill_buffer = planner.fill_buffer_bps / 10_000.0

    gross_notional_consumed = 0.0
    realized_pnl = 0.0
    fills: list[FillResult] = []

    for order in sorted(orders, key=_work_steal_order_key):
        candidate = order.candidate
        remaining_capacity = max_gross_notional - gross_notional_consumed
        notional = min(order.reserved_notional, max(remaining_capacity, 0.0))
        if notional <= 0.0:
            fills.append(
                FillResult(
                    order=order,
                    filled=False,
                    notional=0.0,
                    entry_price=None,
                    exit_price=None,
                    pnl=0.0,
                    return_pct=0.0,
                    hit_take_profit=False,
                    work_steal_priority=candidate.entry_gap_pct,
                )
            )
            continue
        if candidate.realized_low > candidate.entry_price:
            fills.append(
                FillResult(
                    order=order,
                    filled=False,
                    notional=0.0,
                    entry_price=None,
                    exit_price=None,
                    pnl=0.0,
                    return_pct=0.0,
                    hit_take_profit=False,
                    work_steal_priority=candidate.entry_gap_pct,
                )
            )
            continue

        gross_notional_consumed += notional
        entry_price = candidate.entry_price * (1.0 + fill_buffer)
        hit_take_profit = candidate.realized_high >= candidate.take_profit_price
        exit_price = candidate.take_profit_price if hit_take_profit else candidate.realized_close
        return_pct = ((exit_price - entry_price) / entry_price) - (2.0 * fee_rate)
        pnl = notional * return_pct
        realized_pnl += pnl
        fills.append(
            FillResult(
                order=order,
                filled=True,
                notional=notional,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=pnl,
                return_pct=return_pct,
                hit_take_profit=hit_take_profit,
                work_steal_priority=candidate.entry_gap_pct,
            )
        )

    end_equity = account_equity + realized_pnl
    top_symbols = tuple(order.candidate.symbol for order in orders)
    return DaySimulationResult(
        day_index=day_index,
        start_equity=account_equity,
        end_equity=end_equity,
        realized_pnl=realized_pnl,
        max_gross_notional=max_gross_notional,
        top_symbols=top_symbols,
        fills=tuple(fills),
    )


def simulate_wide_strategy(
    candidate_days: Sequence[Sequence[WideCandidate]],
    *,
    starting_equity: float,
    config: WidePlannerConfig | None = None,
) -> ReplaySummary:
    planner = config or WidePlannerConfig()
    equity = starting_equity
    peak_equity = starting_equity
    max_drawdown = 0.0
    day_results: list[DaySimulationResult] = []
    trade_count = 0
    filled_count = 0

    for day_index, day_candidates in enumerate(candidate_days):
        result = simulate_day(day_candidates, account_equity=equity, config=planner, day_index=day_index)
        day_results.append(result)
        equity = result.end_equity
        peak_equity = max(peak_equity, equity)
        if peak_equity > 0:
            max_drawdown = min(max_drawdown, (equity / peak_equity) - 1.0)
        trade_count += len(result.fills)
        filled_count += sum(1 for fill in result.fills if fill.filled)

    total_pnl = equity - starting_equity
    total_return = (equity / starting_equity) - 1.0 if starting_equity > 0 else 0.0
    monthly_return = _monthly_from_total(total_return, len(candidate_days))
    return ReplaySummary(
        start_equity=starting_equity,
        end_equity=equity,
        total_pnl=total_pnl,
        total_return=total_return,
        monthly_return=monthly_return,
        max_drawdown=max_drawdown,
        trade_count=trade_count,
        filled_count=filled_count,
        day_results=tuple(day_results),
    )
