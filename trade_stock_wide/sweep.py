from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import pandas as pd

from .intraday import simulate_intraday_day
from .planner import WidePlannerConfig
from .replay import simulate_wide_strategy
from .selection import WideSelectionConfig, rerank_candidate_days
from .types import WideCandidate


@dataclass(frozen=True)
class WideSweepSpec:
    selection_objective: str
    top_k: int
    watch_activation_pct: float
    steal_protection_pct: float


def _simulate_candidate_days(
    candidate_days: Sequence[Sequence[WideCandidate]],
    *,
    starting_equity: float,
    planner: WidePlannerConfig,
    daily_only: bool,
    hourly_by_symbol: dict[str, pd.DataFrame] | None,
) -> dict[str, float | int | str]:
    if daily_only:
        summary = simulate_wide_strategy(candidate_days, starting_equity=starting_equity, config=planner)
        return {
            "total_return": float(summary.total_return),
            "monthly_return": float(summary.monthly_return),
            "max_drawdown": float(summary.max_drawdown),
            "filled_count": int(summary.filled_count),
            "trade_count": int(summary.trade_count),
        }

    hourly = hourly_by_symbol or {}
    equity = float(starting_equity)
    peak = equity
    max_drawdown = 0.0
    filled_count = 0
    trade_count = 0
    for day_index, day in enumerate(candidate_days):
        result = simulate_intraday_day(
            day,
            account_equity=equity,
            hourly_by_symbol=hourly,
            config=planner,
            day_index=day_index,
            logger=None,
        )
        equity = float(result.end_equity)
        peak = max(peak, equity)
        if peak > 0.0:
            max_drawdown = min(max_drawdown, (equity / peak) - 1.0)
        trade_count += len(result.fills)
        filled_count += sum(1 for fill in result.fills if fill.filled)

    monthly_return = 0.0
    if candidate_days:
        from .replay import _monthly_from_total
        total_return = (equity / float(starting_equity)) - 1.0 if starting_equity > 0 else 0.0
        monthly_return = float(_monthly_from_total(total_return, len(candidate_days)))
    else:
        total_return = 0.0
    return {
        "total_return": float(total_return),
        "monthly_return": float(monthly_return),
        "max_drawdown": float(max_drawdown),
        "filled_count": int(filled_count),
        "trade_count": int(trade_count),
    }


def run_parameter_sweep(
    candidate_days: Sequence[Sequence[WideCandidate]],
    *,
    starting_equity: float,
    selection_objectives: Iterable[str],
    top_ks: Iterable[int],
    watch_activation_pcts: Iterable[float],
    steal_protection_pcts: Iterable[float],
    fee_bps: float = 10.0,
    fill_buffer_bps: float = 5.0,
    pair_notional_fraction: float = 0.5,
    max_pair_notional_fraction: float = 0.5,
    max_leverage: float = 2.0,
    selection_lookback_days: int = 20,
    tiny_net_hidden_dim: int = 8,
    tiny_net_epochs: int = 120,
    tiny_net_learning_rate: float = 0.03,
    tiny_net_l2: float = 1e-4,
    tiny_net_augment_copies: int = 3,
    tiny_net_noise_scale: float = 0.04,
    tiny_net_min_train_samples: int = 12,
    selection_seed: int = 1337,
    selection_torch_device: str = "auto",
    selection_torch_batch_size: int = 256,
    daily_only: bool = False,
    hourly_by_symbol: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for objective in selection_objectives:
        selection = WideSelectionConfig(
            objective=str(objective),
            lookback_days=int(selection_lookback_days),
            tiny_net_hidden_dim=int(tiny_net_hidden_dim),
            tiny_net_epochs=int(tiny_net_epochs),
            tiny_net_learning_rate=float(tiny_net_learning_rate),
            tiny_net_l2=float(tiny_net_l2),
            tiny_net_augment_copies=int(tiny_net_augment_copies),
            tiny_net_noise_scale=float(tiny_net_noise_scale),
            tiny_net_min_train_samples=int(tiny_net_min_train_samples),
            seed=int(selection_seed),
            torch_device=str(selection_torch_device),
            torch_batch_size=int(selection_torch_batch_size),
        )
        reranked = rerank_candidate_days(
            candidate_days,
            config=selection,
            fee_bps=fee_bps,
            fill_buffer_bps=fill_buffer_bps,
        )
        for top_k in top_ks:
            for watch_activation_pct in watch_activation_pcts:
                for steal_protection_pct in steal_protection_pcts:
                    planner = WidePlannerConfig(
                        top_k=int(top_k),
                        pair_notional_fraction=float(pair_notional_fraction),
                        max_pair_notional_fraction=float(max_pair_notional_fraction),
                        max_leverage=float(max_leverage),
                        fee_bps=float(fee_bps),
                        fill_buffer_bps=float(fill_buffer_bps),
                        watch_activation_pct=float(watch_activation_pct),
                        steal_protection_pct=float(steal_protection_pct),
                    )
                    metrics = _simulate_candidate_days(
                        reranked,
                        starting_equity=starting_equity,
                        planner=planner,
                        daily_only=daily_only,
                        hourly_by_symbol=hourly_by_symbol,
                    )
                    row = asdict(
                        WideSweepSpec(
                            selection_objective=str(objective),
                            top_k=int(top_k),
                            watch_activation_pct=float(watch_activation_pct),
                            steal_protection_pct=float(steal_protection_pct),
                        )
                    )
                    row.update(metrics)
                    rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        by=["monthly_return", "total_return", "max_drawdown", "filled_count"],
        ascending=[False, False, False, False],
        ignore_index=True,
    )


__all__ = ["WideSweepSpec", "run_parameter_sweep"]
