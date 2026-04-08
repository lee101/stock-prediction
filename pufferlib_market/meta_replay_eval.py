"""Adaptive meta-evaluation for daily pufferlib_market checkpoints.

This script:
1. Evaluates multiple daily checkpoints on the same daily MKTD and hourly market.
2. Converts each candidate's hourly replay equity into realized daily returns.
3. Selects one winner per decision day using the shared meta-selector.
4. Replays the winner-picked action trace to compare the adaptive ensemble
   against each underlying checkpoint on the same window.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from pufferlib_market.evaluate_multiperiod import LoadedPolicy, load_policy, make_policy_fn
from pufferlib_market.hourly_replay import (
    DailySimResult,
    HourlyMarket,
    HourlyReplayResult,
    load_hourly_market,
    read_mktd,
    replay_hourly_frozen_daily_actions,
    simulate_daily_policy,
)
from unified_hourly_experiment.meta_selector import daily_returns_from_equity, select_daily_winners


@dataclass(frozen=True)
class CandidateReplay:
    name: str
    checkpoint: str
    actions: np.ndarray
    daily: DailySimResult
    hourly: HourlyReplayResult
    daily_returns: pd.Series


def align_daily_returns_by_intersection(
    daily_returns_by_name: dict[str, pd.Series],
) -> dict[str, pd.Series]:
    if not daily_returns_by_name:
        raise ValueError("daily_returns_by_name cannot be empty.")
    common_idx: pd.DatetimeIndex | None = None
    aligned: dict[str, pd.Series] = {}
    for name, series in daily_returns_by_name.items():
        idx = pd.DatetimeIndex(series.index).sort_values().unique()
        if idx.empty:
            raise ValueError(f"{name}: daily return series is empty.")
        common_idx = idx if common_idx is None else common_idx.intersection(idx)
        aligned[name] = pd.to_numeric(series, errors="coerce").sort_index()
    if common_idx is None or common_idx.empty:
        raise ValueError("No common daily timestamps across candidate return series.")
    common_idx = common_idx.sort_values()
    return {
        name: pd.to_numeric(series.reindex(common_idx), errors="coerce").fillna(0.0)
        for name, series in aligned.items()
    }


def combine_actions_by_winners(
    actions_by_name: dict[str, np.ndarray],
    winners: pd.Series,
    *,
    decision_days: pd.DatetimeIndex,
) -> np.ndarray:
    if not actions_by_name:
        raise ValueError("actions_by_name cannot be empty.")
    if decision_days.empty:
        raise ValueError("decision_days cannot be empty.")
    normalized_winners = winners.copy()
    normalized_winners.index = pd.DatetimeIndex(normalized_winners.index).floor("D")
    combined = np.zeros((len(decision_days),), dtype=np.int32)
    for idx, day in enumerate(decision_days):
        winner = normalized_winners.get(pd.Timestamp(day).floor("D"))
        if winner is None or (isinstance(winner, float) and np.isnan(winner)):
            combined[idx] = 0
            continue
        name = str(winner)
        if name not in actions_by_name:
            raise ValueError(f"Winner '{name}' is not present in actions_by_name.")
        actions = np.asarray(actions_by_name[name], dtype=np.int32)
        if idx >= actions.shape[0]:
            raise ValueError(f"Winner '{name}' has only {actions.shape[0]} actions, need index {idx}.")
        combined[idx] = int(actions[idx])
    return combined


def summarize_winner_series(winners: pd.Series) -> dict[str, object]:
    if winners.empty:
        return {"switch_count": 0, "winner_counts": {}, "latest_winner": None}
    normalized = winners.copy()
    normalized = normalized.where(pd.notna(normalized), None)
    labels = normalized.map(lambda value: "cash" if value is None else str(value))
    switch_count = int(max(0, int((labels != labels.shift(1)).sum()) - 1))
    winner_counts = {str(name): int(count) for name, count in labels.value_counts().sort_index().items()}
    latest = normalized.iloc[-1]
    latest_winner = None if latest is None else str(latest)
    return {
        "switch_count": switch_count,
        "winner_counts": winner_counts,
        "latest_winner": latest_winner,
    }


def summarize_candidate(candidate: CandidateReplay) -> dict[str, object]:
    return {
        "checkpoint": candidate.checkpoint,
        "daily": {
            "total_return": float(candidate.daily.total_return),
            "sortino": float(candidate.daily.sortino),
            "max_drawdown": float(candidate.daily.max_drawdown),
            "num_trades": int(candidate.daily.num_trades),
            "win_rate": float(candidate.daily.win_rate),
        },
        "hourly_replay": {
            "total_return": float(candidate.hourly.total_return),
            "sortino": float(candidate.hourly.sortino),
            "max_drawdown": float(candidate.hourly.max_drawdown),
            "num_trades": int(candidate.hourly.num_trades),
            "num_orders": int(candidate.hourly.num_orders),
            "win_rate": float(candidate.hourly.win_rate),
        },
    }


def _coerce_loaded_policy(value: object) -> LoadedPolicy:
    if isinstance(value, LoadedPolicy):
        return value
    if isinstance(value, tuple) and len(value) == 3:
        policy, metadata, num_actions = value
        meta = metadata if isinstance(metadata, dict) else {}
        return LoadedPolicy(
            policy=policy,
            arch=str(meta.get("arch", "unknown")),
            hidden_size=int(meta.get("hidden_size", 0) or 0),
            action_allocation_bins=int(meta.get("action_allocation_bins", 1) or 1),
            action_level_bins=int(meta.get("action_level_bins", 1) or 1),
            action_max_offset_bps=float(meta.get("action_max_offset_bps", 0.0) or 0.0),
            num_actions=int(num_actions),
        )
    raise TypeError(f"Unsupported load_policy result: {type(value)!r}")


def _fixed_action_policy(actions: np.ndarray):
    cursor = {"idx": 0}

    def _policy_fn(_obs: np.ndarray) -> int:
        idx = int(cursor["idx"])
        cursor["idx"] = idx + 1
        if idx < int(actions.shape[0]):
            return int(actions[idx])
        return 0

    return _policy_fn


def _evaluate_candidate(
    *,
    name: str,
    checkpoint: str,
    daily_data_path: str,
    market: HourlyMarket,
    start_date: str,
    end_date: str,
    max_steps: int,
    fee_rate: float,
    fill_buffer_bps: float,
    max_leverage: float,
    short_borrow_apr: float,
    daily_periods_per_year: float,
    hourly_periods_per_year: float,
    deterministic: bool,
    device: torch.device,
) -> CandidateReplay:
    data = read_mktd(daily_data_path)
    loaded = _coerce_loaded_policy(load_policy(
        checkpoint,
        data.num_symbols,
        arch="auto",
        hidden_size=None,
        device=device,
    ))
    policy_fn = make_policy_fn(
        loaded.policy,
        num_symbols=int(data.num_symbols),
        deterministic=bool(deterministic),
        device=device,
    )
    daily = simulate_daily_policy(
        data,
        policy_fn,
        max_steps=int(max_steps),
        fee_rate=float(fee_rate),
        fill_buffer_bps=float(fill_buffer_bps),
        max_leverage=float(max_leverage),
        periods_per_year=float(daily_periods_per_year),
        short_borrow_apr=float(short_borrow_apr),
        action_allocation_bins=int(loaded.action_allocation_bins),
        action_level_bins=int(loaded.action_level_bins),
        action_max_offset_bps=float(loaded.action_max_offset_bps),
    )
    hourly = replay_hourly_frozen_daily_actions(
        data=data,
        actions=daily.actions,
        market=market,
        start_date=start_date,
        end_date=end_date,
        max_steps=int(max_steps),
        fee_rate=float(fee_rate),
        max_leverage=float(max_leverage),
        short_borrow_apr=float(short_borrow_apr),
        periods_per_year=float(hourly_periods_per_year),
        action_allocation_bins=int(loaded.action_allocation_bins),
        action_level_bins=int(loaded.action_level_bins),
        action_max_offset_bps=float(loaded.action_max_offset_bps),
    )
    equity = pd.Series(hourly.equity_curve, index=market.index, name=name)
    daily_returns = daily_returns_from_equity(equity)
    return CandidateReplay(
        name=name,
        checkpoint=str(checkpoint),
        actions=np.asarray(daily.actions, dtype=np.int32),
        daily=daily,
        hourly=hourly,
        daily_returns=daily_returns,
    )


def run_meta_replay(
    *,
    checkpoints: Sequence[str],
    labels: Sequence[str] | None,
    daily_data_path: str,
    hourly_data_root: str,
    start_date: str,
    end_date: str,
    max_steps: int,
    fee_rate: float,
    fill_buffer_bps: float,
    max_leverage: float,
    short_borrow_apr: float,
    daily_periods_per_year: float,
    hourly_periods_per_year: float,
    deterministic: bool,
    lookback_days: int,
    metric: str,
    fallback_strategy: str | None,
    selection_mode: str,
    switch_margin: float,
    min_score_gap: float,
    sit_out_threshold: float | None,
    recency_halflife_days: float | None,
    require_full_window: bool,
    device: torch.device,
) -> dict[str, object]:
    data = read_mktd(daily_data_path)
    market = load_hourly_market(
        data.symbols,
        hourly_data_root,
        start=f"{start_date} 00:00",
        end=f"{end_date} 23:00",
    )
    start_day = pd.to_datetime(start_date, utc=True).floor("D")
    end_day = pd.to_datetime(end_date, utc=True).floor("D")
    daily_days = pd.date_range(start_day, end_day, freq="D", tz="UTC")
    if len(daily_days) != data.num_timesteps:
        raise ValueError(
            f"Date range mismatch for daily data: days={len(daily_days)} timesteps={data.num_timesteps}. "
            "Provide the same start/end used during export_data_daily.py."
        )
    if int(max_steps) >= int(data.num_timesteps):
        raise ValueError(f"max_steps must be < num_timesteps (got {max_steps}, timesteps={data.num_timesteps}).")

    checkpoint_paths = [str(Path(path)) for path in checkpoints]
    if not checkpoint_paths:
        raise ValueError("At least one checkpoint is required.")
    candidate_names = list(labels) if labels else [Path(path).parent.name for path in checkpoint_paths]
    if len(candidate_names) != len(checkpoint_paths):
        raise ValueError("labels length must match checkpoints length.")

    candidates: list[CandidateReplay] = []
    for name, checkpoint in zip(candidate_names, checkpoint_paths):
        candidates.append(
            _evaluate_candidate(
                name=name,
                checkpoint=checkpoint,
                daily_data_path=daily_data_path,
                market=market,
                start_date=start_date,
                end_date=end_date,
                max_steps=int(max_steps),
                fee_rate=float(fee_rate),
                fill_buffer_bps=float(fill_buffer_bps),
                max_leverage=float(max_leverage),
                short_borrow_apr=float(short_borrow_apr),
                daily_periods_per_year=float(daily_periods_per_year),
                hourly_periods_per_year=float(hourly_periods_per_year),
                deterministic=bool(deterministic),
                device=device,
            )
        )

    candidate_by_name = {candidate.name: candidate for candidate in candidates}
    daily_returns = align_daily_returns_by_intersection(
        {candidate.name: candidate.daily_returns for candidate in candidates}
    )
    ordered_names = [candidate.name for candidate in candidates]
    fallback = str(fallback_strategy or ordered_names[0])
    winners = select_daily_winners(
        daily_returns,
        lookback_days=int(lookback_days),
        metric=str(metric),
        fallback_strategy=fallback,
        tie_break_order=ordered_names,
        require_full_window=bool(require_full_window),
        sit_out_threshold=sit_out_threshold,
        selection_mode=str(selection_mode),
        switch_margin=float(switch_margin),
        min_score_gap=float(min_score_gap),
        recency_halflife_days=recency_halflife_days,
    )
    decision_days = daily_days[: int(max_steps)]
    winners = winners.reindex(decision_days, fill_value=fallback)
    meta_actions = combine_actions_by_winners(
        {name: candidate.actions for name, candidate in candidate_by_name.items()},
        winners,
        decision_days=decision_days,
    )
    meta_daily = simulate_daily_policy(
        data,
        _fixed_action_policy(meta_actions),
        max_steps=int(max_steps),
        fee_rate=float(fee_rate),
        fill_buffer_bps=float(fill_buffer_bps),
        max_leverage=float(max_leverage),
        periods_per_year=float(daily_periods_per_year),
        short_borrow_apr=float(short_borrow_apr),
    )
    meta_hourly = replay_hourly_frozen_daily_actions(
        data=data,
        actions=meta_actions,
        market=market,
        start_date=start_date,
        end_date=end_date,
        max_steps=int(max_steps),
        fee_rate=float(fee_rate),
        max_leverage=float(max_leverage),
        short_borrow_apr=float(short_borrow_apr),
        periods_per_year=float(hourly_periods_per_year),
    )

    report = {
        "daily_data_path": str(daily_data_path),
        "hourly_data_root": str(hourly_data_root),
        "date_range": {"start": start_date, "end": end_date},
        "symbols": list(data.symbols),
        "fill_buffer_bps": float(fill_buffer_bps),
        "selector": {
            "lookback_days": int(lookback_days),
            "metric": str(metric),
            "fallback_strategy": fallback,
            "selection_mode": str(selection_mode),
            "switch_margin": float(switch_margin),
            "min_score_gap": float(min_score_gap),
            "sit_out_threshold": None if sit_out_threshold is None else float(sit_out_threshold),
            "recency_halflife_days": None if recency_halflife_days is None else float(recency_halflife_days),
            "require_full_window": bool(require_full_window),
        },
        "candidates": {candidate.name: summarize_candidate(candidate) for candidate in candidates},
        "meta": {
            "daily": {
                "total_return": float(meta_daily.total_return),
                "sortino": float(meta_daily.sortino),
                "max_drawdown": float(meta_daily.max_drawdown),
                "num_trades": int(meta_daily.num_trades),
                "win_rate": float(meta_daily.win_rate),
            },
            "hourly_replay": {
                "total_return": float(meta_hourly.total_return),
                "sortino": float(meta_hourly.sortino),
                "max_drawdown": float(meta_hourly.max_drawdown),
                "num_trades": int(meta_hourly.num_trades),
                "num_orders": int(meta_hourly.num_orders),
                "win_rate": float(meta_hourly.win_rate),
            },
            "winners": summarize_winner_series(winners),
        },
        "decision_winners": [
            {
                "date": str(pd.Timestamp(day).date()),
                "winner": None if winner is None or (isinstance(winner, float) and np.isnan(winner)) else str(winner),
            }
            for day, winner in winners.items()
        ],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive meta replay evaluation for daily RL checkpoints")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", action="append", dest="checkpoint_paths",
                       help="Checkpoint path; pass multiple times for multiple candidates")
    group.add_argument("--checkpoints", dest="checkpoints_csv", help="Comma-separated checkpoint paths")
    parser.add_argument("--labels", default="", help="Optional comma-separated candidate labels")
    parser.add_argument("--daily-data-path", required=True)
    parser.add_argument("--hourly-data-root", default="trainingdatahourly")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--max-leverage", type=float, default=1.0)
    parser.add_argument("--short-borrow-apr", type=float, default=0.0)
    parser.add_argument("--daily-periods-per-year", type=float, default=365.0)
    parser.add_argument("--hourly-periods-per-year", type=float, default=8760.0)
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--metric", default="goodness")
    parser.add_argument("--fallback-strategy", default=None)
    parser.add_argument("--selection-mode", choices=["winner", "winner_cash", "sticky"], default="sticky")
    parser.add_argument("--switch-margin", type=float, default=0.0)
    parser.add_argument("--min-score-gap", type=float, default=0.0)
    parser.add_argument("--sit-out-threshold", type=float, default=None)
    parser.add_argument("--recency-halflife-days", type=float, default=None)
    parser.add_argument("--allow-partial-window", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    checkpoints = [value.strip() for value in str(args.checkpoints_csv or "").split(",") if value.strip()]
    if not checkpoints:
        checkpoints = list(args.checkpoint_paths or [])
    labels = [value.strip() for value in str(args.labels).split(",") if value.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    report = run_meta_replay(
        checkpoints=checkpoints,
        labels=labels if labels else None,
        daily_data_path=str(args.daily_data_path),
        hourly_data_root=str(args.hourly_data_root),
        start_date=str(args.start_date),
        end_date=str(args.end_date),
        max_steps=int(args.max_steps),
        fee_rate=float(args.fee_rate),
        fill_buffer_bps=float(args.fill_buffer_bps),
        max_leverage=float(args.max_leverage),
        short_borrow_apr=float(args.short_borrow_apr),
        daily_periods_per_year=float(args.daily_periods_per_year),
        hourly_periods_per_year=float(args.hourly_periods_per_year),
        deterministic=bool(args.deterministic),
        lookback_days=int(args.lookback_days),
        metric=str(args.metric),
        fallback_strategy=args.fallback_strategy,
        selection_mode=str(args.selection_mode),
        switch_margin=float(args.switch_margin),
        min_score_gap=float(args.min_score_gap),
        sit_out_threshold=args.sit_out_threshold,
        recency_halflife_days=args.recency_halflife_days,
        require_full_window=not bool(args.allow_partial_window),
        device=device,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)


if __name__ == "__main__":
    main()
