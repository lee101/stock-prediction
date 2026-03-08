#!/usr/bin/env python3
"""Sweep daily winner-take-all meta selection across Binance model candidates.

The selector chooses one candidate per day using trailing daily performance
(previous-day style, no lookahead), then compounds hourly returns from the
selected candidate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from loguru import logger

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from binanceleveragesui.sim_meta_switcher import compute_stats, load_model_and_actions, per_bar_sim
from unified_hourly_experiment.meta_selector import (
    SUPPORTED_META_METRICS,
    daily_returns_from_equity,
    score_trailing_returns,
    select_daily_winners,
)


@dataclass(frozen=True)
class Candidate:
    name: str
    symbol: str
    checkpoint: Path


SUPPORTED_META_MODES = ("winner", "winner_cash", "blend_top2", "softmax_all")


def parse_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(x) for x in parse_csv_list(raw)]


def parse_candidate_spec(spec: str) -> Candidate:
    """Parse NAME:SYMBOL:CHECKPOINT spec."""
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid --candidate '{spec}'. Expected NAME:SYMBOL:CHECKPOINT.")
    name, symbol, path_raw = parts[0].strip(), parts[1].strip().upper(), parts[2].strip()
    if not name or not symbol or not path_raw:
        raise ValueError(f"Invalid --candidate '{spec}'. Empty name/symbol/path.")
    path = Path(path_raw)
    if not path.is_absolute():
        path = (REPO / path).resolve()
    if not path.exists():
        raise ValueError(f"Candidate checkpoint does not exist: {path}")
    return Candidate(name=name, symbol=symbol, checkpoint=path)


def default_candidates() -> list[Candidate]:
    """Curated candidate pool from strong DOGE/AAVE checkpoints."""
    specs = [
        "doge_deployed:DOGEUSD:binanceleveragesui/checkpoints/DOGEUSD_r4_R4_h384_cosine/binanceneural_20260301_065549/epoch_001.pt",
        "doge_wider_mlp8:DOGEUSD:binanceleveragesui/checkpoints/DOGEUSD_gen_wider_mlp8/binanceneural_20260227_031938/epoch_004.pt",
        "doge_dilated_142472:DOGEUSD:binanceleveragesui/checkpoints/DOGEUSD_gen_dilated_1_4_24_72/binanceneural_20260226_073041/epoch_004.pt",
        "doge_deeper6l:DOGEUSD:binanceleveragesui/checkpoints/DOGEUSD_gen_deeper_6L/binanceneural_20260227_014411/epoch_010.pt",
        "doge_r5_drop15:DOGEUSD:binanceleveragesui/checkpoints/r5_DOGE_rw05_drop15/binanceneural_20260303_001154/epoch_003.pt",
        "aave_rw05:AAVEUSD:binanceleveragesui/checkpoints/AAVEUSD_sweep_AAVE_h384_cosine_rw05/binanceneural_20260302_102218/epoch_003.pt",
        "aave_r5_wd04:AAVEUSD:binanceleveragesui/checkpoints/r5_AAVE_rw05_wd04/binanceneural_20260303_010938/epoch_002.pt",
        "aave_base:AAVEUSD:binanceleveragesui/checkpoints/AAVEUSD_sweep_AAVE_h384_cosine/binanceneural_20260302_094405/epoch_001.pt",
    ]
    return [parse_candidate_spec(spec) for spec in specs]


def build_equity_series(candidate: Candidate, *, val_days: int, test_days: int, maker_fee: float, max_leverage: float) -> pd.Series:
    bars, actions = load_model_and_actions(
        candidate.checkpoint,
        candidate.symbol,
        val_days=val_days,
        test_days=test_days,
    )
    sim_df = per_bar_sim(
        bars,
        actions,
        maker_fee=maker_fee,
        max_leverage=max_leverage,
        symbol=candidate.symbol,
    )
    if sim_df.empty:
        raise ValueError(f"{candidate.name}: empty simulation frame")
    ts = pd.to_datetime(sim_df["timestamp"], utc=True)
    eq = pd.to_numeric(sim_df["equity"], errors="coerce")
    series = pd.Series(eq.values, index=ts, name=candidate.name).dropna().sort_index()
    series = series[~series.index.duplicated(keep="last")]
    if len(series) < 48:
        raise ValueError(f"{candidate.name}: insufficient bars after simulation ({len(series)})")
    return series


def align_equity_curves(equity_by_name: Mapping[str, pd.Series]) -> dict[str, pd.Series]:
    if not equity_by_name:
        raise ValueError("No equity curves provided.")
    common_idx: pd.DatetimeIndex | None = None
    for series in equity_by_name.values():
        idx = pd.DatetimeIndex(series.index).sort_values().unique()
        common_idx = idx if common_idx is None else common_idx.intersection(idx)
    if common_idx is None or common_idx.empty:
        raise ValueError("No common timestamps across candidate equity curves.")
    common_idx = common_idx.sort_values()
    return {name: series.reindex(common_idx) for name, series in equity_by_name.items()}


def slice_window(equity_by_name: Mapping[str, pd.Series], window_days: int) -> dict[str, pd.Series]:
    if window_days <= 0:
        return {k: v.copy() for k, v in equity_by_name.items()}
    max_ts = max(v.index.max() for v in equity_by_name.values())
    cutoff = max_ts - pd.Timedelta(days=int(window_days))
    sliced = {k: v[v.index >= cutoff] for k, v in equity_by_name.items()}
    min_len = min(len(v) for v in sliced.values())
    if min_len < 48:
        raise ValueError(f"Window {window_days}d leaves too few bars ({min_len}).")
    return align_equity_curves(sliced)


def compose_daily_winner_equity(
    equity_by_name: Mapping[str, pd.Series],
    *,
    lookback_days: int,
    metric: str,
    fallback_strategy: str,
    tie_break_order: Sequence[str],
) -> tuple[pd.Series, pd.Series, int]:
    """Create meta equity from daily winners chosen on trailing daily returns."""
    if fallback_strategy not in equity_by_name:
        raise ValueError(f"fallback_strategy '{fallback_strategy}' missing from equity set.")

    daily_returns = {name: daily_returns_from_equity(series) for name, series in equity_by_name.items()}
    winners = select_daily_winners(
        daily_returns,
        lookback_days=lookback_days,
        metric=metric,
        fallback_strategy=fallback_strategy,
        tie_break_order=tie_break_order,
        require_full_window=True,
    )

    idx = next(iter(equity_by_name.values())).index
    days = pd.DatetimeIndex(idx.floor("D").unique())
    winners = winners.reindex(days, fill_value=fallback_strategy)

    rel = {
        name: series / (series.shift(1).replace(0.0, np.nan))
        for name, series in equity_by_name.items()
    }
    rel = {name: s.replace([np.inf, -np.inf], np.nan).fillna(1.0) for name, s in rel.items()}

    eq_values: list[float] = []
    selected: list[str] = []
    current = 10_000.0
    for ts in idx:
        day = pd.Timestamp(ts).floor("D")
        winner = str(winners.get(day, fallback_strategy))
        r = float(rel[winner].loc[ts])
        if not np.isfinite(r) or r <= 0:
            r = 1.0
        current *= r
        eq_values.append(current)
        selected.append(winner)

    meta_eq = pd.Series(eq_values, index=idx, name="meta_equity")
    selected_hourly = pd.Series(selected, index=idx, name="winner")
    daily_selected = selected_hourly.groupby(selected_hourly.index.floor("D")).last()
    switches = int((daily_selected != daily_selected.shift(1)).sum() - 1) if not daily_selected.empty else 0
    return meta_eq, winners, max(switches, 0)


def _ordered_names(names: Sequence[str], tie_break_order: Sequence[str]) -> list[str]:
    order = [x for x in tie_break_order if x in names]
    seen = set(order)
    order.extend([x for x in names if x not in seen])
    return order


def _build_daily_allocations(
    daily_returns_by_name: Mapping[str, pd.Series],
    *,
    lookback_days: int,
    metric: str,
    fallback_strategy: str,
    tie_break_order: Sequence[str],
    mode: str,
    cash_threshold: float,
    switch_margin: float,
    softmax_temperature: float,
    min_score_gap: float,
) -> dict[pd.Timestamp, dict[str, float]]:
    names = list(daily_returns_by_name.keys())
    ordered = _ordered_names(names, tie_break_order)
    all_days = sorted(
        {
            pd.Timestamp(day).floor("D")
            for series in daily_returns_by_name.values()
            for day in series.index
        }
    )
    all_idx = pd.DatetimeIndex(all_days)
    aligned = {
        name: pd.to_numeric(series.reindex(all_idx).fillna(0.0), errors="coerce").fillna(0.0)
        for name, series in daily_returns_by_name.items()
    }

    allocations: dict[pd.Timestamp, dict[str, float]] = {}
    prev_choice: str = fallback_strategy
    for day_i, day in enumerate(all_idx):
        window_start = max(0, day_i - lookback_days)
        window_len = day_i - window_start
        if window_len < lookback_days:
            choice = prev_choice if prev_choice else fallback_strategy
            allocations[pd.Timestamp(day)] = {choice: 1.0}
            prev_choice = choice
            continue

        scores: dict[str, float] = {}
        for name in ordered:
            window = aligned[name].iloc[window_start:day_i].to_numpy(dtype=np.float64)
            scores[name] = score_trailing_returns(window, metric)

        best_name = ordered[0]
        best_score = scores[best_name]
        for name in ordered[1:]:
            if scores[name] > best_score:
                best_score = scores[name]
                best_name = name
        ranked_scores = sorted((float(scores[name]), name) for name in ordered)
        second_score = float(ranked_scores[-2][0]) if len(ranked_scores) >= 2 else float("-inf")
        score_gap = float(best_score - second_score) if np.isfinite(second_score) else float("inf")

        if mode == "winner":
            low_confidence = score_gap < min_score_gap
            choice = best_name if not low_confidence else (prev_choice if prev_choice else fallback_strategy)
            if prev_choice in names and choice in names and choice != prev_choice:
                if (scores[choice] - scores[prev_choice]) <= switch_margin:
                    choice = prev_choice
            allocations[pd.Timestamp(day)] = {choice: 1.0}
            prev_choice = choice
            continue

        if mode == "winner_cash":
            low_confidence = score_gap < min_score_gap
            choice = "cash" if (best_score <= cash_threshold or low_confidence) else best_name

            if prev_choice in names and choice in names and choice != prev_choice:
                if (scores[choice] - scores[prev_choice]) <= switch_margin:
                    choice = prev_choice
            elif prev_choice == "cash" and choice in names:
                if best_score <= cash_threshold + switch_margin or score_gap < min_score_gap + switch_margin:
                    choice = "cash"
            elif prev_choice in names and choice == "cash":
                prev_score = scores.get(prev_choice, float("-inf"))
                if (
                    prev_choice == best_name
                    and prev_score > cash_threshold - switch_margin
                    and score_gap >= max(0.0, min_score_gap - switch_margin)
                ):
                    choice = prev_choice

            allocations[pd.Timestamp(day)] = {choice: 1.0}
            prev_choice = choice
            continue

        if mode == "softmax_all":
            # Turn model scores into probabilistic weights across all candidates.
            vals = np.array(
                [
                    1e6 if scores[n] == float("inf") else (-1e6 if scores[n] == float("-inf") else float(scores[n]))
                    for n in ordered
                ],
                dtype=np.float64,
            )
            temp = max(float(softmax_temperature), 1e-6)
            vals = vals / temp
            vals = np.where(np.isfinite(vals), vals, 0.0)
            vmax = float(np.max(vals))
            expv = np.exp(np.clip(vals - vmax, -60.0, 60.0))
            denom = float(np.sum(expv))
            if denom <= 0.0:
                alloc = {ordered[0]: 1.0}
            else:
                w = expv / denom
                alloc = {name: float(weight) for name, weight in zip(ordered, w) if float(weight) > 0.0}

            # Optional cash gate by best score.
            if best_score <= cash_threshold:
                allocations[pd.Timestamp(day)] = {"cash": 1.0}
                prev_choice = "cash"
                continue

            # Optional hysteresis via dominant allocation label.
            top_name = max(alloc, key=alloc.get)
            if prev_choice in names and top_name in names and top_name != prev_choice:
                if (scores[top_name] - scores[prev_choice]) <= switch_margin:
                    # Bias toward previous model by transferring top weight to it.
                    carry = alloc.pop(top_name, 0.0)
                    alloc[prev_choice] = float(alloc.get(prev_choice, 0.0) + carry)
                    top_name = prev_choice
            allocations[pd.Timestamp(day)] = alloc
            prev_choice = top_name
            continue

        if mode == "blend_top2":
            ranked = sorted(ordered, key=lambda n: scores[n], reverse=True)
            top = ranked[:2]
            vals = np.array(
                [
                    1e6 if scores[n] == float("inf") else (-1e6 if scores[n] == float("-inf") else float(scores[n]))
                    for n in top
                ],
                dtype=np.float64,
            )
            vals = vals - np.min(vals) + 1e-6
            vals = np.where(np.isfinite(vals), vals, 0.0)
            denom = float(np.sum(vals))
            if denom <= 0.0:
                allocations[pd.Timestamp(day)] = {top[0]: 0.5, top[1]: 0.5}
            else:
                w = vals / denom
                allocations[pd.Timestamp(day)] = {top[0]: float(w[0]), top[1]: float(w[1])}
            continue

        raise ValueError(f"Unsupported mode '{mode}'. Allowed: {SUPPORTED_META_MODES}")

    return allocations


def compose_meta_equity_mode(
    equity_by_name: Mapping[str, pd.Series],
    *,
    lookback_days: int,
    metric: str,
    fallback_strategy: str,
    tie_break_order: Sequence[str],
    mode: str,
    cash_threshold: float,
    switch_margin: float,
    softmax_temperature: float,
    min_score_gap: float = 0.0,
) -> tuple[pd.Series, pd.Series, int]:
    if mode == "winner":
        return compose_daily_winner_equity(
            equity_by_name,
            lookback_days=lookback_days,
            metric=metric,
            fallback_strategy=fallback_strategy,
            tie_break_order=tie_break_order,
        )

    if fallback_strategy not in equity_by_name:
        raise ValueError(f"fallback_strategy '{fallback_strategy}' missing from equity set.")

    daily_returns = {name: daily_returns_from_equity(series) for name, series in equity_by_name.items()}
    allocations = _build_daily_allocations(
        daily_returns,
        lookback_days=lookback_days,
        metric=metric,
        fallback_strategy=fallback_strategy,
        tie_break_order=tie_break_order,
        mode=mode,
        cash_threshold=cash_threshold,
        switch_margin=switch_margin,
        softmax_temperature=softmax_temperature,
        min_score_gap=min_score_gap,
    )

    idx = next(iter(equity_by_name.values())).index
    days = pd.DatetimeIndex(idx.floor("D").unique())
    if not days.empty and pd.Timestamp(days[0]) not in allocations:
        allocations[pd.Timestamp(days[0])] = {fallback_strategy: 1.0}

    rel = {
        name: series / (series.shift(1).replace(0.0, np.nan))
        for name, series in equity_by_name.items()
    }
    rel = {name: s.replace([np.inf, -np.inf], np.nan).fillna(1.0) for name, s in rel.items()}

    eq_values: list[float] = []
    day_labels: list[str] = []
    current = 10_000.0
    for ts in idx:
        day = pd.Timestamp(ts).floor("D")
        alloc = allocations.get(day, {fallback_strategy: 1.0})
        rel_val = 0.0
        w_sum = 0.0
        for name, weight in alloc.items():
            w = max(0.0, float(weight))
            if w <= 0:
                continue
            if name == "cash":
                rel_component = 1.0
            else:
                rel_component = float(rel[name].loc[ts])
                if not np.isfinite(rel_component) or rel_component <= 0:
                    rel_component = 1.0
            rel_val += w * rel_component
            w_sum += w
        if w_sum <= 0:
            rel_val = 1.0
        else:
            rel_val = rel_val / w_sum
        current *= rel_val
        eq_values.append(current)

        if alloc.keys() == {"cash"}:
            day_labels.append("cash")
        elif len(alloc) == 1:
            day_labels.append(next(iter(alloc.keys())))
        else:
            top_names = sorted(alloc.items(), key=lambda kv: kv[1], reverse=True)
            day_labels.append("+".join(name for name, _ in top_names))

    meta_eq = pd.Series(eq_values, index=idx, name="meta_equity")
    labels = pd.Series(day_labels, index=idx, name="allocation")
    day_labels_only = labels.groupby(labels.index.floor("D")).last()
    switches = int((day_labels_only != day_labels_only.shift(1)).sum() - 1) if not day_labels_only.empty else 0
    return meta_eq, day_labels_only, max(switches, 0)


def summarize_windows(rows: Sequence[dict]) -> dict:
    sortinos = [float(r["meta"]["sortino"]) for r in rows]
    returns = [float(r["meta"]["total_return"]) for r in rows]
    dds = [float(r["meta"]["max_dd"]) for r in rows]
    beats = [bool(r["meta"]["sortino"] > r["best_baseline"]["sortino"]) for r in rows]
    switches = [int(r["switches"]) for r in rows]
    return {
        "min_sortino": float(np.min(sortinos)),
        "mean_sortino": float(np.mean(sortinos)),
        "min_return_pct": float(np.min(returns)),
        "mean_return_pct": float(np.mean(returns)),
        "mean_max_dd_pct": float(np.mean(dds)),
        "beats_baseline_windows": int(np.sum(beats)),
        "mean_switches": float(np.mean(switches)),
    }


def rank_key(item: dict) -> tuple[float, float, float, float, float, float]:
    return (
        float(item["summary"]["min_sortino"]),
        float(item["summary"]["beats_baseline_windows"]),
        float(item["summary"]["mean_sortino"]),
        float(item["summary"]["min_return_pct"]),
        float(item["summary"]["mean_return_pct"]),
        -float(item["summary"]["mean_max_dd_pct"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep daily winner meta-selection on Binance candidates.")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate spec NAME:SYMBOL:CHECKPOINT. Can be passed multiple times.",
    )
    parser.add_argument("--windows", default="30,60,90,120")
    parser.add_argument("--lookbacks", default="1,2,3,5,7,10,14")
    parser.add_argument("--metrics", default="return,sortino,calmar,sharpe")
    parser.add_argument("--modes", default="winner,winner_cash,blend_top2")
    parser.add_argument("--cash-thresholds", default="0.0,0.02")
    parser.add_argument("--switch-margins", default="0.0,0.01,0.02")
    parser.add_argument("--min-score-gaps", default="0.0")
    parser.add_argument("--softmax-temperatures", default="0.25,0.5,1.0,2.0")
    parser.add_argument("--val-days", type=int, default=30)
    parser.add_argument("--max-leverage", type=float, default=2.0)
    parser.add_argument("--maker-fee", type=float, default=0.001)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--strict-candidates",
        action="store_true",
        help="fail immediately if any candidate cannot be loaded/simulated",
    )
    args = parser.parse_args()

    windows = parse_int_list(args.windows)
    lookbacks = parse_int_list(args.lookbacks)
    metrics = [m.lower() for m in parse_csv_list(args.metrics)]
    modes = [m.lower() for m in parse_csv_list(args.modes)]
    cash_thresholds = [float(x) for x in parse_csv_list(args.cash_thresholds)]
    switch_margins = [float(x) for x in parse_csv_list(args.switch_margins)]
    min_score_gaps = [float(x) for x in parse_csv_list(args.min_score_gaps)]
    softmax_temperatures = [float(x) for x in parse_csv_list(args.softmax_temperatures)]
    invalid = [m for m in metrics if m not in SUPPORTED_META_METRICS]
    if invalid:
        raise ValueError(f"Unsupported metric(s): {invalid}. Allowed: {SUPPORTED_META_METRICS}")
    invalid_modes = [m for m in modes if m not in SUPPORTED_META_MODES]
    if invalid_modes:
        raise ValueError(f"Unsupported mode(s): {invalid_modes}. Allowed: {SUPPORTED_META_MODES}")
    if any(w <= 0 for w in windows):
        raise ValueError(f"windows must be > 0, got {windows}")
    if any(lb <= 0 for lb in lookbacks):
        raise ValueError(f"lookbacks must be > 0, got {lookbacks}")
    if any(t < 0 for t in cash_thresholds):
        raise ValueError(f"cash-thresholds must be >= 0, got {cash_thresholds}")
    if any(m < 0 for m in switch_margins):
        raise ValueError(f"switch-margins must be >= 0, got {switch_margins}")
    if any(g < 0 for g in min_score_gaps):
        raise ValueError(f"min-score-gaps must be >= 0, got {min_score_gaps}")
    if any(t <= 0 for t in softmax_temperatures):
        raise ValueError(f"softmax-temperatures must be > 0, got {softmax_temperatures}")

    candidates = [parse_candidate_spec(x) for x in args.candidate] if args.candidate else default_candidates()
    if len(candidates) < 2:
        raise ValueError("Need at least two candidates for meta selection.")
    names = [c.name for c in candidates]
    if len(set(names)) != len(names):
        raise ValueError(f"Candidate names must be unique, got {names}")

    max_window = max(windows)
    logger.info("Loading candidate equity curves ({} candidates, {}d max window)...", len(candidates), max_window)
    curves: dict[str, pd.Series] = {}
    for cand in candidates:
        logger.info("  loading {} ({})", cand.name, cand.symbol)
        try:
            curves[cand.name] = build_equity_series(
                cand,
                val_days=args.val_days,
                test_days=max_window,
                maker_fee=args.maker_fee,
                max_leverage=args.max_leverage,
            )
        except Exception as exc:
            if args.strict_candidates:
                raise
            logger.warning("  skipping {} due to load/sim error: {}", cand.name, exc)

    if len(curves) < 2:
        raise ValueError(f"Need at least 2 valid candidates after loading, got {len(curves)}")

    aligned_full = align_equity_curves(curves)
    logger.info("Common bars across candidates: {}", len(next(iter(aligned_full.values()))))

    fallback = candidates[0].name
    tie_break = [c.name for c in candidates]

    all_results: list[dict] = []
    for mode in modes:
        threshold_grid = [0.0] if mode == "winner" else cash_thresholds
        margin_grid = [0.0] if mode == "blend_top2" else switch_margins
        min_gap_grid = [0.0] if mode in ("blend_top2", "softmax_all") else min_score_gaps
        temp_grid = [1.0] if mode != "softmax_all" else softmax_temperatures
        for min_score_gap in min_gap_grid:
            for softmax_temperature in temp_grid:
                for switch_margin in margin_grid:
                    for cash_threshold in threshold_grid:
                        for metric in metrics:
                            for lookback in lookbacks:
                                window_rows: list[dict] = []
                                for days in windows:
                                    win_curves = slice_window(aligned_full, days)
                                    meta_eq, winners, switches = compose_meta_equity_mode(
                                        win_curves,
                                        lookback_days=lookback,
                                        metric=metric,
                                        fallback_strategy=fallback,
                                        tie_break_order=tie_break,
                                        mode=mode,
                                        cash_threshold=float(cash_threshold),
                                        switch_margin=float(switch_margin),
                                        softmax_temperature=float(softmax_temperature),
                                        min_score_gap=float(min_score_gap),
                                    )

                                    candidate_stats = {
                                        name: compute_stats(series.values, name)
                                        for name, series in win_curves.items()
                                    }
                                    best_baseline_name = max(
                                        candidate_stats,
                                        key=lambda n: (
                                            float(candidate_stats[n]["sortino"]),
                                            float(candidate_stats[n]["total_return"]),
                                            -float(candidate_stats[n]["max_dd"]),
                                        ),
                                    )
                                    best_baseline = {"name": best_baseline_name, **candidate_stats[best_baseline_name]}
                                    meta_stats = compute_stats(meta_eq.values, "meta")
                                    window_rows.append(
                                        {
                                            "window_days": int(days),
                                            "meta": meta_stats,
                                            "best_baseline": best_baseline,
                                            "switches": int(switches),
                                            "winner_days": int(len(winners)),
                                        }
                                    )

                                summary = summarize_windows(window_rows)
                                all_results.append(
                                    {
                                        "mode": mode,
                                        "cash_threshold": float(cash_threshold),
                                        "switch_margin": float(switch_margin),
                                        "min_score_gap": float(min_score_gap),
                                        "softmax_temperature": float(softmax_temperature),
                                        "metric": metric,
                                        "lookback_days": int(lookback),
                                        "summary": summary,
                                        "windows": window_rows,
                                    }
                                )

    all_results.sort(key=rank_key, reverse=True)
    top_k = max(1, int(args.top_k))
    logger.info(
        "\nTop {} configs by robustness (min sortino, baseline beats, mean sortino):",
        min(top_k, len(all_results)),
    )
    for i, row in enumerate(all_results[:top_k], start=1):
        s = row["summary"]
        logger.info(
            "{:>2}. {:<11} ct={:<4.2f} sm={:<4.2f} mg={:<4.2f} tp={:<4.2f} {:<8} lb={:<2d} minS={:>6.2f} meanS={:>6.2f} minR={:>+7.2f}% meanR={:>+7.2f}% meanDD={:>6.2f}% beats={}/{} sw={:>5.1f}",
            i,
            row["mode"],
            row.get("cash_threshold", 0.0),
            row.get("switch_margin", 0.0),
            row.get("min_score_gap", 0.0),
            row.get("softmax_temperature", 1.0),
            row["metric"],
            row["lookback_days"],
            s["min_sortino"],
            s["mean_sortino"],
            s["min_return_pct"],
            s["mean_return_pct"],
            s["mean_max_dd_pct"],
            s["beats_baseline_windows"],
            len(windows),
            s["mean_switches"],
        )

    ts = time.strftime("%Y%m%d_%H%M%S")
    output = args.output or Path(f"binanceleveragesui/meta_daily_winners_sweep_{ts}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "candidates": [
            {"name": c.name, "symbol": c.symbol, "checkpoint": str(c.checkpoint)}
            for c in candidates
        ],
        "settings": {
            "windows": windows,
            "lookbacks": lookbacks,
            "metrics": metrics,
            "modes": modes,
            "cash_thresholds": cash_thresholds,
            "switch_margins": switch_margins,
            "min_score_gaps": min_score_gaps,
            "softmax_temperatures": softmax_temperatures,
            "val_days": args.val_days,
            "max_leverage": args.max_leverage,
            "maker_fee": args.maker_fee,
        },
        "results": all_results,
    }
    output.write_text(json.dumps(payload, indent=2))
    logger.info("Saved sweep results: {}", output)


if __name__ == "__main__":
    main()
