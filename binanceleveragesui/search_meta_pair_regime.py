#!/usr/bin/env python3
"""Regime-adaptive meta pair search for DOGE/AAVE selectors."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from binanceleveragesui.sim_meta_switcher import compute_stats
from binanceleveragesui.sweep_meta_daily_winners import (
    align_equity_curves,
    build_equity_series,
    parse_candidate_spec,
    parse_csv_list,
    parse_int_list,
    slice_window,
)
from unified_hourly_experiment.meta_selector import daily_returns_from_equity, score_trailing_returns


def _parse_float_list(raw: str) -> list[float]:
    return [float(x) for x in parse_csv_list(raw)]


def _ordered_names(names: Sequence[str], tie_break_order: Sequence[str]) -> list[str]:
    order = [x for x in tie_break_order if x in names]
    seen = set(order)
    order.extend([x for x in names if x not in seen])
    return order


def _compute_daily_market_abs_returns(
    daily_returns_by_name: Mapping[str, pd.Series],
    days: pd.DatetimeIndex,
) -> pd.Series:
    parts = []
    for name, series in daily_returns_by_name.items():
        s = pd.to_numeric(series.reindex(days).fillna(0.0), errors="coerce").fillna(0.0)
        parts.append(s.abs())
    if not parts:
        return pd.Series(0.0, index=days)
    stacked = pd.concat(parts, axis=1)
    return stacked.mean(axis=1)


def _select_daily_allocations_regime(
    daily_returns_by_name: Mapping[str, pd.Series],
    *,
    low_metric: str,
    high_metric: str,
    low_lookback: int,
    high_lookback: int,
    vol_lookback: int,
    vol_threshold: float,
    fallback_strategy: str,
    tie_break_order: Sequence[str],
    cash_threshold: float,
    switch_margin: float,
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
    market_abs = _compute_daily_market_abs_returns(aligned, all_idx)

    allocations: dict[pd.Timestamp, dict[str, float]] = {}
    prev_choice = fallback_strategy
    for day_i, day in enumerate(all_idx):
        if day_i <= 0:
            allocations[pd.Timestamp(day)] = {prev_choice: 1.0}
            continue

        vol_start = max(0, day_i - vol_lookback)
        vol_window = market_abs.iloc[vol_start:day_i].to_numpy(dtype=np.float64)
        if len(vol_window) < vol_lookback:
            allocations[pd.Timestamp(day)] = {prev_choice: 1.0}
            continue
        realized_vol = float(np.mean(vol_window))
        is_high_vol = realized_vol >= vol_threshold
        metric = high_metric if is_high_vol else low_metric
        lookback = high_lookback if is_high_vol else low_lookback

        window_start = max(0, day_i - lookback)
        if day_i - window_start < lookback:
            allocations[pd.Timestamp(day)] = {prev_choice: 1.0}
            continue

        scores: dict[str, float] = {}
        for name in ordered:
            window = aligned[name].iloc[window_start:day_i].to_numpy(dtype=np.float64)
            scores[name] = float(score_trailing_returns(window, metric))

        ranked = sorted(((scores[name], name) for name in ordered), reverse=True)
        best_score, best_name = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else float("-inf")
        score_gap = float(best_score - second_score) if np.isfinite(second_score) else float("inf")
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
        prev_choice = choice if choice in names else "cash"
    return allocations


def compose_regime_meta_equity(
    equity_by_name: Mapping[str, pd.Series],
    *,
    low_metric: str,
    high_metric: str,
    low_lookback: int,
    high_lookback: int,
    vol_lookback: int,
    vol_threshold: float,
    fallback_strategy: str,
    tie_break_order: Sequence[str],
    cash_threshold: float,
    switch_margin: float,
    min_score_gap: float,
) -> tuple[pd.Series, pd.Series, int]:
    if fallback_strategy not in equity_by_name:
        raise ValueError(f"fallback_strategy '{fallback_strategy}' missing from equity set.")
    daily_returns = {name: daily_returns_from_equity(series) for name, series in equity_by_name.items()}
    allocations = _select_daily_allocations_regime(
        daily_returns,
        low_metric=low_metric,
        high_metric=high_metric,
        low_lookback=low_lookback,
        high_lookback=high_lookback,
        vol_lookback=vol_lookback,
        vol_threshold=vol_threshold,
        fallback_strategy=fallback_strategy,
        tie_break_order=tie_break_order,
        cash_threshold=cash_threshold,
        switch_margin=switch_margin,
        min_score_gap=min_score_gap,
    )

    idx = next(iter(equity_by_name.values())).index
    rel = {
        name: series / (series.shift(1).replace(0.0, np.nan))
        for name, series in equity_by_name.items()
    }
    rel = {name: s.replace([np.inf, -np.inf], np.nan).fillna(1.0) for name, s in rel.items()}

    eq_values: list[float] = []
    labels: list[str] = []
    current = 10_000.0
    for ts in idx:
        day = pd.Timestamp(ts).floor("D")
        alloc = allocations.get(day, {fallback_strategy: 1.0})
        choice = next(iter(alloc.keys()))
        if choice == "cash":
            r = 1.0
        else:
            r = float(rel[choice].loc[ts])
            if not np.isfinite(r) or r <= 0:
                r = 1.0
        current *= r
        eq_values.append(current)
        labels.append(choice)

    meta_eq = pd.Series(eq_values, index=idx, name="meta_equity")
    day_labels = pd.Series(labels, index=idx).groupby(idx.floor("D")).last()
    switches = int((day_labels != day_labels.shift(1)).sum() - 1) if not day_labels.empty else 0
    return meta_eq, day_labels, max(0, switches)


def rank_key(row: dict) -> tuple[float, float, float, float, float, float]:
    return (
        float(row["min_sortino"]),
        float(row["beats"]),
        float(row["mean_sortino"]),
        float(row["min_return_pct"]),
        float(row["mean_return_pct"]),
        -float(row["mean_dd_pct"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Regime-adaptive DOGE/AAVE pair search")
    parser.add_argument("--doge-candidate", action="append", required=True, help="NAME:SYMBOL:CHECKPOINT")
    parser.add_argument("--aave-candidate", action="append", required=True, help="NAME:SYMBOL:CHECKPOINT")
    parser.add_argument("--low-metrics", default="sortino,calmar")
    parser.add_argument("--high-metrics", default="calmar,sortino")
    parser.add_argument("--low-lookbacks", default="1,2,3")
    parser.add_argument("--high-lookbacks", default="1,2,3")
    parser.add_argument("--vol-lookbacks", default="2,3,5,7")
    parser.add_argument("--vol-thresholds", default="0.01,0.015,0.02,0.03")
    parser.add_argument("--cash-thresholds", default="0.0,0.01")
    parser.add_argument("--switch-margins", default="0.0,0.005,0.01")
    parser.add_argument("--min-score-gaps", default="0.0,0.002,0.005")
    parser.add_argument("--windows", default="30,60,90,120")
    parser.add_argument("--val-days", type=int, default=30)
    parser.add_argument("--max-leverage", type=float, default=2.0)
    parser.add_argument("--maker-fee", type=float, default=0.001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    doge_specs = args.doge_candidate
    aave_specs = args.aave_candidate
    low_metrics = [m.lower() for m in parse_csv_list(args.low_metrics)]
    high_metrics = [m.lower() for m in parse_csv_list(args.high_metrics)]
    low_lookbacks = parse_int_list(args.low_lookbacks)
    high_lookbacks = parse_int_list(args.high_lookbacks)
    vol_lookbacks = parse_int_list(args.vol_lookbacks)
    vol_thresholds = _parse_float_list(args.vol_thresholds)
    cash_thresholds = _parse_float_list(args.cash_thresholds)
    switch_margins = _parse_float_list(args.switch_margins)
    min_score_gaps = _parse_float_list(args.min_score_gaps)
    windows = parse_int_list(args.windows)

    candidates = [parse_candidate_spec(x) for x in (doge_specs + aave_specs)]
    max_window = max(windows)
    curves = {}
    logger.info("Loading {} unique candidates...", len(candidates))
    for c in candidates:
        logger.info("  loading {}", c.name)
        curves[c.name] = build_equity_series(
            c,
            val_days=args.val_days,
            test_days=max_window,
            maker_fee=args.maker_fee,
            max_leverage=args.max_leverage,
        )
    curves = align_equity_curves(curves)

    rows = []
    for dspec, aspec in itertools.product(doge_specs, aave_specs):
        d = parse_candidate_spec(dspec)
        a = parse_candidate_spec(aspec)
        for low_metric, high_metric, lb_low, lb_high, lb_vol, vth, ct, sm, mg in itertools.product(
            low_metrics,
            high_metrics,
            low_lookbacks,
            high_lookbacks,
            vol_lookbacks,
            vol_thresholds,
            cash_thresholds,
            switch_margins,
            min_score_gaps,
        ):
            window_rows = []
            for days in windows:
                sub = slice_window({d.name: curves[d.name], a.name: curves[a.name]}, days)
                eq, labels, switches = compose_regime_meta_equity(
                    sub,
                    low_metric=low_metric,
                    high_metric=high_metric,
                    low_lookback=lb_low,
                    high_lookback=lb_high,
                    vol_lookback=lb_vol,
                    vol_threshold=vth,
                    fallback_strategy=d.name,
                    tie_break_order=[d.name, a.name],
                    cash_threshold=ct,
                    switch_margin=sm,
                    min_score_gap=mg,
                )
                meta = compute_stats(eq.values, "meta")
                b1 = compute_stats(sub[d.name].values, d.name)
                b2 = compute_stats(sub[a.name].values, a.name)
                best = max((b1, b2), key=lambda x: (x["sortino"], x["total_return"], -x["max_dd"]))
                window_rows.append(
                    {
                        "days": int(days),
                        "meta": meta,
                        "best": best,
                        "switches": int(switches),
                        "winner_days": int(len(labels)),
                    }
                )

            s = [w["meta"]["sortino"] for w in window_rows]
            r = [w["meta"]["total_return"] for w in window_rows]
            dd = [w["meta"]["max_dd"] for w in window_rows]
            beats = sum(1 for w in window_rows if w["meta"]["sortino"] > w["best"]["sortino"])
            rows.append(
                {
                    "doge": d.name,
                    "aave": a.name,
                    "low_metric": low_metric,
                    "high_metric": high_metric,
                    "low_lookback_days": int(lb_low),
                    "high_lookback_days": int(lb_high),
                    "vol_lookback_days": int(lb_vol),
                    "vol_threshold": float(vth),
                    "cash_threshold": float(ct),
                    "switch_margin": float(sm),
                    "min_score_gap": float(mg),
                    "min_sortino": float(np.min(s)),
                    "mean_sortino": float(np.mean(s)),
                    "min_return_pct": float(np.min(r)),
                    "mean_return_pct": float(np.mean(r)),
                    "mean_dd_pct": float(np.mean(dd)),
                    "beats": int(beats),
                    "windows": window_rows,
                }
            )

    rows.sort(key=rank_key, reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2))
    logger.info("Saved {}", args.output)
    logger.info("Top 20:")
    for i, r in enumerate(rows[:20], 1):
        logger.info(
            "{:>2}. {} + {} | low={}({}) high={}({}) volLb={} vTh={} ct={} sm={} mg={} | minS={:.2f} meanS={:.2f} minR={:+.2f}% meanR={:+.2f}% meanDD={:.2f}% beats={}",
            i,
            r["doge"],
            r["aave"],
            r["low_metric"],
            r["low_lookback_days"],
            r["high_metric"],
            r["high_lookback_days"],
            r["vol_lookback_days"],
            r["vol_threshold"],
            r["cash_threshold"],
            r["switch_margin"],
            r["min_score_gap"],
            r["min_sortino"],
            r["mean_sortino"],
            r["min_return_pct"],
            r["mean_return_pct"],
            r["mean_dd_pct"],
            r["beats"],
        )


if __name__ == "__main__":
    main()
