#!/usr/bin/env python3
"""Search DOGE/AAVE meta-pair configurations over selector hyperparameters."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from loguru import logger

from binanceleveragesui.sim_meta_switcher import compute_stats
from binanceleveragesui.sweep_meta_daily_winners import (
    align_equity_curves,
    build_equity_series,
    compose_meta_equity_mode,
    parse_candidate_spec,
    parse_csv_list,
    parse_int_list,
    slice_window,
)


def _parse_float_list(raw: str) -> list[float]:
    return [float(x) for x in parse_csv_list(raw)]


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
    parser = argparse.ArgumentParser(description="Meta pair grid search")
    parser.add_argument("--doge-candidate", action="append", required=True, help="NAME:SYMBOL:CHECKPOINT")
    parser.add_argument("--aave-candidate", action="append", required=True, help="NAME:SYMBOL:CHECKPOINT")
    parser.add_argument("--mode", default="winner_cash")
    parser.add_argument("--metrics", default="calmar,sortino")
    parser.add_argument("--lookbacks", default="1,2,3")
    parser.add_argument("--cash-thresholds", default="0.0,0.01")
    parser.add_argument("--switch-margins", default="0.0,0.005,0.01")
    parser.add_argument("--min-score-gaps", default="0.0")
    parser.add_argument("--windows", default="30,60,90,120")
    parser.add_argument("--val-days", type=int, default=30)
    parser.add_argument("--max-leverage", type=float, default=2.0)
    parser.add_argument("--maker-fee", type=float, default=0.001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    doge_specs = args.doge_candidate
    aave_specs = args.aave_candidate
    metrics = [m.lower() for m in parse_csv_list(args.metrics)]
    lookbacks = parse_int_list(args.lookbacks)
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
        min_gap_grid = [0.0] if args.mode in ("blend_top2", "softmax_all") else min_score_gaps
        for metric, lb, ct, sm, mg in itertools.product(metrics, lookbacks, cash_thresholds, switch_margins, min_gap_grid):
            if args.mode == "winner" and ct != 0.0:
                continue
            window_rows = []
            for days in windows:
                sub = slice_window({d.name: curves[d.name], a.name: curves[a.name]}, days)
                eq, winners, switches = compose_meta_equity_mode(
                    sub,
                    lookback_days=lb,
                    metric=metric,
                    fallback_strategy=d.name,
                    tie_break_order=[d.name, a.name],
                    mode=args.mode,
                    cash_threshold=ct,
                    switch_margin=sm,
                    softmax_temperature=1.0,
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
                        "winner_days": int(len(winners)),
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
                    "mode": args.mode,
                    "metric": metric,
                    "lookback_days": int(lb),
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
            "{:>2}. {} + {} | {} lb={} ct={} sm={} mg={} | minS={:.2f} meanS={:.2f} minR={:+.2f}% meanR={:+.2f}% meanDD={:.2f}% beats={}",
            i,
            r["doge"],
            r["aave"],
            r["metric"],
            r["lookback_days"],
            r["cash_threshold"],
            r["switch_margin"],
            r.get("min_score_gap", 0.0),
            r["min_sortino"],
            r["mean_sortino"],
            r["min_return_pct"],
            r["mean_return_pct"],
            r["mean_dd_pct"],
            r["beats"],
        )


if __name__ == "__main__":
    main()
