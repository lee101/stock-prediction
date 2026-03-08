#!/usr/bin/env python3
"""Two-layer selector search for DOGE/AAVE meta policies.

Layer 1 builds a pool of base winner_cash selectors over DOGE/AAVE.
Layer 2 runs a daily winner_cash selector over those base selectors.
"""

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


def _profile_name(metric: str, lookback: int, cash_threshold: float, switch_margin: float, min_score_gap: float) -> str:
    return (
        f"p_{metric}_lb{int(lookback)}"
        f"_ct{cash_threshold:.4f}"
        f"_sm{switch_margin:.4f}"
        f"_mg{min_score_gap:.4f}"
    )


def rank_key(row: dict) -> tuple[float, float, float, float, float, float]:
    return (
        float(row["min_sortino"]),
        float(row["beats"]),
        float(row["mean_sortino"]),
        float(row["min_return_pct"]),
        float(row["mean_return_pct"]),
        -float(row["mean_dd_pct"]),
    )


def _build_base_profile_curves(
    pair_curves: dict[str, object],
    *,
    fallback_strategy: str,
    tie_break_order: list[str],
    base_metrics: list[str],
    base_lookbacks: list[int],
    base_cash_thresholds: list[float],
    base_switch_margins: list[float],
    base_min_score_gaps: list[float],
) -> dict[str, object]:
    out: dict[str, object] = {}
    for metric, lb, ct, sm, mg in itertools.product(
        base_metrics,
        base_lookbacks,
        base_cash_thresholds,
        base_switch_margins,
        base_min_score_gaps,
    ):
        name = _profile_name(metric, lb, ct, sm, mg)
        eq, _, _ = compose_meta_equity_mode(
            pair_curves,
            lookback_days=int(lb),
            metric=metric,
            fallback_strategy=fallback_strategy,
            tie_break_order=tie_break_order,
            mode="winner_cash",
            cash_threshold=float(ct),
            switch_margin=float(sm),
            softmax_temperature=1.0,
            min_score_gap=float(mg),
        )
        out[name] = eq
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-layer DOGE/AAVE selector-of-selectors search")
    parser.add_argument("--doge-candidate", required=True, help="NAME:SYMBOL:CHECKPOINT")
    parser.add_argument("--aave-candidate", required=True, help="NAME:SYMBOL:CHECKPOINT")

    parser.add_argument("--base-metrics", default="calmar,sortino")
    parser.add_argument("--base-lookbacks", default="1,2,3")
    parser.add_argument("--base-cash-thresholds", default="0.0,0.01")
    parser.add_argument("--base-switch-margins", default="0.002,0.005,0.01")
    parser.add_argument("--base-min-score-gaps", default="0.0,0.002")

    parser.add_argument("--meta-mode", default="winner_cash")
    parser.add_argument("--meta-metrics", default="calmar,sortino")
    parser.add_argument("--meta-lookbacks", default="1,2,3,5")
    parser.add_argument("--meta-cash-thresholds", default="0.0,0.01")
    parser.add_argument("--meta-switch-margins", default="0.0,0.005")
    parser.add_argument("--meta-min-score-gaps", default="0.0,0.002")

    parser.add_argument("--windows", default="30,60,90,120")
    parser.add_argument("--val-days", type=int, default=30)
    parser.add_argument("--max-leverage", type=float, default=2.30)
    parser.add_argument("--maker-fee", type=float, default=0.001)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    doge = parse_candidate_spec(args.doge_candidate)
    aave = parse_candidate_spec(args.aave_candidate)

    base_metrics = [x.lower() for x in parse_csv_list(args.base_metrics)]
    base_lookbacks = parse_int_list(args.base_lookbacks)
    base_cash_thresholds = _parse_float_list(args.base_cash_thresholds)
    base_switch_margins = _parse_float_list(args.base_switch_margins)
    base_min_score_gaps = _parse_float_list(args.base_min_score_gaps)

    meta_metrics = [x.lower() for x in parse_csv_list(args.meta_metrics)]
    meta_lookbacks = parse_int_list(args.meta_lookbacks)
    meta_cash_thresholds = _parse_float_list(args.meta_cash_thresholds)
    meta_switch_margins = _parse_float_list(args.meta_switch_margins)
    meta_min_score_gaps = _parse_float_list(args.meta_min_score_gaps)
    windows = parse_int_list(args.windows)

    logger.info("Loading base pair curves for {} + {} ...", doge.name, aave.name)
    max_window = max(windows)
    curves = {
        doge.name: build_equity_series(
            doge,
            val_days=args.val_days,
            test_days=max_window,
            maker_fee=args.maker_fee,
            max_leverage=args.max_leverage,
        ),
        aave.name: build_equity_series(
            aave,
            val_days=args.val_days,
            test_days=max_window,
            maker_fee=args.maker_fee,
            max_leverage=args.max_leverage,
        ),
    }
    curves = align_equity_curves(curves)

    rows_by_key: dict[tuple, dict] = {}
    for days in windows:
        logger.info("Window {}d: building layer-1 selector pool ...", days)
        pair_sub = slice_window(curves, days)
        base_curves = _build_base_profile_curves(
            pair_sub,
            fallback_strategy=doge.name,
            tie_break_order=[doge.name, aave.name],
            base_metrics=base_metrics,
            base_lookbacks=base_lookbacks,
            base_cash_thresholds=base_cash_thresholds,
            base_switch_margins=base_switch_margins,
            base_min_score_gaps=base_min_score_gaps,
        )
        base_names = sorted(base_curves.keys())
        if not base_names:
            raise ValueError("No base profiles produced.")

        preferred_fallback = _profile_name("calmar", 1, 0.0, 0.005, 0.0)
        fallback_profile = preferred_fallback if preferred_fallback in base_curves else base_names[0]

        for metric, lb, ct, sm, mg in itertools.product(
            meta_metrics,
            meta_lookbacks,
            meta_cash_thresholds,
            meta_switch_margins,
            meta_min_score_gaps,
        ):
            if args.meta_mode == "winner" and ct != 0.0:
                continue
            eq, labels, switches = compose_meta_equity_mode(
                base_curves,
                lookback_days=int(lb),
                metric=metric,
                fallback_strategy=fallback_profile,
                tie_break_order=base_names,
                mode=args.meta_mode,
                cash_threshold=float(ct),
                switch_margin=float(sm),
                softmax_temperature=1.0,
                min_score_gap=float(mg),
            )
            meta = compute_stats(eq.values, "stacked_meta")
            b1 = compute_stats(pair_sub[doge.name].values, doge.name)
            b2 = compute_stats(pair_sub[aave.name].values, aave.name)
            best = max((b1, b2), key=lambda x: (x["sortino"], x["total_return"], -x["max_dd"]))

            key = (metric, int(lb), float(ct), float(sm), float(mg))
            row = rows_by_key.get(key)
            if row is None:
                row = {
                    "doge": doge.name,
                    "aave": aave.name,
                    "meta_mode": args.meta_mode,
                    "meta_metric": metric,
                    "meta_lookback_days": int(lb),
                    "meta_cash_threshold": float(ct),
                    "meta_switch_margin": float(sm),
                    "meta_min_score_gap": float(mg),
                    "base_profile_count": int(len(base_curves)),
                    "base_profile_space": {
                        "metrics": base_metrics,
                        "lookbacks": base_lookbacks,
                        "cash_thresholds": base_cash_thresholds,
                        "switch_margins": base_switch_margins,
                        "min_score_gaps": base_min_score_gaps,
                    },
                    "windows": [],
                }
                rows_by_key[key] = row

            row["windows"].append(
                {
                    "days": int(days),
                    "meta": meta,
                    "best": best,
                    "switches": int(switches),
                    "winner_days": int(len(labels)),
                }
            )

    rows = list(rows_by_key.values())
    for row in rows:
        wins = sorted(row["windows"], key=lambda x: x["days"])
        s = [w["meta"]["sortino"] for w in wins]
        r = [w["meta"]["total_return"] for w in wins]
        dd = [w["meta"]["max_dd"] for w in wins]
        beats = sum(1 for w in wins if w["meta"]["sortino"] > w["best"]["sortino"])
        row["windows"] = wins
        row["min_sortino"] = float(np.min(s))
        row["mean_sortino"] = float(np.mean(s))
        row["min_return_pct"] = float(np.min(r))
        row["mean_return_pct"] = float(np.mean(r))
        row["mean_dd_pct"] = float(np.mean(dd))
        row["beats"] = int(beats)

    rows.sort(key=rank_key, reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2))

    logger.info("Saved {}", args.output)
    logger.info("Top {}:", min(args.top_k, len(rows)))
    for i, row in enumerate(rows[: args.top_k], 1):
        logger.info(
            "{:>2}. {} lb={} ct={} sm={} mg={} | minS={:.2f} meanS={:.2f} minR={:+.2f}% meanR={:+.2f}% meanDD={:.2f}% beats={} baseN={}",
            i,
            row["meta_metric"],
            row["meta_lookback_days"],
            row["meta_cash_threshold"],
            row["meta_switch_margin"],
            row["meta_min_score_gap"],
            row["min_sortino"],
            row["mean_sortino"],
            row["min_return_pct"],
            row["mean_return_pct"],
            row["mean_dd_pct"],
            row["beats"],
            row["base_profile_count"],
        )


if __name__ == "__main__":
    main()
