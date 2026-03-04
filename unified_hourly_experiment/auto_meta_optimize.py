#!/usr/bin/env python3
"""Autonomous meta-selector optimization runner for stock portfolio strategies."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class MetaRun:
    edge: float
    sit_out_threshold: float
    selection_mode: str
    switch_margin: float
    min_score_gap: float
    output_path: Path


def parse_csv_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in parse_csv_list(value)]


def rank_key(summary: dict) -> tuple[float, float, float, float, float]:
    return (
        float(summary["min_sortino"]),
        float(summary["mean_sortino"]),
        float(summary["min_return_pct"]),
        float(summary["mean_return_pct"]),
        -float(summary["mean_max_drawdown_pct"]),
    )


def eligible_summary(summary: dict, *, min_num_buys: int) -> bool:
    return int(summary.get("min_num_buys", 0)) >= int(min_num_buys)


def _float_token(value: float) -> str:
    return str(value).replace(".", "p")


def run_once(args: argparse.Namespace) -> dict:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (Path("experiments") / f"auto_meta_opt_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = parse_csv_list(args.metrics)
    lookbacks = parse_csv_list(args.lookback_days)
    holdouts = parse_csv_list(args.holdout_days)
    edges = parse_float_list(args.min_edges)
    thresholds = parse_float_list(args.sit_out_thresholds)
    selection_modes = [x.lower() for x in parse_csv_list(args.selection_modes)]
    switch_margins = parse_float_list(args.switch_margins)
    min_score_gaps = parse_float_list(args.min_score_gaps)

    runs: list[MetaRun] = []
    for edge in edges:
        for threshold in thresholds:
            for mode in selection_modes:
                for switch_margin in switch_margins:
                    for min_score_gap in min_score_gaps:
                        edge_token = _float_token(edge)
                        th_token = _float_token(threshold)
                        sm_token = _float_token(switch_margin)
                        mg_token = _float_token(min_score_gap)
                        output_path = out_dir / f"meta_edge{edge_token}_th{th_token}_m{mode}_sm{sm_token}_mg{mg_token}.json"
                        runs.append(
                            MetaRun(
                                edge=edge,
                                sit_out_threshold=threshold,
                                selection_mode=mode,
                                switch_margin=switch_margin,
                                min_score_gap=min_score_gap,
                                output_path=output_path,
                            )
                        )

    for idx, run in enumerate(runs, start=1):
        strategy_args = [item for spec in args.strategy for item in ("--strategy", spec)]
        cmd = [
            sys.executable,
            "unified_hourly_experiment/sweep_meta_portfolio.py",
            *strategy_args,
            "--symbols",
            args.symbols,
            "--metrics",
            ",".join(metrics),
            "--selection-modes",
            run.selection_mode,
            "--switch-margins",
            str(run.switch_margin),
            "--min-score-gaps",
            str(run.min_score_gap),
            "--lookback-days",
            ",".join(lookbacks),
            "--holdout-days",
            ",".join(holdouts),
            "--max-positions",
            str(args.max_positions),
            "--min-edge",
            str(run.edge),
            "--max-hold-hours",
            str(args.max_hold_hours),
            "--decision-lag-bars",
            str(args.decision_lag_bars),
            "--bar-margin",
            str(args.bar_margin),
            "--leverage",
            str(args.leverage),
            "--fee-rate",
            str(args.fee_rate),
            "--margin-rate",
            str(args.margin_rate),
            "--sim-backend",
            str(args.sim_backend),
            "--sit-out-if-negative",
            "--sit-out-threshold",
            str(run.sit_out_threshold),
            "--output",
            str(run.output_path),
        ]
        print(
            f"[{idx}/{len(runs)}] edge={run.edge} threshold={run.sit_out_threshold} "
            f"mode={run.selection_mode} switch_margin={run.switch_margin} min_gap={run.min_score_gap} "
            f"-> {run.output_path}"
        )
        subprocess.run(cmd, check=True)

    ranked_rows = []
    skipped_for_activity = 0
    for run in runs:
        payload = json.loads(run.output_path.read_text())
        best = payload["best"]
        if not eligible_summary(best, min_num_buys=args.min_num_buys):
            skipped_for_activity += 1
            continue
        ranked_rows.append(
            {
                "edge": run.edge,
                "sit_out_threshold": run.sit_out_threshold,
                "selection_mode": run.selection_mode,
                "switch_margin": run.switch_margin,
                "min_score_gap": run.min_score_gap,
                "output": str(run.output_path),
                **best,
            }
        )

    if not ranked_rows:
        raise RuntimeError(
            "No eligible sweep rows after applying activity filter "
            f"(min_num_buys={args.min_num_buys}, skipped={skipped_for_activity})."
        )

    ranked_rows.sort(key=rank_key, reverse=True)
    best = ranked_rows[0]

    recommendation = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "symbols": args.symbols,
        "strategies": args.strategy,
        "search_space": {
            "metrics": metrics,
            "lookback_days": lookbacks,
            "holdout_days": holdouts,
            "min_edges": edges,
            "sit_out_thresholds": thresholds,
            "selection_modes": selection_modes,
            "switch_margins": switch_margins,
            "min_score_gaps": min_score_gaps,
            "min_num_buys": int(args.min_num_buys),
        },
        "skipped_for_activity": int(skipped_for_activity),
        "best": best,
        "top5": ranked_rows[:5],
        "deploy_command": (
            "python unified_hourly_experiment/trade_unified_hourly_meta.py "
            + " ".join(f"--strategy {spec}" for spec in args.strategy)
            + f" --stock-symbols {args.symbols}"
            + f" --min-edge {best['edge']}"
            + f" --max-hold-hours {args.max_hold_hours}"
            + f" --max-positions {args.max_positions}"
            + f" --meta-metric {best['metric']}"
            + f" --meta-lookback-days {best['lookback_days']}"
            + f" --meta-selection-mode {best['selection_mode']}"
            + f" --meta-switch-margin {best['switch_margin']}"
            + f" --meta-min-score-gap {best['min_score_gap']}"
            + " --meta-history-days 120 --sit-out-if-negative"
            + f" --sit-out-threshold {best['sit_out_threshold']}"
            + f" --bar-margin {args.bar_margin}"
            + f" --fee-rate {args.fee_rate}"
            + f" --margin-rate {args.margin_rate}"
            + f" --sim-backend {args.sim_backend}"
            + " --live --loop"
        ),
    }

    summary_path = out_dir / "auto_meta_recommendation.json"
    summary_path.write_text(json.dumps(recommendation, indent=2))
    print(f"Saved recommendation -> {summary_path}")
    print(f"BEST: {best}")
    return recommendation


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous meta-selector sweep orchestrator.")
    parser.add_argument("--strategy", action="append", required=True, help="Repeatable NAME=PATH[:EPOCH] strategy spec")
    parser.add_argument("--symbols", default="NVDA,PLTR,GOOG,DBX,TRIP,MTCH")
    parser.add_argument("--metrics", default="sharpe,sortino,calmar")
    parser.add_argument("--lookback-days", default="5,7,10,14")
    parser.add_argument("--holdout-days", default="30,60,90")
    parser.add_argument("--min-edges", default="0.004,0.005,0.006,0.007,0.008")
    parser.add_argument("--sit-out-thresholds", default="0.2,0.3,0.4,0.5,0.7")
    parser.add_argument("--selection-modes", default="winner")
    parser.add_argument("--switch-margins", default="0.0")
    parser.add_argument("--min-score-gaps", default="0.0")
    parser.add_argument(
        "--min-num-buys",
        type=int,
        default=0,
        help="Require best row from each sweep run to have at least this many buys in every holdout period.",
    )
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-hold-hours", type=int, default=5)
    parser.add_argument("--decision-lag-bars", type=int, default=1)
    parser.add_argument("--bar-margin", type=float, default=0.0013)
    parser.add_argument("--leverage", type=float, default=2.0)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--margin-rate", type=float, default=0.0625)
    parser.add_argument(
        "--sim-backend",
        type=str,
        default="auto",
        choices=["python", "native", "auto"],
        help="Portfolio simulator backend for sweep_meta_portfolio calls.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if len(args.strategy) < 2:
        raise ValueError("Need at least two --strategy specs.")
    modes = [x.lower() for x in parse_csv_list(args.selection_modes)]
    invalid_modes = [x for x in modes if x not in ("winner", "winner_cash", "sticky")]
    if invalid_modes:
        raise ValueError(f"Invalid selection mode(s): {invalid_modes}")
    if any(x < 0 for x in parse_float_list(args.switch_margins)):
        raise ValueError("--switch-margins values must all be >= 0")
    if any(x < 0 for x in parse_float_list(args.min_score_gaps)):
        raise ValueError("--min-score-gaps values must all be >= 0")

    run_once(args)


if __name__ == "__main__":
    main()
