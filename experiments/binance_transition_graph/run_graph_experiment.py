#!/usr/bin/env python3
"""Run the Binance transition-graph experiment.

The default command optimizes candidate graph policies on the pre-holdout tail
and evaluates the best candidates on the final unseen window across
0/5/10/20 bps slippage.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.binance_transition_graph.graph_policy import (
    DEFAULT_SYMBOLS,
    baseline_configs,
    evaluate_slippage_sweep,
    load_hourly_panel,
    optimize_on_train,
    sample_configs,
    summarize_sweep_rows,
    train_eval_split,
    write_result_json,
)


def _markdown_report(*, output_json: Path, output_csv: Path, rows, train_results) -> str:
    best = rows[0] if rows else None
    lines = [
        "# Binance Transition Graph Experiment",
        "",
        f"- JSON: `{output_json}`",
        f"- CSV: `{output_csv}`",
        "- Simulator: sequential hourly graph policy, decision lag 2, fees plus slippage, optional leverage/shorts.",
        "- Promotion target: worst slippage monthly return >= 27% and max drawdown <= 20%.",
        "",
    ]
    if best is not None:
        lines.extend(
            [
                "## Best Holdout Candidate",
                "",
                f"- Config: `{best.config.name}`",
                f"- Worst slippage monthly return: `{best.worst_monthly_return * 100:+.2f}%`",
                f"- Worst slippage max drawdown: `{best.worst_max_drawdown * 100:.2f}%`",
                "",
            ]
        )
    lines.extend(["## Train Leaders", ""])
    for result in train_results[:8]:
        lines.append(
            "- "
            f"`{result.config.name}` train monthly={result.metrics.monthly_return * 100:+.2f}% "
            f"dd={result.metrics.max_drawdown * 100:.2f}% "
            f"sortino={result.metrics.sortino:.2f} trades={result.metrics.num_trades}"
        )
    lines.extend(["", "## Holdout Slippage Sweep", ""])
    if rows:
        df = summarize_sweep_rows(rows)
        cols = ["name", "worst_monthly_pct", "worst_max_dd_pct", "slip20_monthly_pct", "slip20_dd_pct"]
        lines.append("| name | worst_monthly_pct | worst_max_dd_pct | slip20_monthly_pct | slip20_dd_pct |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in df[cols].to_dict(orient="records"):
            lines.append(
                f"| {row['name']} "
                f"| {float(row['worst_monthly_pct']):.2f} "
                f"| {float(row['worst_max_dd_pct']):.2f} "
                f"| {float(row['slip20_monthly_pct']):.2f} "
                f"| {float(row['slip20_dd_pct']):.2f} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="trainingdatahourlybinance")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--train-hours", type=int, default=24 * 90)
    parser.add_argument("--eval-hours", type=int, default=24 * 30)
    parser.add_argument("--trials", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-leverage", type=float, default=2.0)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--eval-warmup-hours", type=int, default=240)
    parser.add_argument("--output-dir", default="experiments/binance_transition_graph/results")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_json = output_dir / f"graph_experiment_{stamp}.json"
    output_csv = output_dir / f"graph_experiment_{stamp}.csv"
    output_md = output_dir / f"graph_experiment_{stamp}.md"

    eval_warmup_hours = min(int(args.eval_warmup_hours), int(args.train_hours))
    total_tail = int(args.train_hours) + int(args.eval_hours) + eval_warmup_hours
    frames = load_hourly_panel(args.data_root, args.symbols, tail_hours=total_tail)
    train_frames, eval_frames = train_eval_split(
        frames,
        train_hours=int(args.train_hours),
        eval_hours=int(args.eval_hours),
        eval_warmup_hours=eval_warmup_hours,
    )
    configs = baseline_configs(max_leverage=float(args.max_leverage), allow_short=bool(args.allow_short))
    configs.extend(
        sample_configs(
            n=int(args.trials),
            seed=int(args.seed),
            max_leverage=float(args.max_leverage),
            allow_short=bool(args.allow_short),
        )
    )
    train_results = optimize_on_train(train_frames, configs, top_k=int(args.top_k))
    holdout_rows = [
        evaluate_slippage_sweep(eval_frames, result.config, warmup_rows=eval_warmup_hours)
        for result in train_results
    ]
    holdout_rows.sort(
        key=lambda row: (
            row.worst_monthly_return - 0.5 * max(0.0, row.worst_max_drawdown - 0.20),
            -row.worst_max_drawdown,
        ),
        reverse=True,
    )

    df = summarize_sweep_rows(holdout_rows)
    df.to_csv(output_csv, index=False)
    payload = {
        "data_root": str(args.data_root),
        "symbols": list(args.symbols),
        "train_hours": int(args.train_hours),
        "eval_hours": int(args.eval_hours),
        "trials": int(args.trials),
        "top_k": int(args.top_k),
        "seed": int(args.seed),
        "max_leverage": float(args.max_leverage),
        "allow_short": bool(args.allow_short),
        "eval_warmup_hours": eval_warmup_hours,
        "train_results": [
            {
                "config": asdict(result.config),
                "metrics": asdict(result.metrics),
                "state_counts": result.state_counts,
            }
            for result in train_results
        ],
        "holdout_slippage": [row.to_json_dict() for row in holdout_rows],
    }
    write_result_json(output_json, payload)
    output_md.write_text(
        _markdown_report(
            output_json=output_json,
            output_csv=output_csv,
            rows=holdout_rows,
            train_results=train_results,
        ),
        encoding="utf-8",
    )

    if holdout_rows:
        best = holdout_rows[0]
        print(
            f"best={best.config.name} "
            f"worst_monthly={best.worst_monthly_return * 100:+.2f}% "
            f"worst_dd={best.worst_max_drawdown * 100:.2f}%"
        )
    print(f"wrote {output_json}")
    print(f"wrote {output_csv}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
