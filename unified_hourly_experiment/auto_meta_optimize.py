#!/usr/bin/env python3
"""Autonomous meta-selector optimization runner for stock portfolio strategies."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.trade_directions import DEFAULT_ALPACA_LIVE8_STOCKS


@dataclass
class MetaRun:
    edge: float
    selection_mode: str
    switch_margin: float
    min_score_gap: float
    recency_halflife_days: float
    trade_amount_scale: float
    min_buy_amount: float
    entry_intensity_power: float
    entry_min_intensity_fraction: float
    long_intensity_multiplier: float
    short_intensity_multiplier: float
    entry_allocator_mode: str
    entry_allocator_edge_power: float
    entry_allocator_max_single_position_fraction: float
    entry_allocator_reserve_fraction: float
    output_path: Path


def parse_csv_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in parse_csv_list(value)]


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in parse_csv_list(value)]


def rank_key(summary: dict) -> tuple[float, float, float, float, float]:
    return (
        float(summary.get("min_goodness_score", summary["min_sortino"])),
        float(summary.get("mean_goodness_score", summary["mean_sortino"])),
        float(summary["min_sortino"]),
        float(summary["mean_sortino"]),
        float(summary["min_return_pct"]),
    )


def eligible_summary(summary: dict, *, min_num_buys: int) -> bool:
    return int(summary.get("min_num_buys", 0)) >= int(min_num_buys)


def build_entry_allocator_mode_summary(ranked_rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in ranked_rows:
        grouped[str(row.get("entry_allocator_mode", "unknown"))].append(row)

    summaries: list[dict] = []
    for mode, rows in grouped.items():
        ordered_rows = sorted(rows, key=rank_key, reverse=True)
        best = ordered_rows[0]
        count = len(ordered_rows)
        summaries.append(
            {
                "entry_allocator_mode": mode,
                "count": count,
                "best": {
                    "metric": best["metric"],
                    "lookback_days": best["lookback_days"],
                    "selection_mode": best["selection_mode"],
                    "switch_margin": best["switch_margin"],
                    "min_score_gap": best["min_score_gap"],
                    "recency_halflife_days": best["recency_halflife_days"],
                    "sit_out_threshold": best["sit_out_threshold"],
                    "trade_amount_scale": best["trade_amount_scale"],
                    "min_buy_amount": best["min_buy_amount"],
                    "entry_intensity_power": best["entry_intensity_power"],
                    "entry_min_intensity_fraction": best["entry_min_intensity_fraction"],
                    "long_intensity_multiplier": best["long_intensity_multiplier"],
                    "short_intensity_multiplier": best["short_intensity_multiplier"],
                    "entry_allocator_edge_power": best["entry_allocator_edge_power"],
                    "entry_allocator_max_single_position_fraction": best[
                        "entry_allocator_max_single_position_fraction"
                    ],
                    "entry_allocator_reserve_fraction": best["entry_allocator_reserve_fraction"],
                    "min_sortino": best["min_sortino"],
                    "mean_sortino": best["mean_sortino"],
                    "min_return_pct": best["min_return_pct"],
                    "mean_return_pct": best["mean_return_pct"],
                    "min_num_buys": best.get("min_num_buys", 0),
                    "output": best["output"],
                },
                "mean_min_sortino": float(sum(float(row["min_sortino"]) for row in ordered_rows) / count),
                "mean_mean_sortino": float(sum(float(row["mean_sortino"]) for row in ordered_rows) / count),
                "mean_min_return_pct": float(sum(float(row["min_return_pct"]) for row in ordered_rows) / count),
                "mean_mean_return_pct": float(sum(float(row["mean_return_pct"]) for row in ordered_rows) / count),
                "mean_min_num_buys": float(sum(float(row.get("min_num_buys", 0)) for row in ordered_rows) / count),
            }
        )

    summaries.sort(key=lambda row: rank_key(row["best"]), reverse=True)
    return summaries


def iter_entry_allocator_grid(
    *,
    mode: str,
    edge_powers: list[float],
    max_single_position_fractions: list[float],
    reserve_fractions: list[float],
) -> list[tuple[float, float, float]]:
    if mode == "legacy":
        return [
            (
                float(edge_powers[0]),
                float(max_single_position_fractions[0]),
                float(reserve_fractions[0]),
            )
        ]
    return [
        (float(edge_power), float(max_single_position_fraction), float(reserve_fraction))
        for edge_power in edge_powers
        for max_single_position_fraction in max_single_position_fractions
        for reserve_fraction in reserve_fractions
    ]


def _float_token(value: float) -> str:
    return str(value).replace(".", "p")


def _threshold_token(thresholds: list[float]) -> str:
    if len(thresholds) == 1:
        return f"th{_float_token(thresholds[0])}"
    return "thm" + "_".join(_float_token(v) for v in thresholds)


def build_deploy_command(
    *,
    strategy_specs: list[str],
    symbols: str,
    decision_lag_bars: int,
    entry_selection_mode: str,
    max_hold_hours: int,
    max_positions: int,
    bar_margin: float,
    entry_order_ttl_hours: int,
    fee_rate: float,
    margin_rate: float,
    market_order_entry: bool,
    best: dict,
) -> str:
    recency_halflife = best.get("recency_halflife_days", 0.0)
    if recency_halflife is None:
        recency_halflife = 0.0
    cmd = (
        "python unified_hourly_experiment/trade_unified_hourly_meta.py "
        + " ".join(f"--strategy {spec}" for spec in strategy_specs)
        + f" --stock-symbols {symbols}"
        + f" --min-edge {best['edge']}"
        + f" --max-hold-hours {max_hold_hours}"
        + f" --max-positions {max_positions}"
        + f" --trade-amount-scale {best['trade_amount_scale']}"
        + f" --min-buy-amount {best['min_buy_amount']}"
        + f" --entry-intensity-power {best['entry_intensity_power']}"
        + f" --entry-min-intensity-fraction {best['entry_min_intensity_fraction']}"
        + f" --long-intensity-multiplier {best['long_intensity_multiplier']}"
        + f" --short-intensity-multiplier {best['short_intensity_multiplier']}"
        + f" --entry-allocator-mode {best['entry_allocator_mode']}"
        + f" --entry-allocator-edge-power {best['entry_allocator_edge_power']}"
        + f" --entry-allocator-max-single-position-fraction {best['entry_allocator_max_single_position_fraction']}"
        + f" --entry-allocator-reserve-fraction {best['entry_allocator_reserve_fraction']}"
        + f" --meta-metric {best['metric']}"
        + f" --meta-lookback-days {best['lookback_days']}"
        + f" --meta-selection-mode {best['selection_mode']}"
        + f" --meta-switch-margin {best['switch_margin']}"
        + f" --meta-min-score-gap {best['min_score_gap']}"
        + f" --meta-recency-halflife-days {recency_halflife}"
        + " --meta-history-days 120 --sit-out-if-negative"
        + f" --sit-out-threshold {best['sit_out_threshold']}"
        + f" --decision-lag-bars {int(decision_lag_bars)}"
        + f" --entry-selection-mode {entry_selection_mode}"
        + f" --bar-margin {bar_margin}"
        + f" --entry-order-ttl-hours {int(entry_order_ttl_hours)}"
        + f" --fee-rate {fee_rate}"
        + f" --margin-rate {margin_rate}"
        + " --live --loop"
    )
    if market_order_entry:
        cmd += " --market-order-entry"
    return cmd


def run_once(args: argparse.Namespace) -> dict:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (Path("experiments") / f"auto_meta_opt_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    action_cache_dir = args.action_cache_dir or (out_dir / "action_cache")
    action_cache_dir.mkdir(parents=True, exist_ok=True)

    metrics = parse_csv_list(args.metrics)
    lookbacks = parse_csv_list(args.lookback_days)
    holdouts = parse_csv_list(args.holdout_days)
    edges = parse_float_list(args.min_edges)
    thresholds = parse_float_list(args.sit_out_thresholds)
    selection_modes = [x.lower() for x in parse_csv_list(args.selection_modes)]
    switch_margins = parse_float_list(args.switch_margins)
    min_score_gaps = parse_float_list(args.min_score_gaps)
    recency_halflife_days = parse_float_list(args.recency_halflife_days)
    trade_amount_scales = parse_float_list(args.trade_amount_scales)
    min_buy_amounts = parse_float_list(args.min_buy_amounts)
    entry_intensity_powers = parse_float_list(args.entry_intensity_powers)
    entry_min_intensity_fractions = parse_float_list(args.entry_min_intensity_fractions)
    long_intensity_multipliers = parse_float_list(args.long_intensity_multipliers)
    short_intensity_multipliers = parse_float_list(args.short_intensity_multipliers)
    entry_allocator_modes = [x.lower() for x in parse_csv_list(args.entry_allocator_modes)]
    entry_allocator_edge_powers = parse_float_list(args.entry_allocator_edge_powers)
    entry_allocator_max_single_position_fractions = parse_float_list(
        args.entry_allocator_max_single_position_fractions
    )
    entry_allocator_reserve_fractions = parse_float_list(args.entry_allocator_reserve_fractions)

    if any(x < 0 for x in recency_halflife_days):
        raise ValueError(f"recency-halflife-days must all be >= 0, got {recency_halflife_days}")
    invalid_allocator_modes = [x for x in entry_allocator_modes if x not in {"legacy", "concentrated"}]
    if invalid_allocator_modes:
        raise ValueError(f"entry-allocator-modes must be legacy/concentrated, got {invalid_allocator_modes}")
    if any(x < 0 for x in entry_allocator_edge_powers):
        raise ValueError(f"entry-allocator-edge-powers must all be >= 0, got {entry_allocator_edge_powers}")
    if any(x < 0 or x > 1 for x in entry_allocator_max_single_position_fractions):
        raise ValueError(
            "entry-allocator-max-single-position-fractions must all be in [0, 1], "
            f"got {entry_allocator_max_single_position_fractions}"
        )
    if any(x < 0 or x > 1 for x in entry_allocator_reserve_fractions):
        raise ValueError(
            f"entry-allocator-reserve-fractions must all be in [0, 1], got {entry_allocator_reserve_fractions}"
        )

    runs: list[MetaRun] = []
    for edge in edges:
        for mode in selection_modes:
            for switch_margin in switch_margins:
                for min_score_gap in min_score_gaps:
                    for recency_halflife in recency_halflife_days:
                        for trade_amount_scale in trade_amount_scales:
                            for min_buy_amount in min_buy_amounts:
                                for entry_intensity_power in entry_intensity_powers:
                                    for entry_min_intensity_fraction in entry_min_intensity_fractions:
                                        for long_intensity_multiplier in long_intensity_multipliers:
                                            for short_intensity_multiplier in short_intensity_multipliers:
                                                for entry_allocator_mode in entry_allocator_modes:
                                                    allocator_grid = iter_entry_allocator_grid(
                                                        mode=entry_allocator_mode,
                                                        edge_powers=entry_allocator_edge_powers,
                                                        max_single_position_fractions=entry_allocator_max_single_position_fractions,
                                                        reserve_fractions=entry_allocator_reserve_fractions,
                                                    )
                                                    for (
                                                        entry_allocator_edge_power,
                                                        entry_allocator_max_single_position_fraction,
                                                        entry_allocator_reserve_fraction,
                                                    ) in allocator_grid:
                                                                edge_token = _float_token(edge)
                                                                th_token = _threshold_token(thresholds)
                                                                sm_token = _float_token(switch_margin)
                                                                mg_token = _float_token(min_score_gap)
                                                                hl_token = _float_token(recency_halflife)
                                                                tas_token = _float_token(trade_amount_scale)
                                                                mba_token = _float_token(min_buy_amount)
                                                                eip_token = _float_token(entry_intensity_power)
                                                                emif_token = _float_token(entry_min_intensity_fraction)
                                                                lim_token = _float_token(long_intensity_multiplier)
                                                                sim_token = _float_token(short_intensity_multiplier)
                                                                eam_token = str(entry_allocator_mode)
                                                                eaep_token = _float_token(entry_allocator_edge_power)
                                                                easp_token = _float_token(entry_allocator_max_single_position_fraction)
                                                                earm_token = _float_token(entry_allocator_reserve_fraction)
                                                                output_path = out_dir / (
                                                                    "meta_edge"
                                                                    f"{edge_token}_{th_token}_m{mode}"
                                                                    f"_sm{sm_token}_mg{mg_token}_hl{hl_token}"
                                                                    f"_tas{tas_token}_mba{mba_token}"
                                                                    f"_pow{eip_token}_minf{emif_token}"
                                                                    f"_lm{lim_token}_smul{sim_token}"
                                                                    f"_eam{eam_token}_eaep{eaep_token}"
                                                                    f"_easp{easp_token}_earm{earm_token}.json"
                                                                )
                                                                runs.append(
                                                                    MetaRun(
                                                                        edge=edge,
                                                                        selection_mode=mode,
                                                                        switch_margin=switch_margin,
                                                                        min_score_gap=min_score_gap,
                                                                        recency_halflife_days=recency_halflife,
                                                                        trade_amount_scale=trade_amount_scale,
                                                                        min_buy_amount=min_buy_amount,
                                                                        entry_intensity_power=entry_intensity_power,
                                                                        entry_min_intensity_fraction=entry_min_intensity_fraction,
                                                                        long_intensity_multiplier=long_intensity_multiplier,
                                                                        short_intensity_multiplier=short_intensity_multiplier,
                                                                        entry_allocator_mode=entry_allocator_mode,
                                                                        entry_allocator_edge_power=entry_allocator_edge_power,
                                                                        entry_allocator_max_single_position_fraction=entry_allocator_max_single_position_fraction,
                                                                        entry_allocator_reserve_fraction=entry_allocator_reserve_fraction,
                                                                        output_path=output_path,
                                                                    )
                                                                )

    for idx, run in enumerate(runs, start=1):
        if args.skip_existing and run.output_path.exists():
            print(f"[{idx}/{len(runs)}] skip existing -> {run.output_path}")
            continue
        strategy_args = [item for spec in args.strategy for item in ("--strategy", spec)]
        cmd = [
            sys.executable,
            "unified_hourly_experiment/sweep_meta_portfolio.py",
            *strategy_args,
            "--symbols",
            args.symbols,
            "--action-cache-dir",
            str(action_cache_dir),
            "--metrics",
            ",".join(metrics),
            "--selection-modes",
            run.selection_mode,
            "--switch-margins",
            str(run.switch_margin),
            "--min-score-gaps",
            str(run.min_score_gap),
            "--recency-halflife-days",
            str(run.recency_halflife_days),
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
            "--trade-amount-scale",
            str(run.trade_amount_scale),
            "--min-buy-amount",
            str(run.min_buy_amount),
            "--entry-intensity-power",
            str(run.entry_intensity_power),
            "--entry-min-intensity-fraction",
            str(run.entry_min_intensity_fraction),
            "--long-intensity-multiplier",
            str(run.long_intensity_multiplier),
            "--short-intensity-multiplier",
            str(run.short_intensity_multiplier),
            "--entry-allocator-mode",
            str(run.entry_allocator_mode),
            "--entry-allocator-edge-power",
            str(run.entry_allocator_edge_power),
            "--entry-allocator-max-single-position-fraction",
            str(run.entry_allocator_max_single_position_fraction),
            "--entry-allocator-reserve-fraction",
            str(run.entry_allocator_reserve_fraction),
            "--decision-lag-bars",
            str(args.decision_lag_bars),
            "--entry-selection-mode",
            str(args.entry_selection_mode),
            "--bar-margin",
            str(args.bar_margin),
            "--execution-bar-margins",
            args.execution_bar_margins,
            "--entry-order-ttl-hours",
            str(int(args.entry_order_ttl_hours)),
            "--execution-entry-order-ttls",
            args.execution_entry_order_ttls,
            "--leverage",
            str(args.leverage),
            "--fee-rate",
            str(args.fee_rate),
            "--margin-rate",
            str(args.margin_rate),
            "--sim-backend",
            str(args.sim_backend),
            "--sit-out-if-negative",
            f"--sit-out-thresholds={",".join(str(x) for x in thresholds)}",
            "--output",
            str(run.output_path),
        ]
        if args.market_order_entry:
            cmd.append("--market-order-entry")
        print(
            f"[{idx}/{len(runs)}] edge={run.edge} thresholds={thresholds} "
            f"mode={run.selection_mode} switch_margin={run.switch_margin} min_gap={run.min_score_gap} "
            f"hl={run.recency_halflife_days} "
            f"scale={run.trade_amount_scale} power={run.entry_intensity_power} "
            f"short_mult={run.short_intensity_multiplier} minf={run.entry_min_intensity_fraction} "
            f"allocator={run.entry_allocator_mode} aedge={run.entry_allocator_edge_power} "
            f"amax={run.entry_allocator_max_single_position_fraction} areserve={run.entry_allocator_reserve_fraction} "
            f"-> {run.output_path}"
        )
        subprocess.run(cmd, check=True)

    ranked_rows = []
    skipped_for_activity = 0
    for run in runs:
        payload = json.loads(run.output_path.read_text())
        summaries = payload.get("summaries") or []
        if not summaries:
            summaries = [payload["best"]]

        for summary in summaries:
            if not eligible_summary(summary, min_num_buys=args.min_num_buys):
                skipped_for_activity += 1
                continue
            ranked_rows.append(
                {
                    "edge": run.edge,
                    "sit_out_threshold": summary.get("sit_out_threshold", thresholds[0]),
                    "selection_mode": run.selection_mode,
                    "switch_margin": run.switch_margin,
                    "min_score_gap": run.min_score_gap,
                    "recency_halflife_days": run.recency_halflife_days,
                    "trade_amount_scale": run.trade_amount_scale,
                    "min_buy_amount": run.min_buy_amount,
                    "entry_intensity_power": run.entry_intensity_power,
                    "entry_min_intensity_fraction": run.entry_min_intensity_fraction,
                    "long_intensity_multiplier": run.long_intensity_multiplier,
                    "short_intensity_multiplier": run.short_intensity_multiplier,
                    "entry_allocator_mode": run.entry_allocator_mode,
                    "entry_allocator_edge_power": run.entry_allocator_edge_power,
                    "entry_allocator_max_single_position_fraction": run.entry_allocator_max_single_position_fraction,
                    "entry_allocator_reserve_fraction": run.entry_allocator_reserve_fraction,
                    "market_order_entry": bool(args.market_order_entry),
                    "entry_selection_mode": str(args.entry_selection_mode),
                    "output": str(run.output_path),
                    **summary,
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
            "recency_halflife_days": recency_halflife_days,
            "trade_amount_scales": trade_amount_scales,
            "min_buy_amounts": min_buy_amounts,
            "entry_intensity_powers": entry_intensity_powers,
            "entry_min_intensity_fractions": entry_min_intensity_fractions,
            "long_intensity_multipliers": long_intensity_multipliers,
            "short_intensity_multipliers": short_intensity_multipliers,
            "entry_allocator_modes": entry_allocator_modes,
            "entry_allocator_edge_powers": entry_allocator_edge_powers,
            "entry_allocator_max_single_position_fractions": entry_allocator_max_single_position_fractions,
            "entry_allocator_reserve_fractions": entry_allocator_reserve_fractions,
            "min_num_buys": int(args.min_num_buys),
            "entry_order_ttl_hours": int(args.entry_order_ttl_hours),
            "execution_bar_margins": parse_float_list(args.execution_bar_margins)
            if str(args.execution_bar_margins).strip()
            else [float(args.bar_margin)],
            "execution_entry_order_ttl_hours": parse_int_list(args.execution_entry_order_ttls)
            if str(args.execution_entry_order_ttls).strip()
            else [int(args.entry_order_ttl_hours)],
            "action_cache_dir": str(action_cache_dir),
            "market_order_entry": bool(args.market_order_entry),
            "entry_selection_mode": str(args.entry_selection_mode),
            "decision_lag_bars": int(args.decision_lag_bars),
        },
        "skipped_for_activity": int(skipped_for_activity),
        "best": best,
        "top5": ranked_rows[:5],
        "entry_allocator_mode_summary": build_entry_allocator_mode_summary(ranked_rows),
        "deploy_command": build_deploy_command(
            strategy_specs=list(args.strategy),
            symbols=args.symbols,
            decision_lag_bars=int(args.decision_lag_bars),
            entry_selection_mode=str(args.entry_selection_mode),
            max_hold_hours=int(args.max_hold_hours),
            max_positions=int(args.max_positions),
            bar_margin=float(args.bar_margin),
            entry_order_ttl_hours=int(args.entry_order_ttl_hours),
            fee_rate=float(args.fee_rate),
            margin_rate=float(args.margin_rate),
            market_order_entry=bool(args.market_order_entry),
            best=best,
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
    parser.add_argument("--symbols", default=",".join(DEFAULT_ALPACA_LIVE8_STOCKS))
    parser.add_argument("--metrics", default="sharpe,sortino,calmar")
    parser.add_argument("--lookback-days", default="5,7,10,14")
    parser.add_argument("--holdout-days", default="30,60,90")
    parser.add_argument("--min-edges", default="0.004,0.005,0.006,0.007,0.008")
    parser.add_argument("--sit-out-thresholds", default="0.2,0.3,0.4,0.5,0.7")
    parser.add_argument("--selection-modes", default="winner")
    parser.add_argument("--switch-margins", default="0.0")
    parser.add_argument("--min-score-gaps", default="0.0")
    parser.add_argument("--recency-halflife-days", default="0.0")
    parser.add_argument(
        "--min-num-buys",
        type=int,
        default=0,
        help="Require best row from each sweep run to have at least this many buys in every holdout period.",
    )
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-hold-hours", type=int, default=5)
    parser.add_argument("--trade-amount-scales", default="100.0")
    parser.add_argument("--min-buy-amounts", default="0.0")
    parser.add_argument("--entry-intensity-powers", default="1.0")
    parser.add_argument("--entry-min-intensity-fractions", default="0.0")
    parser.add_argument("--long-intensity-multipliers", default="1.0")
    parser.add_argument("--short-intensity-multipliers", default="1.0")
    parser.add_argument("--entry-allocator-modes", default="legacy")
    parser.add_argument("--entry-allocator-edge-powers", default="2.0")
    parser.add_argument("--entry-allocator-max-single-position-fractions", default="0.6")
    parser.add_argument("--entry-allocator-reserve-fractions", default="0.1")
    parser.add_argument("--decision-lag-bars", type=int, default=1)
    parser.add_argument(
        "--entry-selection-mode",
        default="edge_rank",
        choices=["edge_rank", "first_trigger"],
        help="How the simulator prioritizes competing fillable entries in sweeps and generated deploy commands.",
    )
    parser.add_argument(
        "--market-order-entry",
        action="store_true",
        help="Use market-order entry fill assumption in sweep simulations and generated deploy command.",
    )
    parser.add_argument("--bar-margin", type=float, default=0.0013)
    parser.add_argument(
        "--execution-bar-margins",
        default="",
        help="Optional comma-separated bar margins used for robustness validation in each sweep run.",
    )
    parser.add_argument(
        "--entry-order-ttl-hours",
        type=int,
        default=0,
        help="How many hourly bars non-filled entry orders remain pending in simulator sweeps (0 disables).",
    )
    parser.add_argument(
        "--execution-entry-order-ttls",
        default="",
        help="Optional comma-separated entry-order TTL values used for robustness validation in each sweep run.",
    )
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
    parser.add_argument(
        "--action-cache-dir",
        type=Path,
        default=None,
        help="Optional shared on-disk cache for generated per-strategy action frames across sweep runs.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip combinations whose output JSON already exists in --output-dir (resume mode).",
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
    execution_bar_margins = (
        parse_float_list(args.execution_bar_margins)
        if str(args.execution_bar_margins).strip()
        else [float(args.bar_margin)]
    )
    execution_entry_order_ttls = (
        parse_int_list(args.execution_entry_order_ttls)
        if str(args.execution_entry_order_ttls).strip()
        else [int(args.entry_order_ttl_hours)]
    )
    if any(x < 0 for x in execution_bar_margins):
        raise ValueError("--execution-bar-margins values must all be >= 0")
    if any(x < 0 for x in execution_entry_order_ttls):
        raise ValueError("--execution-entry-order-ttls values must all be >= 0")
    if any(x <= 0 for x in parse_float_list(args.trade_amount_scales)):
        raise ValueError("--trade-amount-scales values must all be > 0")
    if any(x < 0 for x in parse_float_list(args.min_buy_amounts)):
        raise ValueError("--min-buy-amounts values must all be >= 0")
    if any(x < 0 for x in parse_float_list(args.entry_intensity_powers)):
        raise ValueError("--entry-intensity-powers values must all be >= 0")
    if any(x < 0 for x in parse_float_list(args.entry_min_intensity_fractions)):
        raise ValueError("--entry-min-intensity-fractions values must all be >= 0")
    if any(x < 0 for x in parse_float_list(args.long_intensity_multipliers)):
        raise ValueError("--long-intensity-multipliers values must all be >= 0")
    if any(x < 0 for x in parse_float_list(args.short_intensity_multipliers)):
        raise ValueError("--short-intensity-multipliers values must all be >= 0")
    allocator_modes = [x.lower() for x in parse_csv_list(args.entry_allocator_modes)]
    invalid_allocator_modes = [x for x in allocator_modes if x not in ("legacy", "concentrated")]
    if invalid_allocator_modes:
        raise ValueError(f"--entry-allocator-modes values must be legacy/concentrated, got {invalid_allocator_modes}")
    if any(x < 0 for x in parse_float_list(args.entry_allocator_edge_powers)):
        raise ValueError("--entry-allocator-edge-powers values must all be >= 0")
    if any(x < 0 or x > 1 for x in parse_float_list(args.entry_allocator_max_single_position_fractions)):
        raise ValueError("--entry-allocator-max-single-position-fractions values must all be in [0, 1]")
    if any(x < 0 or x > 1 for x in parse_float_list(args.entry_allocator_reserve_fractions)):
        raise ValueError("--entry-allocator-reserve-fractions values must all be in [0, 1]")

    run_once(args)


if __name__ == "__main__":
    main()
