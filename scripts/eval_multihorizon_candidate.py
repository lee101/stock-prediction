#!/usr/bin/env python3
"""Compare screened32 baseline vs candidate across multiple daily horizons.

This wraps ``pufferlib_market.evaluate_holdout`` so we can answer the real
promotion question for new stock candidates:

1. Is the candidate good on its own?
2. Is it additive to the current live ensemble?
3. Do gains survive multiple horizons plus a recent-tail slice?

The script keeps start indices fixed across scenarios for an apples-to-apples
comparison and reports both raw holdout metrics and derived monthly returns.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pufferlib_market.hourly_replay import read_mktd
from src.daily_stock_defaults import DEFAULT_CHECKPOINT, DEFAULT_EXTRA_CHECKPOINTS


@dataclass(frozen=True)
class Scenario:
    name: str
    checkpoint: str
    extra_checkpoints: tuple[str, ...]


def _parse_int_csv(raw: str) -> list[int]:
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


def _monthly_from_total(total_return: float, window_days: int, trading_days_per_month: float = 21.0) -> float:
    if window_days <= 0:
        return 0.0
    try:
        return math.expm1(math.log1p(float(total_return)) * (trading_days_per_month / float(window_days)))
    except Exception:
        return 0.0


def build_start_indices(
    *,
    num_timesteps: int,
    eval_days: int,
    n_windows: int,
    seed: int,
    recent_within_days: int | None = None,
    exhaustive: bool = False,
) -> list[int]:
    import numpy as np

    steps = int(eval_days)
    if steps < 1:
        raise ValueError("eval_days must be >= 1")
    if n_windows < 1:
        raise ValueError("n_windows must be >= 1")

    window_len = steps + 1
    if window_len > int(num_timesteps):
        raise ValueError(f"Dataset too short for eval_days={steps}: timesteps={num_timesteps}")

    end_min = window_len
    end_max = int(num_timesteps)
    if recent_within_days is not None:
        within = int(recent_within_days)
        if within < 1:
            raise ValueError("recent_within_days must be >= 1")
        end_min = max(end_min, int(num_timesteps) - within)

    candidate_window_count = end_max - end_min + 1
    if candidate_window_count <= 0:
        raise ValueError("No candidate windows available for requested recent tail")

    candidate_start_min = end_min - window_len
    candidate_start_max = candidate_start_min + candidate_window_count - 1

    if exhaustive or candidate_window_count <= n_windows:
        return list(range(candidate_start_min, candidate_start_max + 1))

    rng = np.random.default_rng(int(seed))
    offsets = rng.choice(candidate_window_count, size=int(n_windows), replace=False)
    starts = [candidate_start_min + int(offset) for offset in offsets.tolist()]
    starts.sort()
    return starts


def _run_holdout(
    *,
    scenario: Scenario,
    data_path: Path,
    eval_days: int,
    start_indices: list[int],
    fee_rate: float,
    slippage_bps: int,
    fill_buffer_bps: float,
    decision_lag: int,
    disable_shorts: bool,
) -> dict[str, Any]:
    tmp_root = REPO / ".tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mh_eval_", dir=str(tmp_root)) as tmpdir:
        out_path = Path(tmpdir) / f"{scenario.name}_{eval_days}d_{slippage_bps}bps.json"
        cmd = [
            sys.executable,
            "-m",
            "pufferlib_market.evaluate_holdout",
            "--checkpoint",
            scenario.checkpoint,
            "--data-path",
            str(data_path),
            "--eval-hours",
            str(int(eval_days)),
            "--fee-rate",
            str(float(fee_rate)),
            "--slippage-bps",
            str(int(slippage_bps)),
            "--fill-buffer-bps",
            str(float(fill_buffer_bps)),
            "--decision-lag",
            str(int(decision_lag)),
            "--deterministic",
            "--no-early-stop",
            "--periods-per-year",
            "252",
            "--out",
            str(out_path),
            "--start-indices",
            *[str(int(idx)) for idx in start_indices],
        ]
        if disable_shorts:
            cmd.append("--disable-shorts")
        if scenario.extra_checkpoints:
            cmd.extend(["--extra-checkpoints", *scenario.extra_checkpoints])
        subprocess.run(cmd, cwd=REPO, check=True, capture_output=True, text=True)
        payload = json.loads(out_path.read_text())
    summary = dict(payload.get("summary", {}))
    summary["median_monthly_return"] = _monthly_from_total(float(summary.get("median_total_return", 0.0)), eval_days)
    summary["p10_monthly_return"] = _monthly_from_total(float(summary.get("p10_total_return", 0.0)), eval_days)
    summary["window_days"] = int(eval_days)
    summary["slippage_bps"] = int(slippage_bps)
    summary["start_indices"] = [int(x) for x in start_indices]
    summary["scenario"] = scenario.name
    return summary


def _build_scenarios(args: argparse.Namespace) -> list[Scenario]:
    baseline = Scenario(
        name="baseline",
        checkpoint=str(args.baseline_checkpoint),
        extra_checkpoints=tuple(args.baseline_extra_checkpoints),
    )
    scenarios = [baseline]
    if not args.candidate_checkpoint:
        return scenarios

    candidate = Scenario(
        name="candidate",
        checkpoint=str(args.candidate_checkpoint),
        extra_checkpoints=tuple(args.candidate_extra_checkpoints),
    )
    combo_paths = [str(args.candidate_checkpoint), *list(args.candidate_extra_checkpoints)]
    combo = Scenario(
        name="baseline_plus_candidate",
        checkpoint=str(args.baseline_checkpoint),
        extra_checkpoints=tuple([*args.baseline_extra_checkpoints, *combo_paths]),
    )
    scenarios.extend([candidate, combo])
    return scenarios


def _scenario_aggregate(results: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for regime_name, horizons in results.items():
        for horizon_name, slips in horizons.items():
            for slip_name, summary in slips.items():
                cells.append(
                    {
                        "regime": regime_name,
                        "horizon": horizon_name,
                        "slippage_bps": int(slip_name),
                        "median_monthly_return": float(summary.get("median_monthly_return", 0.0)),
                        "p10_monthly_return": float(summary.get("p10_monthly_return", 0.0)),
                        "negative_windows": int(summary.get("negative_windows", 0)),
                        "median_sortino": float(summary.get("median_sortino", 0.0)),
                    }
                )
    if not cells:
        return {"error": "no cells"}
    worst = min(cells, key=lambda row: row["median_monthly_return"])
    mean_monthly = sum(cell["median_monthly_return"] for cell in cells) / len(cells)
    mean_p10_monthly = sum(cell["p10_monthly_return"] for cell in cells) / len(cells)
    total_neg = sum(cell["negative_windows"] for cell in cells)
    return {
        "cell_count": len(cells),
        "worst_cell": worst,
        "mean_median_monthly_return": mean_monthly,
        "mean_p10_monthly_return": mean_p10_monthly,
        "total_negative_windows": int(total_neg),
    }


def _compare_vs_baseline(
    baseline_results: dict[str, dict[str, dict[str, Any]]],
    other_results: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    wins = 0
    losses = 0
    deltas: list[dict[str, Any]] = []
    for regime_name, horizons in baseline_results.items():
        for horizon_name, slips in horizons.items():
            for slip_name, base_summary in slips.items():
                other_summary = other_results[regime_name][horizon_name][slip_name]
                delta = {
                    "regime": regime_name,
                    "horizon": horizon_name,
                    "slippage_bps": int(slip_name),
                    "delta_median_monthly_return": float(other_summary["median_monthly_return"])
                    - float(base_summary["median_monthly_return"]),
                    "delta_p10_monthly_return": float(other_summary["p10_monthly_return"])
                    - float(base_summary["p10_monthly_return"]),
                    "delta_negative_windows": int(other_summary["negative_windows"])
                    - int(base_summary["negative_windows"]),
                    "delta_median_sortino": float(other_summary["median_sortino"])
                    - float(base_summary["median_sortino"]),
                }
                if delta["delta_median_monthly_return"] > 0:
                    wins += 1
                elif delta["delta_median_monthly_return"] < 0:
                    losses += 1
                deltas.append(delta)
    worst = min(deltas, key=lambda row: row["delta_median_monthly_return"])
    return {
        "cells": deltas,
        "wins": int(wins),
        "losses": int(losses),
        "worst_delta_cell": worst,
        "mean_delta_median_monthly_return": sum(row["delta_median_monthly_return"] for row in deltas) / len(deltas),
        "mean_delta_p10_monthly_return": sum(row["delta_p10_monthly_return"] for row in deltas) / len(deltas),
        "mean_delta_negative_windows": sum(row["delta_negative_windows"] for row in deltas) / len(deltas),
    }


def choose_recommendation(report: dict[str, Any]) -> dict[str, Any]:
    scenarios = report.get("scenarios", {})
    baseline = scenarios.get("baseline", {}).get("aggregate", {})
    candidate = scenarios.get("candidate", {}).get("aggregate", {})
    combo = scenarios.get("baseline_plus_candidate", {}).get("aggregate", {})
    comparisons = report.get("comparisons", {})
    combo_cmp = comparisons.get("baseline_plus_candidate_vs_baseline", {})
    candidate_cmp = comparisons.get("candidate_vs_baseline", {})

    if combo and combo_cmp:
        combo_worst = float(combo.get("worst_cell", {}).get("median_monthly_return", -1e9))
        base_worst = float(baseline.get("worst_cell", {}).get("median_monthly_return", -1e9))
        if (
            combo_worst > base_worst
            and float(combo_cmp.get("mean_delta_median_monthly_return", 0.0)) > 0.0
            and float(combo_cmp.get("mean_delta_negative_windows", 0.0)) <= 0.0
        ):
            return {
                "status": "promising_additive",
                "reason": "candidate improves the baseline ensemble across the multihorizon panel",
            }

    if candidate and candidate_cmp:
        cand_worst = float(candidate.get("worst_cell", {}).get("median_monthly_return", -1e9))
        base_worst = float(baseline.get("worst_cell", {}).get("median_monthly_return", -1e9))
        if (
            cand_worst > base_worst
            and float(candidate_cmp.get("mean_delta_median_monthly_return", 0.0)) > 0.0
            and float(candidate_cmp.get("mean_delta_negative_windows", 0.0)) <= 0.0
        ):
            return {
                "status": "promising_solo",
                "reason": "candidate beats the current baseline on its own but additivity is not yet proven",
            }

    return {
        "status": "not_proven",
        "reason": "candidate does not consistently improve the baseline across horizons/regimes",
    }


def _print_summary(report: dict[str, Any]) -> None:
    print("Multihorizon candidate eval")
    print(f"  data: {report['config']['data_path']}")
    print(f"  horizons_days: {report['config']['horizons_days']}")
    print(f"  slippage_bps: {report['config']['slippage_bps']}")
    print(f"  recent_within_days: {report['config']['recent_within_days']}")
    print()

    for scenario_name, scenario_payload in report["scenarios"].items():
        agg = scenario_payload["aggregate"]
        worst = agg["worst_cell"]
        print(
            f"{scenario_name:>24s}  "
            f"worst_monthly={worst['median_monthly_return'] * 100:+6.2f}%  "
            f"mean_monthly={agg['mean_median_monthly_return'] * 100:+6.2f}%  "
            f"neg_windows={agg['total_negative_windows']:4d}  "
            f"worst={worst['regime']}/{worst['horizon']}/{worst['slippage_bps']}bps"
        )

    if report.get("comparisons"):
        print()
        for name, payload in report["comparisons"].items():
            worst = payload["worst_delta_cell"]
            print(
                f"{name:>24s}  "
                f"mean_delta_monthly={payload['mean_delta_median_monthly_return'] * 100:+6.2f}%  "
                f"mean_delta_neg={payload['mean_delta_negative_windows']:+5.2f}  "
                f"wins={payload['wins']:2d} losses={payload['losses']:2d}  "
                f"worst_delta={worst['delta_median_monthly_return'] * 100:+6.2f}% "
                f"@ {worst['regime']}/{worst['horizon']}/{worst['slippage_bps']}bps"
            )

    rec = report["recommendation"]
    print()
    print(f"recommendation: {rec['status']}  ({rec['reason']})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        default="pufferlib_market/data/screened32_single_offset_val_full.bin",
        help="Daily MKTD validation dataset",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Primary checkpoint for the current baseline ensemble",
    )
    parser.add_argument(
        "--baseline-extra-checkpoints",
        nargs="*",
        default=list(DEFAULT_EXTRA_CHECKPOINTS),
        help="Extra checkpoints for the current baseline ensemble",
    )
    parser.add_argument("--candidate-checkpoint", default=None, help="Primary candidate checkpoint")
    parser.add_argument(
        "--candidate-extra-checkpoints",
        nargs="*",
        default=[],
        help="Optional extra checkpoints for the candidate ensemble",
    )
    parser.add_argument("--horizons-days", default="30,60,100,120")
    parser.add_argument("--slippage-bps", default="0,5,10,20")
    parser.add_argument("--n-windows", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--recent-within-days", type=int, default=140)
    parser.add_argument("--exhaustive", action="store_true", help="Use all possible windows per horizon/regime")
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--decision-lag", type=int, default=2)
    parser.add_argument("--allow-shorts", action="store_false", dest="disable_shorts")
    parser.set_defaults(disable_shorts=True)
    parser.add_argument("--out", default="reports/multihorizon_candidate_eval.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = REPO / data_path
    if not data_path.exists():
        raise FileNotFoundError(f"data path not found: {data_path}")

    dataset = read_mktd(data_path)
    horizons = _parse_int_csv(args.horizons_days)
    slippages = _parse_int_csv(args.slippage_bps)
    scenarios = _build_scenarios(args)
    regimes = {
        "full": None,
        "recent": int(args.recent_within_days) if args.recent_within_days and args.recent_within_days > 0 else None,
    }
    if regimes["recent"] is None:
        regimes = {"full": None}

    report: dict[str, Any] = {
        "config": {
            "data_path": str(data_path),
            "num_timesteps": int(dataset.num_timesteps),
            "horizons_days": horizons,
            "slippage_bps": slippages,
            "n_windows": int(args.n_windows),
            "seed": int(args.seed),
            "recent_within_days": regimes.get("recent"),
            "fee_rate": float(args.fee_rate),
            "fill_buffer_bps": float(args.fill_buffer_bps),
            "decision_lag": int(args.decision_lag),
            "disable_shorts": bool(args.disable_shorts),
        },
        "scenarios": {},
        "comparisons": {},
    }

    for scenario in scenarios:
        scenario_results: dict[str, dict[str, dict[str, Any]]] = {}
        for regime_name, recent_days in regimes.items():
            regime_results: dict[str, dict[str, Any]] = {}
            for horizon in horizons:
                start_indices = build_start_indices(
                    num_timesteps=int(dataset.num_timesteps),
                    eval_days=int(horizon),
                    n_windows=int(args.n_windows),
                    seed=int(args.seed),
                    recent_within_days=recent_days,
                    exhaustive=bool(args.exhaustive),
                )
                slip_results: dict[str, Any] = {}
                for slip in slippages:
                    slip_results[str(int(slip))] = _run_holdout(
                        scenario=scenario,
                        data_path=data_path,
                        eval_days=int(horizon),
                        start_indices=start_indices,
                        fee_rate=float(args.fee_rate),
                        slippage_bps=int(slip),
                        fill_buffer_bps=float(args.fill_buffer_bps),
                        decision_lag=int(args.decision_lag),
                        disable_shorts=bool(args.disable_shorts),
                    )
                regime_results[f"{int(horizon)}d"] = slip_results
            scenario_results[regime_name] = regime_results
        report["scenarios"][scenario.name] = {
            "results": scenario_results,
            "aggregate": _scenario_aggregate(scenario_results),
        }

    baseline_results = report["scenarios"]["baseline"]["results"]
    if "candidate" in report["scenarios"]:
        candidate_results = report["scenarios"]["candidate"]["results"]
        report["comparisons"]["candidate_vs_baseline"] = _compare_vs_baseline(
            baseline_results=baseline_results,
            other_results=candidate_results,
        )
    if "baseline_plus_candidate" in report["scenarios"]:
        combo_results = report["scenarios"]["baseline_plus_candidate"]["results"]
        report["comparisons"]["baseline_plus_candidate_vs_baseline"] = _compare_vs_baseline(
            baseline_results=baseline_results,
            other_results=combo_results,
        )

    report["recommendation"] = choose_recommendation(report)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    _print_summary(report)
    print(f"\nreport: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
