#!/usr/bin/env python3
"""Refit Chronos2 preaug, hyperparams, and LoRA configs per symbol."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


DEFAULT_PREAUG_STRATEGIES = (
    "baseline",
    "percent_change",
    "log_returns",
    "differencing",
    "detrending",
    "robust_scaling",
    "minmax_standard",
    "rolling_norm",
)


def _csv_tokens(raw: str | None) -> List[str]:
    if raw is None:
        return []
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def _positive_int_tokens(raw: str) -> List[int]:
    values = [int(token) for token in _csv_tokens(raw)]
    return [value for value in values if value > 0]


def _bool_mode_tokens(raw: str) -> List[str]:
    values: List[str] = []
    for token in _csv_tokens(raw):
        normalized = token.lower()
        if normalized not in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
            raise ValueError(f"invalid bool mode '{token}'")
        values.append(normalized)
    if not values:
        raise ValueError("expected at least one bool mode")
    return values


def _discover_symbols(symbols_arg: str | None, data_dir: Path, limit: int | None) -> List[str]:
    if symbols_arg:
        symbols = [token.upper() for token in _csv_tokens(symbols_arg)]
    else:
        symbols = sorted(path.stem.upper() for path in data_dir.glob("*.csv"))
    if limit is not None:
        symbols = symbols[: max(0, limit)]
    return symbols


@dataclass(frozen=True)
class RefitPaths:
    data_dir: Path
    hyperparam_dir: Path
    preaug_dir: Path
    preaug_best_dir: Path
    benchmark_output_dir: Path
    lora_output_root: Path
    lora_results_dir: Path
    refit_report_dir: Path


def _resolve_paths(
    *,
    frequency: str,
    asset_kind: str,
    report_root: Path,
) -> RefitPaths:
    if frequency == "hourly":
        if asset_kind == "stocks":
            data_dir = Path("trainingdatahourly/stocks")
        else:
            data_dir = Path("trainingdatahourly/crypto")
        hyperparam_dir = Path("hyperparams/chronos2/hourly")
        preaug_dir = Path("preaugstrategies/chronos2/hourly")
        preaug_best_dir = Path("preaugstrategies/best/hourly")
        lora_results_dir = Path("hyperparams/chronos2/hourly_lora")
    else:
        data_dir = Path("trainingdata")
        hyperparam_dir = Path("hyperparams/chronos2")
        preaug_dir = Path("preaugstrategies/chronos2")
        preaug_best_dir = Path("preaugstrategies/best")
        lora_results_dir = Path("hyperparams/chronos2/daily_lora")

    run_root = report_root / frequency / asset_kind
    return RefitPaths(
        data_dir=data_dir,
        hyperparam_dir=hyperparam_dir,
        preaug_dir=preaug_dir,
        preaug_best_dir=preaug_best_dir,
        benchmark_output_dir=run_root / "benchmarks",
        lora_output_root=Path("chronos2_finetuned") / frequency / asset_kind,
        lora_results_dir=lora_results_dir,
        refit_report_dir=run_root,
    )


def build_preaug_command(
    *,
    python_bin: str,
    symbols: Sequence[str],
    paths: RefitPaths,
    args: argparse.Namespace,
) -> List[str]:
    cmd = [
        python_bin,
        "preaug_sweeps/evaluate_preaug_chronos.py",
        "--symbols",
        *symbols,
        "--hyperparam-root",
        str(paths.hyperparam_dir),
        "--output-dir",
        str(paths.preaug_dir),
        "--mirror-best-dir",
        str(paths.preaug_best_dir),
        "--data-dir",
        str(paths.data_dir),
        "--frequency",
        args.frequency,
        "--selection-metric",
        args.preaug_selection_metric,
        "--benchmark-cache-dir",
        str(paths.refit_report_dir / "preaug_cache"),
        "--report-dir",
        str(paths.refit_report_dir / "preaug_reports"),
        "--strategies",
        *args.preaug_strategies,
    ]
    if args.device_map:
        cmd.extend(["--device-map", args.device_map])
    if args.torch_dtype:
        cmd.extend(["--torch-dtype", args.torch_dtype])
    if args.torch_compile:
        cmd.append("--torch-compile")
    if args.pipeline_backend:
        cmd.extend(["--pipeline-backend", args.pipeline_backend])
    return cmd


def build_benchmark_command(
    *,
    python_bin: str,
    symbols: Sequence[str],
    paths: RefitPaths,
    args: argparse.Namespace,
) -> List[str]:
    cmd = [
        python_bin,
        "benchmark_chronos2.py",
        "--symbols",
        *symbols,
        "--data-dir",
        str(paths.data_dir),
        "--context-lengths",
        *[str(value) for value in args.context_lengths],
        "--batch-sizes",
        *[str(value) for value in args.batch_sizes],
        "--aggregations",
        *args.aggregations,
        "--sample-counts",
        *[str(value) for value in args.sample_counts],
        "--scalers",
        *args.scalers,
        "--multivariate-modes",
        *args.multivariate_modes,
        "--output-dir",
        str(paths.benchmark_output_dir),
        "--hyperparam-root",
        "hyperparams",
        "--update-hyperparams",
        "--force-update",
    ]
    if args.frequency != "daily":
        cmd.extend(["--chronos2-subdir", args.frequency])
    if args.search_method != "grid":
        cmd.extend(["--search-method", args.search_method])
    if args.device_map:
        cmd.extend(["--device-map", args.device_map])
    if args.torch_dtype:
        cmd.extend(["--torch-dtype", args.torch_dtype])
    if args.pipeline_backend:
        cmd.extend(["--pipeline-backend", args.pipeline_backend])
    if args.predict_batches_jointly:
        cmd.append("--predict-batches-jointly")
    if args.torch_compile:
        cmd.append("--torch-compile")
    return cmd


def build_lora_command(
    *,
    python_bin: str,
    symbols: Sequence[str],
    paths: RefitPaths,
    args: argparse.Namespace,
) -> List[str]:
    run_id = args.run_id
    return [
        python_bin,
        "scripts/chronos2_lora_improvement_sweep.py",
        "--run-id",
        run_id,
        "--symbols",
        ",".join(symbols),
        "--data-root",
        str(paths.data_dir),
        "--output-root",
        str(paths.lora_output_root),
        "--results-dir",
        str(paths.lora_results_dir),
        "--preaugs",
        ",".join(args.lora_preaugs),
        "--context-lengths",
        ",".join(str(value) for value in args.lora_context_lengths),
        "--learning-rates",
        ",".join(args.lora_learning_rates),
        "--lora-rs",
        ",".join(str(value) for value in args.lora_rs),
        "--batch-size",
        str(args.lora_batch_size),
        "--num-steps",
        str(args.lora_num_steps),
        "--prediction-length",
        str(args.lora_prediction_length),
        "--improvement-threshold",
        str(args.lora_improvement_threshold),
    ]


def build_promote_command(
    *,
    python_bin: str,
    symbols: Sequence[str],
    paths: RefitPaths,
    args: argparse.Namespace,
) -> List[str]:
    cmd = [
        python_bin,
        "scripts/promote_chronos2_lora_reports.py",
        "--report-dir",
        str(paths.lora_results_dir),
        "--output-dir",
        str(paths.hyperparam_dir),
        "--run-id",
        args.run_id,
        "--symbols",
        *symbols,
        "--selection-strategy",
        args.promote_selection_strategy,
    ]
    if args.promote_metric:
        cmd.extend(["--metric", args.promote_metric])
    return cmd


def _run_command(cmd: Sequence[str], *, cwd: Path) -> int:
    proc = subprocess.run(list(cmd), cwd=cwd)
    return int(proc.returncode)


def _record_plan(path: Path, *, symbols: Sequence[str], commands: Sequence[Sequence[str]], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "symbols": list(symbols),
        "args": {key: value for key, value in vars(args).items() if key != "execute"},
        "commands": [list(command) for command in commands],
    }
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols. Defaults to all CSV stems in the resolved data dir.")
    parser.add_argument("--limit-symbols", type=int, default=None)
    parser.add_argument("--frequency", choices=("daily", "hourly"), default="hourly")
    parser.add_argument("--asset-kind", choices=("stocks", "crypto"), default="stocks")
    parser.add_argument("--run-id", default=time.strftime("chronos2_refit_%Y%m%d_%H%M%S"))
    parser.add_argument("--report-root", type=Path, default=Path("reports/chronos2_pair_refits"))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--execute", action="store_true", help="Run the generated commands instead of only printing/saving them.")
    parser.add_argument("--skip-preaug", action="store_true")
    parser.add_argument("--skip-hyperparams", action="store_true")
    parser.add_argument("--skip-lora", action="store_true")
    parser.add_argument("--skip-promote", action="store_true")
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--pipeline-backend", choices=("chronos", "cutechronos", "auto"), default="chronos")
    parser.add_argument("--predict-batches-jointly", action="store_true")
    parser.add_argument("--search-method", choices=("grid", "direct"), default="grid")
    parser.add_argument("--context-lengths", default="256,512,1024")
    parser.add_argument("--batch-sizes", default="32,64")
    parser.add_argument("--aggregations", default="median")
    parser.add_argument("--sample-counts", default="0")
    parser.add_argument("--scalers", default="none,meanstd")
    parser.add_argument("--multivariate-modes", default="false,true")
    parser.add_argument("--preaug-selection-metric", choices=("mae_percent", "mae", "pct_return_mae", "rmse", "mape"), default="mae_percent")
    parser.add_argument("--preaug-strategies", default=",".join(DEFAULT_PREAUG_STRATEGIES))
    parser.add_argument("--lora-preaugs", default="baseline,rolling_norm,detrending,robust_scaling")
    parser.add_argument("--lora-context-lengths", default="128,256,512")
    parser.add_argument("--lora-learning-rates", default="1e-5,5e-5")
    parser.add_argument("--lora-rs", default="8,16")
    parser.add_argument("--lora-batch-size", type=int, default=32)
    parser.add_argument("--lora-num-steps", type=int, default=1000)
    parser.add_argument("--lora-prediction-length", type=int, default=24)
    parser.add_argument("--lora-improvement-threshold", type=float, default=5.0)
    parser.add_argument("--promote-selection-strategy", choices=("best_single", "stable_family"), default="stable_family")
    parser.add_argument("--promote-metric", choices=("val_mae_percent", "val_pct_return_mae", "val_mae"), default="val_mae_percent")
    args = parser.parse_args(argv)

    args.context_lengths = _positive_int_tokens(args.context_lengths)
    args.batch_sizes = _positive_int_tokens(args.batch_sizes)
    args.aggregations = _csv_tokens(args.aggregations)
    args.sample_counts = [int(token) for token in _csv_tokens(args.sample_counts)]
    args.scalers = _csv_tokens(args.scalers)
    args.multivariate_modes = _bool_mode_tokens(args.multivariate_modes)
    args.preaug_strategies = _csv_tokens(args.preaug_strategies)
    args.lora_preaugs = _csv_tokens(args.lora_preaugs)
    args.lora_context_lengths = _positive_int_tokens(args.lora_context_lengths)
    args.lora_learning_rates = _csv_tokens(args.lora_learning_rates)
    args.lora_rs = _positive_int_tokens(args.lora_rs)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    paths = _resolve_paths(frequency=args.frequency, asset_kind=args.asset_kind, report_root=args.report_root)
    symbols = _discover_symbols(args.symbols, paths.data_dir, args.limit_symbols)
    if not symbols:
        print(f"No symbols discovered under {paths.data_dir}", flush=True)
        return 2

    commands: List[List[str]] = []
    if not args.skip_preaug:
        commands.append(build_preaug_command(python_bin=args.python_bin, symbols=symbols, paths=paths, args=args))
    if not args.skip_hyperparams:
        commands.append(build_benchmark_command(python_bin=args.python_bin, symbols=symbols, paths=paths, args=args))
    if not args.skip_lora:
        commands.append(build_lora_command(python_bin=args.python_bin, symbols=symbols, paths=paths, args=args))
    if not args.skip_promote:
        commands.append(build_promote_command(python_bin=args.python_bin, symbols=symbols, paths=paths, args=args))

    plan_path = paths.refit_report_dir / f"{args.run_id}_plan.json"
    _record_plan(plan_path, symbols=symbols, commands=commands, args=args)

    print(f"Symbols ({len(symbols)}): {', '.join(symbols)}", flush=True)
    print(f"Plan saved to {plan_path}", flush=True)
    for idx, cmd in enumerate(commands, start=1):
        print(f"[{idx}/{len(commands)}] {' '.join(cmd)}", flush=True)

    if not args.execute:
        return 0

    for cmd in commands:
        code = _run_command(cmd, cwd=repo_root)
        if code != 0:
            print(f"Stage failed with exit code {code}: {' '.join(cmd)}", flush=True)
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
