#!/usr/bin/env python3
# ruff: noqa: E402
"""Run production-faithful holdout evaluation for the Binance hybrid launch config."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.binance_hybrid_eval_defaults import (
    DEFAULT_DAILY_PERIODS_PER_YEAR,
    DEFAULT_HOURLY_PERIODS_PER_YEAR,
    DEFAULT_PROD_EVAL_ALLOW_SHORTS,
    DEFAULT_PROD_EVAL_DATA_PATH,
    DEFAULT_PROD_EVAL_DECISION_LAG,
    DEFAULT_PROD_EVAL_FEE_RATE,
    DEFAULT_PROD_EVAL_FILL_BUFFER_BPS,
    DEFAULT_PROD_EVAL_HOURS,
    DEFAULT_PROD_EVAL_REQUIRE_RUNTIME_HEALTH,
    DEFAULT_PROD_EVAL_REQUIRE_RUNTIME_MATCH,
    DEFAULT_PROD_EVAL_RUNTIME_AUDIT_HOURS,
    DEFAULT_PROD_EVAL_RUNTIME_MAX_DEGRADED_FALLBACK_COUNT,
    DEFAULT_PROD_EVAL_RUNTIME_MAX_DEGRADED_STATUS_COUNT,
    DEFAULT_PROD_EVAL_RUNTIME_MAX_GEMINI_SKIPPED_COUNT,
    DEFAULT_PROD_EVAL_RUNTIME_MIN_HEALTHY_COMPLETED,
    DEFAULT_PROD_EVAL_RUNTIME_TRACE_DIR,
    DEFAULT_PROD_EVAL_SEED,
    DEFAULT_PROD_EVAL_SKIP_REPLAY_EVAL,
    DEFAULT_PROD_EVAL_SKIP_RUNTIME_AUDIT,
    DEFAULT_PROD_EVAL_SLIPPAGE_BPS,
    DEFAULT_PROD_EVAL_WINDOWS,
    DEFAULT_REPLAY_EVAL_END_DATE,
    DEFAULT_REPLAY_EVAL_FILL_BUFFER_BPS,
    DEFAULT_REPLAY_EVAL_HOURLY_ROOT,
    DEFAULT_REPLAY_EVAL_START_DATE,
    DEFAULT_REPLAY_ROBUST_START_STATES,
)
from src.binance_hybrid_launch import (
    DEFAULT_LAUNCH_SCRIPT,
    BinanceHybridLaunchConfig,
    resolve_target_launch_config,
)
from src.binance_hybrid_launch import (
    parse_launch_script as _parse_launch_script,
)
from src.binance_hybrid_runtime_audit import (
    BinanceHybridRuntimeAuditResult,
    audit_binance_hybrid_runtime,
    build_runtime_audit_health_issues,
    build_runtime_audit_launch_mismatch_issues,
)


parse_launch_script = _parse_launch_script


@dataclass(frozen=True)
class ReplayRunResult:
    output_path: str
    daily_total_return: float | None
    daily_sortino: float | None
    hourly_total_return: float | None
    hourly_sortino: float | None
    hourly_goodness_score: float | None
    robust_worst_hourly_return: float | None


@dataclass(frozen=True)
class EvalRunResult:
    checkpoint: str
    holdout_output_path: str
    median_total_return: float
    median_sortino: float
    median_max_drawdown: float
    p10_total_return: float
    replay: ReplayRunResult | None = None


@dataclass(frozen=True)
class EvalCommandPlan:
    checkpoint: str
    holdout_command: list[str]
    holdout_output_path: str
    replay_command: list[str] | None = None
    replay_output_path: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the live Binance hybrid launch config")
    parser.add_argument("--launch-script", default=str(DEFAULT_LAUNCH_SCRIPT))
    parser.add_argument("--data-path", default=str(DEFAULT_PROD_EVAL_DATA_PATH))
    parser.add_argument("--eval-hours", type=int, default=DEFAULT_PROD_EVAL_HOURS)
    parser.add_argument("--n-windows", type=int, default=DEFAULT_PROD_EVAL_WINDOWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_PROD_EVAL_SEED)
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_PROD_EVAL_FEE_RATE)
    parser.add_argument("--slippage-bps", type=float, default=DEFAULT_PROD_EVAL_SLIPPAGE_BPS)
    parser.add_argument("--fill-buffer-bps", type=float, default=DEFAULT_PROD_EVAL_FILL_BUFFER_BPS)
    parser.add_argument("--decision-lag", type=int, default=DEFAULT_PROD_EVAL_DECISION_LAG)
    parser.add_argument("--periods-per-year", type=float, default=DEFAULT_DAILY_PERIODS_PER_YEAR)
    parser.add_argument("--replay-eval-hourly-root", default=DEFAULT_REPLAY_EVAL_HOURLY_ROOT)
    parser.add_argument("--replay-eval-start-date", default=DEFAULT_REPLAY_EVAL_START_DATE)
    parser.add_argument("--replay-eval-end-date", default=DEFAULT_REPLAY_EVAL_END_DATE)
    parser.add_argument("--replay-eval-fill-buffer-bps", type=float, default=DEFAULT_REPLAY_EVAL_FILL_BUFFER_BPS)
    parser.add_argument("--replay-eval-hourly-periods-per-year", type=float, default=DEFAULT_HOURLY_PERIODS_PER_YEAR)
    parser.add_argument("--replay-robust-start-states", default=DEFAULT_REPLAY_ROBUST_START_STATES)
    parser.add_argument("--skip-replay-eval", action="store_true", default=DEFAULT_PROD_EVAL_SKIP_REPLAY_EVAL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--candidate-checkpoint", action="append", default=[])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--leverage", type=float, default=None)
    parser.add_argument("--runtime-trace-dir", default=str(DEFAULT_PROD_EVAL_RUNTIME_TRACE_DIR))
    parser.add_argument("--runtime-audit-hours", type=float, default=DEFAULT_PROD_EVAL_RUNTIME_AUDIT_HOURS)
    parser.add_argument("--skip-runtime-audit", action=argparse.BooleanOptionalAction, default=DEFAULT_PROD_EVAL_SKIP_RUNTIME_AUDIT)
    parser.add_argument("--require-runtime-match", action=argparse.BooleanOptionalAction, default=DEFAULT_PROD_EVAL_REQUIRE_RUNTIME_MATCH)
    parser.add_argument("--require-runtime-health", action=argparse.BooleanOptionalAction, default=DEFAULT_PROD_EVAL_REQUIRE_RUNTIME_HEALTH)
    parser.add_argument("--runtime-min-healthy-completed", type=int, default=DEFAULT_PROD_EVAL_RUNTIME_MIN_HEALTHY_COMPLETED)
    parser.add_argument("--runtime-max-degraded-status-count", type=int, default=DEFAULT_PROD_EVAL_RUNTIME_MAX_DEGRADED_STATUS_COUNT)
    parser.add_argument("--runtime-max-degraded-fallback-count", type=int, default=DEFAULT_PROD_EVAL_RUNTIME_MAX_DEGRADED_FALLBACK_COUNT)
    parser.add_argument("--runtime-max-gemini-skipped-count", type=int, default=DEFAULT_PROD_EVAL_RUNTIME_MAX_GEMINI_SKIPPED_COUNT)
    parser.add_argument("--allow-shorts", action="store_true", default=DEFAULT_PROD_EVAL_ALLOW_SHORTS)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _default_output_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    return REPO_ROOT / "analysis" / f"current_binance_prod_eval_{stamp}"


def build_manifest_eval_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "data_path": str(Path(args.data_path).expanduser().resolve(strict=False)),
        "eval_hours": int(args.eval_hours),
        "n_windows": int(args.n_windows),
        "seed": int(args.seed),
        "fee_rate": float(args.fee_rate),
        "slippage_bps": float(args.slippage_bps),
        "fill_buffer_bps": float(args.fill_buffer_bps),
        "decision_lag": int(args.decision_lag),
        "periods_per_year": float(args.periods_per_year),
        "replay_eval_hourly_root": str(args.replay_eval_hourly_root),
        "replay_eval_start_date": str(args.replay_eval_start_date),
        "replay_eval_end_date": str(args.replay_eval_end_date),
        "replay_eval_fill_buffer_bps": float(args.replay_eval_fill_buffer_bps),
        "replay_eval_hourly_periods_per_year": float(args.replay_eval_hourly_periods_per_year),
        "replay_robust_start_states": str(args.replay_robust_start_states),
        "runtime_trace_dir": str(Path(args.runtime_trace_dir).expanduser().resolve(strict=False)),
        "runtime_audit_hours": float(args.runtime_audit_hours),
        "skip_runtime_audit": bool(args.skip_runtime_audit),
        "require_runtime_match": bool(args.require_runtime_match),
        "require_runtime_health": bool(args.require_runtime_health),
        "runtime_min_healthy_completed": int(args.runtime_min_healthy_completed),
        "runtime_max_degraded_status_count": int(args.runtime_max_degraded_status_count),
        "runtime_max_degraded_fallback_count": int(args.runtime_max_degraded_fallback_count),
        "runtime_max_gemini_skipped_count": int(args.runtime_max_gemini_skipped_count),
        "allow_shorts": bool(args.allow_shorts),
        "skip_replay_eval": bool(args.skip_replay_eval),
    }


def run_current_runtime_audit(
    args: argparse.Namespace,
    *,
    launch_script: str | Path,
) -> tuple[BinanceHybridRuntimeAuditResult | None, list[str], list[str]]:
    if bool(args.skip_runtime_audit):
        return None, [], []
    runtime_audit = audit_binance_hybrid_runtime(
        launch_script=launch_script,
        trace_dir=args.runtime_trace_dir,
        hours=float(args.runtime_audit_hours),
    )
    live_launch_config = parse_launch_script(launch_script, require_rl_checkpoint=False)
    drift_issues = build_runtime_audit_launch_mismatch_issues(
        runtime_audit,
        require_recent_snapshots=bool(args.require_runtime_match),
        require_checkpoint_match=bool(live_launch_config.rl_checkpoint),
    )
    health_issues = build_runtime_audit_health_issues(
        runtime_audit,
        require_recent_snapshots=bool(args.require_runtime_health),
        min_healthy_completed_count=int(args.runtime_min_healthy_completed),
        max_degraded_status_count=int(args.runtime_max_degraded_status_count),
        max_degraded_fallback_count=int(args.runtime_max_degraded_fallback_count),
        max_gemini_call_skipped_count=int(args.runtime_max_gemini_skipped_count),
    )
    return runtime_audit, drift_issues, health_issues


def _print_runtime_audit_summary(
    runtime_audit: BinanceHybridRuntimeAuditResult | None,
    drift_issues: list[str],
    health_issues: list[str],
) -> None:
    if runtime_audit is None:
        return
    print("")
    print("Current runtime audit")
    print(f"  snapshots:           {runtime_audit.snapshot_count}")
    print(f"  healthy completed:   {runtime_audit.healthy_completed_count}")
    print(f"  degraded statuses:   {runtime_audit.degraded_status_count}")
    print(f"  degraded fallbacks:  {runtime_audit.degraded_allocation_source_count}")
    print(f"  override cycles:     {runtime_audit.override_allocation_source_count}")
    if drift_issues:
        print("  launch drift issues:")
        for issue in drift_issues:
            print(f"    - {issue}")
    if health_issues:
        print("  runtime health issues:")
        for issue in health_issues:
            print(f"    - {issue}")




def _sanitize_checkpoint_name(checkpoint: str | Path) -> str:
    checkpoint_path = Path(checkpoint)
    parent_name = checkpoint_path.parent.name.strip()
    if parent_name:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", parent_name)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", checkpoint_path.stem)


def build_holdout_command(
    launch_config: BinanceHybridLaunchConfig,
    *,
    data_path: str | Path,
    checkpoint: str | Path,
    eval_hours: int,
    n_windows: int,
    seed: int,
    fee_rate: float,
    slippage_bps: float,
    fill_buffer_bps: float,
    decision_lag: int,
    periods_per_year: float,
    device: str,
    output_path: str | Path,
    disable_shorts: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "pufferlib_market.evaluate_holdout",
        "--checkpoint",
        str(checkpoint),
        "--data-path",
        str(data_path),
        "--eval-hours",
        str(int(eval_hours)),
        "--n-windows",
        str(int(n_windows)),
        "--seed",
        str(int(seed)),
        "--fee-rate",
        str(float(fee_rate)),
        "--slippage-bps",
        str(float(slippage_bps)),
        "--fill-buffer-bps",
        str(float(fill_buffer_bps)),
        "--max-leverage",
        str(float(launch_config.leverage)),
        "--periods-per-year",
        str(float(periods_per_year)),
        "--decision-lag",
        str(int(decision_lag)),
        "--deterministic",
        "--tradable-symbols",
        ",".join(launch_config.symbols),
        "--device",
        str(device),
        "--out",
        str(output_path),
    ]
    if disable_shorts:
        command.append("--disable-shorts")
    return command


def build_replay_command(
    launch_config: BinanceHybridLaunchConfig,
    *,
    data_path: str | Path,
    checkpoint: str | Path,
    max_steps: int,
    fee_rate: float,
    fill_buffer_bps: float,
    daily_periods_per_year: float,
    hourly_periods_per_year: float,
    hourly_data_root: str | Path,
    start_date: str,
    end_date: str,
    output_path: str | Path,
    disable_shorts: bool,
    tradable_symbols: list[str] | None = None,
    robust_start_states: str = "",
    device: str = "cpu",
) -> list[str]:
    resolved_tradable_symbols = tradable_symbols or launch_config.symbols
    command = [
        sys.executable,
        "-m",
        "pufferlib_market.replay_eval",
        "--checkpoint",
        str(checkpoint),
        "--daily-data-path",
        str(data_path),
        "--hourly-data-root",
        str(hourly_data_root),
        "--start-date",
        str(start_date),
        "--end-date",
        str(end_date),
        "--max-steps",
        str(int(max_steps)),
        "--fee-rate",
        str(float(fee_rate)),
        "--fill-buffer-bps",
        str(float(fill_buffer_bps)),
        "--max-leverage",
        str(float(launch_config.leverage)),
        "--daily-periods-per-year",
        str(float(daily_periods_per_year)),
        "--hourly-periods-per-year",
        str(float(hourly_periods_per_year)),
        "--deterministic",
        "--tradable-symbols",
        ",".join(resolved_tradable_symbols),
        "--output-json",
        str(output_path),
    ]
    if disable_shorts:
        command.append("--disable-shorts")
    if robust_start_states.strip():
        command.extend(["--robust-start-states", robust_start_states.strip()])
    if str(device).strip().lower() == "cpu":
        command.append("--cpu")
    return command


def _load_eval_summary(path: str | Path) -> EvalRunResult:
    payload = json.loads(Path(path).read_text())
    summary = payload["summary"]
    return EvalRunResult(
        checkpoint=str(payload["checkpoint"]),
        holdout_output_path=str(Path(path)),
        median_total_return=float(summary["median_total_return"]),
        median_sortino=float(summary["median_sortino"]),
        median_max_drawdown=float(summary["median_max_drawdown"]),
        p10_total_return=float(summary["p10_total_return"]),
    )


def _load_replay_summary(path: str | Path) -> ReplayRunResult:
    payload = json.loads(Path(path).read_text())
    daily = payload.get("daily", {})
    hourly = payload.get("hourly_replay", {})
    robust = payload.get("robust_start_summary", {})
    robust_hourly = robust.get("hourly_replay", {}) if isinstance(robust, dict) else {}
    return ReplayRunResult(
        output_path=str(Path(path)),
        daily_total_return=float(daily.get("total_return")) if "total_return" in daily else None,
        daily_sortino=float(daily.get("sortino")) if "sortino" in daily else None,
        hourly_total_return=float(hourly.get("total_return")) if "total_return" in hourly else None,
        hourly_sortino=float(hourly.get("sortino")) if "sortino" in hourly else None,
        hourly_goodness_score=float(hourly.get("goodness_score")) if "goodness_score" in hourly else None,
        robust_worst_hourly_return=(
            float(robust_hourly.get("worst_total_return")) if "worst_total_return" in robust_hourly else None
        ),
    )


def _print_plan(launch_config: BinanceHybridLaunchConfig, commands: list[list[str]]) -> None:
    print("Binance hybrid production evaluation")
    print(f"  launch:      {launch_config.launch_script}")
    print(f"  model:       {launch_config.model}")
    print(f"  symbols:     {' '.join(launch_config.symbols)}")
    print(f"  exec mode:   {launch_config.execution_mode}")
    print(f"  leverage:    {launch_config.leverage}")
    print(f"  checkpoint:  {launch_config.rl_checkpoint or 'RL disabled'}")
    print("")
    for index, command in enumerate(commands, start=1):
        print(f"[{index}] {' '.join(shlex.quote(token) for token in command)}")


def build_eval_plans(
    launch_config: BinanceHybridLaunchConfig,
    args: argparse.Namespace,
    *,
    output_dir: str | Path,
) -> list[EvalCommandPlan]:
    output_dir_path = Path(output_dir)
    checkpoints: list[str] = []
    if launch_config.rl_checkpoint:
        checkpoints.append(str(launch_config.rl_checkpoint))
    checkpoints.extend(str(checkpoint) for checkpoint in args.candidate_checkpoint)
    if not checkpoints:
        raise ValueError("launch has no rl_checkpoint; provide at least one --candidate-checkpoint to evaluate")
    plans: list[EvalCommandPlan] = []
    for checkpoint in checkpoints:
        holdout_output_path = output_dir_path / (
            f"{_sanitize_checkpoint_name(checkpoint)}_"
            f"{int(args.eval_hours)}h_{int(args.n_windows)}w_lag{int(args.decision_lag)}_livecfg.json"
        )
        holdout_command = build_holdout_command(
            launch_config,
            data_path=args.data_path,
            checkpoint=checkpoint,
            eval_hours=args.eval_hours,
            n_windows=args.n_windows,
            seed=args.seed,
            fee_rate=args.fee_rate,
            slippage_bps=args.slippage_bps,
            fill_buffer_bps=args.fill_buffer_bps,
            decision_lag=args.decision_lag,
            periods_per_year=args.periods_per_year,
            device=args.device,
            output_path=holdout_output_path,
            disable_shorts=not args.allow_shorts,
        )
        replay_command: list[str] | None = None
        replay_output_path: Path | None = None
        if not args.skip_replay_eval:
            replay_output_path = output_dir_path / (
                f"{_sanitize_checkpoint_name(checkpoint)}_"
                f"replay_{int(args.eval_hours)}h_lag{int(args.decision_lag)}_livecfg.json"
            )
            replay_command = build_replay_command(
                launch_config,
                data_path=args.data_path,
                checkpoint=checkpoint,
                max_steps=args.eval_hours,
                fee_rate=args.fee_rate,
                fill_buffer_bps=args.replay_eval_fill_buffer_bps,
                daily_periods_per_year=args.periods_per_year,
                hourly_periods_per_year=args.replay_eval_hourly_periods_per_year,
                hourly_data_root=args.replay_eval_hourly_root,
                start_date=args.replay_eval_start_date,
                end_date=args.replay_eval_end_date,
                output_path=replay_output_path,
                disable_shorts=not args.allow_shorts,
                robust_start_states=args.replay_robust_start_states,
                device=args.device,
            )
        plans.append(
            EvalCommandPlan(
                checkpoint=str(checkpoint),
                holdout_command=holdout_command,
                holdout_output_path=str(holdout_output_path),
                replay_command=replay_command,
                replay_output_path=str(replay_output_path) if replay_output_path is not None else None,
            )
        )
    return plans


def run_eval(args: argparse.Namespace) -> int:
    launch_config = resolve_target_launch_config(
        args.launch_script,
        symbols_override=args.symbols or None,
        leverage_override=args.leverage,
    )
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_audit, runtime_audit_issues, runtime_health_issues = run_current_runtime_audit(
        args,
        launch_script=args.launch_script,
    )
    runtime_failure_header: str | None = None
    runtime_failure_issues: list[str] = []
    if args.require_runtime_match and runtime_audit_issues:
        runtime_failure_header = "current live runtime does not match launch config:"
        runtime_failure_issues = runtime_audit_issues
    elif args.require_runtime_health and runtime_health_issues:
        runtime_failure_header = "current live runtime is too degraded to trust as a production baseline:"
        runtime_failure_issues = runtime_health_issues

    try:
        plans = build_eval_plans(launch_config, args, output_dir=output_dir)
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    commands: list[list[str]] = []
    for plan in plans:
        commands.append(plan.holdout_command)
        if plan.replay_command is not None:
            commands.append(plan.replay_command)
    _print_plan(launch_config, commands)
    _print_runtime_audit_summary(runtime_audit, runtime_audit_issues, runtime_health_issues)
    if runtime_failure_header is not None:
        sys.stderr.write(f"{runtime_failure_header}\n")
        for issue in runtime_failure_issues:
            sys.stderr.write(f"- {issue}\n")
        if not args.dry_run:
            return 2
    if args.dry_run:
        print("\nDRY RUN -- no evaluations executed")
        return 0

    results: list[EvalRunResult] = []
    for plan in plans:
        completed = subprocess.run(
            plan.holdout_command,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            sys.stderr.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        result = _load_eval_summary(plan.holdout_output_path)
        if plan.replay_command is not None and plan.replay_output_path is not None:
            replay_completed = subprocess.run(
                plan.replay_command,
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            if replay_completed.returncode != 0:
                sys.stderr.write(replay_completed.stdout)
                sys.stderr.write(replay_completed.stderr)
                return replay_completed.returncode
            result = EvalRunResult(
                checkpoint=result.checkpoint,
                holdout_output_path=result.holdout_output_path,
                median_total_return=result.median_total_return,
                median_sortino=result.median_sortino,
                median_max_drawdown=result.median_max_drawdown,
                p10_total_return=result.p10_total_return,
                replay=_load_replay_summary(plan.replay_output_path),
            )
        results.append(result)

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "launch_config": asdict(launch_config),
        "eval_config": build_manifest_eval_config(args),
        "current_runtime_audit": runtime_audit.to_dict() if runtime_audit is not None else None,
        "current_runtime_audit_issues": runtime_audit_issues,
        "current_runtime_health_issues": runtime_health_issues,
        "evaluations": [asdict(result) for result in results],
    }
    manifest_path = output_dir / "prod_launch_eval_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    print("\nSummary")
    for result in results:
        summary_line = (
            f"  {Path(result.checkpoint).parent.name:28s} "
            f"holdout_med={result.median_total_return * 100:+6.2f}% "
            f"holdout_sort={result.median_sortino:+5.2f} "
            f"p10={result.p10_total_return * 100:+6.2f}% "
            f"med_dd={result.median_max_drawdown * 100:5.2f}%"
        )
        if result.replay is None:
            print(summary_line)
            continue
        print(summary_line)
        print(
            f"  {'':28s} "
            f"replay_hourly={100 * (result.replay.hourly_total_return or 0.0):+6.2f}% "
            f"replay_sort={result.replay.hourly_sortino or 0.0:+5.2f} "
            f"goodness={result.replay.hourly_goodness_score or 0.0:+6.2f} "
            f"robust_worst={100 * (result.replay.robust_worst_hourly_return or 0.0):+6.2f}%"
        )
    print(f"\nManifest: {manifest_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_eval(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
