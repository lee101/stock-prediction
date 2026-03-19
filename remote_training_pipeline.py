from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
import csv
import shlex

import pandas as pd


DEFAULT_REMOTE_HOST = "administrator@93.127.141.100"
DEFAULT_REMOTE_DIR = "/nvme0n1-disk/code/stock-prediction"
DEFAULT_REMOTE_ENV = ".venv313"
DEFAULT_REMOTE_RUN_ROOT = "analysis/remote_runs"


def parse_csv_tokens(raw: str | Sequence[str] | None, *, cast=str) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, str):
        tokens = raw.split(",")
    else:
        tokens = []
        for value in raw:
            tokens.extend(str(value).split(","))
    parsed: list[Any] = []
    for token in tokens:
        token = str(token).strip()
        if not token:
            continue
        parsed.append(cast(token))
    return parsed


def normalize_symbols(symbols: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        clean = str(symbol).strip().upper()
        if not clean or clean in seen:
            continue
        normalized.append(clean)
        seen.add(clean)
    return normalized


def _candidate_hourly_paths(symbol: str, data_root: Path) -> list[Path]:
    upper = symbol.upper()
    root = Path(data_root)
    return [
        root / f"{upper}.csv",
        root / "crypto" / f"{upper}.csv",
        root / "stocks" / f"{upper}.csv",
    ]


def resolve_hourly_symbol_path(symbol: str, data_root: Path) -> Path:
    for path in _candidate_hourly_paths(symbol, data_root):
        if path.exists():
            return path
    raise FileNotFoundError(f"No hourly CSV found for {symbol} under {data_root}")


def load_hourly_index(symbol: str, data_root: Path) -> pd.DatetimeIndex:
    path = resolve_hourly_symbol_path(symbol, data_root)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header_map = {str(name).strip().lower(): str(name) for name in (reader.fieldnames or [])}
    if "timestamp" in header_map:
        ts_col = header_map["timestamp"]
    elif "date" in header_map:
        ts_col = header_map["date"]
    else:
        raise ValueError(f"{path} must contain 'timestamp' or 'date' column")

    frame = pd.read_csv(path, usecols=[ts_col])
    index = pd.to_datetime(frame[ts_col], utc=True, errors="coerce")
    index = index.dropna().dt.floor("h")
    if index.empty:
        raise ValueError(f"{path} contains no valid timestamps")
    return pd.DatetimeIndex(index.sort_values().drop_duplicates())


@dataclass(frozen=True)
class HourlyTrainValWindow:
    earliest_common: str
    latest_common: str
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    train_hours: int
    val_hours: int
    gap_hours: int


def build_hourly_train_val_window_from_bounds(
    *,
    earliest_common: str | pd.Timestamp,
    latest_common: str | pd.Timestamp,
    train_hours: int,
    val_hours: int,
    gap_hours: int = 0,
) -> HourlyTrainValWindow:
    if train_hours < 1 or val_hours < 1:
        raise ValueError("train_hours and val_hours must be >= 1")
    if gap_hours < 0:
        raise ValueError("gap_hours must be >= 0")

    earliest_ts = pd.Timestamp(earliest_common)
    latest_ts = pd.Timestamp(latest_common)
    if earliest_ts.tzinfo is None:
        earliest_ts = earliest_ts.tz_localize("UTC")
    else:
        earliest_ts = earliest_ts.tz_convert("UTC")
    if latest_ts.tzinfo is None:
        latest_ts = latest_ts.tz_localize("UTC")
    else:
        latest_ts = latest_ts.tz_convert("UTC")

    earliest_ts = earliest_ts.floor("h")
    latest_ts = latest_ts.floor("h")
    if latest_ts <= earliest_ts:
        raise ValueError(
            f"No overlapping hourly window: start={earliest_ts} end={latest_ts}"
        )

    val_end = latest_ts
    val_start = val_end - pd.Timedelta(hours=int(val_hours) - 1)
    train_end = val_start - pd.Timedelta(hours=int(gap_hours) + 1)
    train_start = train_end - pd.Timedelta(hours=int(train_hours) - 1)
    if train_start < earliest_ts:
        raise ValueError(
            "Not enough shared history for requested train/val split: "
            f"need train_start={train_start}, earliest_common={earliest_ts}"
        )

    return HourlyTrainValWindow(
        earliest_common=earliest_ts.isoformat(),
        latest_common=latest_ts.isoformat(),
        train_start=train_start.isoformat(),
        train_end=train_end.isoformat(),
        val_start=val_start.isoformat(),
        val_end=val_end.isoformat(),
        train_hours=int(train_hours),
        val_hours=int(val_hours),
        gap_hours=int(gap_hours),
    )


def compute_hourly_overlap_bounds(
    *,
    symbols: Sequence[str],
    data_root: Path,
) -> tuple[str, str]:
    resolved = normalize_symbols(symbols)
    if not resolved:
        raise ValueError("At least one symbol is required")

    indices = [load_hourly_index(symbol, data_root) for symbol in resolved]
    earliest_common = max(index.min() for index in indices).floor("h")
    latest_common = min(index.max() for index in indices).floor("h")
    return earliest_common.isoformat(), latest_common.isoformat()


def compute_hourly_train_val_window(
    *,
    symbols: Sequence[str],
    data_root: Path,
    train_hours: int,
    val_hours: int,
    gap_hours: int = 0,
) -> HourlyTrainValWindow:
    earliest_common, latest_common = compute_hourly_overlap_bounds(symbols=symbols, data_root=data_root)
    return build_hourly_train_val_window_from_bounds(
        earliest_common=earliest_common,
        latest_common=latest_common,
        train_hours=train_hours,
        val_hours=val_hours,
        gap_hours=gap_hours,
    )


def build_run_crypto_lora_batch_cmd(
    *,
    run_id: str,
    symbols: Sequence[str],
    data_root: str,
    output_root: str,
    results_dir: str,
    preaugs: Sequence[str],
    context_lengths: Sequence[int],
    learning_rates: Sequence[float],
    num_steps: int,
    prediction_length: int,
    lora_r: int,
) -> list[str]:
    cmd = [
        "python",
        "-u",
        "scripts/run_crypto_lora_batch.py",
        "--run-id",
        str(run_id),
        "--symbols",
        ",".join(normalize_symbols(symbols)),
        "--data-root",
        str(data_root),
        "--output-root",
        str(output_root),
        "--results-dir",
        str(results_dir),
        "--preaugs",
        ",".join(str(value).strip() for value in preaugs if str(value).strip()),
        "--context-lengths",
        ",".join(str(int(value)) for value in context_lengths),
        "--learning-rates",
        ",".join(str(float(value)) for value in learning_rates),
        "--num-steps",
        str(int(num_steps)),
        "--prediction-length",
        str(int(prediction_length)),
        "--lora-r",
        str(int(lora_r)),
    ]
    return cmd


def build_promote_lora_cmd(
    *,
    report_dir: str,
    output_dir: str,
    symbols: Sequence[str],
    run_id: str,
    metric: str = "val_mae_percent",
) -> list[str]:
    cmd = [
        "python",
        "-u",
        "scripts/promote_chronos2_lora_reports.py",
        "--report-dir",
        str(report_dir),
        "--output-dir",
        str(output_dir),
        "--run-id",
        str(run_id),
        "--metric",
        str(metric),
    ]
    if symbols:
        cmd.extend(["--symbols", *normalize_symbols(symbols)])
    return cmd


def build_forecast_cache_cmd(
    *,
    symbols: Sequence[str],
    data_root: str,
    forecast_cache_root: str,
    lookback_hours: float,
    output_json: str | None = None,
    force_rebuild: bool = False,
) -> list[str]:
    cmd = [
        "python",
        "-u",
        "scripts/build_hourly_forecast_caches.py",
        "--symbols",
        ",".join(normalize_symbols(symbols)),
        "--data-root",
        str(data_root),
        "--forecast-cache-root",
        str(forecast_cache_root),
        "--horizons",
        "1,24",
        "--lookback-hours",
        str(float(lookback_hours)),
    ]
    if output_json:
        cmd.extend(["--output-json", str(output_json)])
    if force_rebuild:
        cmd.append("--force-rebuild")
    return cmd


def build_export_hourly_forecast_cmd(
    *,
    symbols: Sequence[str],
    data_root: str,
    forecast_cache_root: str,
    output_path: str,
    start_date: str,
    end_date: str,
    feature_lag: int,
    min_hours: int,
    min_coverage: float,
) -> list[str]:
    return [
        "python",
        "-u",
        "-m",
        "pufferlib_market.export_data_hourly_forecast",
        "--symbols",
        ",".join(normalize_symbols(symbols)),
        "--data-root",
        str(data_root),
        "--forecast-cache-root",
        str(forecast_cache_root),
        "--output",
        str(output_path),
        "--start-date",
        str(start_date),
        "--end-date",
        str(end_date),
        "--feature-lag",
        str(int(feature_lag)),
        "--min-hours",
        str(int(min_hours)),
        "--min-coverage",
        str(float(min_coverage)),
    ]


def build_autoresearch_cmd(
    *,
    train_data: str,
    val_data: str,
    leaderboard: str,
    checkpoint_root: str,
    time_budget: int,
    max_trials: int,
    descriptions: Sequence[str],
    rank_metric: str = "auto",
) -> list[str]:
    cmd = [
        "python",
        "-u",
        "-m",
        "pufferlib_market.autoresearch_rl",
        "--train-data",
        str(train_data),
        "--val-data",
        str(val_data),
        "--time-budget",
        str(int(time_budget)),
        "--max-trials",
        str(int(max_trials)),
        "--leaderboard",
        str(leaderboard),
        "--checkpoint-root",
        str(checkpoint_root),
        "--rank-metric",
        str(rank_metric),
    ]
    if descriptions:
        cmd.extend(["--descriptions", ",".join(str(item).strip() for item in descriptions if str(item).strip())])
    return cmd


@dataclass(frozen=True)
class RemotePipelinePlan:
    run_id: str
    symbols: tuple[str, ...]
    window: HourlyTrainValWindow
    remote_run_dir: str
    remote_script_path: str
    remote_log_path: str
    remote_pid_path: str
    lora_results_dir: str
    forecast_cache_root: str
    forecast_mae_path: str
    train_data_path: str
    val_data_path: str
    leaderboard_path: str
    checkpoint_root: str
    commands: tuple[tuple[str, ...], ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["commands"] = [list(command) for command in self.commands]
        return payload


def build_remote_hourly_chronos_rl_plan(
    *,
    run_id: str,
    symbols: Sequence[str],
    local_data_root: Path,
    remote_data_root: str,
    train_hours: int,
    val_hours: int,
    gap_hours: int,
    preaugs: Sequence[str],
    context_lengths: Sequence[int],
    learning_rates: Sequence[float],
    num_steps: int,
    prediction_length: int,
    lora_r: int,
    feature_lag: int,
    min_coverage: float,
    time_budget: int,
    max_trials: int,
    descriptions: Sequence[str],
    remote_run_root: str = DEFAULT_REMOTE_RUN_ROOT,
    forecast_lookback_hours: float | None = None,
    earliest_common_override: str | None = None,
    latest_common_override: str | None = None,
) -> RemotePipelinePlan:
    if earliest_common_override is not None and latest_common_override is not None:
        window = build_hourly_train_val_window_from_bounds(
            earliest_common=earliest_common_override,
            latest_common=latest_common_override,
            train_hours=train_hours,
            val_hours=val_hours,
            gap_hours=gap_hours,
        )
    else:
        window = compute_hourly_train_val_window(
            symbols=symbols,
            data_root=Path(local_data_root),
            train_hours=train_hours,
            val_hours=val_hours,
            gap_hours=gap_hours,
        )
    normalized = tuple(normalize_symbols(symbols))
    run_dir = f"{remote_run_root.rstrip('/')}/{run_id}"
    lora_results_dir = f"{run_dir}/lora_results"
    forecast_cache_root = f"{run_dir}/forecast_cache"
    forecast_mae_path = f"{run_dir}/forecast_mae.json"
    train_data_path = f"pufferlib_market/data/{run_id}_train.bin"
    val_data_path = f"pufferlib_market/data/{run_id}_val.bin"
    leaderboard_path = f"pufferlib_market/{run_id}_leaderboard.csv"
    checkpoint_root = f"pufferlib_market/checkpoints/{run_id}"
    remote_script_path = f"{run_dir}/pipeline.sh"
    remote_log_path = f"{run_dir}/pipeline.log"
    remote_pid_path = f"{run_dir}/pipeline.pid"
    lookback_hours = float(forecast_lookback_hours) if forecast_lookback_hours is not None else float(train_hours + val_hours + 168)

    commands = [
        build_run_crypto_lora_batch_cmd(
            run_id=run_id,
            symbols=normalized,
            data_root=remote_data_root,
            output_root="chronos2_finetuned",
            results_dir=lora_results_dir,
            preaugs=preaugs,
            context_lengths=context_lengths,
            learning_rates=learning_rates,
            num_steps=num_steps,
            prediction_length=prediction_length,
            lora_r=lora_r,
        ),
        build_promote_lora_cmd(
            report_dir=lora_results_dir,
            output_dir="hyperparams/chronos2/hourly",
            symbols=normalized,
            run_id=run_id,
        ),
        build_forecast_cache_cmd(
            symbols=normalized,
            data_root=remote_data_root,
            forecast_cache_root=forecast_cache_root,
            lookback_hours=lookback_hours,
            output_json=forecast_mae_path,
            force_rebuild=True,
        ),
        build_export_hourly_forecast_cmd(
            symbols=normalized,
            data_root=remote_data_root,
            forecast_cache_root=forecast_cache_root,
            output_path=train_data_path,
            start_date=window.train_start,
            end_date=window.train_end,
            feature_lag=feature_lag,
            min_hours=train_hours,
            min_coverage=min_coverage,
        ),
        build_export_hourly_forecast_cmd(
            symbols=normalized,
            data_root=remote_data_root,
            forecast_cache_root=forecast_cache_root,
            output_path=val_data_path,
            start_date=window.val_start,
            end_date=window.val_end,
            feature_lag=feature_lag,
            min_hours=val_hours,
            min_coverage=min_coverage,
        ),
        build_autoresearch_cmd(
            train_data=train_data_path,
            val_data=val_data_path,
            leaderboard=leaderboard_path,
            checkpoint_root=checkpoint_root,
            time_budget=time_budget,
            max_trials=max_trials,
            descriptions=descriptions,
        ),
    ]
    return RemotePipelinePlan(
        run_id=str(run_id),
        symbols=normalized,
        window=window,
        remote_run_dir=run_dir,
        remote_script_path=remote_script_path,
        remote_log_path=remote_log_path,
        remote_pid_path=remote_pid_path,
        lora_results_dir=lora_results_dir,
        forecast_cache_root=forecast_cache_root,
        forecast_mae_path=forecast_mae_path,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        leaderboard_path=leaderboard_path,
        checkpoint_root=checkpoint_root,
        commands=tuple(tuple(command) for command in commands),
    )


def shell_join(cmd: Sequence[str]) -> str:
    return shlex.join([str(token) for token in cmd])


def render_remote_pipeline_script(
    *,
    remote_dir: str,
    remote_env: str,
    plan: RemotePipelinePlan,
) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(remote_dir))}",
        f"source {shlex.quote(str(remote_env).rstrip('/'))}/bin/activate",
        'export PYTHONPATH="$PWD:$PWD/PufferLib:${PYTHONPATH:-}"',
        f"mkdir -p {shlex.quote(plan.remote_run_dir)}",
        "echo \"[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting remote hourly Chronos2/RL pipeline\"",
    ]
    for command in plan.commands:
        lines.append(f"echo \"+ {shell_join(command)}\"")
        lines.append(shell_join(command))
    lines.append("echo \"[$(date -u +%Y-%m-%dT%H:%M:%SZ)] pipeline complete\"")
    return "\n".join(lines) + "\n"


__all__ = [
    "DEFAULT_REMOTE_ENV",
    "DEFAULT_REMOTE_DIR",
    "DEFAULT_REMOTE_HOST",
    "DEFAULT_REMOTE_RUN_ROOT",
    "HourlyTrainValWindow",
    "RemotePipelinePlan",
    "build_hourly_train_val_window_from_bounds",
    "build_autoresearch_cmd",
    "build_export_hourly_forecast_cmd",
    "build_forecast_cache_cmd",
    "build_promote_lora_cmd",
    "build_remote_hourly_chronos_rl_plan",
    "build_run_crypto_lora_batch_cmd",
    "compute_hourly_overlap_bounds",
    "compute_hourly_train_val_window",
    "load_hourly_index",
    "normalize_symbols",
    "parse_csv_tokens",
    "render_remote_pipeline_script",
    "resolve_hourly_symbol_path",
    "shell_join",
]
