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


def _candidate_daily_paths(symbol: str, data_root: Path) -> list[Path]:
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


def resolve_daily_symbol_path(symbol: str, data_root: Path) -> Path:
    for path in _candidate_daily_paths(symbol, data_root):
        if path.exists():
            return path
    raise FileNotFoundError(f"No daily CSV found for {symbol} under {data_root}")


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


def load_daily_index(symbol: str, data_root: Path) -> pd.DatetimeIndex:
    path = resolve_daily_symbol_path(symbol, data_root)
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
    index = index.dropna().dt.floor("D")
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


def compute_daily_overlap_bounds(
    *,
    symbols: Sequence[str],
    data_root: Path,
) -> tuple[str, str]:
    resolved = normalize_symbols(symbols)
    if not resolved:
        raise ValueError("At least one symbol is required")

    indices = [load_daily_index(symbol, data_root) for symbol in resolved]
    earliest_common = max(index.min() for index in indices).floor("D")
    latest_common = min(index.max() for index in indices).floor("D")
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


def build_export_daily_fused_cmd(
    *,
    symbols: Sequence[str],
    data_root: str,
    hourly_root: str,
    daily_forecast_root: str,
    hourly_forecast_root: str,
    output_path: str,
    start_date: str,
    end_date: str,
    min_days: int,
    zscore_window: int = 60,
) -> list[str]:
    return [
        "python",
        "-u",
        "-m",
        "pufferlib_market.export_data_daily_v4",
        "--symbols",
        ",".join(normalize_symbols(symbols)),
        "--data-root",
        str(data_root),
        "--hourly-root",
        str(hourly_root),
        "--daily-forecast-root",
        str(daily_forecast_root),
        "--hourly-forecast-root",
        str(hourly_forecast_root),
        "--single-output",
        str(output_path),
        "--start-date",
        str(start_date),
        "--end-date",
        str(end_date),
        "--min-days",
        str(int(min_days)),
        "--zscore-window",
        str(int(zscore_window)),
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
    max_timesteps_per_sample: int = 0,
    stocks_mode: bool = False,
    start_from: int = 0,
    seed_only: bool = False,
    poly_prune: bool = True,
    init_best_config: str = "",
    lock_best_config: bool = False,
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
    if stocks_mode:
        cmd.append("--stocks")
    if descriptions:
        cmd.extend(["--descriptions", ",".join(str(item).strip() for item in descriptions if str(item).strip())])
    if max_timesteps_per_sample > 0:
        cmd.extend(["--max-timesteps-per-sample", str(int(max_timesteps_per_sample))])
    if start_from > 0:
        cmd.extend(["--start-from", str(int(start_from))])
    if seed_only:
        cmd.append("--seed-only")
    if not poly_prune:
        cmd.append("--no-poly-prune")
    if init_best_config:
        cmd.extend(["--init-best-config", str(init_best_config)])
    if lock_best_config:
        cmd.append("--lock-best-config")
    return cmd


@dataclass(frozen=True)
class RemoteAutoresearchPlan:
    run_id: str
    remote_run_dir: str
    remote_script_path: str
    remote_log_path: str
    remote_pid_path: str
    train_data_path: str
    val_data_path: str
    leaderboard_path: str
    checkpoint_root: str
    post_eval_output_path: str | None
    commands: tuple[tuple[str, ...], ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["commands"] = [list(command) for command in self.commands]
        return payload


def _append_optional_cli_arg(cmd: list[str], flag: str, value: object) -> None:
    if value is None:
        return
    if isinstance(value, str):
        if not value:
            return
        cmd.extend([flag, value])
        return
    if isinstance(value, (list, tuple)):
        if not value:
            return
        cmd.extend([flag, ",".join(str(item) for item in value)])
        return
    cmd.extend([flag, str(value)])


def build_remote_autoresearch_plan(
    *,
    run_id: str,
    train_data_path: str,
    val_data_path: str,
    time_budget: int,
    max_trials: int,
    descriptions: Sequence[str],
    rank_metric: str = "auto",
    leaderboard_path: str | None = None,
    checkpoint_root: str | None = None,
    periods_per_year: float | None = None,
    max_steps_override: int = 0,
    max_timesteps_per_sample: int = 0,
    fee_rate_override: float = -1.0,
    stocks_mode: bool = False,
    start_from: int = 0,
    seed_only: bool = False,
    holdout_data: str | None = None,
    holdout_eval_steps: int = 0,
    holdout_n_windows: int = 0,
    holdout_seed: int = 1337,
    holdout_end_within_steps: int = 0,
    holdout_fee_rate: float = -1.0,
    holdout_fill_buffer_bps: float = 5.0,
    holdout_max_leverage: float = 1.0,
    holdout_short_borrow_apr: float = 0.0,
    replay_eval_data: str | None = None,
    replay_eval_hourly_root: str = "",
    replay_eval_start_date: str = "",
    replay_eval_end_date: str = "",
    replay_eval_fill_buffer_bps: float = 5.0,
    replay_eval_run_hourly_policy: bool = False,
    replay_eval_robust_start_states: str = "",
    replay_eval_hourly_periods_per_year: float = 8760.0,
    market_validation_asset_class: str = "",
    market_validation_days: int = 0,
    market_validation_cash: float = 10_000.0,
    market_validation_decision_cadence: str = "hourly",
    market_validation_symbols: Sequence[str] | None = None,
    remote_run_root: str = DEFAULT_REMOTE_RUN_ROOT,
    post_eval_periods: Sequence[int] | None = None,
    post_eval_sort_period: int = 120,
    post_eval_max_workers: int = 4,
    post_eval_cache_path: str | None = None,
    post_eval_use_compile: bool = False,
    post_eval_parallel: bool = False,
    init_best_config: str = "",
    lock_best_config: bool = False,
) -> RemoteAutoresearchPlan:
    run_dir = f"{remote_run_root.rstrip('/')}/{run_id}"
    leaderboard = leaderboard_path or f"pufferlib_market/{run_id}_leaderboard.csv"
    checkpoints = checkpoint_root or f"pufferlib_market/checkpoints/{run_id}"
    remote_script_path = f"{run_dir}/pipeline.sh"
    remote_log_path = f"{run_dir}/pipeline.log"
    remote_pid_path = f"{run_dir}/pipeline.pid"

    # stocks_mode: polynomial early stopping is ineffective (combined_score gap
    # between degenerate and escaped seeds is only ~4%), so use fixed 25% threshold
    # rejection instead (--no-poly-prune), which cuts degenerate trials from ~60s
    # to ~30s on H100.
    cmd = build_autoresearch_cmd(
        train_data=train_data_path,
        val_data=val_data_path,
        leaderboard=leaderboard,
        checkpoint_root=checkpoints,
        time_budget=time_budget,
        max_trials=max_trials,
        descriptions=descriptions,
        rank_metric=rank_metric,
        max_timesteps_per_sample=max_timesteps_per_sample,
        stocks_mode=stocks_mode,
        start_from=start_from,
        seed_only=seed_only,
        poly_prune=not stocks_mode,
        init_best_config=init_best_config,
        lock_best_config=lock_best_config,
    )
    _append_optional_cli_arg(cmd, "--periods-per-year", periods_per_year)
    if max_steps_override > 0:
        _append_optional_cli_arg(cmd, "--max-steps-override", max_steps_override)
    if fee_rate_override >= 0.0:
        _append_optional_cli_arg(cmd, "--fee-rate-override", fee_rate_override)
    if holdout_data:
        _append_optional_cli_arg(cmd, "--holdout-data", holdout_data)
        if holdout_eval_steps > 0:
            _append_optional_cli_arg(cmd, "--holdout-eval-steps", holdout_eval_steps)
        if holdout_n_windows > 0:
            _append_optional_cli_arg(cmd, "--holdout-n-windows", holdout_n_windows)
        _append_optional_cli_arg(cmd, "--holdout-seed", holdout_seed)
        if holdout_end_within_steps > 0:
            _append_optional_cli_arg(cmd, "--holdout-end-within-steps", holdout_end_within_steps)
        if holdout_fee_rate >= 0.0:
            _append_optional_cli_arg(cmd, "--holdout-fee-rate", holdout_fee_rate)
        _append_optional_cli_arg(cmd, "--holdout-fill-buffer-bps", holdout_fill_buffer_bps)
        _append_optional_cli_arg(cmd, "--holdout-max-leverage", holdout_max_leverage)
        _append_optional_cli_arg(cmd, "--holdout-short-borrow-apr", holdout_short_borrow_apr)
    if market_validation_asset_class:
        _append_optional_cli_arg(cmd, "--market-validation-asset-class", market_validation_asset_class)
        if market_validation_days > 0:
            _append_optional_cli_arg(cmd, "--market-validation-days", market_validation_days)
        _append_optional_cli_arg(cmd, "--market-validation-cash", market_validation_cash)
        _append_optional_cli_arg(
            cmd,
            "--market-validation-decision-cadence",
            market_validation_decision_cadence,
        )
        if market_validation_symbols:
            _append_optional_cli_arg(
                cmd,
                "--market-validation-symbols",
                normalize_symbols(market_validation_symbols),
            )
    replay_enabled = bool(replay_eval_hourly_root and replay_eval_start_date and replay_eval_end_date)
    if replay_enabled:
        _append_optional_cli_arg(cmd, "--replay-eval-data", replay_eval_data or holdout_data or val_data_path)
        _append_optional_cli_arg(cmd, "--replay-eval-hourly-root", replay_eval_hourly_root)
        _append_optional_cli_arg(cmd, "--replay-eval-start-date", replay_eval_start_date)
        _append_optional_cli_arg(cmd, "--replay-eval-end-date", replay_eval_end_date)
        _append_optional_cli_arg(cmd, "--replay-eval-fill-buffer-bps", replay_eval_fill_buffer_bps)
        _append_optional_cli_arg(
            cmd,
            "--replay-eval-robust-start-states",
            replay_eval_robust_start_states,
        )
        _append_optional_cli_arg(
            cmd,
            "--replay-eval-hourly-periods-per-year",
            replay_eval_hourly_periods_per_year,
        )
        if replay_eval_run_hourly_policy:
            cmd.append("--replay-eval-run-hourly-policy")

    commands: list[list[str]] = [cmd]
    post_eval_output_path: str | None = None
    if post_eval_periods:
        periods = [int(period) for period in post_eval_periods if int(period) > 0]
        if periods:
            post_eval_output_path = f"{run_dir}/marketsim_{'_'.join(str(period) for period in periods)}.csv"
            fast_eval_cmd = [
                "python",
                "-u",
                "pufferlib_market/fast_marketsim_eval.py",
                "--root",
                ".",
                "--output",
                post_eval_output_path,
                "--periods",
                ",".join(str(period) for period in periods),
                "--sort-period",
                str(int(post_eval_sort_period)),
                "--checkpoint-dirs",
                checkpoints,
                "--max-workers",
                str(int(post_eval_max_workers)),
            ]
            cache_path = post_eval_cache_path or f"{run_dir}/marketsim_cache.json"
            fast_eval_cmd.extend(["--cache-path", cache_path])
            if not post_eval_use_compile:
                fast_eval_cmd.append("--no-compile")
            if not post_eval_parallel:
                fast_eval_cmd.append("--sequential")
            commands.append(fast_eval_cmd)

    return RemoteAutoresearchPlan(
        run_id=str(run_id),
        remote_run_dir=run_dir,
        remote_script_path=remote_script_path,
        remote_log_path=remote_log_path,
        remote_pid_path=remote_pid_path,
        train_data_path=str(train_data_path),
        val_data_path=str(val_data_path),
        leaderboard_path=leaderboard,
        checkpoint_root=checkpoints,
        post_eval_output_path=post_eval_output_path,
        commands=tuple(tuple(command) for command in commands),
    )


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


@dataclass(frozen=True)
class RemoteChronosComparisonPlan:
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
    hourly_train_data_path: str
    hourly_val_data_path: str
    daily_train_data_path: str
    daily_val_data_path: str
    hourly_leaderboard_path: str
    hourly_checkpoint_root: str
    daily_leaderboard_path: str
    daily_checkpoint_root: str
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


def build_remote_chronos_compare_plan(
    *,
    run_id: str,
    symbols: Sequence[str],
    local_hourly_data_root: Path,
    remote_hourly_data_root: str,
    local_daily_data_root: Path,
    remote_daily_data_root: str,
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
    daily_forecast_root: str = "strategytraining/forecast_cache",
    zscore_window: int = 60,
    hourly_periods_per_year: float = 8760.0,
    daily_periods_per_year: float = 365.0,
    hourly_rank_metric: str = "generalization_score",
    daily_rank_metric: str = "generalization_score",
    hourly_max_steps_override: int = 720,
    daily_max_steps_override: int = 90,
    hourly_holdout_eval_steps: int = 168,
    daily_holdout_eval_steps: int = 30,
    holdout_n_windows: int = 12,
    holdout_fee_rate: float = 0.001,
    holdout_fill_buffer_bps: float = 5.0,
    remote_run_root: str = DEFAULT_REMOTE_RUN_ROOT,
    forecast_lookback_hours: float | None = None,
    earliest_common_override: str | None = None,
    latest_common_override: str | None = None,
) -> RemoteChronosComparisonPlan:
    if train_hours % 24 != 0 or val_hours % 24 != 0 or gap_hours % 24 != 0:
        raise ValueError("Chronos daily/hourly comparison requires train_hours, val_hours, gap_hours to be multiples of 24")

    if earliest_common_override is not None and latest_common_override is not None:
        effective_earliest = str(earliest_common_override)
        effective_latest = str(latest_common_override)
    else:
        hourly_earliest, hourly_latest = compute_hourly_overlap_bounds(
            symbols=symbols,
            data_root=Path(local_hourly_data_root),
        )
        daily_earliest_date, daily_latest_date = compute_daily_overlap_bounds(
            symbols=symbols,
            data_root=Path(local_daily_data_root),
        )
        daily_earliest = pd.Timestamp(daily_earliest_date).tz_convert("UTC").floor("D")
        daily_latest = pd.Timestamp(daily_latest_date).tz_convert("UTC").floor("D") + pd.Timedelta(hours=23)
        effective_earliest = max(pd.Timestamp(hourly_earliest), daily_earliest).isoformat()
        effective_latest = min(pd.Timestamp(hourly_latest), daily_latest).isoformat()

    window = build_hourly_train_val_window_from_bounds(
        earliest_common=effective_earliest,
        latest_common=effective_latest,
        train_hours=train_hours,
        val_hours=val_hours,
        gap_hours=gap_hours,
    )
    normalized = tuple(normalize_symbols(symbols))
    run_dir = f"{remote_run_root.rstrip('/')}/{run_id}"
    lora_results_dir = f"{run_dir}/lora_results"
    forecast_cache_root = f"{run_dir}/forecast_cache"
    forecast_mae_path = f"{run_dir}/forecast_mae.json"
    hourly_train_data_path = f"pufferlib_market/data/{run_id}_hourly_train.bin"
    hourly_val_data_path = f"pufferlib_market/data/{run_id}_hourly_val.bin"
    daily_train_data_path = f"pufferlib_market/data/{run_id}_daily_train.bin"
    daily_val_data_path = f"pufferlib_market/data/{run_id}_daily_val.bin"
    remote_script_path = f"{run_dir}/pipeline.sh"
    remote_log_path = f"{run_dir}/pipeline.log"
    remote_pid_path = f"{run_dir}/pipeline.pid"
    lookback_hours = float(forecast_lookback_hours) if forecast_lookback_hours is not None else float(train_hours + val_hours + 168)

    hourly_holdout_steps = min(int(hourly_holdout_eval_steps), max(int(val_hours) - 1, 1))
    daily_val_days = max(int(val_hours // 24), 1)
    daily_holdout_steps = min(int(daily_holdout_eval_steps), max(daily_val_days - 1, 1))
    daily_min_days = max(int(train_hours // 24), 30)
    daily_val_min_days = max(daily_val_days, 10)
    train_start_day = str(pd.Timestamp(window.train_start).date())
    train_end_day = str(pd.Timestamp(window.train_end).date())
    val_start_day = str(pd.Timestamp(window.val_start).date())
    val_end_day = str(pd.Timestamp(window.val_end).date())

    hourly_plan = build_remote_autoresearch_plan(
        run_id=f"{run_id}_hourly",
        train_data_path=hourly_train_data_path,
        val_data_path=hourly_val_data_path,
        time_budget=time_budget,
        max_trials=max_trials,
        descriptions=descriptions,
        rank_metric=hourly_rank_metric,
        periods_per_year=hourly_periods_per_year,
        max_steps_override=hourly_max_steps_override,
        holdout_data=hourly_val_data_path,
        holdout_eval_steps=hourly_holdout_steps,
        holdout_n_windows=holdout_n_windows,
        holdout_fee_rate=holdout_fee_rate,
        holdout_fill_buffer_bps=holdout_fill_buffer_bps,
        leaderboard_path=f"analysis/remote_runs/{run_id}/hourly_leaderboard.csv",
        checkpoint_root=f"pufferlib_market/checkpoints/{run_id}_hourly",
    )
    daily_plan = build_remote_autoresearch_plan(
        run_id=f"{run_id}_daily",
        train_data_path=daily_train_data_path,
        val_data_path=daily_val_data_path,
        time_budget=time_budget,
        max_trials=max_trials,
        descriptions=descriptions,
        rank_metric=daily_rank_metric,
        periods_per_year=daily_periods_per_year,
        max_steps_override=daily_max_steps_override,
        holdout_data=daily_val_data_path,
        holdout_eval_steps=daily_holdout_steps,
        holdout_n_windows=holdout_n_windows,
        holdout_fee_rate=holdout_fee_rate,
        holdout_fill_buffer_bps=holdout_fill_buffer_bps,
        leaderboard_path=f"analysis/remote_runs/{run_id}/daily_leaderboard.csv",
        checkpoint_root=f"pufferlib_market/checkpoints/{run_id}_daily",
    )

    commands: list[tuple[str, ...]] = [
        tuple(
            build_run_crypto_lora_batch_cmd(
                run_id=run_id,
                symbols=normalized,
                data_root=remote_hourly_data_root,
                output_root="chronos2_finetuned",
                results_dir=lora_results_dir,
                preaugs=preaugs,
                context_lengths=context_lengths,
                learning_rates=learning_rates,
                num_steps=num_steps,
                prediction_length=prediction_length,
                lora_r=lora_r,
            )
        ),
        tuple(
            build_promote_lora_cmd(
                report_dir=lora_results_dir,
                output_dir="hyperparams/chronos2/hourly",
                symbols=normalized,
                run_id=run_id,
            )
        ),
        tuple(
            build_forecast_cache_cmd(
                symbols=normalized,
                data_root=remote_hourly_data_root,
                forecast_cache_root=forecast_cache_root,
                lookback_hours=lookback_hours,
                output_json=forecast_mae_path,
                force_rebuild=True,
            )
        ),
        tuple(
            build_export_hourly_forecast_cmd(
                symbols=normalized,
                data_root=remote_hourly_data_root,
                forecast_cache_root=forecast_cache_root,
                output_path=hourly_train_data_path,
                start_date=window.train_start,
                end_date=window.train_end,
                feature_lag=feature_lag,
                min_hours=train_hours,
                min_coverage=min_coverage,
            )
        ),
        tuple(
            build_export_hourly_forecast_cmd(
                symbols=normalized,
                data_root=remote_hourly_data_root,
                forecast_cache_root=forecast_cache_root,
                output_path=hourly_val_data_path,
                start_date=window.val_start,
                end_date=window.val_end,
                feature_lag=feature_lag,
                min_hours=val_hours,
                min_coverage=min_coverage,
            )
        ),
        tuple(
            build_export_daily_fused_cmd(
                symbols=normalized,
                data_root=remote_daily_data_root,
                hourly_root=remote_hourly_data_root,
                daily_forecast_root=daily_forecast_root,
                hourly_forecast_root=forecast_cache_root,
                output_path=daily_train_data_path,
                start_date=train_start_day,
                end_date=train_end_day,
                min_days=daily_min_days,
                zscore_window=zscore_window,
            )
        ),
        tuple(
            build_export_daily_fused_cmd(
                symbols=normalized,
                data_root=remote_daily_data_root,
                hourly_root=remote_hourly_data_root,
                daily_forecast_root=daily_forecast_root,
                hourly_forecast_root=forecast_cache_root,
                output_path=daily_val_data_path,
                start_date=val_start_day,
                end_date=val_end_day,
                min_days=daily_val_min_days,
                zscore_window=zscore_window,
            )
        ),
    ]
    commands.extend(hourly_plan.commands)
    commands.extend(daily_plan.commands)

    return RemoteChronosComparisonPlan(
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
        hourly_train_data_path=hourly_train_data_path,
        hourly_val_data_path=hourly_val_data_path,
        daily_train_data_path=daily_train_data_path,
        daily_val_data_path=daily_val_data_path,
        hourly_leaderboard_path=hourly_plan.leaderboard_path,
        hourly_checkpoint_root=hourly_plan.checkpoint_root,
        daily_leaderboard_path=daily_plan.leaderboard_path,
        daily_checkpoint_root=daily_plan.checkpoint_root,
        commands=tuple(commands),
    )


def shell_join(cmd: Sequence[str]) -> str:
    return shlex.join([str(token) for token in cmd])


def render_remote_pipeline_script(
    *,
    remote_dir: str,
    remote_env: str,
    plan: RemotePipelinePlan | RemoteAutoresearchPlan,
) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(remote_dir))}",
        f"source {shlex.quote(str(remote_env).rstrip('/'))}/bin/activate",
        'export PYTHONPATH="$PWD:$PWD/PufferLib:${PYTHONPATH:-}"',
        f"mkdir -p {shlex.quote(plan.remote_run_dir)}",
        "echo \"[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting remote pipeline\"",
        'echo "+ rm -rf pufferlib_market/build pufferlib_market/binding*.so"',
        "rm -rf pufferlib_market/build pufferlib_market/binding*.so",
        'echo "+ python pufferlib_market/setup.py build_ext --inplace --force"',
        "python pufferlib_market/setup.py build_ext --inplace --force",
        'echo "+ python -c \'import pufferlib_market.binding; print(\\\"binding OK\\\")\'"',
        "python -c 'import pufferlib_market.binding; print(\"binding OK\")'",
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
    "RemoteAutoresearchPlan",
    "RemoteChronosComparisonPlan",
    "RemotePipelinePlan",
    "build_remote_autoresearch_plan",
    "build_export_daily_fused_cmd",
    "build_hourly_train_val_window_from_bounds",
    "build_autoresearch_cmd",
    "build_export_hourly_forecast_cmd",
    "build_forecast_cache_cmd",
    "build_promote_lora_cmd",
    "build_remote_chronos_compare_plan",
    "build_remote_hourly_chronos_rl_plan",
    "build_run_crypto_lora_batch_cmd",
    "compute_daily_overlap_bounds",
    "compute_hourly_overlap_bounds",
    "compute_hourly_train_val_window",
    "load_daily_index",
    "load_hourly_index",
    "normalize_symbols",
    "parse_csv_tokens",
    "render_remote_pipeline_script",
    "resolve_daily_symbol_path",
    "resolve_hourly_symbol_path",
    "shell_join",
]
