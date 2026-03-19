#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_SHORTABLE_STOCKS: tuple[str, ...] = (
    "YELP",
    "EBAY",
    "TRIP",
    "MTCH",
    "KIND",
    "ANGI",
    "Z",
    "EXPE",
    "BKNG",
    "NWSA",
    "NYT",
)


@dataclass
class RunSummary:
    symbol: str
    stage: str
    success: bool
    return_code: int
    log_json: Path | None = None
    gross_pnl: float | None = None
    total_reward: float | None = None
    train_steps_per_second: float | None = None
    error: str | None = None


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def _parse_csv_list(raw: str) -> list[str]:
    values: list[str] = []
    for token in str(raw).split(","):
        token = token.strip().upper()
        if token:
            values.append(token)
    # Preserve order while deduplicating
    seen: set[str] = set()
    ordered: list[str] = []
    for token in values:
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def _check_symbols_exist(symbols: Sequence[str], data_root: Path) -> list[str]:
    available: list[str] = []
    for symbol in symbols:
        path = data_root / f"{symbol}.csv"
        if path.exists():
            available.append(symbol)
        else:
            logger.warning("Skipping {} (missing data: {})", symbol, path)
    return available


def _run_command(cmd: Sequence[str], *, cwd: Path) -> int:
    quoted = " ".join(cmd)
    logger.info("Running command: {}", quoted)
    result = subprocess.run(list(cmd), cwd=str(cwd), check=False)
    if result.returncode != 0:
        logger.error("Command failed (rc={}): {}", result.returncode, quoted)
    return int(result.returncode)


def _safe_read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _extract_metrics(path: Path) -> dict[str, float]:
    payload = _safe_read_json(path)
    if not payload:
        return {}
    out: dict[str, float] = {}
    for key in ("gross_pnl", "total_reward", "train_steps_per_second"):
        value = payload.get(key)
        try:
            out[key] = float(value)
        except Exception:
            continue
    return out


def _rank_symbols(stage1: Iterable[RunSummary], top_k: int) -> list[str]:
    scored: list[tuple[float, str]] = []
    for row in stage1:
        if not row.success:
            continue
        score = row.gross_pnl
        if score is None:
            score = row.total_reward
        if score is None:
            continue
        scored.append((float(score), row.symbol))
    scored.sort(reverse=True)
    return [symbol for _, symbol in scored[: max(1, int(top_k))]]


def _write_summary(run_dir: Path, payload: dict[str, Any]) -> Path:
    out = run_dir / "summary.json"
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return out


def _stage_train(
    *,
    stage_name: str,
    symbols: Sequence[str],
    stage_dir: Path,
    repo_root: Path,
    python_exec: str,
    data_root: Path,
    forecast_cache_root: Path,
    forecast_horizons: str,
    timesteps: int,
    validation_interval: int,
    device: str,
    env_backend: str,
    auto_correlated_count: int,
    correlation_min_abs: float,
    explicit_correlated_map: dict[str, str] | None = None,
) -> list[RunSummary]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    rows: list[RunSummary] = []
    history_csv = stage_dir / "history.csv"

    for symbol in symbols:
        log_json = stage_dir / f"{symbol}.json"
        cmd = [
            python_exec,
            "training/run_fastppo.py",
            "--symbol",
            symbol,
            "--data-root",
            str(data_root),
            "--context-len",
            "128",
            "--horizon",
            "1",
            "--total-timesteps",
            str(int(timesteps)),
            "--num-envs",
            "4",
            "--learning-rate",
            "3e-4",
            "--gamma",
            "0.995",
            "--device",
            device,
            "--env-backend",
            env_backend,
            "--forecast-cache-root",
            str(forecast_cache_root),
            "--forecast-horizons",
            forecast_horizons,
            "--auto-correlated-count",
            str(int(auto_correlated_count)),
            "--correlation-min-abs",
            f"{float(correlation_min_abs):.6f}",
            "--validation-interval-timesteps",
            str(int(validation_interval)),
            "--log-json",
            str(log_json),
            "--history-csv",
            str(history_csv),
        ]
        explicit = None
        if explicit_correlated_map is not None:
            explicit = explicit_correlated_map.get(symbol)
        if explicit:
            cmd.extend(["--correlated-symbols", explicit])

        rc = _run_command(cmd, cwd=repo_root)
        if rc != 0:
            rows.append(
                RunSummary(
                    symbol=symbol,
                    stage=stage_name,
                    success=False,
                    return_code=rc,
                    log_json=log_json if log_json.exists() else None,
                    error=f"training command failed (rc={rc})",
                )
            )
            continue

        metrics = _extract_metrics(log_json)
        rows.append(
            RunSummary(
                symbol=symbol,
                stage=stage_name,
                success=True,
                return_code=0,
                log_json=log_json,
                gross_pnl=metrics.get("gross_pnl"),
                total_reward=metrics.get("total_reward"),
                train_steps_per_second=metrics.get("train_steps_per_second"),
            )
        )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Full-auto Chronos2 + fast PPO loop for short-only Alpaca stock universe."
    )
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SHORTABLE_STOCKS),
        help="Comma-separated short-only stock symbols.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--forecast-cache-root", type=Path, default=Path("binanceneural/forecast_cache_shortable_stocks"))
    parser.add_argument("--forecast-horizons", default="1,4,24")
    parser.add_argument("--cache-lookback-hours", type=float, default=float(24 * 365))
    parser.add_argument("--cache-no-compute-mae", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("experiments/shortable_full_auto"))
    parser.add_argument("--python-exec", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--env-backend", choices=("fast", "python"), default="fast")

    parser.add_argument("--skip-tune", action="store_true")
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--skip-stage2", action="store_true")

    parser.add_argument("--tune-grid", choices=("ultraquick", "quick", "full"), default="quick")
    parser.add_argument("--prediction-length", type=int, default=24)
    parser.add_argument("--stage1-timesteps", type=int, default=65_536)
    parser.add_argument("--stage2-timesteps", type=int, default=131_072)
    parser.add_argument("--stage1-validation-interval", type=int, default=16_384)
    parser.add_argument("--stage2-validation-interval", type=int, default=32_768)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--cohort-size", type=int, default=3)
    parser.add_argument("--cohort-min-abs-corr", type=float, default=0.2)
    parser.add_argument("--auto-correlated-count", type=int, default=3)
    args = parser.parse_args(argv)

    symbols = _parse_csv_list(args.symbols)
    symbols = _check_symbols_exist(symbols, args.data_root)
    if not symbols:
        raise SystemExit("No symbols with available data.")

    run_dir = (args.output_root / f"run_{_now_tag()}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: {}", run_dir)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "symbols": symbols,
        "short_only_universe": list(DEFAULT_SHORTABLE_STOCKS),
        "data_root": str(args.data_root.resolve()),
        "forecast_cache_root": str(args.forecast_cache_root.resolve()),
        "forecast_horizons": args.forecast_horizons,
        "cache_lookback_hours": float(args.cache_lookback_hours),
        "cache_no_compute_mae": bool(args.cache_no_compute_mae),
        "device": args.device,
        "env_backend": args.env_backend,
        "stage1_timesteps": int(args.stage1_timesteps),
        "stage2_timesteps": int(args.stage2_timesteps),
        "prediction_length": int(args.prediction_length),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    rc_tune = 0
    if not args.skip_tune:
        tune_cmd = [
            args.python_exec,
            "hyperparam_chronos_hourly.py",
            "--symbols",
            *symbols,
            "--prediction-length",
            str(int(args.prediction_length)),
            "--objective",
            "composite",
            "--smoothness-weight",
            "0.35",
            "--direction-bonus",
            "0.05",
            "--cohort-size",
            str(int(args.cohort_size)),
            "--cohort-min-abs-corr",
            f"{float(args.cohort_min_abs_corr):.6f}",
            "--enable-cross-learning",
            "--save-hyperparams",
            "--device",
            args.device,
            "--output",
            str(run_dir / "chronos_tuning.json"),
        ]
        if args.tune_grid == "quick":
            tune_cmd.append("--quick")
        elif args.tune_grid == "ultraquick":
            tune_cmd.append("--ultraquick")
        rc_tune = _run_command(tune_cmd, cwd=PROJECT_ROOT)

    rc_cache = 0
    if not args.skip_cache:
        cache_cmd = [
            args.python_exec,
            "scripts/build_hourly_forecast_caches.py",
            "--symbols",
            ",".join(symbols),
            "--data-root",
            str(args.data_root),
            "--forecast-cache-root",
            str(args.forecast_cache_root),
            "--horizons",
            args.forecast_horizons,
            "--lookback-hours",
            str(float(args.cache_lookback_hours)),
            "--output-json",
            str(run_dir / "forecast_mae.json"),
        ]
        if args.cache_no_compute_mae:
            cache_cmd.append("--no-compute-mae")
        rc_cache = _run_command(cache_cmd, cwd=PROJECT_ROOT)

    stage1_rows: list[RunSummary] = []
    if not args.skip_stage1:
        stage1_rows = _stage_train(
            stage_name="stage1",
            symbols=symbols,
            stage_dir=run_dir / "stage1",
            repo_root=PROJECT_ROOT,
            python_exec=args.python_exec,
            data_root=args.data_root,
            forecast_cache_root=args.forecast_cache_root,
            forecast_horizons=args.forecast_horizons,
            timesteps=args.stage1_timesteps,
            validation_interval=args.stage1_validation_interval,
            device=args.device,
            env_backend=args.env_backend,
            auto_correlated_count=args.auto_correlated_count,
            correlation_min_abs=args.cohort_min_abs_corr,
        )

    top_symbols = _rank_symbols(stage1_rows, args.top_k) if stage1_rows else []
    if not top_symbols:
        top_symbols = symbols[: max(1, int(args.top_k))]

    stage2_rows: list[RunSummary] = []
    if not args.skip_stage2:
        correlated_map: dict[str, str] = {}
        for symbol in top_symbols:
            peers = [s for s in top_symbols if s != symbol]
            correlated_map[symbol] = ",".join(peers)

        stage2_rows = _stage_train(
            stage_name="stage2",
            symbols=top_symbols,
            stage_dir=run_dir / "stage2",
            repo_root=PROJECT_ROOT,
            python_exec=args.python_exec,
            data_root=args.data_root,
            forecast_cache_root=args.forecast_cache_root,
            forecast_horizons=args.forecast_horizons,
            timesteps=args.stage2_timesteps,
            validation_interval=args.stage2_validation_interval,
            device=args.device,
            env_backend=args.env_backend,
            auto_correlated_count=max(args.auto_correlated_count, 2),
            correlation_min_abs=args.cohort_min_abs_corr,
            explicit_correlated_map=correlated_map,
        )

    summary = {
        "run_dir": str(run_dir),
        "symbols": symbols,
        "top_symbols_stage2": top_symbols,
        "tune_return_code": rc_tune,
        "cache_return_code": rc_cache,
        "stage1": [row.__dict__ for row in stage1_rows],
        "stage2": [row.__dict__ for row in stage2_rows],
    }
    summary_path = _write_summary(run_dir, summary)
    logger.info("Summary written: {}", summary_path)
    logger.info("Completed full-auto shortable run for {} symbols.", len(symbols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
