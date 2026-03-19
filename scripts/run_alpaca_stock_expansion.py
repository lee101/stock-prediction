#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from newnanoalpacahourlyexp.data import _effective_min_history_bars
from src.alpaca_stock_expansion import (
    StockExpansionCandidate,
    build_hourly_return_correlation_cohorts,
    candidate_training_plan,
    count_candidate_history_rows,
    default_stock_expansion_candidates,
    load_stock_expansion_manifest,
    split_candidates_by_history,
    write_stock_expansion_manifest,
)


DEFAULT_CHECKPOINT = (
    "binanceneural/checkpoints/"
    "alpaca_cross_global_mixed14_robust_short_seq128_lb4000_20260205_2319/epoch_004.pt"
)
LIVE_20260318_BASE_STOCK_SYMBOLS = "NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT,AAPL,MSFT,META,TSLA,NET,BKNG,EBAY,EXPE"
LIVE_20260319_BASE_STOCK_SYMBOLS = f"{LIVE_20260318_BASE_STOCK_SYMBOLS},ITUB"
LIVE_20260319_POST_BTG_BASE_STOCK_SYMBOLS = f"{LIVE_20260319_BASE_STOCK_SYMBOLS},BTG"
DEFAULT_BASE_STOCK_SYMBOLS = f"{LIVE_20260319_POST_BTG_BASE_STOCK_SYMBOLS},ABEV"
BASE_STOCK_UNIVERSE_ALIASES = {
    "stock19": LIVE_20260318_BASE_STOCK_SYMBOLS,
    "live20260318": LIVE_20260318_BASE_STOCK_SYMBOLS,
    "live20260319": LIVE_20260319_BASE_STOCK_SYMBOLS,
    "live20260319_post_btg": LIVE_20260319_POST_BTG_BASE_STOCK_SYMBOLS,
    "live20260319_post_abev": DEFAULT_BASE_STOCK_SYMBOLS,
}


def _parse_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [token.strip().upper() for token in str(raw).split(",") if token.strip()]


def _resolve_base_symbols(raw: str | None) -> list[str]:
    value = str(raw or "").strip()
    if not value:
        value = DEFAULT_BASE_STOCK_SYMBOLS
    value = BASE_STOCK_UNIVERSE_ALIASES.get(value, value)
    symbols = _parse_csv(value)
    if not symbols:
        raise ValueError(f"Unable to resolve base stock symbols from {raw!r}")
    return symbols


def _run_command(cmd: Sequence[str]) -> None:
    completed = subprocess.run(list(cmd), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(cmd)}")


def _build_forecast_caches(
    *,
    symbols: Iterable[str],
    data_root: Path,
    forecast_cache_root: Path,
    lookback_hours: float,
    output_json: Path,
    force_rebuild: bool,
) -> None:
    symbol_csv = ",".join(sorted({str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()}))
    if not symbol_csv:
        return
    cmd = [
        sys.executable,
        "scripts/build_hourly_forecast_caches.py",
        "--symbols",
        symbol_csv,
        "--data-root",
        str(data_root),
        "--forecast-cache-root",
        str(forecast_cache_root),
        "--horizons",
        "1,24",
        "--lookback-hours",
        str(float(lookback_hours)),
        "--output-json",
        str(output_json),
    ]
    if force_rebuild:
        cmd.append("--force-rebuild")
    _run_command(cmd)


def _read_metrics(output_dir: Path) -> dict[str, object]:
    metrics_path = Path(output_dir) / "metrics.json"
    return json.loads(metrics_path.read_text())


def _trial_done(output_dir: Path) -> bool:
    return (Path(output_dir) / "metrics.json").exists()


def _resolve_baseline_metrics_path(*, output_dir: Path, baseline_source_dir: Path | None) -> Path:
    source_dir = Path(baseline_source_dir) if baseline_source_dir is not None else Path(output_dir) / "baseline"
    return Path(source_dir) / "metrics.json"


def _load_forecast_cache_mae(path: Path) -> dict[str, dict[int, float]]:
    path = Path(path)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    rows = payload.get("metrics") or []
    result: dict[str, dict[int, float]] = {}
    for row in rows:
        symbol = str((row or {}).get("symbol") or "").strip().upper()
        if not symbol:
            continue
        try:
            horizon = int((row or {}).get("horizon_hours"))
            mae_percent = float((row or {}).get("mae_percent"))
        except (TypeError, ValueError):
            continue
        result.setdefault(symbol, {})[horizon] = mae_percent
    return result


def _coerce_forecast_cache_mae_source_path(source: Path | None) -> Path | None:
    if source is None:
        return None
    source_path = Path(source)
    if source_path.is_dir():
        source_path = source_path / "forecast_cache_mae.json"
    return source_path


def _resolve_forecast_cache_mae_path(
    *,
    output_dir: Path,
    skip_cache_build: bool,
    forecast_cache_mae_source: Path | None,
    require_existing_summary: bool,
) -> Path:
    output_path = Path(output_dir) / "forecast_cache_mae.json"
    if output_path.exists():
        return output_path

    source_path = _coerce_forecast_cache_mae_source_path(forecast_cache_mae_source)
    if source_path is not None:
        if not source_path.exists():
            raise SystemExit(f"Forecast cache MAE source does not exist: {source_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.resolve() != output_path.resolve():
            shutil.copyfile(source_path, output_path)
        return output_path

    if skip_cache_build and require_existing_summary:
        raise SystemExit(
            "Missing forecast cache MAE summary for --skip-cache-build with candidate MAE gates. "
            "Pass --forecast-cache-mae-source pointing to an existing forecast_cache_mae.json, "
            "or rerun without --skip-cache-build."
        )

    return output_path


def _candidate_mae_snapshot(mae_by_symbol: dict[str, dict[int, float]], symbol: str) -> dict[str, float]:
    horizons = mae_by_symbol.get(str(symbol or "").strip().upper(), {})
    return {
        "candidate_mae_h1_percent": float(horizons.get(1, 0.0) or 0.0),
        "candidate_mae_h24_percent": float(horizons.get(24, 0.0) or 0.0),
    }


def _candidate_mae_gate_reason(
    *,
    symbol: str,
    mae_snapshot: dict[str, float],
    max_h1_mae_percent: float,
    max_h24_mae_percent: float,
) -> str:
    failures: list[str] = []
    h1 = float(mae_snapshot.get("candidate_mae_h1_percent", 0.0) or 0.0)
    h24 = float(mae_snapshot.get("candidate_mae_h24_percent", 0.0) or 0.0)
    if float(max_h1_mae_percent) > 0.0 and h1 > float(max_h1_mae_percent):
        failures.append(f"h1 MAE {h1:.4f}% exceeds gate {float(max_h1_mae_percent):.4f}%")
    if float(max_h24_mae_percent) > 0.0 and h24 > float(max_h24_mae_percent):
        failures.append(f"h24 MAE {h24:.4f}% exceeds gate {float(max_h24_mae_percent):.4f}%")
    if not failures:
        return ""
    return f"forecast cache gate for {str(symbol or '').strip().upper()}: {'; '.join(failures)}"


def _baseline_metric(metrics: dict[str, object], key: str) -> float:
    try:
        return float(metrics.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _candidate_direction_lists(
    candidate: StockExpansionCandidate,
    *,
    base_long_only_symbols: Sequence[str],
    base_short_only_symbols: Sequence[str],
) -> tuple[list[str], list[str]]:
    long_only = [str(symbol).strip().upper() for symbol in base_long_only_symbols if str(symbol).strip()]
    short_only = [str(symbol).strip().upper() for symbol in base_short_only_symbols if str(symbol).strip()]
    if candidate.side == "long" and candidate.symbol not in long_only:
        long_only.append(candidate.symbol)
    elif candidate.side == "short" and candidate.symbol not in short_only:
        short_only.append(candidate.symbol)
    return long_only, short_only


def _resolve_cache_symbols(
    *,
    base_symbols: Sequence[str],
    ready_candidates: Sequence[StockExpansionCandidate],
    candidate_only_cache_build: bool,
) -> list[str]:
    cache_symbols = [candidate.symbol for candidate in ready_candidates]
    if not candidate_only_cache_build:
        cache_symbols = list(base_symbols) + cache_symbols
    return cache_symbols


def _run_trial(
    *,
    symbols: Sequence[str],
    checkpoint: Path,
    stock_data_root: Path,
    forecast_cache_root: Path,
    output_dir: Path,
    moving_average_windows: str,
    min_history_hours: int,
    eval_days: float,
    long_only_symbols: Sequence[str],
    short_only_symbols: Sequence[str],
    baseline_metrics: dict[str, object] | None,
    enable_baseline_gate: bool,
    baseline_early_exit_min_steps: int,
    baseline_stage1_progress: float,
    baseline_stage2_progress: float,
    baseline_stage3_progress: float,
    baseline_return_tolerance: float,
    baseline_sortino_tolerance: float,
    baseline_max_drawdown_tolerance: float,
    initial_state: Path | None = None,
) -> dict[str, object]:
    cmd = [
        sys.executable,
        "-m",
        "newnanoalpacahourlyexp.run_hourly_trader_sim",
        "--symbols",
        ",".join(symbols),
        "--checkpoint",
        str(checkpoint),
        "--sequence-length",
        "128",
        "--horizon",
        "1",
        "--forecast-horizons",
        "1,24",
        "--context-lengths",
        "64,96,192",
        "--forecast-cache-root",
        str(forecast_cache_root),
        "--stock-data-root",
        str(stock_data_root),
        "--allocation-pct",
        "0.2",
        "--allocation-mode",
        "portfolio",
        "--cache-only",
        "--allow-short",
        "--eval-days",
        str(float(eval_days)),
        "--moving-average-windows",
        str(moving_average_windows),
        "--min-history-hours",
        str(int(min_history_hours)),
        "--no-drawdown-profit-early-exit",
        "--output-dir",
        str(output_dir),
    ]
    if long_only_symbols:
        cmd.extend(["--long-only-symbols", ",".join(long_only_symbols)])
    if short_only_symbols:
        cmd.extend(["--short-only-symbols", ",".join(short_only_symbols)])
    if initial_state is not None:
        cmd.extend(["--initial-state", str(initial_state)])
    if enable_baseline_gate and baseline_metrics is not None:
        cmd.extend(
            [
                "--baseline-comparability-early-exit",
                "--baseline-total-return",
                str(_baseline_metric(baseline_metrics, "total_return")),
                "--baseline-sortino",
                str(_baseline_metric(baseline_metrics, "sortino")),
                "--baseline-max-drawdown",
                str(_baseline_metric(baseline_metrics, "max_drawdown")),
                "--baseline-early-exit-min-steps",
                str(int(baseline_early_exit_min_steps)),
                "--baseline-stage1-progress",
                str(float(baseline_stage1_progress)),
                "--baseline-stage2-progress",
                str(float(baseline_stage2_progress)),
                "--baseline-stage3-progress",
                str(float(baseline_stage3_progress)),
                "--baseline-return-tolerance",
                str(float(baseline_return_tolerance)),
                "--baseline-sortino-tolerance",
                str(float(baseline_sortino_tolerance)),
                "--baseline-max-drawdown-tolerance",
                str(float(baseline_max_drawdown_tolerance)),
            ]
        )
    _run_command(cmd)
    return _read_metrics(output_dir)


def _result_row(
    *,
    symbol: str,
    side: str,
    sector: str,
    priority: int,
    thesis: str,
    metrics: dict[str, object],
    summary_path: Path,
    baseline_metrics: dict[str, object] | None,
    sim_ran: bool = True,
    mae_snapshot: dict[str, float] | None = None,
    training_plan: dict[str, object] | None = None,
) -> dict[str, object]:
    total_return = _baseline_metric(metrics, "total_return")
    sortino = _baseline_metric(metrics, "sortino")
    max_drawdown = _baseline_metric(metrics, "max_drawdown")
    baseline_total_return = _baseline_metric(baseline_metrics or {}, "total_return")
    baseline_sortino = _baseline_metric(baseline_metrics or {}, "sortino")
    baseline_max_drawdown = _baseline_metric(baseline_metrics or {}, "max_drawdown")
    mae_snapshot = dict(mae_snapshot or {})
    training_plan = dict(training_plan or {})
    return {
        "symbol": symbol,
        "side": side,
        "sector": sector,
        "priority": int(priority),
        "sim_ran": bool(sim_ran),
        "total_return": total_return,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "final_equity": _baseline_metric(metrics, "final_equity"),
        "num_fills": int(_baseline_metric(metrics, "num_fills")),
        "terminated_early": bool(metrics.get("terminated_early", False)),
        "termination_reason": str(metrics.get("termination_reason") or ""),
        "termination_progress_fraction": _baseline_metric(metrics, "termination_progress_fraction"),
        "total_return_delta_vs_baseline": total_return - baseline_total_return,
        "sortino_delta_vs_baseline": sortino - baseline_sortino,
        "max_drawdown_delta_vs_baseline": max_drawdown - baseline_max_drawdown,
        "summary_path": str(summary_path),
        "hourly_config_exists": bool(training_plan.get("hourly_config_exists", False)),
        "recommended_next_step": str(training_plan.get("recommended_next_step") or ""),
        "recommended_reason": str(training_plan.get("recommended_reason") or ""),
        "recommended_peer_symbols": list(training_plan.get("recommended_peer_symbols") or []),
        "hourly_tuning_command": str(training_plan.get("hourly_tuning_command") or ""),
        "lora_command": str(training_plan.get("lora_command") or ""),
        "thesis": thesis,
        "candidate_mae_h1_percent": float(mae_snapshot.get("candidate_mae_h1_percent", 0.0) or 0.0),
        "candidate_mae_h24_percent": float(mae_snapshot.get("candidate_mae_h24_percent", 0.0) or 0.0),
    }


def _write_cache_gate_summary(
    *,
    candidate_output_dir: Path,
    symbol: str,
    side: str,
    mae_snapshot: dict[str, float],
    gate_reason: str,
    max_h1_mae_percent: float,
    max_h24_mae_percent: float,
) -> Path:
    candidate_output_dir = Path(candidate_output_dir)
    candidate_output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = candidate_output_dir / "cache_gate.json"
    payload = {
        "symbol": str(symbol).strip().upper(),
        "side": str(side).strip(),
        "gate_reason": gate_reason,
        "candidate_mae_h1_percent": float(mae_snapshot.get("candidate_mae_h1_percent", 0.0) or 0.0),
        "candidate_mae_h24_percent": float(mae_snapshot.get("candidate_mae_h24_percent", 0.0) or 0.0),
        "max_h1_mae_percent": float(max_h1_mae_percent),
        "max_h24_mae_percent": float(max_h24_mae_percent),
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")
    return summary_path


def _skipped_result_row(
    *,
    candidate: StockExpansionCandidate,
    baseline_metrics: dict[str, object] | None,
    summary_path: Path,
    mae_snapshot: dict[str, float],
    gate_reason: str,
    training_plan: dict[str, object] | None = None,
) -> dict[str, object]:
    return _result_row(
        symbol=candidate.symbol,
        side=candidate.side,
        sector=candidate.sector,
        priority=candidate.priority,
        thesis=candidate.thesis,
        metrics={
            "total_return": -1.0,
            "sortino": -999.0,
            "max_drawdown": 1.0,
            "final_equity": 0.0,
            "num_fills": 0.0,
            "terminated_early": True,
            "termination_reason": gate_reason,
            "termination_progress_fraction": 0.0,
        },
        summary_path=summary_path,
        baseline_metrics=baseline_metrics,
        sim_ran=False,
        mae_snapshot=mae_snapshot,
        training_plan=training_plan,
    )


def _assess_promotion(
    row: dict[str, object],
    *,
    min_return_delta: float,
    min_sortino_delta: float,
    max_drawdown_delta: float,
) -> dict[str, object]:
    if str(row.get("symbol") or "").upper() == "BASE":
        return {
            "promotable": False,
            "promotion_reason": "Baseline row is not a candidate for promotion.",
            "promotion_failures": ["baseline row"],
        }

    failures: list[str] = []
    total_return_delta = float(row.get("total_return_delta_vs_baseline", 0.0) or 0.0)
    sortino_delta = float(row.get("sortino_delta_vs_baseline", 0.0) or 0.0)
    max_drawdown_delta_vs_baseline = float(row.get("max_drawdown_delta_vs_baseline", 0.0) or 0.0)
    terminated_early = bool(row.get("terminated_early", False))
    termination_reason = str(row.get("termination_reason") or "").strip()

    if terminated_early:
        failures.append(termination_reason or "candidate terminated early")
    if total_return_delta < float(min_return_delta):
        failures.append(
            f"return delta {total_return_delta:.6f} below promotion floor {float(min_return_delta):.6f}"
        )
    if sortino_delta < float(min_sortino_delta):
        failures.append(f"sortino delta {sortino_delta:.4f} below promotion floor {float(min_sortino_delta):.4f}")
    if max_drawdown_delta_vs_baseline > float(max_drawdown_delta):
        failures.append(
            f"max drawdown delta {max_drawdown_delta_vs_baseline:.6f} exceeds promotion cap "
            f"{float(max_drawdown_delta):.6f}"
        )

    if failures:
        return {
            "promotable": False,
            "promotion_reason": "; ".join(failures),
            "promotion_failures": failures,
        }

    return {
        "promotable": True,
        "promotion_reason": (
            f"return delta {total_return_delta:.6f}, "
            f"sortino delta {sortino_delta:.4f}, "
            f"max drawdown delta {max_drawdown_delta_vs_baseline:.6f}"
        ),
        "promotion_failures": [],
    }


def _build_promotion_summary(
    rows: Sequence[dict[str, object]],
    *,
    min_return_delta: float,
    min_sortino_delta: float,
    max_drawdown_delta: float,
) -> dict[str, object]:
    thresholds = {
        "min_return_delta": float(min_return_delta),
        "min_sortino_delta": float(min_sortino_delta),
        "max_drawdown_delta": float(max_drawdown_delta),
    }
    promotable_candidates = [row for row in rows if bool(row.get("promotable", False))]
    promoted_row = promotable_candidates[0] if promotable_candidates else None
    if promoted_row is not None:
        reason = str(promoted_row.get("promotion_reason") or "").strip()
        promotion_reason = (
            f"{promoted_row['symbol']} cleared promotion thresholds. {reason}".strip()
        )
    else:
        rejected_symbols = [str(row.get("symbol") or "") for row in rows if str(row.get("symbol") or "").upper() != "BASE"]
        joined = ",".join(rejected_symbols)
        promotion_reason = f"No candidate met promotion thresholds. Rejected candidates: {joined}"
    return {
        "promote": promoted_row is not None,
        "promoted_symbol": None if promoted_row is None else str(promoted_row["symbol"]),
        "promotion_reason": promotion_reason,
        "thresholds": thresholds,
        "promotable_candidates": [str(row["symbol"]) for row in promotable_candidates],
    }


def _trial_sort_key(row: dict[str, object]) -> tuple[float, float, float, float, float]:
    return (
        1.0 if bool(row.get("sim_ran", True)) else 0.0,
        float(row.get("sortino_delta_vs_baseline", float("-inf"))),
        float(row.get("total_return_delta_vs_baseline", float("-inf"))),
        -float(row.get("max_drawdown_delta_vs_baseline", float("inf"))),
        float(row.get("sortino", float("-inf"))),
    )


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "symbol",
        "side",
        "sector",
        "priority",
        "sim_ran",
        "total_return",
        "sortino",
        "max_drawdown",
        "final_equity",
        "num_fills",
        "terminated_early",
        "termination_progress_fraction",
        "total_return_delta_vs_baseline",
        "sortino_delta_vs_baseline",
        "max_drawdown_delta_vs_baseline",
        "promotable",
        "promotion_reason",
        "candidate_mae_h1_percent",
        "candidate_mae_h24_percent",
        "summary_path",
        "hourly_config_exists",
        "recommended_next_step",
        "recommended_reason",
        "recommended_peer_symbols",
        "hourly_tuning_command",
        "lora_command",
        "thesis",
        "termination_reason",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate stock-universe expansion candidates by adding one symbol at a time to the current live base universe.",
    )
    parser.add_argument("--base-stock-universe", default="live20260319_post_abev")
    parser.add_argument("--base-long-only-symbols", default=None)
    parser.add_argument("--base-short-only-symbols", default=None)
    parser.add_argument("--default-checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--stock-data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--forecast-cache-root", type=Path, default=Path("unified_hourly_experiment/forecast_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/alpaca_stock_expansion_20260318"))
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument(
        "--forecast-cache-mae-source",
        type=Path,
        default=None,
        help="Optional existing forecast_cache_mae.json file or source directory to reuse when skipping cache builds.",
    )
    parser.add_argument("--candidate-symbols", default=None, help="Optional comma-separated subset of candidate symbols.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--moving-average-windows", default="168,600,720")
    parser.add_argument("--min-history-hours", type=int, default=480)
    parser.add_argument("--cache-lookback-hours", type=float, default=5000.0)
    parser.add_argument("--eval-days", type=float, default=120.0)
    parser.add_argument("--skip-cache-build", action="store_true")
    parser.add_argument("--force-cache-rebuild", action="store_true")
    parser.add_argument(
        "--candidate-only-cache-build",
        action="store_true",
        help="Only refresh forecast caches for candidate symbols instead of rebuilding the full base basket.",
    )
    parser.add_argument("--reuse-baseline", action="store_true", default=True)
    parser.add_argument("--force-baseline-rerun", action="store_true")
    parser.add_argument(
        "--baseline-source-dir",
        type=Path,
        default=None,
        help="Optional directory containing an existing baseline metrics.json to reuse across separate candidate runs.",
    )
    parser.add_argument("--reuse-candidate-results", action="store_true", default=True)
    parser.add_argument("--force-candidate-rerun", action="store_true")
    parser.add_argument("--disable-baseline-gate", action="store_true")
    parser.add_argument("--baseline-early-exit-min-steps", type=int, default=40)
    parser.add_argument("--baseline-stage1-progress", type=float, default=0.30)
    parser.add_argument("--baseline-stage2-progress", type=float, default=0.50)
    parser.add_argument("--baseline-stage3-progress", type=float, default=0.75)
    parser.add_argument("--baseline-return-tolerance", type=float, default=0.02)
    parser.add_argument("--baseline-sortino-tolerance", type=float, default=0.50)
    parser.add_argument("--baseline-max-drawdown-tolerance", type=float, default=0.02)
    parser.add_argument("--promotion-min-return-delta", type=float, default=0.0)
    parser.add_argument("--promotion-min-sortino-delta", type=float, default=0.0)
    parser.add_argument("--promotion-max-drawdown-delta", type=float, default=0.0)
    parser.add_argument("--candidate-max-h1-mae-percent", type=float, default=0.0)
    parser.add_argument("--candidate-max-h24-mae-percent", type=float, default=0.0)
    parser.add_argument("--initial-state", type=Path, default=None)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.manifest_path or output_dir / "candidate_manifest.json"
    if manifest_path.exists():
        base_stock_universe, default_checkpoint, candidates = load_stock_expansion_manifest(manifest_path)
        if not base_stock_universe:
            base_stock_universe = str(args.base_stock_universe)
        if not default_checkpoint:
            default_checkpoint = str(args.default_checkpoint)
    else:
        base_stock_universe = str(args.base_stock_universe)
        default_checkpoint = str(args.default_checkpoint)
        candidates = default_stock_expansion_candidates()
        write_stock_expansion_manifest(
            manifest_path,
            base_stock_universe=base_stock_universe,
            default_checkpoint=default_checkpoint,
            candidates=candidates,
        )

    requested = set(_parse_csv(args.candidate_symbols))
    if requested:
        candidates = [candidate for candidate in candidates if candidate.symbol in requested]
    if args.limit and args.limit > 0:
        candidates = list(candidates)[: int(args.limit)]

    base_symbols = _resolve_base_symbols(base_stock_universe)
    base_symbol_set = set(base_symbols)
    base_long_only_symbols = _parse_csv(args.base_long_only_symbols)
    base_short_only_symbols = _parse_csv(args.base_short_only_symbols)

    already_in_base = [candidate for candidate in candidates if candidate.symbol in base_symbol_set]
    if already_in_base:
        payload = [{"symbol": candidate.symbol, "side": candidate.side} for candidate in already_in_base]
        (output_dir / "already_in_base_candidates.json").write_text(json.dumps(payload, indent=2) + "\n")
    candidates = [candidate for candidate in candidates if candidate.symbol not in base_symbol_set]

    required_history_rows = _effective_min_history_bars("AAPL", int(args.min_history_hours))
    ready, insufficient_history, missing = split_candidates_by_history(
        candidates,
        data_root=args.stock_data_root,
        min_history_rows=required_history_rows,
    )
    if missing:
        missing_payload = [{"symbol": candidate.symbol, "side": candidate.side} for candidate in missing]
        (output_dir / "missing_candidates.json").write_text(json.dumps(missing_payload, indent=2) + "\n")
    if insufficient_history:
        insufficient_payload = [
            {
                "symbol": candidate.symbol,
                "side": candidate.side,
                "history_rows": count_candidate_history_rows(candidate.symbol, data_root=args.stock_data_root),
                "required_history_rows": required_history_rows,
            }
            for candidate in insufficient_history
        ]
        (output_dir / "insufficient_history_candidates.json").write_text(
            json.dumps(insufficient_payload, indent=2) + "\n"
        )
    if not ready:
        raise SystemExit("No non-base candidates have sufficient hourly stock data.")

    training_correlation_cohorts = build_hourly_return_correlation_cohorts(
        [*base_symbols, *(candidate.symbol for candidate in ready)],
        data_root=args.stock_data_root,
    )
    training_plans = {
        candidate.symbol: candidate_training_plan(
            candidate.symbol,
            comparison_symbols=base_symbols,
            data_root=args.stock_data_root,
            correlation_cohorts=training_correlation_cohorts,
        )
        for candidate in ready
    }

    mae_summary_required = (
        float(args.candidate_max_h1_mae_percent) > 0.0 or float(args.candidate_max_h24_mae_percent) > 0.0
    )
    mae_summary_path = output_dir / "forecast_cache_mae.json"
    if not args.skip_cache_build:
        cache_symbols = _resolve_cache_symbols(
            base_symbols=base_symbols,
            ready_candidates=ready,
            candidate_only_cache_build=bool(args.candidate_only_cache_build),
        )
        _build_forecast_caches(
            symbols=cache_symbols,
            data_root=args.stock_data_root,
            forecast_cache_root=args.forecast_cache_root,
            lookback_hours=float(args.cache_lookback_hours),
            output_json=mae_summary_path,
            force_rebuild=bool(args.force_cache_rebuild),
        )
    mae_summary_path = _resolve_forecast_cache_mae_path(
        output_dir=output_dir,
        skip_cache_build=bool(args.skip_cache_build),
        forecast_cache_mae_source=args.forecast_cache_mae_source,
        require_existing_summary=mae_summary_required,
    )
    mae_by_symbol = _load_forecast_cache_mae(mae_summary_path)

    checkpoint_path = Path(default_checkpoint).expanduser().resolve()
    baseline_dir = output_dir / "baseline"
    baseline_metrics_path = _resolve_baseline_metrics_path(
        output_dir=output_dir,
        baseline_source_dir=args.baseline_source_dir,
    )
    baseline_summary_path = baseline_metrics_path
    if bool(args.reuse_baseline) and not bool(args.force_baseline_rerun) and baseline_metrics_path.exists():
        baseline_metrics = json.loads(baseline_metrics_path.read_text())
    else:
        baseline_metrics = _run_trial(
            symbols=base_symbols,
            checkpoint=checkpoint_path,
            stock_data_root=args.stock_data_root,
            forecast_cache_root=args.forecast_cache_root,
            output_dir=baseline_dir,
            moving_average_windows=str(args.moving_average_windows),
            min_history_hours=int(args.min_history_hours),
            eval_days=float(args.eval_days),
            long_only_symbols=base_long_only_symbols,
            short_only_symbols=base_short_only_symbols,
            baseline_metrics=None,
            enable_baseline_gate=False,
            baseline_early_exit_min_steps=int(args.baseline_early_exit_min_steps),
            baseline_stage1_progress=float(args.baseline_stage1_progress),
            baseline_stage2_progress=float(args.baseline_stage2_progress),
            baseline_stage3_progress=float(args.baseline_stage3_progress),
            baseline_return_tolerance=float(args.baseline_return_tolerance),
            baseline_sortino_tolerance=float(args.baseline_sortino_tolerance),
            baseline_max_drawdown_tolerance=float(args.baseline_max_drawdown_tolerance),
            initial_state=args.initial_state,
        )
        baseline_summary_path = baseline_dir / "metrics.json"

    baseline_row = _result_row(
        symbol="BASE",
        side="mixed",
        sector="base",
        priority=999,
        thesis="Current live base stock universe baseline.",
        metrics=baseline_metrics,
        summary_path=baseline_summary_path,
        baseline_metrics=None,
        sim_ran=True,
        training_plan=None,
    )
    baseline_row["lora_command"] = ""
    baseline_row["hourly_tuning_command"] = ""
    rows: list[dict[str, object]] = [baseline_row]

    failed_candidates: list[dict[str, object]] = []
    for candidate in ready:
        long_only_symbols, short_only_symbols = _candidate_direction_lists(
            candidate,
            base_long_only_symbols=base_long_only_symbols,
            base_short_only_symbols=base_short_only_symbols,
        )
        candidate_output_dir = output_dir / candidate.symbol
        mae_snapshot = _candidate_mae_snapshot(mae_by_symbol, candidate.symbol)
        training_plan = training_plans.get(candidate.symbol, {})
        gate_reason = _candidate_mae_gate_reason(
            symbol=candidate.symbol,
            mae_snapshot=mae_snapshot,
            max_h1_mae_percent=float(args.candidate_max_h1_mae_percent),
            max_h24_mae_percent=float(args.candidate_max_h24_mae_percent),
        )
        if gate_reason:
            summary_path = _write_cache_gate_summary(
                candidate_output_dir=candidate_output_dir,
                symbol=candidate.symbol,
                side=candidate.side,
                mae_snapshot=mae_snapshot,
                gate_reason=gate_reason,
                max_h1_mae_percent=float(args.candidate_max_h1_mae_percent),
                max_h24_mae_percent=float(args.candidate_max_h24_mae_percent),
            )
            rows.append(
                _skipped_result_row(
                    candidate=candidate,
                    baseline_metrics=baseline_metrics,
                    summary_path=summary_path,
                    mae_snapshot=mae_snapshot,
                    gate_reason=gate_reason,
                    training_plan=training_plan,
                )
            )
            continue
        try:
            if bool(args.reuse_candidate_results) and not bool(args.force_candidate_rerun) and _trial_done(candidate_output_dir):
                metrics = _read_metrics(candidate_output_dir)
            else:
                metrics = _run_trial(
                    symbols=list(base_symbols) + [candidate.symbol],
                    checkpoint=checkpoint_path,
                    stock_data_root=args.stock_data_root,
                    forecast_cache_root=args.forecast_cache_root,
                    output_dir=candidate_output_dir,
                    moving_average_windows=str(args.moving_average_windows),
                    min_history_hours=int(args.min_history_hours),
                    eval_days=float(args.eval_days),
                    long_only_symbols=long_only_symbols,
                    short_only_symbols=short_only_symbols,
                    baseline_metrics=baseline_metrics,
                    enable_baseline_gate=not bool(args.disable_baseline_gate),
                    baseline_early_exit_min_steps=int(args.baseline_early_exit_min_steps),
                    baseline_stage1_progress=float(args.baseline_stage1_progress),
                    baseline_stage2_progress=float(args.baseline_stage2_progress),
                    baseline_stage3_progress=float(args.baseline_stage3_progress),
                    baseline_return_tolerance=float(args.baseline_return_tolerance),
                    baseline_sortino_tolerance=float(args.baseline_sortino_tolerance),
                    baseline_max_drawdown_tolerance=float(args.baseline_max_drawdown_tolerance),
                    initial_state=args.initial_state,
                )
        except RuntimeError as exc:
            failed_candidates.append({"symbol": candidate.symbol, "side": candidate.side, "error": str(exc)})
            continue
        rows.append(
            _result_row(
                symbol=candidate.symbol,
                side=candidate.side,
                sector=candidate.sector,
                priority=candidate.priority,
                thesis=candidate.thesis,
                metrics=metrics,
                summary_path=candidate_output_dir / "metrics.json",
                baseline_metrics=baseline_metrics,
                sim_ran=True,
                mae_snapshot=mae_snapshot,
                training_plan=training_plan,
            )
        )

    sorted_rows = [rows[0]] + sorted(rows[1:], key=_trial_sort_key, reverse=True)
    for row in sorted_rows:
        assessment = _assess_promotion(
            row,
            min_return_delta=float(args.promotion_min_return_delta),
            min_sortino_delta=float(args.promotion_min_sortino_delta),
            max_drawdown_delta=float(args.promotion_max_drawdown_delta),
        )
        row.update(assessment)
    (output_dir / "expansion_results.json").write_text(json.dumps(sorted_rows, indent=2) + "\n")
    _write_csv(output_dir / "expansion_results.csv", sorted_rows)
    (output_dir / "baseline_metrics.json").write_text(json.dumps(baseline_metrics, indent=2) + "\n")
    if args.baseline_source_dir is not None:
        (output_dir / "baseline_source.json").write_text(
            json.dumps({"baseline_source_dir": str(Path(args.baseline_source_dir))}, indent=2) + "\n"
        )
    if failed_candidates:
        (output_dir / "failed_candidates.json").write_text(json.dumps(failed_candidates, indent=2) + "\n")
    promotion_summary = _build_promotion_summary(
        sorted_rows,
        min_return_delta=float(args.promotion_min_return_delta),
        min_sortino_delta=float(args.promotion_min_sortino_delta),
        max_drawdown_delta=float(args.promotion_max_drawdown_delta),
    )
    (output_dir / "promotion_summary.json").write_text(json.dumps(promotion_summary, indent=2) + "\n")

    top_lora_commands = [
        {
            "symbol": row["symbol"],
            "recommended_next_step": row.get("recommended_next_step"),
            "recommended_reason": row.get("recommended_reason"),
            "recommended_peer_symbols": row.get("recommended_peer_symbols"),
            "total_return_delta_vs_baseline": row.get("total_return_delta_vs_baseline"),
            "sortino_delta_vs_baseline": row.get("sortino_delta_vs_baseline"),
            "max_drawdown_delta_vs_baseline": row.get("max_drawdown_delta_vs_baseline"),
            "hourly_tuning_command": row.get("hourly_tuning_command"),
            "lora_command": row.get("lora_command"),
        }
        for row in sorted_rows[1:6]
    ]
    (output_dir / "top_lora_commands.json").write_text(json.dumps(top_lora_commands, indent=2) + "\n")
    (output_dir / "top_training_commands.json").write_text(json.dumps(top_lora_commands, indent=2) + "\n")

    for row in sorted_rows:
        print(
            f"{row['symbol']:>18} "
            f"sortino={float(row.get('sortino', 0.0)):.4f} "
            f"return={float(row.get('total_return', 0.0)):.6f} "
            f"max_dd={float(row.get('max_drawdown', 0.0)):.6f} "
            f"delta_sortino={float(row.get('sortino_delta_vs_baseline', 0.0)):.4f} "
            f"promotable={bool(row.get('promotable', False))} "
            f"terminated_early={bool(row.get('terminated_early', False))}"
        )
    print(promotion_summary["promotion_reason"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
