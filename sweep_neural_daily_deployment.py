#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neuraldailymarketsimulator.simulator import NeuralDailyMarketSimulator
from neuraldailytraining import DailyTradingRuntime
from src.neural_daily_deployment import (
    DEFAULT_ACTIVE_CHECKPOINT_LINK,
    DEFAULT_ACTIVE_CONFIG_PATH,
    build_recent_window_start_dates,
    build_scenario_row,
    load_deployment_config,
    parse_optional_float_grid,
    selection_metric_sort_key,
    selection_metric_value,
    should_promote_candidate,
    summarize_threshold_scenarios,
    write_deployment_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep neural-daily deployment thresholds and optionally promote.")
    parser.add_argument(
        "--checkpoint",
        nargs="+",
        required=True,
        help="Candidate checkpoint paths to evaluate.",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        help="Optional explicit baseline checkpoint. Defaults to the active deployment config or active symlink if present.",
    )
    parser.add_argument("--symbols", nargs="*", help="Optional explicit symbol universe for all checkpoints.")
    parser.add_argument("--data-root", help="Optional override for dataset data_root.")
    parser.add_argument("--forecast-cache", help="Optional override for forecast cache directory.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--window-count", type=int, default=3)
    parser.add_argument("--window-stride-days", type=int, default=20)
    parser.add_argument("--start-date", nargs="*", help="Optional explicit start dates instead of auto recent windows.")
    parser.add_argument("--risk-thresholds", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--confidence-thresholds", default="none,0.2,0.35,0.5")
    parser.add_argument("--stock-fee", type=float, default=0.0008)
    parser.add_argument("--crypto-fee", type=float, default=0.0008)
    parser.add_argument("--initial-cash", type=float, default=1.0)
    parser.add_argument("--sortino-clip", type=float, default=10.0)
    parser.add_argument(
        "--selection-metric",
        choices=(
            "robust_score",
            "sortino_p25",
            "return_p25_pct",
            "goodness_score_mean",
            "pnl_smoothness_mean",
        ),
        default="robust_score",
        help="Metric used to choose the best parameter pair and checkpoint.",
    )
    parser.add_argument("--min-robust-improvement", type=float, default=0.0)
    parser.add_argument("--min-return-p25-pct", type=float, default=0.0)
    parser.add_argument("--min-sortino-p25", type=float, default=0.0)
    parser.add_argument("--promote", action="store_true", help="Update the active checkpoint symlink and JSON config.")
    parser.add_argument("--active-link", default=str(DEFAULT_ACTIVE_CHECKPOINT_LINK))
    parser.add_argument("--active-config", default=str(DEFAULT_ACTIVE_CONFIG_PATH))
    parser.add_argument("--output-json", help="Optional output path for the sweep report.")
    return parser.parse_args()


def _resolve_dataset_runtime(
    checkpoint_path: Path,
    *,
    symbols: tuple[str, ...] | None,
    data_root: str | None,
    forecast_cache: str | None,
    device: str | None,
    risk_threshold: float | None = None,
    confidence_threshold: float | None = None,
) -> DailyTradingRuntime:
    base_runtime = DailyTradingRuntime(checkpoint_path, device=device)
    dataset_cfg = copy.deepcopy(base_runtime.dataset_config)
    if symbols:
        dataset_cfg.symbols = tuple(symbols)
    if data_root:
        dataset_cfg.data_root = Path(data_root)
    if forecast_cache:
        dataset_cfg.forecast_cache_dir = Path(forecast_cache)
    return DailyTradingRuntime(
        checkpoint_path,
        dataset_config=dataset_cfg,
        device=device,
        risk_threshold=risk_threshold,
        confidence_threshold=confidence_threshold,
    )


def _baseline_checkpoint_from_active(
    *,
    active_config_path: Path,
    active_link_path: Path,
) -> Path | None:
    if active_config_path.exists():
        try:
            payload = load_deployment_config(active_config_path)
            checkpoint = payload.get("checkpoint")
            if checkpoint:
                return Path(str(checkpoint))
        except Exception:
            pass
    if active_link_path.exists() or active_link_path.is_symlink():
        return active_link_path.resolve()
    return None


def _resolve_start_dates(
    checkpoint_path: Path,
    *,
    symbols: tuple[str, ...] | None,
    data_root: str | None,
    forecast_cache: str | None,
    device: str | None,
    days: int,
    window_count: int,
    window_stride_days: int,
    explicit_start_dates: list[str] | None,
    stock_fee: float,
    crypto_fee: float,
    initial_cash: float,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    runtime = _resolve_dataset_runtime(
        checkpoint_path,
        symbols=symbols,
        data_root=data_root,
        forecast_cache=forecast_cache,
        device=device,
    )
    resolved_symbols = symbols or tuple(runtime.dataset_config.symbols)
    simulator = NeuralDailyMarketSimulator(
        runtime,
        resolved_symbols,
        stock_fee=stock_fee,
        crypto_fee=crypto_fee,
        initial_cash=initial_cash,
    )
    start_dates = build_recent_window_start_dates(
        simulator._available_dates(),  # type: ignore[attr-defined]
        window_days=days,
        max_windows=window_count,
        stride_days=window_stride_days,
        explicit_start_dates=explicit_start_dates,
    )
    return tuple(resolved_symbols), tuple(start_dates)


def evaluate_checkpoint_grid(
    checkpoint_path: Path,
    *,
    symbols: tuple[str, ...] | None,
    data_root: str | None,
    forecast_cache: str | None,
    device: str | None,
    risk_thresholds: tuple[float | None, ...],
    confidence_thresholds: tuple[float | None, ...],
    start_dates: tuple[str, ...],
    days: int,
    stock_fee: float,
    crypto_fee: float,
    initial_cash: float,
    sortino_clip: float,
    selection_metric: str,
) -> dict[str, Any]:
    scenario_rows: list[dict[str, Any]] = []
    runtime = _resolve_dataset_runtime(
        checkpoint_path,
        symbols=symbols,
        data_root=data_root,
        forecast_cache=forecast_cache,
        device=device,
        risk_threshold=risk_thresholds[0],
        confidence_threshold=confidence_thresholds[0],
    )
    resolved_symbols = symbols or tuple(runtime.dataset_config.symbols)
    simulator = NeuralDailyMarketSimulator(
        runtime,
        resolved_symbols,
        stock_fee=stock_fee,
        crypto_fee=crypto_fee,
        initial_cash=initial_cash,
    )

    total_pairs = len(risk_thresholds) * len(confidence_thresholds)
    pair_index = 0
    for risk_threshold in risk_thresholds:
        runtime.risk_threshold = float(risk_threshold or 0.0)
        for confidence_threshold in confidence_thresholds:
            pair_index += 1
            runtime.confidence_threshold = None if confidence_threshold is None else float(confidence_threshold)
            print(
                f"  pair {pair_index}/{total_pairs}: risk={runtime.risk_threshold} "
                f"confidence={runtime.confidence_threshold}"
            )
            for start_date in start_dates:
                _, summary = simulator.run(start_date=start_date, days=days)
                scenario_rows.append(
                    build_scenario_row(
                        start_date=start_date,
                        days=days,
                        summary=summary,
                        risk_threshold=risk_threshold,
                        confidence_threshold=confidence_threshold,
                    )
                )

    parameter_summaries = summarize_threshold_scenarios(scenario_rows, sortino_clip=sortino_clip)
    if not parameter_summaries:
        raise RuntimeError(f"No parameter summaries produced for {checkpoint_path}")
    best_parameters = max(parameter_summaries, key=lambda item: selection_metric_sort_key(item, selection_metric))

    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "symbols": list(resolved_symbols),
        "start_dates": list(start_dates),
        "scenario_rows": scenario_rows,
        "parameter_summaries": parameter_summaries,
        "best_parameters": best_parameters,
    }


def _promote_checkpoint(
    checkpoint_path: Path,
    *,
    active_link_path: Path,
    active_config_path: Path,
    symbols: tuple[str, ...],
    best_parameters: dict[str, Any],
    selection_metric: str,
    metadata: dict[str, Any],
) -> None:
    active_link_path.parent.mkdir(parents=True, exist_ok=True)
    if active_link_path.exists() or active_link_path.is_symlink():
        active_link_path.unlink()
    active_link_path.symlink_to(checkpoint_path.resolve())
    write_deployment_config(
        active_config_path,
        checkpoint=checkpoint_path,
        risk_threshold=best_parameters.get("risk_threshold"),
        confidence_threshold=best_parameters.get("confidence_threshold"),
        symbols=symbols,
        selection_metric=selection_metric,
        selection_value=selection_metric_value(best_parameters, selection_metric),
        summary=best_parameters,
        metadata=metadata,
    )


def main() -> int:
    args = parse_args()

    candidate_paths = tuple(Path(path).expanduser().resolve() for path in args.checkpoint)
    for checkpoint_path in candidate_paths:
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

    risk_thresholds = parse_optional_float_grid(args.risk_thresholds, allow_none=False)
    confidence_thresholds = parse_optional_float_grid(args.confidence_thresholds, allow_none=True)
    if not risk_thresholds:
        raise ValueError("At least one risk threshold is required.")
    if not confidence_thresholds:
        raise ValueError("At least one confidence threshold is required.")

    explicit_symbols = tuple(symbol.upper() for symbol in args.symbols) if args.symbols else None
    reference_checkpoint = candidate_paths[0]
    resolved_symbols, start_dates = _resolve_start_dates(
        reference_checkpoint,
        symbols=explicit_symbols,
        data_root=args.data_root,
        forecast_cache=args.forecast_cache,
        device=args.device,
        days=args.days,
        window_count=args.window_count,
        window_stride_days=args.window_stride_days,
        explicit_start_dates=args.start_date,
        stock_fee=args.stock_fee,
        crypto_fee=args.crypto_fee,
        initial_cash=args.initial_cash,
    )

    baseline_checkpoint = (
        Path(args.baseline_checkpoint).expanduser().resolve()
        if args.baseline_checkpoint
        else _baseline_checkpoint_from_active(
            active_config_path=Path(args.active_config).expanduser(),
            active_link_path=Path(args.active_link).expanduser(),
        )
    )
    evaluation_paths = list(candidate_paths)
    if baseline_checkpoint is not None and baseline_checkpoint not in evaluation_paths:
        evaluation_paths.append(baseline_checkpoint)

    checkpoint_results: list[dict[str, Any]] = []
    for checkpoint_path in evaluation_paths:
        result = evaluate_checkpoint_grid(
            checkpoint_path,
            symbols=resolved_symbols,
            data_root=args.data_root,
            forecast_cache=args.forecast_cache,
            device=args.device,
            risk_thresholds=risk_thresholds,
            confidence_thresholds=confidence_thresholds,
            start_dates=start_dates,
            days=args.days,
            stock_fee=args.stock_fee,
            crypto_fee=args.crypto_fee,
            initial_cash=args.initial_cash,
            sortino_clip=args.sortino_clip,
            selection_metric=args.selection_metric,
        )
        checkpoint_results.append(result)
        best = result["best_parameters"]
        print(
            f"{Path(result['checkpoint']).name}: {args.selection_metric}="
            f"{selection_metric_value(best, args.selection_metric):.4f} "
            f"robust={best['robust_score']:.4f} return_p25={best['return_p25_pct']:.2f}% sortino_p25={best['sortino_p25']:.4f} "
            f"risk={best['risk_threshold']} confidence={best['confidence_threshold']}"
        )

    checkpoint_results.sort(key=lambda item: selection_metric_sort_key(item["best_parameters"], args.selection_metric), reverse=True)

    best_overall = checkpoint_results[0]
    best_overall_summary = best_overall["best_parameters"]
    baseline_result = None
    if baseline_checkpoint is not None:
        baseline_result = next((item for item in checkpoint_results if item["checkpoint"] == str(baseline_checkpoint)), None)

    promote_decision = should_promote_candidate(
        best_overall_summary,
        baseline_result["best_parameters"] if baseline_result else None,
        min_robust_improvement=args.min_robust_improvement,
        min_return_p25_pct=args.min_return_p25_pct,
        min_sortino_p25=args.min_sortino_p25,
    )

    active_link_path = Path(args.active_link).expanduser()
    active_config_path = Path(args.active_config).expanduser()
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbols": list(resolved_symbols),
        "days": int(args.days),
        "start_dates": list(start_dates),
        "risk_thresholds": [None if value is None else float(value) for value in risk_thresholds],
        "confidence_thresholds": [None if value is None else float(value) for value in confidence_thresholds],
        "baseline_checkpoint": str(baseline_checkpoint) if baseline_checkpoint else None,
        "best_overall": {
            "checkpoint": best_overall["checkpoint"],
            "best_parameters": best_overall_summary,
        },
        "selection_metric": args.selection_metric,
        "results": checkpoint_results,
        "promotion": {
            "requested": bool(args.promote),
            "allowed": bool(promote_decision),
            "active_link": str(active_link_path),
            "active_config": str(active_config_path),
        },
    }

    if args.promote and promote_decision:
        _promote_checkpoint(
            Path(best_overall["checkpoint"]),
            active_link_path=active_link_path,
            active_config_path=active_config_path,
            symbols=resolved_symbols,
            best_parameters=best_overall_summary,
            selection_metric=args.selection_metric,
            metadata={
                "generated_at": report["generated_at"],
                "start_dates": list(start_dates),
                "days": int(args.days),
                "stock_fee": float(args.stock_fee),
                "crypto_fee": float(args.crypto_fee),
                "baseline_checkpoint": str(baseline_checkpoint) if baseline_checkpoint else None,
                "selection_metric": args.selection_metric,
            },
        )
        report["promotion"]["performed"] = True
        print(f"Promoted {best_overall['checkpoint']} -> {active_link_path}")
    else:
        report["promotion"]["performed"] = False
        if args.promote:
            print("Promotion skipped because the candidate did not beat the baseline gates.")

    if args.output_json:
        output_path = Path(args.output_json)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = Path("analysis") / f"neural_daily_deploy_sweep_{stamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote sweep report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
