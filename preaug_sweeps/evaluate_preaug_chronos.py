#!/usr/bin/env python3
"""
Chronos2 pre-augmentation sweep driven by existing hyperparameter configs.

This utility reuses the Chronos2 benchmarking harness to measure each
augmentation strategy's MAE% on the standard validation/test windows used
throughout the repo (20/20 split, forecast horizon 1). It writes the winning
strategy per symbol to ``preaugstrategies/chronos2/<symbol>.json`` so that the
runtime selector can pick it up without running the slower Kronos fine-tuning
loop.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_chronos2 import (
    Chronos2Benchmark,
    Chronos2Candidate,
    CandidateReport,
    _prepare_series,
    _window_indices,
)
from preaug_sweeps.augmentations import AUGMENTATION_REGISTRY
from src.preaug import PreAugmentationChoice


SELECTION_METRICS = ("mae_percent", "mae", "pct_return_mae", "rmse", "mape")


class _StaticPreAugSelector:
    """Minimal selector that always returns the provided choice."""

    def __init__(self, choice: PreAugmentationChoice) -> None:
        self._choice = choice

    def get_choice(self, symbol: str) -> PreAugmentationChoice:
        return self._choice


def _load_chronos_symbols(best_dir: Path) -> List[str]:
    symbols: List[str] = []
    for path in sorted(best_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        if payload.get("model") == "chronos2":
            symbols.append(path.stem)
    return symbols


def _load_candidate(config_path: Path) -> Chronos2Candidate:
    with config_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    cfg = payload.get("config") or {}
    quantiles = tuple(cfg.get("quantile_levels") or (0.1, 0.5, 0.9))
    return Chronos2Candidate(
        name=str(cfg.get("name", f"{payload.get('symbol', config_path.stem)}_chronos2")),
        context_length=int(cfg.get("context_length", 1024)),
        batch_size=int(cfg.get("batch_size", 128)),
        quantile_levels=quantiles,
        aggregation=str(cfg.get("aggregation", "median")),
        sample_count=int(cfg.get("sample_count", 0)),
        scaler=str(cfg.get("scaler", "none")),
        predict_kwargs=dict(cfg.get("predict_kwargs") or {}),
    )


def _build_benchmark(args: argparse.Namespace) -> Chronos2Benchmark:
    bench_args = SimpleNamespace(
        symbols=[],
        data_dir=str(args.data_dir),
        model_id=args.model_id,
        device_map=args.device_map,
        quantile_levels=args.base_quantiles,
        predict_batches_jointly=args.predict_batches_jointly,
        limit_prediction_length=args.limit_prediction_length,
        max_output_patches=args.max_output_patches,
        unrolled_quantiles=args.unrolled_quantiles,
        seed=args.seed,
        torch_dtype=args.torch_dtype,
        torch_compile=args.torch_compile,
        verbose=args.verbose,
        output_dir=str(args.benchmark_cache_dir),
    )
    return Chronos2Benchmark(bench_args)


def _select_value(metrics: Mapping[str, float], metric: str) -> float:
    value = metrics.get(metric)
    if value is None:
        return math.inf
    return float(value)


def _metrics_dict(result: CandidateReport) -> Dict[str, Dict[str, float]]:
    return {
        "validation": {
            "mae": result.validation.price_mae,
            "mae_percent": result.validation.mae_percent,
            "rmse": result.validation.rmse,
            "pct_return_mae": result.validation.pct_return_mae,
            "latency_s": result.validation.latency_s,
        },
        "test": {
            "mae": result.test.price_mae,
            "mae_percent": result.test.mae_percent,
            "rmse": result.test.rmse,
            "pct_return_mae": result.test.pct_return_mae,
            "latency_s": result.test.latency_s,
        },
    }


def _evaluate_strategy(
    benchmark: Chronos2Benchmark,
    symbol: str,
    df,
    val_indices: Iterable[int],
    test_indices: Iterable[int],
    candidate: Chronos2Candidate,
    strategy: str,
) -> CandidateReport:
    wrapper = benchmark._get_wrapper(candidate.context_length, candidate.quantile_levels)
    original_selector = getattr(wrapper, "_preaug_selector", None)
    choice = PreAugmentationChoice(
        symbol=symbol,
        strategy=strategy,
        params={},
        metric="manual",
        metric_value=0.0,
        source_path=Path(f"inline:{strategy}"),
    )
    wrapper._preaug_selector = _StaticPreAugSelector(choice)
    try:
        return benchmark._evaluate_candidate(symbol, df, val_indices, test_indices, candidate)
    finally:
        wrapper._preaug_selector = original_selector


def _persist_best(
    symbol: str,
    best_strategy: str,
    metrics: Mapping[str, Mapping[str, float]],
    comparison: Mapping[str, Mapping[str, Mapping[str, float]]],
    args: argparse.Namespace,
    selection_value: float,
) -> Path:
    payload = {
        "symbol": symbol,
        "best_strategy": best_strategy,
        "mae": metrics["mae"],
        "mae_percent": metrics["mae_percent"],
        "rmse": metrics["rmse"],
        "pct_return_mae": metrics["pct_return_mae"],
        "config": {"name": best_strategy, "params": {}},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selection_metric": args.selection_metric,
        "selection_value": selection_value,
        "comparison": {
            name: {
                "mae": data["test"]["mae"],
                "mae_percent": data["test"]["mae_percent"],
                "pct_return_mae": data["test"]["pct_return_mae"],
            }
            for name, data in comparison.items()
        },
        "metadata": {
            "source": "chronos2_preaug_eval",
            "hyperparam_root": str(args.hyperparam_root),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / f"{symbol}.json"
    with target.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    if args.mirror_best_dir:
        args.mirror_best_dir.mkdir(parents=True, exist_ok=True)
        mirror = args.mirror_best_dir / f"{symbol}.json"
        with mirror.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", help="Symbols to evaluate (default: chronos best selections).")
    parser.add_argument("--hyperparam-root", type=Path, default=Path("hyperparams/chronos2"))
    parser.add_argument("--best-selection-root", type=Path, default=Path("hyperparams/best"))
    parser.add_argument("--strategies", nargs="+", help="Augmentation strategies to test (default: all).")
    parser.add_argument(
        "--selection-metric",
        choices=SELECTION_METRICS,
        default="mae_percent",
        help="Metric used to pick the best augmentation (based on TEST window).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("preaugstrategies/chronos2"))
    parser.add_argument("--mirror-best-dir", type=Path, default=Path("preaugstrategies/best"))
    parser.add_argument("--data-dir", type=Path, default=Path("trainingdata"))
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument("--torch-compile", action="store_true", help="Enable torch.compile for Chronos2.")
    parser.add_argument("--predict-batches-jointly", action="store_true")
    parser.add_argument("--limit-prediction-length", action="store_true")
    parser.add_argument("--max-output-patches", type=int, default=None)
    parser.add_argument("--unrolled-quantiles", nargs="+", type=float)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--base-quantiles", nargs="+", type=float, default=(0.1, 0.5, 0.9))
    parser.add_argument("--benchmark-cache-dir", type=Path, default=Path("chronos2_benchmarks/preaug_cache"))
    parser.add_argument("--report-dir", type=Path, default=Path("preaug_sweeps/reports"))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    strategies = args.strategies or list(AUGMENTATION_REGISTRY.keys())
    symbols = args.symbols or _load_chronos_symbols(args.best_selection_root)
    if not symbols:
        raise SystemExit("No symbols to evaluate. Provide --symbols or ensure hyperparams/best has chronos entries.")

    benchmark = _build_benchmark(args)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    summary_payload: Dict[str, Dict[str, object]] = {}
    start = time.time()

    for idx, symbol in enumerate(symbols, start=1):
        config_path = args.hyperparam_root / f"{symbol}.json"
        if not config_path.exists():
            print(f"[WARN] Skipping {symbol}: missing {config_path}")
            continue
        csv_path = args.data_dir / f"{symbol}.csv"
        if not csv_path.exists():
            print(f"[WARN] Skipping {symbol}: missing price data {csv_path}")
            continue

        candidate = _load_candidate(config_path)
        df = _prepare_series(csv_path)
        try:
            val_indices, test_indices = _window_indices(len(df))
        except ValueError as exc:
            print(f"[WARN] Skipping {symbol}: {exc}")
            continue

        comparison: Dict[str, Dict[str, Dict[str, float]]] = {}
        for strategy in strategies:
            try:
                report = _evaluate_strategy(benchmark, symbol, df, val_indices, test_indices, candidate, strategy)
            except Exception as exc:
                print(f"[ERROR] {symbol} strategy {strategy} failed: {exc}")
                continue
            comparison[strategy] = _metrics_dict(report)

        if not comparison:
            print(f"[WARN] No successful strategies for {symbol}")
            continue

        best_strategy = min(
            comparison.items(),
            key=lambda item: _select_value(item[1]["test"], args.selection_metric),
        )[0]
        selection_value = _select_value(comparison[best_strategy]["test"], args.selection_metric)
        best_metrics = comparison[best_strategy]["test"]

        target_path = _persist_best(symbol, best_strategy, best_metrics, comparison, args, selection_value)
        summary_payload[symbol] = {
            "best_strategy": best_strategy,
            "selection_value": selection_value,
            "selection_metric": args.selection_metric,
            "comparison": comparison,
            "output_path": str(target_path),
        }
        print(
            f"[{idx}/{len(symbols)}] {symbol}: best={best_strategy} "
            f"{args.selection_metric}={selection_value:.4f} ({target_path})"
        )

    if not summary_payload:
        print("No symbols evaluated successfully.")
        return 1

    report_path = args.report_dir / f"chronos_preaug_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with report_path.open("w", encoding="utf-8") as fp:
        json.dump(summary_payload, fp, indent=2)

    elapsed = time.time() - start
    print(f"\nCompleted Chronos2 pre-augmentation scan for {len(summary_payload)} symbols in {elapsed/60:.2f} minutes.")
    print(f"Summary written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
