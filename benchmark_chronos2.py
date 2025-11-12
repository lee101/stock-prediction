#!/usr/bin/env python3
"""
Chronos2 benchmarking harness with reversible pre/post transforms and synthetic sampling.

This script evaluates Chronos2 inference configurations on historical OHLC data using
the same validation/test windows as ``test_hyperparameters_extended`` (20/20 split).
It supports:

* context length sweeps
* batch-size tweaks
* reversible column-wise scaling (mean/std)
* synthetic sampling (e.g. 4096 draws) derived from Chronos quantiles
* Toto-style aggregations (mean, mean±std, quantiles, trimmed means, etc.)

Results are written to JSON under ``chronos2_benchmarks/{symbol}/`` and optionally
update ``hyperparams/chronos2/{symbol}.json`` when a configuration wins on
``test_pct_return_mae``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, TypeVar

import numpy as np
import pandas as pd
from scipy.optimize import direct as scipy_direct
from sklearn.metrics import mean_absolute_error

try:
    import chronos_compile_config
except ModuleNotFoundError:  # pragma: no cover - optional compile tweaks
    chronos_compile_config = None  # type: ignore

from src.models.chronos2_postprocessing import (
    Chronos2AggregationSpec,
    ColumnScaler,
    aggregate_quantile_forecasts,
    resolve_quantile_levels,
)
from src.models.chronos2_wrapper import Chronos2OHLCWrapper, DEFAULT_QUANTILE_LEVELS
from kronostraining.metrics_utils import compute_mae_percent

logger = logging.getLogger(__name__)

# Sliding window constants (align with test_hyperparameters_extended)
FORECAST_HORIZON = 1
VAL_WINDOW = 20
TEST_WINDOW = 20
MIN_CONTEXT = 128

TARGET_COLUMNS = ("open", "high", "low", "close")
DEFAULT_DEVICE_MAP = "cuda"


def _parse_torch_dtype(value: Optional[str]):
    if value is None:
        return None
    normalized = value.strip().lower()
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - torch unavailable
        raise RuntimeError("Torch is required to set --torch-dtype.") from exc
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype '{value}'.")
    return mapping[normalized]


@dataclass
class Chronos2Candidate:
    name: str
    context_length: int
    batch_size: int
    quantile_levels: Tuple[float, ...] = DEFAULT_QUANTILE_LEVELS
    aggregation: str = "median"
    sample_count: int = 0
    scaler: str = "none"
    predict_kwargs: Dict[str, float | int | bool | Sequence[float]] = field(default_factory=dict)


@dataclass(frozen=True)
class DirectSearchSpace:
    """Discrete hyperparameter domains explored by DIRECT."""

    context_lengths: Tuple[int, ...]
    batch_sizes: Tuple[int, ...]
    aggregations: Tuple[str, ...]
    sample_counts: Tuple[int, ...]
    scalers: Tuple[str, ...]


@dataclass
class EvaluationResult:
    price_mae: float
    rmse: float
    pct_return_mae: float
    latency_s: float
    mae_percent: float
    predictions: List[float]


@dataclass
class CandidateReport:
    symbol: str
    candidate: Chronos2Candidate
    validation: EvaluationResult
    test: EvaluationResult
    windows: Dict[str, int]

    def to_payload(self) -> Dict[str, object]:
        return {
            "symbol": self.symbol,
            "candidate": asdict(self.candidate),
            "validation": asdict(self.validation),
            "test": asdict(self.test),
            "windows": self.windows,
        }


def _prepare_series(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns or "close" not in df.columns:
        raise KeyError(f"{csv_path.name} missing 'timestamp' or 'close'")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _window_indices(length: int) -> Tuple[range, range]:
    if length < MIN_CONTEXT + VAL_WINDOW + TEST_WINDOW:
        raise ValueError(
            f"dataset too short ({length} rows); need at least {MIN_CONTEXT + VAL_WINDOW + TEST_WINDOW}"
        )
    val_start = length - (TEST_WINDOW + VAL_WINDOW)
    return range(val_start, length - TEST_WINDOW), range(length - TEST_WINDOW, length)

_T = TypeVar("_T")


def _unique_sequence(values: Sequence[_T]) -> Tuple[_T, ...]:
    seen: List[_T] = []
    for value in values:
        if value in seen:
            continue
        seen.append(value)
    return tuple(seen)


def _resolve_context_lengths(series_length: int, args: argparse.Namespace) -> Tuple[int, ...]:
    guard = getattr(args, "auto_context_guard", VAL_WINDOW + TEST_WINDOW)
    max_allowed = max(MIN_CONTEXT, series_length - guard)
    if getattr(args, "auto_context_lengths", False):
        start = max(MIN_CONTEXT, getattr(args, "auto_context_min", MIN_CONTEXT))
        stop = min(max_allowed, getattr(args, "auto_context_max", max_allowed))
        step = max(1, getattr(args, "auto_context_step", 64))
        values: List[int] = []
        ctx = start
        while ctx <= stop:
            values.append(int(ctx))
            ctx += step
        values.extend([start, stop, max_allowed])
    else:
        values = [int(v) for v in getattr(args, "context_lengths", []) if MIN_CONTEXT <= v <= max_allowed]
    unique = sorted({v for v in values if MIN_CONTEXT <= v <= max_allowed})
    if not unique:
        fallback = max(MIN_CONTEXT, min(max_allowed, series_length - guard // 2))
        unique = [fallback]
    return tuple(unique)


def _build_direct_search_space(series_length: int, args: argparse.Namespace) -> DirectSearchSpace:
    contexts = _resolve_context_lengths(series_length, args)
    batch_source = getattr(args, "direct_batch_sizes", None) or getattr(args, "batch_sizes", [])
    batches = tuple(int(max(1, b)) for b in _unique_sequence(batch_source) if int(b) > 0)
    aggs_source = getattr(args, "direct_aggregations", None) or getattr(args, "aggregations", [])
    aggregations = tuple(agg for agg in _unique_sequence(aggs_source) if isinstance(agg, str))
    samples_source = getattr(args, "direct_sample_counts", None) or getattr(args, "sample_counts", [])
    sample_counts = tuple(int(max(0, s)) for s in _unique_sequence(samples_source) if int(s) >= 0)
    scalers_source = getattr(args, "direct_scalers", None) or getattr(args, "scalers", [] )
    scalers = tuple(str(s) for s in _unique_sequence(scalers_source)) or ("none",)
    if not contexts:
        raise ValueError("No context lengths available for DIRECT search")
    if not batches:
        raise ValueError("No batch sizes available for DIRECT search")
    if not aggregations:
        raise ValueError("No aggregations provided for DIRECT search")
    if not sample_counts:
        sample_counts = (0,)
    return DirectSearchSpace(
        context_lengths=contexts,
        batch_sizes=batches,
        aggregations=aggregations,
        sample_counts=sample_counts,
        scalers=scalers,
    )


def _option_bounds(count: int) -> Tuple[float, float]:
    if count <= 1:
        return (0.0, 1.0)
    return (0.0, float(count - 1))


def _pick_option(options: Tuple[_T, ...], raw_value: float) -> Tuple[_T, int]:
    if len(options) == 1:
        return options[0], 0
    idx = int(round(raw_value))
    idx = max(0, min(len(options) - 1, idx))
    return options[idx], idx


def _hashable(value: object) -> object:
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(v) for v in value)
    return value


def _candidate_signature(candidate: Chronos2Candidate) -> Tuple[object, ...]:
    return (
        candidate.context_length,
        candidate.batch_size,
        candidate.aggregation,
        candidate.sample_count,
        candidate.scaler,
        tuple(candidate.quantile_levels),
        _hashable(candidate.predict_kwargs),
    )


def _select_direct_metric(report: CandidateReport, mode: str) -> float:
    if mode == "test_pct_mae":
        return report.test.pct_return_mae
    if mode == "avg_pct_mae":
        return 0.5 * (report.validation.pct_return_mae + report.test.pct_return_mae)
    if mode == "validation_price_mae":
        return report.validation.price_mae
    if mode == "avg_price_mae":
        return 0.5 * (report.validation.price_mae + report.test.price_mae)
    return report.validation.pct_return_mae


class Chronos2Benchmark:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.symbols = args.symbols
        self.data_dir = Path(args.data_dir)
        self.model_id = args.model_id
        self.device_map = args.device_map
        self.quantile_levels = tuple(sorted(set(args.quantile_levels)))
        self.predict_kwargs = self._build_predict_kwargs(args)
        self.torch_dtype = _parse_torch_dtype(args.torch_dtype)
        self.wrapper_cache: MutableMapping[Tuple[int, Tuple[float, ...]], Chronos2OHLCWrapper] = {}
        self.rng = np.random.default_rng(args.seed)
        self.output_root = Path(args.output_dir)
        self.output_root.mkdir(parents=True, exist_ok=True)
        if args.torch_compile and chronos_compile_config is not None:
            chronos_compile_config.apply(verbose=args.verbose)

    @staticmethod
    def _build_predict_kwargs(args: argparse.Namespace) -> Dict[str, object]:
        kwargs: Dict[str, object] = {}
        if args.predict_batches_jointly:
            kwargs["predict_batches_jointly"] = True
        if args.limit_prediction_length:
            kwargs["limit_prediction_length"] = True
        if args.max_output_patches is not None:
            kwargs["max_output_patches"] = args.max_output_patches
        if args.unrolled_quantiles:
            kwargs["unrolled_quantiles"] = list(args.unrolled_quantiles)
        return kwargs

    def _get_wrapper(self, context_length: int, quantile_levels: Tuple[float, ...]) -> Chronos2OHLCWrapper:
        cache_key = (context_length, quantile_levels)
        cached = self.wrapper_cache.get(cache_key)
        if cached is not None:
            return cached
        wrapper = Chronos2OHLCWrapper.from_pretrained(
            model_id=self.model_id,
            device_map=self.device_map,
            target_columns=TARGET_COLUMNS,
            default_context_length=context_length,
            quantile_levels=quantile_levels,
            torch_dtype=self.torch_dtype,
        )
        self.wrapper_cache[cache_key] = wrapper
        return wrapper

    def _release_wrappers(self) -> None:
        for wrapper in self.wrapper_cache.values():
            try:
                wrapper.unload()
            except Exception:  # pragma: no cover - unload best-effort
                continue
        self.wrapper_cache.clear()

    def run(self, candidates: Sequence[Chronos2Candidate]) -> List[CandidateReport]:
        reports: List[CandidateReport] = []
        for symbol in self.symbols:
            csv_path = self.data_dir / f"{symbol}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing data for {symbol}: {csv_path}")
            df = _prepare_series(csv_path)
            val_indices, test_indices = _window_indices(len(df))
            for candidate in candidates:
                report = self._evaluate_candidate(symbol, df, val_indices, test_indices, candidate)
                reports.append(report)
                if self.args.verbose:
                    print(
                        f"[{symbol}] {candidate.name} "
                        f"val_pct_mae={report.validation.pct_return_mae:.4f} "
                        f"test_pct_mae={report.test.pct_return_mae:.4f} "
                        f"latency={report.test.latency_s:.2f}s"
                    )
            self._release_wrappers()
        return reports

    def run_direct(self) -> List[CandidateReport]:
        reports: List[CandidateReport] = []
        for symbol in self.symbols:
            csv_path = self.data_dir / f"{symbol}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing data for {symbol}: {csv_path}")
            df = _prepare_series(csv_path)
            val_indices, test_indices = _window_indices(len(df))
            search_space = _build_direct_search_space(len(df), self.args)
            symbol_reports = self._optimize_symbol_with_direct(symbol, df, val_indices, test_indices, search_space)
            reports.extend(symbol_reports)
            self._release_wrappers()
        return reports

    def _optimize_symbol_with_direct(
        self,
        symbol: str,
        df: pd.DataFrame,
        val_indices: Iterable[int],
        test_indices: Iterable[int],
        search_space: DirectSearchSpace,
    ) -> List[CandidateReport]:
        args = self.args
        bounds = [
            _option_bounds(len(search_space.context_lengths)),
            _option_bounds(len(search_space.batch_sizes)),
            _option_bounds(len(search_space.aggregations)),
            _option_bounds(len(search_space.sample_counts)),
            _option_bounds(len(search_space.scalers)),
        ]
        eval_history: List[CandidateReport] = []
        cache: Dict[Tuple[object, ...], Optional[CandidateReport]] = {}
        eval_counter = 0

        def build_candidate(vector: Sequence[float]) -> Chronos2Candidate:
            nonlocal eval_counter
            ctx, _ = _pick_option(search_space.context_lengths, vector[0])
            batch, _ = _pick_option(search_space.batch_sizes, vector[1])
            agg, _ = _pick_option(search_space.aggregations, vector[2])
            samples, _ = _pick_option(search_space.sample_counts, vector[3])
            scaler, _ = _pick_option(search_space.scalers, vector[4])
            eval_counter += 1
            name_parts = ["direct", f"ctx{ctx}", f"bs{batch}", agg.replace(" ", "")]
            if scaler != "none":
                name_parts.append(f"scale_{scaler}")
            if samples > 0:
                name_parts.append(f"s{samples}")
            name_parts.append(f"eval{eval_counter}")
            return Chronos2Candidate(
                name="_".join(name_parts),
                context_length=ctx,
                batch_size=batch,
                quantile_levels=self.quantile_levels,
                aggregation=agg,
                sample_count=samples,
                scaler=scaler,
                predict_kwargs=dict(self.predict_kwargs),
            )

        def evaluate(candidate: Chronos2Candidate) -> CandidateReport:
            signature = _candidate_signature(candidate)
            if signature in cache:
                cached = cache[signature]
                if cached is None:
                    raise RuntimeError("candidate previously failed")
                return cached
            try:
                report = self._evaluate_candidate(symbol, df, val_indices, test_indices, candidate)
            except Exception:
                cache[signature] = None
                raise
            cache[signature] = report
            eval_history.append(report)
            if args.verbose:
                print(
                    f"[DIRECT:{symbol}] {candidate.name} val_pct_mae={report.validation.pct_return_mae:.4f} "
                    f"test_pct_mae={report.test.pct_return_mae:.4f} latency={report.test.latency_s:.2f}s"
                )
            return report

        latency_weight = getattr(args, "direct_latency_weight", 0.0) or 0.0

        def objective(vector: Sequence[float]) -> float:
            candidate = build_candidate(vector)
            try:
                report = evaluate(candidate)
            except Exception as exc:  # pragma: no cover - defensive path
                logger.warning("DIRECT evaluation failed for %s/%s: %s", symbol, candidate.name, exc)
                return float("inf")
            metric = _select_direct_metric(report, args.direct_objective)
            if latency_weight:
                metric += latency_weight * report.test.latency_s
            return metric

        try:
            scipy_direct(
                objective,
                bounds,
                maxfun=getattr(args, "direct_maxfun", 40),
                maxiter=getattr(args, "direct_maxiter", 200),
                locally_biased=getattr(args, "direct_locally_biased", True),
            )
        except ValueError as exc:
            raise RuntimeError(f"DIRECT search failed for {symbol}: {exc}") from exc

        return eval_history

    def _evaluate_candidate(
        self,
        symbol: str,
        df: pd.DataFrame,
        val_indices: Iterable[int],
        test_indices: Iterable[int],
        candidate: Chronos2Candidate,
    ) -> CandidateReport:
        wrapper = self._get_wrapper(candidate.context_length, candidate.quantile_levels)
        evaluation_kwargs = dict(
            symbol=symbol,
            df=df,
            candidate=candidate,
            wrapper=wrapper,
        )
        val_result = self._evaluate_indices(val_indices, **evaluation_kwargs)
        test_result = self._evaluate_indices(test_indices, **evaluation_kwargs)
        return CandidateReport(
            symbol=symbol,
            candidate=candidate,
            validation=val_result,
            test=test_result,
            windows={"val_window": VAL_WINDOW, "test_window": TEST_WINDOW, "forecast_horizon": FORECAST_HORIZON},
        )

    def _evaluate_indices(
            self,
            indices: Iterable[int],
            *,
            symbol: str,
            df: pd.DataFrame,
            candidate: Chronos2Candidate,
            wrapper: Chronos2OHLCWrapper,
        ) -> EvaluationResult:
        preds: List[float] = []
        returns: List[float] = []
        actual_returns: List[float] = []
        actual_prices: List[float] = []
        total_latency = 0.0

        spec = Chronos2AggregationSpec(
            aggregation=candidate.aggregation,
            sample_count=candidate.sample_count,
            scaler=candidate.scaler,
        )
        quantile_tuple = resolve_quantile_levels(candidate.quantile_levels, candidate.sample_count)

        for idx in indices:
            context_start = max(0, idx - candidate.context_length)
            context_df = df.iloc[context_start:idx].copy()
            if context_df.empty:
                raise RuntimeError(f"{symbol} context window empty at idx={idx}")

            scaler = ColumnScaler(candidate.scaler, context_df[list(TARGET_COLUMNS)], TARGET_COLUMNS)
            transformed_context = scaler.transform(context_df)

            predict_kwargs = dict(self.predict_kwargs)
            predict_kwargs.update(candidate.predict_kwargs or {})

            start = time.perf_counter()
            batch = wrapper.predict_ohlc(
                transformed_context,
                symbol=symbol,
                prediction_length=FORECAST_HORIZON,
                context_length=candidate.context_length,
                quantile_levels=quantile_tuple,
                batch_size=candidate.batch_size,
                predict_kwargs=predict_kwargs,
            )
            total_latency += time.perf_counter() - start

            quantile_frames: Dict[float, pd.DataFrame] = {
                level: scaler.inverse(batch.quantile(level)) for level in quantile_tuple
            }

            aggregated = aggregate_quantile_forecasts(
                quantile_frames,
                columns=("close",),
                spec=spec,
                rng=self.rng,
            )
            price_pred = float(aggregated.get("close", np.nan))

            preds.append(price_pred)
            prev_price = float(df["close"].iloc[idx - 1])
            actual_price = float(df["close"].iloc[idx])
            actual_prices.append(actual_price)

            if prev_price == 0.0:
                returns.append(0.0)
                actual_returns.append(0.0)
            else:
                returns.append((price_pred - prev_price) / prev_price)
                actual_returns.append((actual_price - prev_price) / prev_price)

        price_mae = mean_absolute_error(actual_prices, preds)
        price_rmse = float(
            np.sqrt(np.mean((np.asarray(actual_prices, dtype=np.float64) - np.asarray(preds, dtype=np.float64)) ** 2))
        )
        pct_return_mae = mean_absolute_error(actual_returns, returns)
        mae_percent = compute_mae_percent(price_mae, np.asarray(actual_prices, dtype=np.float64))
        return EvaluationResult(
            price_mae=price_mae,
            rmse=price_rmse,
            pct_return_mae=pct_return_mae,
            latency_s=total_latency,
            mae_percent=mae_percent,
            predictions=preds,
        )



def _load_candidates_from_file(
    path: Optional[Path],
    default_quantiles: Tuple[float, ...],
    base_predict_kwargs: Mapping[str, object],
) -> List[Chronos2Candidate]:
    if path is None:
        return []
    with path.open() as fp:
        payload = json.load(fp)
    candidates: List[Chronos2Candidate] = []
    for entry in payload:
        quantiles = tuple(entry.get("quantile_levels", default_quantiles))
        predict_kwargs = dict(base_predict_kwargs)
        predict_kwargs.update(entry.get("predict_kwargs", {}))
        candidate = Chronos2Candidate(
            name=str(entry["name"]),
            context_length=int(entry.get("context_length", 512)),
            batch_size=int(entry.get("batch_size", 128)),
            quantile_levels=quantiles,
            aggregation=str(entry.get("aggregation", "median")),
            sample_count=int(entry.get("sample_count", 0)),
            scaler=str(entry.get("scaler", "none")),
            predict_kwargs=predict_kwargs,
        )
        candidates.append(candidate)
    return candidates


def _build_cross_product_candidates(
    args: argparse.Namespace,
    base_predict_kwargs: Mapping[str, object],
) -> List[Chronos2Candidate]:
    candidates: List[Chronos2Candidate] = []
    idx = 0
    for ctx in args.context_lengths:
        for batch in args.batch_sizes:
            for agg in args.aggregations:
                for scaler in args.scalers:
                    for sample_count in args.sample_counts:
                        idx += 1
                        name_parts = [f"ctx{ctx}", f"bs{batch}", agg.replace(" ", "")]
                        if scaler != "none":
                            name_parts.append(f"scale_{scaler}")
                        if sample_count > 0:
                            name_parts.append(f"s{sample_count}")
                        candidate = Chronos2Candidate(
                            name="_".join(name_parts),
                            context_length=ctx,
                            batch_size=batch,
                            quantile_levels=tuple(args.quantile_levels),
                            aggregation=agg,
                            sample_count=sample_count,
                            scaler=scaler,
                            predict_kwargs=dict(base_predict_kwargs),
                        )
                        candidates.append(candidate)
                        if args.max_candidates and len(candidates) >= args.max_candidates:
                            return candidates
    return candidates


def _persist_reports(reports: Sequence[CandidateReport], output_root: Path) -> List[Path]:
    if not reports:
        raise RuntimeError("No benchmark reports to persist.")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    grouped: Dict[str, List[CandidateReport]] = {}
    for report in reports:
        grouped.setdefault(report.symbol, []).append(report)
    output_paths: List[Path] = []
    for symbol, symbol_reports in grouped.items():
        output_dir = output_root / symbol
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{symbol}_chronos2_bench_{timestamp}.json"
        payload = [entry.to_payload() for entry in symbol_reports]
        with output_path.open("w") as fp:
            json.dump(payload, fp, indent=2)
        output_paths.append(output_path)
    return output_paths


def _maybe_update_hyperparams(reports: Sequence[CandidateReport], args: argparse.Namespace) -> None:
    if not args.update_hyperparams:
        return
    by_symbol: Dict[str, List[CandidateReport]] = {}
    for report in reports:
        by_symbol.setdefault(report.symbol, []).append(report)
    for symbol, symbol_reports in by_symbol.items():
        best_report = min(symbol_reports, key=lambda r: r.validation.pct_return_mae)
        target_path = Path(args.hyperparam_root) / "chronos2" / f"{symbol}.json"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        existing_val: Optional[float] = None
        if target_path.exists():
            try:
                with target_path.open() as fp:
                    existing_payload = json.load(fp)
                existing_val = existing_payload.get("validation", {}).get("pct_return_mae")
            except Exception:
                existing_val = None

        best_val = best_report.validation.pct_return_mae
        if existing_val is not None and existing_val <= best_val:
            print(
                f"[SKIP] {symbol}: validation pct MAE {best_val:.6f} is not better than existing {existing_val:.6f}; keeping current config"
            )
            continue
        payload = {
            "symbol": symbol,
            "model": "chronos2",
            "config": {
                "name": best_report.candidate.name,
                "model_id": args.model_id,
                "device_map": args.device_map,
                "context_length": best_report.candidate.context_length,
                "batch_size": best_report.candidate.batch_size,
                "quantile_levels": list(best_report.candidate.quantile_levels),
                "aggregation": best_report.candidate.aggregation,
                "sample_count": best_report.candidate.sample_count,
                "scaler": best_report.candidate.scaler,
                "predict_kwargs": best_report.candidate.predict_kwargs,
            },
            "validation": {
                "price_mae": best_report.validation.price_mae,
                "pct_return_mae": best_report.validation.pct_return_mae,
                "latency_s": best_report.validation.latency_s,
            },
            "test": {
                "price_mae": best_report.test.price_mae,
                "pct_return_mae": best_report.test.pct_return_mae,
                "latency_s": best_report.test.latency_s,
            },
            "windows": best_report.windows,
            "metadata": {
                "source": "chronos2_benchmark",
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "selection_metric": "validation_pct_return_mae",
                "selection_value": best_val,
            },
        }
        with target_path.open("w") as fp:
            json.dump(payload, fp, indent=2)
        print(
            f"[INFO] Updated {target_path} with candidate '{best_report.candidate.name}' (val pct MAE {best_val:.6f})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=["ADSK"], help="Symbols to evaluate")
    parser.add_argument(
        "--search-method",
        choices=["grid", "direct"],
        default="grid",
        help="Choose between exhaustive grid search or SciPy DIRECT optimization",
    )
    parser.add_argument("--data-dir", default="trainingdata", help="Directory containing {symbol}.csv files")
    parser.add_argument("--model-id", default="amazon/chronos-2", help="Chronos2 model identifier")
    parser.add_argument("--device-map", default=DEFAULT_DEVICE_MAP, help="Device map (default: cuda)")
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[512])
    parser.add_argument("--auto-context-lengths", action="store_true", help="Derive context lengths per symbol")
    parser.add_argument("--auto-context-min", type=int, default=512)
    parser.add_argument("--auto-context-max", type=int, default=8192)
    parser.add_argument("--auto-context-step", type=int, default=128)
    parser.add_argument(
        "--auto-context-guard",
        type=int,
        default=VAL_WINDOW + TEST_WINDOW,
        help="Rows reserved for validation/test when deriving contexts",
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[128])
    parser.add_argument("--aggregations", nargs="+", default=["median"])
    parser.add_argument("--sample-counts", type=int, nargs="+", default=[0])
    parser.add_argument("--scalers", nargs="+", default=["none"])
    parser.add_argument("--direct-sample-counts", type=int, nargs="+", help="Override sample counts for DIRECT")
    parser.add_argument("--direct-batch-sizes", type=int, nargs="+", help="Override batch sizes for DIRECT")
    parser.add_argument("--direct-aggregations", nargs="+", help="Override aggregations for DIRECT")
    parser.add_argument("--direct-scalers", nargs="+", help="Override scalers for DIRECT")
    parser.add_argument("--quantile-levels", type=float, nargs="+", default=list(DEFAULT_QUANTILE_LEVELS))
    parser.add_argument("--sample-seed", dest="seed", type=int, default=1337)
    parser.add_argument("--predict-batches-jointly", action="store_true")
    parser.add_argument("--limit-prediction-length", action="store_true")
    parser.add_argument("--max-output-patches", type=int)
    parser.add_argument("--unrolled-quantiles", type=float, nargs="+")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument(
        "--torch-dtype",
        choices=["float32", "fp32", "float16", "fp16", "half", "bfloat16", "bf16"],
        help="Optional torch dtype override for Chronos2 weights",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--candidates-file", type=Path, help="JSON file defining candidate configs")
    parser.add_argument("--max-candidates", type=int, help="Optional limit when using cross-product candidates")
    parser.add_argument("--output-dir", default="chronos2_benchmarks")
    parser.add_argument("--hyperparam-root", default="hyperparams")
    parser.add_argument("--update-hyperparams", action="store_true")
    parser.add_argument("--data-only", action="store_true", help="Skip hyperparam updates even if flag is set")
    parser.add_argument("--predict-kwargs", type=str, help="JSON dict merged into predict kwargs")
    parser.add_argument("--batch-size", dest="deprecated_batch_size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--direct-maxfun", type=int, default=48, help="Max function evals per symbol for DIRECT")
    parser.add_argument("--direct-maxiter", type=int, default=200, help="Max iterations for DIRECT")
    parser.add_argument(
        "--direct-objective",
        choices=[
            "validation_pct_mae",
            "test_pct_mae",
            "avg_pct_mae",
            "validation_price_mae",
            "avg_price_mae",
        ],
        default="validation_pct_mae",
        help="Metric minimized during DIRECT search",
    )
    parser.add_argument(
        "--direct-latency-weight",
        type=float,
        default=0.0,
        help="Optional multiplier applied to latency_s during DIRECT objective",
    )
    parser.add_argument(
        "--disable-direct-local-bias",
        dest="direct_locally_biased",
        action="store_false",
        help="Use globally biased DIRECT search instead of locally biased",
    )
    parser.set_defaults(direct_locally_biased=True)
    args = parser.parse_args()
    if args.deprecated_batch_size and args.deprecated_batch_size not in args.batch_sizes:
        args.batch_sizes.append(args.deprecated_batch_size)
    args.quantile_levels = sorted(set(args.quantile_levels))
    return args


def main() -> None:
    args = parse_args()
    base_predict_kwargs = Chronos2Benchmark._build_predict_kwargs(args)
    if args.predict_kwargs:
        base_predict_kwargs.update(json.loads(args.predict_kwargs))

    benchmark = Chronos2Benchmark(args)
    benchmark.predict_kwargs = dict(base_predict_kwargs)

    if args.search_method == "direct":
        reports = benchmark.run_direct()
    else:
        candidates = _load_candidates_from_file(args.candidates_file, tuple(args.quantile_levels), base_predict_kwargs)
        if not candidates:
            candidates = _build_cross_product_candidates(args, base_predict_kwargs)
        if not candidates:
            raise RuntimeError("No candidates specified for benchmarking.")
        reports = benchmark.run(candidates)
    output_paths = _persist_reports(reports, benchmark.output_root)
    for path in output_paths:
        print(f"[INFO] Saved benchmark report -> {path}")

    if args.update_hyperparams and not args.data_only:
        _maybe_update_hyperparams(reports, args)


if __name__ == "__main__":
    main()
