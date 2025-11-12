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
import math
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

try:
    import chronos_compile_config
except ModuleNotFoundError:  # pragma: no cover - optional compile tweaks
    chronos_compile_config = None  # type: ignore

from src.models.chronos2_wrapper import Chronos2OHLCWrapper, DEFAULT_QUANTILE_LEVELS
from src.models.toto_aggregation import aggregate_with_spec

# Sliding window constants (align with test_hyperparameters_extended)
FORECAST_HORIZON = 1
VAL_WINDOW = 20
TEST_WINDOW = 20
MIN_CONTEXT = 128

TARGET_COLUMNS = ("open", "high", "low", "close")
DEFAULT_DEVICE_MAP = "cuda"
GAUSSIAN_Q_FACTOR = 2.0 * 1.2815515655446004  # Difference between q90 and q10 in std units for Normal dist.


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


@dataclass
class EvaluationResult:
    price_mae: float
    pct_return_mae: float
    latency_s: float
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


class ColumnScaler:
    """Simple reversible column-wise scaler (currently supports identity + mean/std)."""

    def __init__(self, method: str, frame: pd.DataFrame, columns: Sequence[str]) -> None:
        self.method = (method or "none").lower()
        self.columns = tuple(columns)
        self.params: Dict[str, Dict[str, float]] = {}
        if self.method == "none":
            return
        if self.method == "meanstd":
            for column in self.columns:
                if column not in frame.columns:
                    continue
                series = frame[column].astype("float64")
                mean = float(series.mean())
                std = float(series.std(ddof=0))
                if not math.isfinite(std) or std < 1e-6:
                    std = max(abs(mean) * 1e-3, 1.0)
                self.params[column] = {"mean": mean, "std": std}
            return
        raise ValueError(f"Unsupported scaler '{self.method}'")

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.method == "none":
            return frame
        result = frame.copy()
        for column, stats in self.params.items():
            if column not in result.columns:
                continue
            result[column] = (result[column] - stats["mean"]) / stats["std"]
        return result

    def inverse(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.method == "none":
            return frame
        result = frame.copy()
        for column, stats in self.params.items():
            if column not in result.columns:
                continue
            result[column] = result[column] * stats["std"] + stats["mean"]
        return result


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


def _gaussian_sample_matrix(
    median: np.ndarray,
    q10: np.ndarray,
    q90: np.ndarray,
    sample_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if sample_count <= 0:
        return median.reshape(1, -1)

    spread = (q90 - q10) / GAUSSIAN_Q_FACTOR
    eps = np.maximum(1e-6, np.abs(median) * 1e-4)
    std = np.clip(spread, eps, None)
    samples = rng.normal(loc=median, scale=std, size=(sample_count, median.size))
    return samples.astype(np.float64)


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
            # unload wrappers between symbols to free GPU memory
            for wrapper in self.wrapper_cache.values():
                wrapper.unload()
            self.wrapper_cache.clear()
        return reports

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

        quantiles_needed = set(candidate.quantile_levels)
        quantiles_needed.add(0.5)
        if candidate.sample_count > 0:
            quantiles_needed.update((0.1, 0.5, 0.9))

        # ensure consistent order
        quantile_tuple = tuple(sorted(quantiles_needed))

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

            quantile_frames: Dict[float, pd.DataFrame] = {}
            for level in quantile_tuple:
                quantile_frames[level] = scaler.inverse(batch.quantile(level))

            median_series = quantile_frames.get(0.5, next(iter(quantile_frames.values())))["close"].to_numpy(
                dtype=np.float64
            )

            if candidate.sample_count > 0:
                q10_series = quantile_frames.get(0.1, quantile_frames[min(quantile_frames.keys())])[
                    "close"
                ].to_numpy(dtype=np.float64)
                q90_series = quantile_frames.get(0.9, quantile_frames[max(quantile_frames.keys())])[
                    "close"
                ].to_numpy(dtype=np.float64)
                samples = _gaussian_sample_matrix(
                    median_series,
                    q10_series,
                    q90_series,
                    candidate.sample_count,
                    rng=self.rng,
                )
            else:
                samples = median_series

            samples_matrix = np.asarray(samples, dtype=np.float64)
            if samples_matrix.ndim == 0:
                samples_matrix = samples_matrix.reshape(1, 1)
            elif samples_matrix.ndim == 1:
                samples_matrix = samples_matrix.reshape(1, -1)

            if candidate.sample_count <= 1:
                price_pred = float(np.atleast_1d(median_series)[0])
            else:
                try:
                    aggregated = aggregate_with_spec(samples_matrix, candidate.aggregation)
                except ValueError as exc:
                    raise RuntimeError(
                        f"Aggregation failed for {candidate.aggregation} with samples shape={samples_matrix.shape}"
                    ) from exc
                price_pred = float(np.atleast_1d(aggregated)[0])

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
        pct_return_mae = mean_absolute_error(actual_returns, returns)
        return EvaluationResult(
            price_mae=price_mae,
            pct_return_mae=pct_return_mae,
            latency_s=total_latency,
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


def _persist_reports(reports: Sequence[CandidateReport], output_root: Path) -> Path:
    if not reports:
        raise RuntimeError("No benchmark reports to persist.")
    symbol = reports[0].symbol
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / symbol
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{symbol}_chronos2_bench_{timestamp}.json"
    payload = [report.to_payload() for report in reports]
    with output_path.open("w") as fp:
        json.dump(payload, fp, indent=2)
    return output_path


def _maybe_update_hyperparams(reports: Sequence[CandidateReport], args: argparse.Namespace) -> None:
    if not args.update_hyperparams:
        return
    by_symbol: Dict[str, List[CandidateReport]] = {}
    for report in reports:
        by_symbol.setdefault(report.symbol, []).append(report)
    for symbol, symbol_reports in by_symbol.items():
        best_report = min(symbol_reports, key=lambda r: r.test.pct_return_mae)
        target_path = Path(args.hyperparam_root) / "chronos2" / f"{symbol}.json"
        target_path.parent.mkdir(parents=True, exist_ok=True)
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
            },
        }
        with target_path.open("w") as fp:
            json.dump(payload, fp, indent=2)
        print(f"[INFO] Updated {target_path} with candidate '{best_report.candidate.name}'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=["ADSK"], help="Symbols to evaluate")
    parser.add_argument("--data-dir", default="trainingdata", help="Directory containing {symbol}.csv files")
    parser.add_argument("--model-id", default="amazon/chronos-2", help="Chronos2 model identifier")
    parser.add_argument("--device-map", default=DEFAULT_DEVICE_MAP, help="Device map (default: cuda)")
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[512])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[128])
    parser.add_argument("--aggregations", nargs="+", default=["median"])
    parser.add_argument("--sample-counts", type=int, nargs="+", default=[0])
    parser.add_argument("--scalers", nargs="+", default=["none"])
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

    candidates = _load_candidates_from_file(args.candidates_file, tuple(args.quantile_levels), base_predict_kwargs)
    if not candidates:
        candidates = _build_cross_product_candidates(args, base_predict_kwargs)
    if not candidates:
        raise RuntimeError("No candidates specified for benchmarking.")

    reports = benchmark.run(candidates)
    output_path = _persist_reports(reports, benchmark.output_root)
    print(f"[INFO] Saved benchmark report -> {output_path}")

    if args.update_hyperparams and not args.data_only:
        _maybe_update_hyperparams(reports, args)


if __name__ == "__main__":
    main()
