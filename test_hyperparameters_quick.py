#!/usr/bin/env python3
"""
Quick hyperparameter exploration for rapid iteration.

This is a lighter-weight version of test_hyperparameters_extended.py that
tests a strategic subset of hyperparameters for quick feedback.

Useful for:
- Quick validation on new stock pairs
- Testing before running full grid search
- Iterative experimentation
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error

from src.models.kronos_wrapper import KronosForecastingWrapper
from src.models.toto_wrapper import TotoPipeline
from src.models.toto_aggregation import aggregate_with_spec

# --- Configuration ---
FORECAST_HORIZON = 1
VAL_WINDOW = 20
TEST_WINDOW = 20
MIN_CONTEXT = 128

DATA_DIR = Path("trainingdata")
OUTPUT_ROOT = Path("hyperparams_quick")
OUTPUT_ROOT.mkdir(exist_ok=True)
(OUTPUT_ROOT / "kronos").mkdir(exist_ok=True)
(OUTPUT_ROOT / "toto").mkdir(exist_ok=True)


@dataclass(frozen=True)
class KronosRunConfig:
    name: str
    temperature: float
    top_p: float
    top_k: int
    sample_count: int
    max_context: int
    clip: float


@dataclass(frozen=True)
class TotoRunConfig:
    name: str
    num_samples: int
    aggregate: str
    samples_per_batch: int


@dataclass
class EvaluationResult:
    price_mae: float
    pct_return_mae: float
    latency_s: float
    predictions: List[float]


# --- Strategic Quick Test Grids ---
# Focus on the most promising parameter regions

KRONOS_QUICK_GRID = (
    # Very conservative configs - low temp, tight sampling
    KronosRunConfig("kronos_temp0.12_p0.78_s192_k24_clip1.5_ctx224", 0.12, 0.78, 24, 192, 224, 1.5),
    KronosRunConfig("kronos_temp0.12_p0.80_s208_k24_clip1.6_ctx224", 0.12, 0.80, 24, 208, 224, 1.6),
    KronosRunConfig("kronos_temp0.14_p0.78_s192_k20_clip1.6_ctx224", 0.14, 0.78, 20, 192, 224, 1.6),
    KronosRunConfig("kronos_temp0.14_p0.80_s208_k24_clip1.5_ctx256", 0.14, 0.80, 24, 208, 256, 1.5),
    KronosRunConfig("kronos_temp0.15_p0.80_s192_k20_clip1.7_ctx224", 0.15, 0.80, 20, 192, 224, 1.7),

    # Medium conservative - balanced exploration
    KronosRunConfig("kronos_temp0.16_p0.80_s208_k24_clip1.8_ctx224", 0.16, 0.80, 24, 208, 224, 1.8),
    KronosRunConfig("kronos_temp0.16_p0.82_s224_k24_clip1.7_ctx256", 0.16, 0.82, 24, 224, 256, 1.7),
    KronosRunConfig("kronos_temp0.18_p0.80_s224_k20_clip1.8_ctx224", 0.18, 0.80, 20, 224, 224, 1.8),
    KronosRunConfig("kronos_temp0.18_p0.82_s208_k24_clip1.6_ctx256", 0.18, 0.82, 24, 208, 256, 1.6),

    # Moderate temperature for comparison
    KronosRunConfig("kronos_temp0.20_p0.80_s224_k24_clip1.8_ctx224", 0.20, 0.80, 24, 224, 224, 1.8),
    KronosRunConfig("kronos_temp0.22_p0.82_s240_k24_clip2.0_ctx256", 0.22, 0.82, 24, 240, 256, 2.0),
)

TOTO_QUICK_GRID = (
    # Lower sample counts with conservative aggregation
    TotoRunConfig("toto_quantile15_256", 256, "quantile_0.15", 64),
    TotoRunConfig("toto_quantile18_512", 512, "quantile_0.18", 128),
    TotoRunConfig("toto_quantile20_512", 512, "quantile_0.20", 128),
    TotoRunConfig("toto_trimmed10_512", 512, "trimmed_mean_10", 128),
    TotoRunConfig("toto_lower_trim15_512", 512, "lower_trimmed_mean_15", 128),

    # Medium sample counts
    TotoRunConfig("toto_quantile15_1024", 1024, "quantile_0.15", 256),
    TotoRunConfig("toto_quantile18_1024", 1024, "quantile_0.18", 256),
    TotoRunConfig("toto_trimmed10_1024", 1024, "trimmed_mean_10", 256),
    TotoRunConfig("toto_qpstd_015_012_1024", 1024, "quantile_plus_std_0.15_0.12", 256),
    TotoRunConfig("toto_qpstd_015_015_1024", 1024, "quantile_plus_std_0.15_0.15", 256),

    # Higher sample counts
    TotoRunConfig("toto_quantile15_2048", 2048, "quantile_0.15", 256),
    TotoRunConfig("toto_trimmed10_2048", 2048, "trimmed_mean_10", 256),
    TotoRunConfig("toto_qpstd_015_015_2048", 2048, "quantile_plus_std_0.15_0.15", 256),
    TotoRunConfig("toto_mean_qmix_015_030_2048", 2048, "mean_quantile_mix_0.15_0.3", 256),

    # High sample counts for best quality
    TotoRunConfig("toto_quantile15_3072", 3072, "quantile_0.15", 384),
    TotoRunConfig("toto_trimmed10_3072", 3072, "trimmed_mean_10", 384),
)


# --- Evaluation Functions ---
KRONOS_WRAPPER_CACHE: Dict[str, KronosForecastingWrapper] = {}
_TOTO_PIPELINE: Optional[TotoPipeline] = None


def _get_kronos_wrapper(config: KronosRunConfig) -> KronosForecastingWrapper:
    """Get or create Kronos wrapper with caching."""
    key = f"{config.max_context}_{config.clip}"
    wrapper = KRONOS_WRAPPER_CACHE.get(key)
    if wrapper is None:
        wrapper = KronosForecastingWrapper(
            model_name="NeoQuasar/Kronos-base",
            tokenizer_name="NeoQuasar/Kronos-Tokenizer-base",
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            max_context=config.max_context,
            clip=config.clip,
        )
        KRONOS_WRAPPER_CACHE[key] = wrapper
    return wrapper


def _get_toto_pipeline() -> TotoPipeline:
    """Get or create Toto pipeline (singleton)."""
    global _TOTO_PIPELINE
    if _TOTO_PIPELINE is None:
        device_map = "cuda" if torch.cuda.is_available() else "cpu"
        _TOTO_PIPELINE = TotoPipeline.from_pretrained(
            model_id="Datadog/Toto-Open-Base-1.0",
            device_map=device_map,
            compile_model=False,
            torch_compile=False,
        )
    return _TOTO_PIPELINE


def _prepare_series(symbol_path: Path) -> pd.DataFrame:
    """Load and prepare time series data."""
    df = pd.read_csv(symbol_path)
    if "timestamp" not in df.columns or "close" not in df.columns:
        raise KeyError(f"{symbol_path.name} missing 'timestamp' or 'close'")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _sequential_kronos(
    df: pd.DataFrame,
    indices: Iterable[int],
    config: KronosRunConfig,
) -> EvaluationResult:
    """Evaluate Kronos on sequential forecasts."""
    wrapper = _get_kronos_wrapper(config)
    total_latency = 0.0
    preds: List[float] = []
    returns: List[float] = []
    actual_returns: List[float] = []
    actual_prices: List[float] = []

    for idx in indices:
        sub_df = df.iloc[: idx + 1].copy()
        start_time = time.perf_counter()
        result = wrapper.predict_series(
            data=sub_df,
            timestamp_col="timestamp",
            columns=["close"],
            pred_len=FORECAST_HORIZON,
            lookback=config.max_context,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            sample_count=config.sample_count,
        )
        total_latency += time.perf_counter() - start_time

        kronos_close = result.get("close")
        if kronos_close is None or kronos_close.absolute.size == 0:
            raise RuntimeError("Kronos returned no forecasts.")
        preds.append(float(kronos_close.absolute[0]))
        returns.append(float(kronos_close.percent[0]))
        actual_price = float(df["close"].iloc[idx])
        prev_price = float(df["close"].iloc[idx - 1])
        actual_prices.append(actual_price)
        if prev_price == 0.0:
            actual_returns.append(0.0)
        else:
            actual_returns.append((actual_price - prev_price) / prev_price)

    price_mae = mean_absolute_error(actual_prices, preds)
    pct_return_mae = mean_absolute_error(actual_returns, returns)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return EvaluationResult(price_mae, pct_return_mae, total_latency, preds)


def _sequential_toto(
    df: pd.DataFrame,
    indices: Iterable[int],
    config: TotoRunConfig,
) -> EvaluationResult:
    """Evaluate Toto on sequential forecasts."""
    pipeline = _get_toto_pipeline()
    prices = df["close"].to_numpy(dtype=np.float64)
    preds: List[float] = []
    returns: List[float] = []
    actual_returns: List[float] = []
    actual_prices: List[float] = []
    total_latency = 0.0

    for idx in indices:
        context = prices[:idx].astype(np.float32)
        prev_price = prices[idx - 1]

        start_time = time.perf_counter()
        forecasts = pipeline.predict(
            context=context,
            prediction_length=FORECAST_HORIZON,
            num_samples=config.num_samples,
            samples_per_batch=config.samples_per_batch,
        )
        total_latency += time.perf_counter() - start_time

        if not forecasts:
            raise RuntimeError("Toto returned no forecasts.")
        step_values = aggregate_with_spec(forecasts[0].samples, config.aggregate)
        price_pred = float(np.atleast_1d(step_values)[0])
        preds.append(price_pred)
        pred_return = 0.0 if prev_price == 0 else (price_pred - prev_price) / prev_price
        returns.append(pred_return)
        actual_price = prices[idx]
        actual_prices.append(actual_price)
        actual_returns.append(0.0 if prev_price == 0 else (actual_price - prev_price) / prev_price)
        del forecasts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    price_mae = mean_absolute_error(actual_prices, preds)
    pct_return_mae = mean_absolute_error(actual_returns, returns)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return EvaluationResult(price_mae, pct_return_mae, total_latency, preds)


def _select_best(
    evals: Dict[str, EvaluationResult],
) -> Tuple[str, EvaluationResult]:
    """Select best configuration by price MAE."""
    best_name = min(evals.keys(), key=lambda name: evals[name].price_mae)
    return best_name, evals[best_name]


def _persist_result(
    model: str,
    symbol: str,
    config,
    val_result: EvaluationResult,
    test_result: EvaluationResult,
) -> Path:
    """Persist evaluation results to JSON."""
    config_dict = asdict(config)
    validation_payload = {
        "price_mae": val_result.price_mae,
        "pct_return_mae": val_result.pct_return_mae,
        "latency_s": val_result.latency_s,
    }
    test_payload = {
        "price_mae": test_result.price_mae,
        "pct_return_mae": test_result.pct_return_mae,
        "latency_s": test_result.latency_s,
    }
    windows_payload = {
        "val_window": VAL_WINDOW,
        "test_window": TEST_WINDOW,
        "forecast_horizon": FORECAST_HORIZON,
    }

    output_path = OUTPUT_ROOT / model / f"{symbol}.json"
    output_data = {
        "model": model,
        "symbol": symbol,
        "config": config_dict,
        "validation": validation_payload,
        "test": test_payload,
        "windows": windows_payload,
        "metadata": {"source": "hyperparams_quick"},
    }
    with output_path.open("w") as f:
        json.dump(output_data, f, indent=2)

    print(f"[INFO] Saved {model} best config for {symbol} -> {output_path}")
    return output_path


def _evaluate_symbol(
    symbol_path: Path,
    *,
    test_kronos: bool = True,
    test_toto: bool = True,
) -> None:
    """Evaluate hyperparameters for a single symbol."""
    symbol = symbol_path.stem
    df = _prepare_series(symbol_path)
    if len(df) < VAL_WINDOW + TEST_WINDOW + MIN_CONTEXT:
        print(f"[WARN] {symbol}: not enough data, skipping.")
        return

    val_start = len(df) - (TEST_WINDOW + VAL_WINDOW)
    val_indices = range(val_start, len(df) - TEST_WINDOW)
    test_indices = range(len(df) - TEST_WINDOW, len(df))

    # Evaluate Kronos
    if test_kronos:
        print(f"\n[INFO] Testing {len(KRONOS_QUICK_GRID)} Kronos configurations for {symbol}...")
        kronos_val_results: Dict[str, EvaluationResult] = {}

        for idx, cfg in enumerate(KRONOS_QUICK_GRID, 1):
            try:
                print(f"[INFO] Kronos {idx}/{len(KRONOS_QUICK_GRID)}: {cfg.name}")
                result = _sequential_kronos(df, val_indices, cfg)
                kronos_val_results[cfg.name] = result
                print(f"  -> MAE: {result.price_mae:.4f}, Latency: {result.latency_s:.2f}s")
            except Exception as exc:
                print(f"[WARN] Kronos {cfg.name} failed on {symbol}: {exc}")

        if kronos_val_results:
            best_kronos_name, best_kronos_val = _select_best(kronos_val_results)
            print(f"\n[INFO] Best Kronos: {best_kronos_name} (MAE: {best_kronos_val.price_mae:.4f})")

            best_kronos_cfg = next(cfg for cfg in KRONOS_QUICK_GRID if cfg.name == best_kronos_name)
            try:
                kronos_test = _sequential_kronos(df, test_indices, best_kronos_cfg)
                _persist_result("kronos", symbol, best_kronos_cfg, best_kronos_val, kronos_test)
                print(f"[INFO] Test MAE: {kronos_test.price_mae:.4f}")
            except Exception as exc:
                print(f"[WARN] Kronos test evaluation failed for {symbol}: {exc}")

    # Evaluate Toto
    if test_toto:
        print(f"\n[INFO] Testing {len(TOTO_QUICK_GRID)} Toto configurations for {symbol}...")
        toto_val_results: Dict[str, EvaluationResult] = {}

        for idx, cfg in enumerate(TOTO_QUICK_GRID, 1):
            try:
                print(f"[INFO] Toto {idx}/{len(TOTO_QUICK_GRID)}: {cfg.name}")
                result = _sequential_toto(df, val_indices, cfg)
                toto_val_results[cfg.name] = result
                print(f"  -> MAE: {result.price_mae:.4f}, Latency: {result.latency_s:.2f}s")
            except Exception as exc:
                print(f"[WARN] Toto {cfg.name} failed on {symbol}: {exc}")

        if toto_val_results:
            best_toto_name, best_toto_val = _select_best(toto_val_results)
            print(f"\n[INFO] Best Toto: {best_toto_name} (MAE: {best_toto_val.price_mae:.4f})")

            best_toto_cfg = next(cfg for cfg in TOTO_QUICK_GRID if cfg.name == best_toto_name)
            try:
                toto_test = _sequential_toto(df, test_indices, best_toto_cfg)
                _persist_result("toto", symbol, best_toto_cfg, best_toto_val, toto_test)
                print(f"[INFO] Test MAE: {toto_test.price_mae:.4f}")
            except Exception as exc:
                print(f"[WARN] Toto test evaluation failed for {symbol}: {exc}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Quick hyperparameter exploration for Kronos/Toto"
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="Symbols to evaluate (default: all CSVs in trainingdata/)",
    )
    parser.add_argument(
        "--skip-kronos",
        action="store_true",
        help="Skip Kronos evaluation",
    )
    parser.add_argument(
        "--skip-toto",
        action="store_true",
        help="Skip Toto evaluation",
    )
    args = parser.parse_args()

    if args.symbols:
        csv_files = []
        for sym in args.symbols:
            candidate = DATA_DIR / f"{sym}.csv"
            if candidate.exists():
                csv_files.append(candidate)
            else:
                print(f"[WARN] Symbol {sym} not found in {DATA_DIR}")
    else:
        csv_files = sorted(DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    for csv_path in csv_files:
        print(f"\n{'='*60}")
        print(f"Evaluating {csv_path.stem}")
        print(f"{'='*60}")
        try:
            _evaluate_symbol(
                csv_path,
                test_kronos=not args.skip_kronos,
                test_toto=not args.skip_toto,
            )
        except Exception as exc:
            print(f"[ERROR] Failed on {csv_path.stem}: {exc}")


if __name__ == "__main__":
    main()
