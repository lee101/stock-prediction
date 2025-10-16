#!/usr/bin/env python3
"""
Hyperparameter training-style evaluation for Kronos and Toto.

For each symbol in ``trainingdata`` this script:
  1. Splits the series into training/validation/test where the final TEST_WINDOW
     observations are treated as unseen data.
  2. Runs the Kronos and Toto hyperparameter grids, scoring each configuration
     on the validation window.
  3. Selects the best configuration per model (lowest price MAE) and evaluates
     it on the held-out test window.
  4. Persists the best configuration and metrics to JSON files under
     ``hyperparams/{kronos,toto}/<symbol>.json``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error

from src.models.kronos_wrapper import KronosForecastingWrapper
from src.models.toto_wrapper import TotoPipeline
from src.models.toto_aggregation import aggregate_with_spec
from hyperparamstore import save_best_config, save_model_selection
from test_kronos_vs_toto import (
    KRONOS_SWEEP,
    KronosRunConfig,
    TotoRunConfig,
    TOTO_SWEEP,
)
import time


FORECAST_HORIZON = 1
VAL_WINDOW = 20
TEST_WINDOW = 20
MIN_CONTEXT = 128

DATA_DIR = Path("trainingdata")
OUTPUT_ROOT = Path("hyperparams")
OUTPUT_ROOT.mkdir(exist_ok=True)
(OUTPUT_ROOT / "kronos").mkdir(exist_ok=True)
(OUTPUT_ROOT / "toto").mkdir(exist_ok=True)

KRONOS_TRAIN_NAMES = {
    "kronos_temp0.15_p0.82_s208_k16_clip1.8_ctx224",
    "kronos_temp0.16_p0.80_s192_k16_clip2_ctx256",
    "kronos_temp0.14_p0.80_s200_k24_clip1.6_ctx224",
    "kronos_temp0.12_p0.78_s224_k24_clip1.5_ctx224",
    "kronos_temp0.118_p0.755_s288_k26_clip1.35_ctx192",
    "kronos_temp0.145_p0.82_s208_k16_clip1.75_ctx224",
    "kronos_temp0.148_p0.81_s240_k18_clip1.7_ctx224",
    "kronos_temp0.152_p0.83_s192_k20_clip1.85_ctx232",
    "kronos_temp0.155_p0.82_s224_k18_clip1.9_ctx240",
}
KRONOS_TRAIN_SWEEP = tuple(cfg for cfg in KRONOS_SWEEP if cfg.name in KRONOS_TRAIN_NAMES)

# Allow a lightweight Toto sweep when GPU memory is constrained.
USE_COMPACT_TOTO_SWEEP = os.getenv("TOTO_COMPACT_SWEEP", "0").strip().lower() in {"1", "true", "yes", "on"}

if USE_COMPACT_TOTO_SWEEP:
    TOTO_TRAIN_SWEEP = (
        TotoRunConfig(
            name="toto_trimmed10_128",
            num_samples=128,
            aggregate="trimmed_mean_10",
            samples_per_batch=16,
        ),
        TotoRunConfig(
            name="toto_quantile_plus_std_015_015_128",
            num_samples=128,
            aggregate="quantile_plus_std_0.15_0.15",
            samples_per_batch=16,
        ),
        TotoRunConfig(
            name="toto_quantile_plus_std_015_012_128",
            num_samples=128,
            aggregate="quantile_plus_std_0.15_0.12",
            samples_per_batch=16,
        ),
        TotoRunConfig(
            name="toto_mean_quantile_mix_015_030_128",
            num_samples=128,
            aggregate="mean_quantile_mix_0.15_0.3",
            samples_per_batch=16,
        ),
        TotoRunConfig(
            name="toto_quantile15_128",
            num_samples=128,
            aggregate="quantile_0.15",
            samples_per_batch=16,
        ),
    )
else:
    TOTO_TRAIN_NAMES = {
        "toto_quantile_plus_std_015_015",
        "toto_quantile_plus_std_015_012",
        "toto_quantile_plus_std_0145_018",
        "toto_mean_quantile_mix_0.15_0.3",
        "toto_mean_quantile_mix_0.145_0.40",
        "toto_quantile15_3072",
        "toto_trimmed10_3072",
    }
    TOTO_TRAIN_SWEEP = tuple(cfg for cfg in TOTO_SWEEP if cfg.name in TOTO_TRAIN_NAMES)

if not KRONOS_TRAIN_SWEEP or not TOTO_TRAIN_SWEEP:
    raise RuntimeError("Training sweeps could not be constructed from base grids.")

@dataclass
class EvaluationResult:
    price_mae: float
    return_mae: float
    latency_s: float
    predictions: List[float]


def _prepare_series(symbol_path: Path) -> pd.DataFrame:
    df = pd.read_csv(symbol_path)
    if "timestamp" not in df.columns or "close" not in df.columns:
        raise KeyError(f"{symbol_path.name} missing 'timestamp' or 'close'")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


KRONOS_WRAPPER_CACHE: Dict[str, KronosForecastingWrapper] = {}
_TOTO_PIPELINE: Optional[TotoPipeline] = None


def _get_kronos_wrapper(config: KronosRunConfig) -> KronosForecastingWrapper:
    key = (
        f"{config.temperature}_{config.top_p}_{config.top_k}_"
        f"{config.sample_count}_{config.max_context}_{config.clip}"
    )
    wrapper = KRONOS_WRAPPER_CACHE.get(key)
    if wrapper is None:
        wrapper = KronosForecastingWrapper(
            model_name="NeoQuasar/Kronos-base",
            tokenizer_name="NeoQuasar/Kronos-Tokenizer-base",
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            max_context=config.max_context,
            clip=config.clip,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            sample_count=config.sample_count,
        )
        KRONOS_WRAPPER_CACHE[key] = wrapper
    return wrapper


def _get_toto_pipeline() -> TotoPipeline:
    global _TOTO_PIPELINE
    if _TOTO_PIPELINE is None:
        device_map = "cuda" if torch.cuda.is_available() else "cpu"
        _TOTO_PIPELINE = TotoPipeline.from_pretrained(
            model_id="Datadog/Toto-Open-Base-1.0",
            device_map=device_map,
        )
    return _TOTO_PIPELINE


def _sequential_kronos(
    df: pd.DataFrame,
    indices: Iterable[int],
    config: KronosRunConfig,
) -> EvaluationResult:
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
    return_mae = mean_absolute_error(actual_returns, returns)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return EvaluationResult(price_mae, return_mae, total_latency, preds)


def _sequential_toto(
    df: pd.DataFrame,
    indices: Iterable[int],
    config: TotoRunConfig,
) -> EvaluationResult:
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
    return_mae = mean_absolute_error(actual_returns, returns)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return EvaluationResult(price_mae, return_mae, total_latency, preds)


def _select_best(
    evals: Dict[str, EvaluationResult],
) -> Tuple[str, EvaluationResult]:
    best_name = min(evals.keys(), key=lambda name: evals[name].price_mae)
    return best_name, evals[best_name]


def _evaluate_symbol(symbol_path: Path) -> None:
    symbol = symbol_path.stem
    df = _prepare_series(symbol_path)
    if len(df) < VAL_WINDOW + TEST_WINDOW + MIN_CONTEXT:
        print(f"[WARN] {symbol}: not enough data, skipping.")
        return

    val_start = len(df) - (TEST_WINDOW + VAL_WINDOW)
    val_indices = range(val_start, len(df) - TEST_WINDOW)
    test_indices = range(len(df) - TEST_WINDOW, len(df))

    kronos_val_results: Dict[str, EvaluationResult] = {}
    kronos_summary: Optional[Dict[str, Any]] = None
    for cfg in KRONOS_TRAIN_SWEEP:
        try:
            kronos_val_results[cfg.name] = _sequential_kronos(df, val_indices, cfg)
        except Exception as exc:
            print(f"[WARN] Kronos {cfg.name} failed on {symbol}: {exc}")

    if not kronos_val_results:
        print(f"[WARN] {symbol}: no Kronos configs succeeded.")
    else:
        best_kronos_name, best_kronos_val = _select_best(kronos_val_results)
        best_kronos_cfg = next(cfg for cfg in KRONOS_TRAIN_SWEEP if cfg.name == best_kronos_name)
        kronos_test = None
        try:
            kronos_test = _sequential_kronos(df, test_indices, best_kronos_cfg)
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(f"[WARN] Kronos test evaluation failed for {symbol} ({best_kronos_cfg.name}): {exc}")
        if kronos_test is not None:
            config_dict, val_payload, test_payload, path = _persist_result(
                "kronos",
                symbol,
                best_kronos_cfg,
                best_kronos_val,
                kronos_test,
            )
            kronos_summary = {
                "model": "kronos",
                "config": config_dict,
                "validation": val_payload,
                "test": test_payload,
                "path": str(path),
            }

    toto_val_results: Dict[str, EvaluationResult] = {}
    toto_summary: Optional[Dict[str, Any]] = None
    for cfg in TOTO_TRAIN_SWEEP:
        try:
            toto_val_results[cfg.name] = _sequential_toto(df, val_indices, cfg)
        except Exception as exc:
            print(f"[WARN] Toto {cfg.name} failed on {symbol}: {exc}")

    if not toto_val_results:
        print(f"[WARN] {symbol}: no Toto configs succeeded.")
    else:
        best_toto_name, best_toto_val = _select_best(toto_val_results)
        best_toto_cfg = next(cfg for cfg in TOTO_TRAIN_SWEEP if cfg.name == best_toto_name)
        toto_test = None
        try:
            toto_test = _sequential_toto(df, test_indices, best_toto_cfg)
        except Exception as exc:
            print(f"[WARN] Toto test evaluation failed for {symbol} ({best_toto_cfg.name}): {exc}")
        if toto_test is not None:
            config_dict, val_payload, test_payload, path = _persist_result(
                "toto",
                symbol,
                best_toto_cfg,
                best_toto_val,
                toto_test,
            )
            toto_summary = {
                "model": "toto",
                "config": config_dict,
                "validation": val_payload,
                "test": test_payload,
                "path": str(path),
            }

    # Save overall best model selection
    selection = None
    if kronos_summary and toto_summary:
        if kronos_summary["validation"]["price_mae"] <= toto_summary["validation"]["price_mae"]:
            selection = kronos_summary
        else:
            selection = toto_summary
    elif kronos_summary:
        selection = kronos_summary
    elif toto_summary:
        selection = toto_summary

    if selection is not None:
        save_model_selection(
            symbol=symbol,
            model=selection["model"],
            config=selection["config"],
            validation=selection["validation"],
            test=selection["test"],
            windows={
                "val_window": VAL_WINDOW,
                "test_window": TEST_WINDOW,
                "forecast_horizon": FORECAST_HORIZON,
            },
            metadata={"source": "hyperparamtraining"},
            config_path=selection["path"],
        )


def _persist_result(
    model: str,
    symbol: str,
    config,
    val_result: EvaluationResult,
    test_result: EvaluationResult,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Path]:
    config_dict = asdict(config)
    validation_payload = {
        "price_mae": val_result.price_mae,
        "return_mae": val_result.return_mae,
        "latency_s": val_result.latency_s,
    }
    test_payload = {
        "price_mae": test_result.price_mae,
        "return_mae": test_result.return_mae,
        "latency_s": test_result.latency_s,
    }
    windows_payload = {
        "val_window": VAL_WINDOW,
        "test_window": TEST_WINDOW,
        "forecast_horizon": FORECAST_HORIZON,
    }
    path = save_best_config(
        model=model,
        symbol=symbol,
        config=config_dict,
        validation=validation_payload,
        test=test_payload,
        windows=windows_payload,
        metadata={"source": "hyperparamtraining"},
    )
    print(f"[INFO] Saved {model} best config for {symbol} -> {path}")
    return config_dict, validation_payload, test_payload, path


def main(symbols: List[str] | None = None) -> None:
    if symbols:
        csv_files = []
        for sym in symbols:
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
        print(f"\n=== Evaluating {csv_path.stem} ===")
        try:
            _evaluate_symbol(csv_path)
        except Exception as exc:
            print(f"[ERROR] Failed on {csv_path.stem}: {exc}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hyperparameter training for Kronos/Toto.")
    parser.add_argument("--symbols", nargs="*", help="Symbols to evaluate (default: all CSVs)")
    args = parser.parse_args()
    main(args.symbols)
