#!/usr/bin/env python3
"""
Extended hyperparameter exploration for Kronos and Toto models.

This script performs a comprehensive grid search over hyperparameters to
optimize MAE on stock pairs in trainingdata/. It explores:

For Kronos:
- Temperature: wider range from 0.10 to 0.30
- Top-p: range from 0.70 to 0.90
- Sample counts: from 128 to 320
- Context lengths: 192, 224, 256, 288
- Clip values: 1.2 to 2.5
- Top-k: different values for diversity

For Toto:
- Number of samples: from 64 to 4096
- Aggregation strategies: quantile variations, trimmed means, std-based
- Samples per batch: optimized for each configuration

Results are saved to hyperparams_extended/{kronos,toto}/<symbol>.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error

try:  # Optional dependency; required when --search-method=optuna
    import optuna
except ModuleNotFoundError:  # pragma: no cover - optuna optional
    optuna = None  # type: ignore[assignment]

from src.models.kronos_wrapper import KronosForecastingWrapper
from src.models.toto_wrapper import TotoPipeline
from src.models.toto_aggregation import aggregate_with_spec
from hyperparamstore import save_best_config, save_model_selection

# --- Configuration ---
FORECAST_HORIZON = 1
VAL_WINDOW = 20
TEST_WINDOW = 20
MIN_CONTEXT = 128

DATA_DIR = Path("trainingdata")
OUTPUT_ROOT = Path("hyperparams_extended")
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


# --- Hyperparameter domains (shared across search strategies) ---
KRONOS_TEMPERATURES: Sequence[float] = (
    0.10,
    0.12,
    0.14,
    0.15,
    0.16,
    0.18,
    0.20,
    0.22,
    0.24,
    0.28,
    0.30,
)
KRONOS_TOP_PS: Sequence[float] = (0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90)
KRONOS_SAMPLE_COUNTS: Sequence[int] = (128, 160, 192, 208, 224, 256, 288, 320)
KRONOS_CONTEXTS: Sequence[int] = (192, 224, 256, 288)
KRONOS_CLIPS: Sequence[float] = (1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5)
KRONOS_TOP_KS: Sequence[int] = (0, 16, 20, 24, 28, 32)

TOTO_SAMPLE_COUNTS: Sequence[int] = (64, 128, 256, 512, 1024, 2048, 3072, 4096)
TOTO_AGGREGATIONS: Sequence[str] = (
    "mean",
    "median",
    "trimmed_mean_5",
    "trimmed_mean_10",
    "trimmed_mean_15",
    "trimmed_mean_20",
    "lower_trimmed_mean_10",
    "lower_trimmed_mean_15",
    "lower_trimmed_mean_20",
    "quantile_0.10",
    "quantile_0.15",
    "quantile_0.18",
    "quantile_0.20",
    "quantile_0.25",
    "quantile_0.30",
    "quantile_0.35",
    "mean_minus_std_0.3",
    "mean_minus_std_0.5",
    "mean_minus_std_0.7",
    "mean_plus_std_0.3",
    "quantile_plus_std_0.10_0.10",
    "quantile_plus_std_0.10_0.15",
    "quantile_plus_std_0.15_0.10",
    "quantile_plus_std_0.15_0.12",
    "quantile_plus_std_0.15_0.15",
    "quantile_plus_std_0.15_0.18",
    "quantile_plus_std_0.18_0.15",
    "quantile_plus_std_0.20_0.10",
    "mean_quantile_mix_0.10_0.25",
    "mean_quantile_mix_0.15_0.30",
    "mean_quantile_mix_0.15_0.40",
    "mean_quantile_mix_0.20_0.35",
)
TOTO_SPB_INDEX_CHOICES: Sequence[int] = (0, 1, 2, 3)


def _toto_samples_per_batch_options(num_samples: int) -> Sequence[int]:
    if num_samples <= 128:
        return (16, 32)
    if num_samples <= 512:
        return (32, 64, 128)
    if num_samples <= 1024:
        return (64, 128, 256)
    if num_samples <= 2048:
        return (128, 256)
    return (256, 512)


# --- Extended Kronos Hyperparameter Grid ---
# Testing combinations that focus on lower temperatures and tighter sampling
def generate_kronos_grid() -> Tuple[KronosRunConfig, ...]:
    """Generate comprehensive Kronos hyperparameter grid."""
    configs = []

    # Conservative configurations (lower temperature, tighter sampling)
    # Generate a strategic subset (not all combinations to avoid explosion)
    # Focus on promising regions based on existing results

    # Region 1: Very conservative (low temp, high top_p)
    for temp in [0.10, 0.12, 0.14]:
        for top_p in [0.78, 0.80, 0.82]:
            for sample_count in [192, 224, 256]:
                for context in [224, 256]:
                    for clip in [1.4, 1.6, 1.8]:
                        for top_k in [20, 24, 28]:
                            configs.append(KronosRunConfig(
                                name=f"kronos_temp{temp}_p{top_p}_s{sample_count}_k{top_k}_clip{clip}_ctx{context}",
                                temperature=temp,
                                top_p=top_p,
                                top_k=top_k,
                                sample_count=sample_count,
                                max_context=context,
                                clip=clip,
                            ))

    # Region 2: Medium conservative (mid temp, varied top_p)
    for temp in [0.15, 0.16, 0.18]:
        for top_p in [0.75, 0.80, 0.82, 0.85]:
            for sample_count in [160, 192, 208, 240]:
                for context in [192, 224, 256]:
                    for clip in [1.5, 1.7, 2.0]:
                        for top_k in [16, 20, 24]:
                            configs.append(KronosRunConfig(
                                name=f"kronos_temp{temp}_p{top_p}_s{sample_count}_k{top_k}_clip{clip}_ctx{context}",
                                temperature=temp,
                                top_p=top_p,
                                top_k=top_k,
                                sample_count=sample_count,
                                max_context=context,
                                clip=clip,
                            ))

    # Region 3: Moderate exploration (higher temp for comparison)
    for temp in [0.20, 0.22, 0.24]:
        for top_p in [0.78, 0.82, 0.85]:
            for sample_count in [192, 224, 256, 288]:
                for context in [224, 256]:
                    for clip in [1.6, 1.8, 2.0, 2.2]:
                        for top_k in [16, 24, 32]:
                            configs.append(KronosRunConfig(
                                name=f"kronos_temp{temp}_p{top_p}_s{sample_count}_k{top_k}_clip{clip}_ctx{context}",
                                temperature=temp,
                                top_p=top_p,
                                top_k=top_k,
                                sample_count=sample_count,
                                max_context=context,
                                clip=clip,
                            ))

    return tuple(configs)


# --- Extended Toto Hyperparameter Grid ---
def generate_toto_grid() -> Tuple[TotoRunConfig, ...]:
    """Generate comprehensive Toto hyperparameter grid."""
    configs = []

    # Test various sample counts
    for num_samples in TOTO_SAMPLE_COUNTS:
        # Adjust samples_per_batch based on num_samples
        samples_per_batch_options = _toto_samples_per_batch_options(num_samples)

        for aggregate in TOTO_AGGREGATIONS:
            for samples_per_batch in samples_per_batch_options:
                configs.append(TotoRunConfig(
                    name=f"toto_{aggregate}_{num_samples}_spb{samples_per_batch}",
                    num_samples=num_samples,
                    aggregate=aggregate,
                    samples_per_batch=samples_per_batch,
                ))

    return tuple(configs)


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
            compile_model=False,  # Disable compile for faster testing
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
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Path, Path]:
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

    # Save to extended directory
    output_path = OUTPUT_ROOT / model / f"{symbol}.json"
    output_data = {
        "model": model,
        "symbol": symbol,
        "config": config_dict,
        "validation": validation_payload,
        "test": test_payload,
        "windows": windows_payload,
        "metadata": {"source": "hyperparams_extended"},
    }
    with output_path.open("w") as f:
        json.dump(output_data, f, indent=2)

    metadata = {
        "source": "hyperparams_extended",
        "extended_path": str(output_path),
    }
    store_path = save_best_config(
        model=model,
        symbol=symbol,
        config=config_dict,
        validation=validation_payload,
        test=test_payload,
        windows=windows_payload,
        metadata=metadata,
    )

    print(f"[INFO] Saved {model} best config for {symbol} -> {output_path} (store: {store_path})")
    return config_dict, validation_payload, test_payload, output_path, store_path


def _ensure_optuna_available() -> None:
    if optuna is None:
        raise RuntimeError(
            "Optuna is not installed. Install it with 'uv pip install optuna' to use --search-method=optuna."
        )


def _sample_kronos_config_from_trial(trial: "optuna.Trial") -> KronosRunConfig:  # type: ignore[name-defined]
    temperature = trial.suggest_categorical("temperature", list(KRONOS_TEMPERATURES))
    top_p = trial.suggest_categorical("top_p", list(KRONOS_TOP_PS))
    top_k = trial.suggest_categorical("top_k", list(KRONOS_TOP_KS))
    sample_count = trial.suggest_categorical("sample_count", list(KRONOS_SAMPLE_COUNTS))
    max_context = trial.suggest_categorical("max_context", list(KRONOS_CONTEXTS))
    clip = trial.suggest_categorical("clip", list(KRONOS_CLIPS))
    name = (
        f"kronos_opt_temp{temperature:.3f}_p{top_p:.2f}_s{sample_count}"
        f"_k{top_k}_clip{clip:.2f}_ctx{max_context}"
    )
    return KronosRunConfig(
        name=name,
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        sample_count=int(sample_count),
        max_context=int(max_context),
        clip=float(clip),
    )


def _sample_toto_config_from_trial(trial: "optuna.Trial") -> TotoRunConfig:  # type: ignore[name-defined]
    num_samples = trial.suggest_categorical("num_samples", list(TOTO_SAMPLE_COUNTS))
    aggregate = trial.suggest_categorical("aggregate", list(TOTO_AGGREGATIONS))
    spb_index = trial.suggest_categorical("samples_per_batch_idx", list(TOTO_SPB_INDEX_CHOICES))
    spb_candidates = _toto_samples_per_batch_options(int(num_samples))
    selected_idx = min(int(spb_index), len(spb_candidates) - 1)
    samples_per_batch = spb_candidates[selected_idx]
    name = f"toto_opt_{aggregate}_{int(num_samples)}_spb{int(samples_per_batch)}"
    return TotoRunConfig(
        name=name,
        num_samples=int(num_samples),
        aggregate=str(aggregate),
        samples_per_batch=int(samples_per_batch),
    )


def _optuna_optimize_kronos(
    *,
    df: pd.DataFrame,
    symbol: str,
    val_indices: Iterable[int],
    test_indices: Iterable[int],
    trials: int,
) -> Tuple[KronosRunConfig, EvaluationResult, EvaluationResult, "optuna.study.Study"]:  # type: ignore[name-defined]
    _ensure_optuna_available()
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name=f"kronos_{symbol}")

    def objective(trial: "optuna.Trial") -> float:  # type: ignore[name-defined]
        config = _sample_kronos_config_from_trial(trial)
        start = time.perf_counter()
        result = _sequential_kronos(df, val_indices, config)
        latency = time.perf_counter() - start
        trial.set_user_attr("config", config)
        trial.set_user_attr("validation_result", result)
        trial.set_user_attr("latency", latency)
        print(
            f"[OPTUNA][Kronos][{symbol}] Trial {trial.number}: "
            f"MAE={result.price_mae:.4f}, temp={config.temperature:.3f}, "
            f"top_p={config.top_p:.2f}, top_k={config.top_k}, samples={config.sample_count}, "
            f"ctx={config.max_context}, clip={config.clip:.2f}, latency={latency:.2f}s"
        )
        return result.price_mae

    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best_trial = study.best_trial
    best_config = best_trial.user_attrs["config"]
    best_val_result = best_trial.user_attrs["validation_result"]
    best_test_result = _sequential_kronos(df, test_indices, best_config)
    return best_config, best_val_result, best_test_result, study


def _optuna_optimize_toto(
    *,
    df: pd.DataFrame,
    symbol: str,
    val_indices: Iterable[int],
    test_indices: Iterable[int],
    trials: int,
) -> Tuple[TotoRunConfig, EvaluationResult, EvaluationResult, "optuna.study.Study"]:  # type: ignore[name-defined]
    _ensure_optuna_available()
    sampler = optuna.samplers.TPESampler(seed=52)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name=f"toto_{symbol}")

    def objective(trial: "optuna.Trial") -> float:  # type: ignore[name-defined]
        config = _sample_toto_config_from_trial(trial)
        start = time.perf_counter()
        result = _sequential_toto(df, val_indices, config)
        latency = time.perf_counter() - start
        trial.set_user_attr("config", config)
        trial.set_user_attr("validation_result", result)
        trial.set_user_attr("latency", latency)
        print(
            f"[OPTUNA][Toto][{symbol}] Trial {trial.number}: "
            f"MAE={result.price_mae:.4f}, agg={config.aggregate}, samples={config.num_samples}, "
            f"spb={config.samples_per_batch}, latency={latency:.2f}s"
        )
        return result.price_mae

    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best_trial = study.best_trial
    best_config = best_trial.user_attrs["config"]
    best_val_result = best_trial.user_attrs["validation_result"]
    best_test_result = _sequential_toto(df, test_indices, best_config)
    return best_config, best_val_result, best_test_result, study


def _evaluate_symbol(
    symbol_path: Path,
    *,
    test_kronos: bool = True,
    test_toto: bool = True,
    max_kronos_configs: Optional[int] = None,
    max_toto_configs: Optional[int] = None,
    search_method: str = "grid",
    kronos_trials: Optional[int] = None,
    toto_trials: Optional[int] = None,
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
    kronos_summary: Optional[Dict[str, Any]] = None
    if test_kronos:
        if search_method == "optuna":
            kronos_trials = kronos_trials or 200
            print(f"\n[INFO] Running Optuna Kronos search for {symbol} ({kronos_trials} trials)...")
            (
                best_kronos_cfg,
                best_kronos_val,
                kronos_test,
                kronos_study,
            ) = _optuna_optimize_kronos(
                df=df,
                symbol=symbol,
                val_indices=val_indices,
                test_indices=test_indices,
                trials=kronos_trials,
            )
            print(
                f"[INFO] Optuna Kronos best MAE={best_kronos_val.price_mae:.4f} "
                f"(trial {kronos_study.best_trial.number})"
            )
            (
                config_dict,
                val_payload,
                test_payload,
                extended_path,
                store_path,
            ) = _persist_result("kronos", symbol, best_kronos_cfg, best_kronos_val, kronos_test)
            kronos_summary = {
                "model": "kronos",
                "config": config_dict,
                "validation": val_payload,
                "test": test_payload,
                "extended_path": str(extended_path),
                "store_path": str(store_path),
            }
            print(f"[INFO] Kronos test MAE: {kronos_test.price_mae:.4f}")
        else:
            print(f"\n[INFO] Testing Kronos configurations for {symbol}...")
            kronos_configs_full: List[KronosRunConfig] = list(generate_kronos_grid())
            kronos_configs = kronos_configs_full
            if max_kronos_configs:
                total_kronos = len(kronos_configs_full)
                if max_kronos_configs < total_kronos:
                    rng = random.Random(42)
                    rng.shuffle(kronos_configs)
                    print(
                        f"[INFO] Shuffled Kronos grid; sampling first {max_kronos_configs} configs out of {total_kronos}"
                    )
                kronos_configs = kronos_configs[:max_kronos_configs]

            print(f"[INFO] Testing {len(kronos_configs)} Kronos configurations")
            kronos_val_results: Dict[str, EvaluationResult] = {}

            for idx, cfg in enumerate(kronos_configs, 1):
                try:
                    print(f"[INFO] Kronos {idx}/{len(kronos_configs)}: {cfg.name}")
                    result = _sequential_kronos(df, val_indices, cfg)
                    kronos_val_results[cfg.name] = result
                    print(f"  -> MAE: {result.price_mae:.4f}, Latency: {result.latency_s:.2f}s")
                except Exception as exc:
                    print(f"[WARN] Kronos {cfg.name} failed on {symbol}: {exc}")

            if kronos_val_results:
                best_kronos_name, best_kronos_val = _select_best(kronos_val_results)
                print(f"\n[INFO] Best Kronos: {best_kronos_name} (MAE: {best_kronos_val.price_mae:.4f})")

                best_kronos_cfg = next(cfg for cfg in kronos_configs if cfg.name == best_kronos_name)
                try:
                    kronos_test = _sequential_kronos(df, test_indices, best_kronos_cfg)
                    (
                        config_dict,
                        val_payload,
                        test_payload,
                        extended_path,
                        store_path,
                    ) = _persist_result("kronos", symbol, best_kronos_cfg, best_kronos_val, kronos_test)
                    kronos_summary = {
                        "model": "kronos",
                        "config": config_dict,
                        "validation": val_payload,
                        "test": test_payload,
                        "extended_path": str(extended_path),
                        "store_path": str(store_path),
                    }
                    print(f"[INFO] Test MAE: {kronos_test.price_mae:.4f}")
                except Exception as exc:
                    print(f"[WARN] Kronos test evaluation failed for {symbol}: {exc}")

    # Evaluate Toto
    toto_summary: Optional[Dict[str, Any]] = None
    if test_toto:
        if search_method == "optuna":
            toto_trials = toto_trials or 150
            print(f"\n[INFO] Running Optuna Toto search for {symbol} ({toto_trials} trials)...")
            (
                best_toto_cfg,
                best_toto_val,
                toto_test,
                toto_study,
            ) = _optuna_optimize_toto(
                df=df,
                symbol=symbol,
                val_indices=val_indices,
                test_indices=test_indices,
                trials=toto_trials,
            )
            print(
                f"[INFO] Optuna Toto best MAE={best_toto_val.price_mae:.4f} "
                f"(trial {toto_study.best_trial.number})"
            )
            (
                config_dict,
                val_payload,
                test_payload,
                extended_path,
                store_path,
            ) = _persist_result("toto", symbol, best_toto_cfg, best_toto_val, toto_test)
            toto_summary = {
                "model": "toto",
                "config": config_dict,
                "validation": val_payload,
                "test": test_payload,
                "extended_path": str(extended_path),
                "store_path": str(store_path),
            }
            print(f"[INFO] Toto test MAE: {toto_test.price_mae:.4f}")
        else:
            print(f"\n[INFO] Testing Toto configurations for {symbol}...")
            toto_configs_full: List[TotoRunConfig] = list(generate_toto_grid())
            toto_configs = toto_configs_full
            if max_toto_configs:
                total_toto = len(toto_configs_full)
                if max_toto_configs < total_toto:
                    rng = random.Random(1337)
                    rng.shuffle(toto_configs)
                    print(
                        f"[INFO] Shuffled Toto grid; sampling first {max_toto_configs} configs out of {total_toto}"
                    )
                toto_configs = toto_configs[:max_toto_configs]

            print(f"[INFO] Testing {len(toto_configs)} Toto configurations")
            toto_val_results: Dict[str, EvaluationResult] = {}

            for idx, cfg in enumerate(toto_configs, 1):
                try:
                    print(f"[INFO] Toto {idx}/{len(toto_configs)}: {cfg.name}")
                    result = _sequential_toto(df, val_indices, cfg)
                    toto_val_results[cfg.name] = result
                    print(f"  -> MAE: {result.price_mae:.4f}, Latency: {result.latency_s:.2f}s")
                except Exception as exc:
                    print(f"[WARN] Toto {cfg.name} failed on {symbol}: {exc}")

            if toto_val_results:
                best_toto_name, best_toto_val = _select_best(toto_val_results)
                print(f"\n[INFO] Best Toto: {best_toto_name} (MAE: {best_toto_val.price_mae:.4f})")

                best_toto_cfg = next(cfg for cfg in toto_configs if cfg.name == best_toto_name)
                try:
                    toto_test = _sequential_toto(df, test_indices, best_toto_cfg)
                    (
                        config_dict,
                        val_payload,
                        test_payload,
                        extended_path,
                        store_path,
                    ) = _persist_result("toto", symbol, best_toto_cfg, best_toto_val, toto_test)
                    toto_summary = {
                        "model": "toto",
                        "config": config_dict,
                        "validation": val_payload,
                        "test": test_payload,
                        "extended_path": str(extended_path),
                        "store_path": str(store_path),
                    }
                    print(f"[INFO] Test MAE: {toto_test.price_mae:.4f}")
                except Exception as exc:
                    print(f"[WARN] Toto test evaluation failed for {symbol}: {exc}")

    selection: Optional[Dict[str, Any]] = None
    if kronos_summary and toto_summary:
        kronos_test_mae = kronos_summary["test"]["price_mae"]
        toto_test_mae = toto_summary["test"]["price_mae"]
        if kronos_test_mae <= toto_test_mae:
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
            metadata={
                "source": "hyperparams_extended",
                "extended_path": selection.get("extended_path"),
                "selection_metric": "test_price_mae",
                "selection_value": selection["test"]["price_mae"],
            },
            config_path=selection["store_path"],
        )
        print(f"[INFO] Selected {selection['model']} as best model for {symbol}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extended hyperparameter exploration for Kronos/Toto"
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
    parser.add_argument(
        "--max-kronos-configs",
        type=int,
        help="Limit number of Kronos configurations to test",
    )
    parser.add_argument(
        "--max-toto-configs",
        type=int,
        help="Limit number of Toto configurations to test",
    )
    parser.add_argument(
        "--search-method",
        choices=["grid", "optuna"],
        default="grid",
        help="Search strategy to explore hyperparameters (default: grid).",
    )
    parser.add_argument(
        "--kronos-trials",
        type=int,
        default=200,
        help="Number of Optuna trials for Kronos when using --search-method=optuna.",
    )
    parser.add_argument(
        "--toto-trials",
        type=int,
        default=150,
        help="Number of Optuna trials for Toto when using --search-method=optuna.",
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

    if args.search_method == "optuna" and (args.max_kronos_configs or args.max_toto_configs):
        print("[WARN] max-* limits are ignored when using Optuna search.")

    for csv_path in csv_files:
        print(f"\n{'='*60}")
        print(f"Evaluating {csv_path.stem}")
        print(f"{'='*60}")
        try:
            _evaluate_symbol(
                csv_path,
                test_kronos=not args.skip_kronos,
                test_toto=not args.skip_toto,
                max_kronos_configs=args.max_kronos_configs,
                max_toto_configs=args.max_toto_configs,
                search_method=args.search_method,
                kronos_trials=args.kronos_trials,
                toto_trials=args.toto_trials,
            )
        except Exception as exc:
            print(f"[ERROR] Failed on {csv_path.stem}: {exc}")


if __name__ == "__main__":
    main()
