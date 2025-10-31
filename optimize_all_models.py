#!/usr/bin/env python3
"""
Comprehensive model optimization testing all approaches:
- Toto with various aggregations
- Kronos standard
- Kronos ensemble with trimmed_mean aggregation

All optimized for pct_return_mae (the metric that matters for trading).
"""
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error

try:
    import optuna
except ImportError:
    optuna = None

from src.models.kronos_wrapper import KronosForecastingWrapper
from src.models.kronos_ensemble import KronosEnsembleWrapper
from src.models.toto_wrapper import TotoPipeline
from src.models.toto_aggregation import aggregate_with_spec

# Configuration
VAL_WINDOW = 20
TEST_WINDOW = 20
MIN_CONTEXT = 128
DATA_DIR = Path("trainingdata")
OUTPUT_DIR = Path("hyperparams_optimized_all")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


@dataclass
class EvalResult:
    """Evaluation result container."""

    price_mae: float
    pct_return_mae: float
    latency_s: float
    config: dict
    model_type: str


class ComprehensiveOptimizer:
    """Optimizer testing all model variants."""

    def __init__(self, symbol: str, verbose: bool = True):
        self.symbol = symbol
        self.verbose = verbose

        # Load data
        data_path = DATA_DIR / f"{symbol}.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Data not found: {data_path}")

        self.df = pd.read_csv(data_path)
        self.df = self.df.sort_values("timestamp").reset_index(drop=True)

        # Split
        self.val_start = len(self.df) - (TEST_WINDOW + VAL_WINDOW)
        self.val_indices = list(range(self.val_start, len(self.df) - TEST_WINDOW))
        self.test_indices = list(range(len(self.df) - TEST_WINDOW, len(self.df)))

        # Model caches
        self.kronos_wrappers: Dict[str, KronosForecastingWrapper] = {}
        self.kronos_ensemble: Optional[KronosEnsembleWrapper] = None
        self.toto_pipeline: Optional[TotoPipeline] = None

    def log(self, msg: str):
        """Log message if verbose."""
        if self.verbose:
            print(msg)

    def get_kronos_wrapper(self, max_context: int, clip: float) -> KronosForecastingWrapper:
        """Get cached Kronos wrapper."""
        key = f"{max_context}_{clip}"
        if key not in self.kronos_wrappers:
            self.kronos_wrappers[key] = KronosForecastingWrapper(
                model_name="NeoQuasar/Kronos-base",
                tokenizer_name="NeoQuasar/Kronos-Tokenizer-base",
                device="cuda:0" if torch.cuda.is_available() else "cpu",
                max_context=max_context,
                clip=clip,
            )
        return self.kronos_wrappers[key]

    def get_kronos_ensemble(self, max_context: int = 192, clip: float = 1.8) -> KronosEnsembleWrapper:
        """Get cached Kronos ensemble wrapper."""
        if self.kronos_ensemble is None:
            self.kronos_ensemble = KronosEnsembleWrapper(
                max_context=max_context, clip=clip
            )
        return self.kronos_ensemble

    def get_toto_pipeline(self) -> TotoPipeline:
        """Get cached Toto pipeline."""
        if self.toto_pipeline is None:
            device_map = "cuda" if torch.cuda.is_available() else "cpu"
            self.toto_pipeline = TotoPipeline.from_pretrained(
                model_id="Datadog/Toto-Open-Base-1.0",
                device_map=device_map,
                compile_model=False,
                torch_compile=False,
            )
        return self.toto_pipeline

    def eval_kronos_standard(self, config: dict, indices: List[int]) -> EvalResult:
        """Evaluate standard Kronos."""
        wrapper = self.get_kronos_wrapper(config["max_context"], config["clip"])

        preds = []
        returns = []
        actual_returns = []
        actual_prices = []
        total_latency = 0.0

        for idx in indices:
            sub_df = self.df.iloc[: idx + 1].copy()
            start_time = time.perf_counter()
            result = wrapper.predict_series(
                data=sub_df,
                timestamp_col="timestamp",
                columns=["close"],
                pred_len=1,
                lookback=config["max_context"],
                temperature=config["temperature"],
                top_p=config["top_p"],
                top_k=config["top_k"],
                sample_count=config.get("sample_count", 160),
            )
            total_latency += time.perf_counter() - start_time

            kronos_close = result.get("close")
            if kronos_close is None or kronos_close.absolute.size == 0:
                continue

            preds.append(float(kronos_close.absolute[0]))
            returns.append(float(kronos_close.percent[0]))
            actual_price = float(self.df["close"].iloc[idx])
            prev_price = float(self.df["close"].iloc[idx - 1])
            actual_prices.append(actual_price)
            actual_returns.append(
                0.0 if prev_price == 0 else (actual_price - prev_price) / prev_price
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return EvalResult(
            price_mae=mean_absolute_error(actual_prices, preds),
            pct_return_mae=mean_absolute_error(actual_returns, returns),
            latency_s=total_latency,
            config=config,
            model_type="kronos_standard",
        )

    def eval_kronos_ensemble(self, config: dict, indices: List[int]) -> EvalResult:
        """Evaluate Kronos with ensemble aggregation."""
        wrapper = self.get_kronos_ensemble(config["max_context"], config["clip"])

        preds = []
        returns = []
        actual_returns = []
        actual_prices = []
        total_latency = 0.0

        for idx in indices:
            sub_df = self.df.iloc[: idx + 1].copy()
            start_time = time.perf_counter()
            result = wrapper.predict_ensemble(
                data=sub_df,
                timestamp_col="timestamp",
                columns=["close"],
                pred_len=1,
                lookback=config["max_context"],
                num_samples=config.get("num_samples", 10),
                base_temperature=config["temperature"],
                temperature_range=config.get("temperature_range", (0.10, 0.25)),
                top_p=config["top_p"],
                top_k=config["top_k"],
                aggregate=config["aggregate"],
            )
            total_latency += time.perf_counter() - start_time

            if "close" not in result:
                continue

            preds.append(float(result["close"]["absolute"]))
            returns.append(float(result["close"]["percent"]))
            actual_price = float(self.df["close"].iloc[idx])
            prev_price = float(self.df["close"].iloc[idx - 1])
            actual_prices.append(actual_price)
            actual_returns.append(
                0.0 if prev_price == 0 else (actual_price - prev_price) / prev_price
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return EvalResult(
            price_mae=mean_absolute_error(actual_prices, preds),
            pct_return_mae=mean_absolute_error(actual_returns, returns),
            latency_s=total_latency,
            config=config,
            model_type="kronos_ensemble",
        )

    def eval_toto(self, config: dict, indices: List[int]) -> EvalResult:
        """Evaluate Toto."""
        pipeline = self.get_toto_pipeline()
        prices = self.df["close"].to_numpy(dtype=np.float64)

        preds = []
        returns = []
        actual_returns = []
        actual_prices = []
        total_latency = 0.0

        for idx in indices:
            context = prices[:idx].astype(np.float32)
            prev_price = prices[idx - 1]

            start_time = time.perf_counter()
            forecasts = pipeline.predict(
                context=context,
                prediction_length=1,
                num_samples=config["num_samples"],
                samples_per_batch=config["samples_per_batch"],
            )
            total_latency += time.perf_counter() - start_time

            if not forecasts:
                continue

            step_values = aggregate_with_spec(forecasts[0].samples, config["aggregate"])
            price_pred = float(np.atleast_1d(step_values)[0])
            preds.append(price_pred)
            pred_return = 0.0 if prev_price == 0 else (price_pred - prev_price) / prev_price
            returns.append(pred_return)

            actual_price = prices[idx]
            actual_prices.append(actual_price)
            actual_returns.append(
                0.0 if prev_price == 0 else (actual_price - prev_price) / prev_price
            )

            del forecasts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return EvalResult(
            price_mae=mean_absolute_error(actual_prices, preds),
            pct_return_mae=mean_absolute_error(actual_returns, returns),
            latency_s=total_latency,
            config=config,
            model_type="toto",
        )

    def optimize_all(self, trials_per_model: int = 30) -> Dict[str, EvalResult]:
        """
        Optimize all model types and return best configs.

        Args:
            trials_per_model: Number of Optuna trials per model type

        Returns:
            Dictionary of best results by model type
        """
        if optuna is None:
            raise RuntimeError("Optuna required. Run: uv pip install optuna")

        self.log(f"\n{'='*70}")
        self.log(f"Optimizing {self.symbol} - {trials_per_model} trials per model")
        self.log(f"{'='*70}")

        results = {}

        # 1. Optimize Toto
        self.log("\n[1/3] Optimizing Toto...")
        results["toto"] = self._optimize_toto(trials_per_model)

        # 2. Optimize Kronos Standard
        self.log("\n[2/3] Optimizing Kronos Standard...")
        results["kronos_standard"] = self._optimize_kronos_standard(trials_per_model)

        # 3. Optimize Kronos Ensemble
        self.log("\n[3/3] Optimizing Kronos Ensemble...")
        results["kronos_ensemble"] = self._optimize_kronos_ensemble(trials_per_model)

        return results

    def _optimize_toto(self, trials: int) -> EvalResult:
        """Optimize Toto model."""
        AGGREGATIONS = [
            "trimmed_mean_5", "trimmed_mean_10", "trimmed_mean_15", "trimmed_mean_20",
            "lower_trimmed_mean_10", "lower_trimmed_mean_15", "lower_trimmed_mean_20",
            "quantile_0.15", "quantile_0.20", "quantile_0.25",
            "mean", "median",
        ]
        SAMPLE_COUNTS = [64, 128, 256, 512, 1024, 2048]

        # Define all possible SPB values
        ALL_SPB_OPTIONS = [16, 32, 64, 128, 256, 512]

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))

        def objective(trial):
            num_samples = trial.suggest_categorical("num_samples", SAMPLE_COUNTS)
            aggregate = trial.suggest_categorical("aggregate", AGGREGATIONS)

            # Use index-based selection for samples_per_batch to avoid dynamic value space
            spb_index = trial.suggest_int("spb_index", 0, 5)

            # Get valid SPB options for this num_samples
            if num_samples <= 128:
                valid_spb = [16, 32]
            elif num_samples <= 512:
                valid_spb = [32, 64, 128]
            else:
                valid_spb = [64, 128, 256, 512]

            # Map index to valid value (wrap around if index too high)
            spb = valid_spb[spb_index % len(valid_spb)]

            config = {"num_samples": num_samples, "aggregate": aggregate, "samples_per_batch": spb}
            result = self.eval_toto(config, self.val_indices)
            self.log(f"  Trial {trial.number}: mae={result.pct_return_mae:.6f}, {aggregate}, n={num_samples}")
            return result.pct_return_mae

        study.optimize(objective, n_trials=trials, show_progress_bar=False)

        # Reconstruct best config with proper SPB
        best_num_samples = study.best_params["num_samples"]
        if best_num_samples <= 128:
            valid_spb = [16, 32]
        elif best_num_samples <= 512:
            valid_spb = [32, 64, 128]
        else:
            valid_spb = [64, 128, 256, 512]

        best_spb = valid_spb[study.best_params["spb_index"] % len(valid_spb)]

        best_config = {
            "num_samples": best_num_samples,
            "aggregate": study.best_params["aggregate"],
            "samples_per_batch": best_spb,
        }
        best_result = self.eval_toto(best_config, self.val_indices)
        self.log(f"  ✓ Best: mae={best_result.pct_return_mae:.6f}")
        return best_result

    def _optimize_kronos_standard(self, trials: int) -> EvalResult:
        """Optimize standard Kronos."""
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=43))

        def objective(trial):
            config = {
                "temperature": trial.suggest_float("temperature", 0.10, 0.30),
                "top_p": trial.suggest_float("top_p", 0.70, 0.90),
                "top_k": trial.suggest_categorical("top_k", [0, 16, 20, 24, 28, 32]),
                "sample_count": trial.suggest_categorical("sample_count", [128, 160, 192, 224, 256]),
                "max_context": trial.suggest_categorical("max_context", [192, 224, 256]),
                "clip": trial.suggest_float("clip", 1.2, 2.5),
            }
            result = self.eval_kronos_standard(config, self.val_indices)
            self.log(f"  Trial {trial.number}: mae={result.pct_return_mae:.6f}, temp={config['temperature']:.3f}")
            return result.pct_return_mae

        study.optimize(objective, n_trials=trials, show_progress_bar=False)

        best_config = {k: study.best_params[k] for k in study.best_params}
        best_result = self.eval_kronos_standard(best_config, self.val_indices)
        self.log(f"  ✓ Best: mae={best_result.pct_return_mae:.6f}")
        return best_result

    def _optimize_kronos_ensemble(self, trials: int) -> EvalResult:
        """Optimize Kronos ensemble."""
        AGGREGATIONS = [
            "trimmed_mean_5", "trimmed_mean_10", "trimmed_mean_15", "trimmed_mean_20",
            "median", "mean",
        ]

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=44))

        def objective(trial):
            config = {
                "temperature": trial.suggest_float("temperature", 0.10, 0.25),
                "temperature_range": (0.10, trial.suggest_float("temp_max", 0.20, 0.35)),
                "top_p": trial.suggest_float("top_p", 0.75, 0.90),
                "top_k": trial.suggest_categorical("top_k", [0, 16, 24, 32]),
                "num_samples": trial.suggest_categorical("num_samples", [5, 8, 10, 12, 15]),
                "max_context": trial.suggest_categorical("max_context", [192, 224, 256]),
                "clip": trial.suggest_float("clip", 1.4, 2.2),
                "aggregate": trial.suggest_categorical("aggregate", AGGREGATIONS),
            }
            result = self.eval_kronos_ensemble(config, self.val_indices)
            self.log(
                f"  Trial {trial.number}: mae={result.pct_return_mae:.6f}, "
                f"agg={config['aggregate']}, n={config['num_samples']}"
            )
            return result.pct_return_mae

        study.optimize(objective, n_trials=trials, show_progress_bar=False)

        best_config = {
            "temperature": study.best_params["temperature"],
            "temperature_range": (0.10, study.best_params["temp_max"]),
            "top_p": study.best_params["top_p"],
            "top_k": study.best_params["top_k"],
            "num_samples": study.best_params["num_samples"],
            "max_context": study.best_params["max_context"],
            "clip": study.best_params["clip"],
            "aggregate": study.best_params["aggregate"],
        }
        best_result = self.eval_kronos_ensemble(best_config, self.val_indices)
        self.log(f"  ✓ Best: mae={best_result.pct_return_mae:.6f}")
        return best_result


def main():
    parser = argparse.ArgumentParser(description="Comprehensive model optimization")
    parser.add_argument("--symbol", type=str, required=True, help="Stock symbol")
    parser.add_argument("--trials", type=int, default=30, help="Trials per model type")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()

    optimizer = ComprehensiveOptimizer(args.symbol)
    results = optimizer.optimize_all(trials_per_model=args.trials)

    # Find overall best
    best_model = min(results.items(), key=lambda x: x[1].pct_return_mae)
    best_name, best_result = best_model

    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")

    for model_name, result in sorted(results.items(), key=lambda x: x[1].pct_return_mae):
        icon = "🏆" if model_name == best_name else "  "
        print(f"{icon} {model_name:20s} pct_return_mae: {result.pct_return_mae:.6f}")

    print(f"\n✅ Best: {best_name}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    for model_name, result in results.items():
        output_path = output_dir / model_name / f"{args.symbol}.json"
        output_path.parent.mkdir(exist_ok=True, parents=True)

        with output_path.open("w") as f:
            json.dump({
                "symbol": args.symbol,
                "model_type": model_name,
                "config": result.config,
                "validation": {
                    "price_mae": result.price_mae,
                    "pct_return_mae": result.pct_return_mae,
                    "latency_s": result.latency_s,
                },
            }, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
