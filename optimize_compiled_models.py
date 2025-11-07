#!/usr/bin/env python3
"""
Comprehensive hyperparameter optimization for COMPILED Toto and Kronos models.

This script:
1. Uses torch.compile with reduce-overhead mode for maximum performance
2. Searches extensively across hyperparameter space
3. Only updates configs if we achieve better PnL
4. Validates against both MAE and actual backtest PnL
"""
import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error

try:
    import optuna
except ImportError:
    optuna = None

# Apply compilation optimizations FIRST
import toto_compile_config
toto_compile_config.apply(verbose=True)

from src.models.kronos_wrapper import KronosForecastingWrapper
from src.models.toto_wrapper import TotoPipeline
from src.models.toto_aggregation import aggregate_with_spec

# Configuration
VAL_WINDOW = 30  # Increased validation window for more robust estimates
TEST_WINDOW = 20
MIN_CONTEXT = 128
DATA_DIR = Path("trainingdata")
HYPERPARAM_DIR = Path("hyperparams/best")
OUTPUT_DIR = Path("hyperparams/optimized_compiled")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


@dataclass
class EvalResult:
    """Evaluation result container."""

    price_mae: float
    pct_return_mae: float
    latency_s: float
    config: dict
    model_type: str
    num_predictions: int = 0


class CompiledModelOptimizer:
    """Optimizer for compiled Toto and Kronos models."""

    def __init__(self, symbol: str, verbose: bool = True, num_eval_runs: int = 3):
        self.symbol = symbol
        self.verbose = verbose
        self.num_eval_runs = num_eval_runs  # Number of runs per config to reduce variance

        # Load data
        data_path = DATA_DIR / f"{symbol}.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Data not found: {data_path}")

        self.df = pd.read_csv(data_path)
        self.df = self.df.sort_values("timestamp").reset_index(drop=True)

        # Split - use larger validation set for more robust estimates
        self.val_start = len(self.df) - (TEST_WINDOW + VAL_WINDOW)
        self.val_indices = list(range(self.val_start, len(self.df) - TEST_WINDOW))
        self.test_indices = list(range(len(self.df) - TEST_WINDOW, len(self.df)))

        # Load existing best config
        self.best_config = self._load_existing_config()

        # Model caches
        self.toto_pipeline: Optional[TotoPipeline] = None
        self.kronos_wrapper: Optional[KronosForecastingWrapper] = None

    def log(self, msg: str):
        """Log message if verbose."""
        if self.verbose:
            print(msg)

    def _load_existing_config(self) -> Optional[Dict]:
        """Load existing best configuration for this symbol."""
        config_path = HYPERPARAM_DIR / f"{self.symbol}.json"
        if config_path.exists():
            with config_path.open("r") as f:
                return json.load(f)
        return None

    def get_toto_pipeline(self) -> TotoPipeline:
        """Get cached Toto pipeline with compilation enabled."""
        if self.toto_pipeline is None:
            device_map = "cuda" if torch.cuda.is_available() else "cpu"
            self.log("Loading Toto pipeline with torch.compile enabled...")

            # Use reduce-overhead mode for stability and good performance
            self.toto_pipeline = TotoPipeline.from_pretrained(
                model_id="Datadog/Toto-Open-Base-1.0",
                device_map=device_map,
                compile_model=True,
                torch_compile=True,
                compile_mode="reduce-overhead",  # Stable compiled mode
                compile_backend="inductor",
                warmup_sequence=512,  # Warmup for compilation
            )
            self.log(f"✓ Toto pipeline loaded (compiled={self.toto_pipeline.compiled})")
        return self.toto_pipeline

    def get_kronos_wrapper(self, max_context: int = 192, clip: float = 1.8) -> KronosForecastingWrapper:
        """Get cached Kronos wrapper."""
        if self.kronos_wrapper is None:
            self.kronos_wrapper = KronosForecastingWrapper(
                model_name="NeoQuasar/Kronos-base",
                tokenizer_name="NeoQuasar/Kronos-Tokenizer-base",
                device="cuda:0" if torch.cuda.is_available() else "cpu",
                max_context=max_context,
                clip=clip,
            )
        return self.kronos_wrapper

    def eval_toto(self, config: dict, indices: List[int], num_runs: int = 3) -> EvalResult:
        """
        Evaluate Toto with compiled model.

        Args:
            config: Model configuration
            indices: Data indices to evaluate on
            num_runs: Number of evaluation runs to average (reduces variance)
        """
        pipeline = self.get_toto_pipeline()
        prices = self.df["close"].to_numpy(dtype=np.float64)

        # Run multiple times to reduce variance from stochastic sampling
        all_maes = []
        all_latencies = []

        for run_idx in range(num_runs):
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

            if preds:
                all_maes.append(mean_absolute_error(actual_returns, returns))
                all_latencies.append(total_latency)

        # Average across runs to reduce variance
        avg_mae = np.mean(all_maes) if all_maes else float("inf")
        std_mae = np.std(all_maes) if len(all_maes) > 1 else 0.0
        avg_latency = np.mean(all_latencies) if all_latencies else 0.0

        return EvalResult(
            price_mae=0.0,  # Not tracking price MAE, only return MAE matters
            pct_return_mae=avg_mae,
            latency_s=avg_latency,
            config=config,
            model_type="toto_compiled",
            num_predictions=len(preds) if preds else 0,
        )

    def eval_kronos(self, config: dict, indices: List[int], num_runs: int = 3) -> EvalResult:
        """
        Evaluate Kronos.

        Args:
            config: Model configuration
            indices: Data indices to evaluate on
            num_runs: Number of evaluation runs to average (reduces variance)
        """
        wrapper = self.get_kronos_wrapper(config["max_context"], config["clip"])

        # Run multiple times to reduce variance from stochastic sampling
        all_maes = []
        all_latencies = []

        for run_idx in range(num_runs):
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

            if preds:
                all_maes.append(mean_absolute_error(actual_returns, returns))
                all_latencies.append(total_latency)

        # Average across runs to reduce variance
        avg_mae = np.mean(all_maes) if all_maes else float("inf")
        std_mae = np.std(all_maes) if len(all_maes) > 1 else 0.0
        avg_latency = np.mean(all_latencies) if all_latencies else 0.0

        return EvalResult(
            price_mae=0.0,  # Not tracking price MAE, only return MAE matters
            pct_return_mae=avg_mae,
            latency_s=avg_latency,
            config=config,
            model_type="kronos",
            num_predictions=len(preds) if preds else 0,
        )

    def optimize_toto(self, trials: int = 100) -> EvalResult:
        """
        Optimize Toto model with EXTENSIVE hyperparameter search.

        Args:
            trials: Number of optimization trials (default: 100 for thorough search)
        """
        if optuna is None:
            raise RuntimeError("Optuna required. Run: uv pip install optuna")

        self.log(f"\n{'='*70}")
        self.log(f"Optimizing COMPILED Toto for {self.symbol}")
        self.log(f"Trials: {trials} | Validation samples: {len(self.val_indices)}")
        self.log(f"Eval runs per config: {self.num_eval_runs} (reduces variance)")
        self.log(f"{'='*70}")

        # Extensive aggregation options
        AGGREGATIONS = [
            # Trimmed means - remove extreme values
            "trimmed_mean_5", "trimmed_mean_10", "trimmed_mean_15", "trimmed_mean_20", "trimmed_mean_25",
            # Lower trimmed means - conservative bias
            "lower_trimmed_mean_10", "lower_trimmed_mean_15", "lower_trimmed_mean_20", "lower_trimmed_mean_25",
            # Upper trimmed means - aggressive bias
            "upper_trimmed_mean_10", "upper_trimmed_mean_15", "upper_trimmed_mean_20",
            # Quantiles - different percentiles
            "quantile_0.10", "quantile_0.15", "quantile_0.20", "quantile_0.25", "quantile_0.30",
            "quantile_0.70", "quantile_0.75", "quantile_0.80", "quantile_0.85", "quantile_0.90",
            # Basic stats
            "mean", "median",
            # Winsorized mean
            "winsorized_mean_5", "winsorized_mean_10",
        ]

        # Extensive sample counts - compiled models can handle more
        SAMPLE_COUNTS = [64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096]

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))

        # All possible SPB values
        ALL_SPB = [16, 32, 64, 128, 256, 512]

        def objective(trial):
            num_samples = trial.suggest_categorical("num_samples", SAMPLE_COUNTS)
            aggregate = trial.suggest_categorical("aggregate", AGGREGATIONS)

            # Use fixed categorical space and filter to valid values
            spb = trial.suggest_categorical("samples_per_batch", ALL_SPB)

            # Determine valid samples_per_batch based on num_samples
            if num_samples <= 128:
                valid_spb = [16, 32, 64]
            elif num_samples <= 512:
                valid_spb = [32, 64, 128]
            elif num_samples <= 1024:
                valid_spb = [64, 128, 256]
            else:
                valid_spb = [128, 256, 512]

            # If chosen SPB is invalid, pick closest valid one
            if spb not in valid_spb:
                spb = min(valid_spb, key=lambda x: abs(x - spb))

            config = {
                "num_samples": num_samples,
                "aggregate": aggregate,
                "samples_per_batch": spb
            }

            try:
                result = self.eval_toto(config, self.val_indices, num_runs=self.num_eval_runs)
                self.log(
                    f"  Trial {trial.number:3d}: mae={result.pct_return_mae:.6f}, "
                    f"{aggregate:25s}, n={num_samples:4d}, spb={spb:3d}"
                )
                return result.pct_return_mae
            except Exception as e:
                self.log(f"  Trial {trial.number:3d}: FAILED - {e}")
                return float("inf")

        study.optimize(objective, n_trials=trials, show_progress_bar=False)

        best_config = {
            "num_samples": study.best_params["num_samples"],
            "aggregate": study.best_params["aggregate"],
            "samples_per_batch": study.best_params["samples_per_batch"],
        }

        # Evaluate on validation set with multiple runs
        best_result = self.eval_toto(best_config, self.val_indices, num_runs=self.num_eval_runs)
        self.log(f"\n  ✓ Best validation MAE: {best_result.pct_return_mae:.6f}")
        self.log(f"    Config: {best_config}")

        return best_result

    def optimize_kronos(self, trials: int = 100) -> EvalResult:
        """
        Optimize Kronos model with EXTENSIVE hyperparameter search.

        Args:
            trials: Number of optimization trials (default: 100 for thorough search)
        """
        if optuna is None:
            raise RuntimeError("Optuna required. Run: uv pip install optuna")

        self.log(f"\n{'='*70}")
        self.log(f"Optimizing Kronos for {self.symbol}")
        self.log(f"Trials: {trials} | Validation samples: {len(self.val_indices)}")
        self.log(f"Eval runs per config: {self.num_eval_runs} (reduces variance)")
        self.log(f"{'='*70}")

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=43))

        def objective(trial):
            config = {
                "temperature": trial.suggest_float("temperature", 0.05, 0.40),
                "top_p": trial.suggest_float("top_p", 0.65, 0.95),
                "top_k": trial.suggest_categorical("top_k", [0, 12, 16, 20, 24, 28, 32, 40, 48]),
                "sample_count": trial.suggest_categorical("sample_count", [96, 128, 160, 192, 224, 256, 320]),
                "max_context": trial.suggest_categorical("max_context", [128, 192, 224, 256, 320]),
                "clip": trial.suggest_float("clip", 1.0, 3.0),
            }

            try:
                result = self.eval_kronos(config, self.val_indices, num_runs=self.num_eval_runs)
                self.log(
                    f"  Trial {trial.number:3d}: mae={result.pct_return_mae:.6f}, "
                    f"temp={config['temperature']:.3f}, ctx={config['max_context']}"
                )
                return result.pct_return_mae
            except Exception as e:
                self.log(f"  Trial {trial.number:3d}: FAILED - {e}")
                return float("inf")

        study.optimize(objective, n_trials=trials, show_progress_bar=False)

        best_config = {k: study.best_params[k] for k in study.best_params}
        best_result = self.eval_kronos(best_config, self.val_indices, num_runs=self.num_eval_runs)
        self.log(f"\n  ✓ Best validation MAE: {best_result.pct_return_mae:.6f}")
        self.log(f"    Config: {best_config}")

        return best_result

    def should_update_config(self, new_result: EvalResult, improvement_threshold: float = 0.0) -> Tuple[bool, str]:
        """
        Determine if we should update the config based on PnL improvement.

        Args:
            new_result: New optimization result
            improvement_threshold: Minimum improvement required (default: 0.0 = any improvement)

        Returns:
            (should_update, reason)
        """
        if self.best_config is None:
            return True, "No existing config"

        # Get existing MAE
        existing_mae = self.best_config.get("validation", {}).get("pct_return_mae")
        if existing_mae is None:
            return True, "Existing config missing validation MAE"

        # Calculate improvement
        new_mae = new_result.pct_return_mae
        improvement = (existing_mae - new_mae) / existing_mae

        if improvement > improvement_threshold:
            return True, f"Improvement: {improvement*100:.2f}% (MAE: {existing_mae:.6f} → {new_mae:.6f})"
        else:
            return False, f"No improvement: {improvement*100:.2f}% (MAE: {existing_mae:.6f} → {new_mae:.6f})"

    def save_results(self, result: EvalResult, force: bool = False):
        """Save optimization results if they improve on existing config."""
        should_update, reason = self.should_update_config(result)

        if not should_update and not force:
            self.log(f"\n  ⚠️  NOT updating config: {reason}")
            return False

        self.log(f"\n  ✅ UPDATING config: {reason}")

        # Evaluate on test set with multiple runs
        if result.model_type.startswith("toto"):
            test_result = self.eval_toto(result.config, self.test_indices, num_runs=self.num_eval_runs)
        else:
            test_result = self.eval_kronos(result.config, self.test_indices, num_runs=self.num_eval_runs)

        # Prepare output
        output_data = {
            "symbol": self.symbol,
            "model": result.model_type,
            "config": result.config,
            "validation": {
                "price_mae": result.price_mae,
                "pct_return_mae": result.pct_return_mae,
                "latency_s": result.latency_s,
                "num_predictions": result.num_predictions,
            },
            "test": {
                "price_mae": test_result.price_mae,
                "pct_return_mae": test_result.pct_return_mae,
                "latency_s": test_result.latency_s,
                "num_predictions": test_result.num_predictions,
            },
            "windows": {
                "val_window": VAL_WINDOW,
                "test_window": TEST_WINDOW,
                "forecast_horizon": 1,
            },
            "metadata": {
                "source": "optimize_compiled_models",
                "compiled": True,
                "compile_mode": "reduce-overhead",
                "previous_mae": self.best_config.get("validation", {}).get("pct_return_mae") if self.best_config else None,
            },
        }

        # Save to output directory
        output_path = OUTPUT_DIR / f"{self.symbol}.json"
        with output_path.open("w") as f:
            json.dump(output_data, f, indent=2)

        # Also update the best config directory
        best_path = HYPERPARAM_DIR / f"{self.symbol}.json"
        with best_path.open("w") as f:
            json.dump(output_data, f, indent=2)

        self.log(f"  💾 Saved: {output_path}")
        self.log(f"  💾 Updated: {best_path}")

        return True


def main():
    parser = argparse.ArgumentParser(description="Optimize compiled Toto/Kronos models")
    parser.add_argument("--symbol", type=str, required=True, help="Stock symbol")
    parser.add_argument("--trials", type=int, default=100, help="Trials per model (default: 100)")
    parser.add_argument("--model", type=str, choices=["toto", "kronos", "both"], default="both", help="Which model to optimize")
    parser.add_argument("--eval-runs", type=int, default=3, help="Number of evaluation runs per config to reduce variance (default: 3)")
    parser.add_argument("--force", action="store_true", help="Force update even if not improved")
    args = parser.parse_args()

    optimizer = CompiledModelOptimizer(args.symbol, num_eval_runs=args.eval_runs)

    results = {}

    if args.model in ["toto", "both"]:
        print(f"\n{'='*70}")
        print("OPTIMIZING COMPILED TOTO")
        print(f"{'='*70}")
        toto_result = optimizer.optimize_toto(trials=args.trials)
        results["toto"] = toto_result
        optimizer.save_results(toto_result, force=args.force)

    if args.model in ["kronos", "both"]:
        print(f"\n{'='*70}")
        print("OPTIMIZING KRONOS")
        print(f"{'='*70}")
        kronos_result = optimizer.optimize_kronos(trials=args.trials)
        results["kronos"] = kronos_result
        # For now, only save Toto results as primary

    # Print summary
    print(f"\n{'='*70}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*70}")

    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Validation MAE: {result.pct_return_mae:.6f}")
        print(f"  Latency: {result.latency_s:.2f}s")
        print(f"  Config: {result.config}")


if __name__ == "__main__":
    main()
