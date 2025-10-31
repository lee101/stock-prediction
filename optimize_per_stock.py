#!/usr/bin/env python3
"""
Per-stock targeted optimization focusing on pct_return_mae.

This script:
1. Loads existing best configs as baselines
2. Performs targeted search around promising hyperparameter regions
3. Uses pct_return_mae as the PRIMARY metric
4. Saves improved configs per stock
"""
import argparse
import json
import time
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

from src.models.kronos_wrapper import KronosForecastingWrapper
from src.models.toto_wrapper import TotoPipeline
from src.models.toto_aggregation import aggregate_with_spec

# Configuration
VAL_WINDOW = 20
TEST_WINDOW = 20
MIN_CONTEXT = 128
DATA_DIR = Path("trainingdata")
OUTPUT_DIR = Path("hyperparams_optimized")
OUTPUT_DIR.mkdir(exist_ok=True)


class StockOptimizer:
    """Optimizer for individual stock pairs."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.kronos_wrappers: Dict[str, KronosForecastingWrapper] = {}
        self.toto_pipeline: Optional[TotoPipeline] = None

        # Load data
        data_path = DATA_DIR / f"{symbol}.csv"
        self.df = pd.read_csv(data_path)
        self.df = self.df.sort_values("timestamp").reset_index(drop=True)

        # Split indices
        self.val_start = len(self.df) - (TEST_WINDOW + VAL_WINDOW)
        self.val_indices = list(range(self.val_start, len(self.df) - TEST_WINDOW))
        self.test_indices = list(range(len(self.df) - TEST_WINDOW, len(self.df)))

    def get_kronos_wrapper(self, max_context: int, clip: float) -> KronosForecastingWrapper:
        """Get or create Kronos wrapper."""
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

    def get_toto_pipeline(self) -> TotoPipeline:
        """Get or create Toto pipeline."""
        if self.toto_pipeline is None:
            device_map = "cuda" if torch.cuda.is_available() else "cpu"
            self.toto_pipeline = TotoPipeline.from_pretrained(
                model_id="Datadog/Toto-Open-Base-1.0",
                device_map=device_map,
                compile_model=False,
                torch_compile=False,
            )
        return self.toto_pipeline

    def evaluate_kronos_config(self, config: dict, indices: List[int]) -> dict:
        """Evaluate a Kronos configuration."""
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
                sample_count=config["sample_count"],
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

        return {
            "price_mae": mean_absolute_error(actual_prices, preds),
            "pct_return_mae": mean_absolute_error(actual_returns, returns),
            "latency_s": total_latency,
        }

    def evaluate_toto_config(self, config: dict, indices: List[int]) -> dict:
        """Evaluate a Toto configuration."""
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

        return {
            "price_mae": mean_absolute_error(actual_prices, preds),
            "pct_return_mae": mean_absolute_error(actual_returns, returns),
            "latency_s": total_latency,
        }

    def optimize_toto_focused(self, trials: int = 100, baseline_config: Optional[dict] = None):
        """Focused Toto optimization using Optuna."""
        if optuna is None:
            raise RuntimeError("Optuna not installed. Run: uv pip install optuna")

        print(f"\n[Toto Optimization] {self.symbol} - {trials} trials")

        # Enhanced aggregation strategies based on what works
        AGGREGATIONS = [
            "trimmed_mean_5", "trimmed_mean_10", "trimmed_mean_15", "trimmed_mean_20",
            "lower_trimmed_mean_10", "lower_trimmed_mean_15", "lower_trimmed_mean_20",
            "quantile_0.15", "quantile_0.18", "quantile_0.20", "quantile_0.25",
            "mean_minus_std_0.3", "mean_minus_std_0.5",
            "quantile_plus_std_0.15_0.15", "quantile_plus_std_0.18_0.15",
            "mean", "median",
        ]

        SAMPLE_COUNTS = [64, 128, 256, 512, 1024, 2048]

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        def objective(trial):
            num_samples = trial.suggest_categorical("num_samples", SAMPLE_COUNTS)
            aggregate = trial.suggest_categorical("aggregate", AGGREGATIONS)

            # Smart samples_per_batch selection
            if num_samples <= 128:
                spb_options = [16, 32]
            elif num_samples <= 512:
                spb_options = [32, 64, 128]
            else:
                spb_options = [64, 128, 256]

            samples_per_batch = trial.suggest_categorical("samples_per_batch", spb_options)

            config = {
                "num_samples": num_samples,
                "aggregate": aggregate,
                "samples_per_batch": samples_per_batch,
            }

            result = self.evaluate_toto_config(config, self.val_indices)

            print(
                f"  Trial {trial.number}: pct_return_mae={result['pct_return_mae']:.6f}, "
                f"agg={aggregate}, samples={num_samples}"
            )

            return result["pct_return_mae"]  # Optimize for return MAE!

        study.optimize(objective, n_trials=trials, show_progress_bar=False)

        best_config = {
            "num_samples": study.best_params["num_samples"],
            "aggregate": study.best_params["aggregate"],
            "samples_per_batch": study.best_params["samples_per_batch"],
        }

        # Evaluate on test set
        val_result = self.evaluate_toto_config(best_config, self.val_indices)
        test_result = self.evaluate_toto_config(best_config, self.test_indices)

        print(f"\n  Best config: {best_config}")
        print(f"  Val pct_return_mae: {val_result['pct_return_mae']:.6f}")
        print(f"  Test pct_return_mae: {test_result['pct_return_mae']:.6f}")

        return best_config, val_result, test_result

    def optimize_kronos_focused(self, trials: int = 100, baseline_config: Optional[dict] = None):
        """Focused Kronos optimization using Optuna."""
        if optuna is None:
            raise RuntimeError("Optuna not installed. Run: uv pip install optuna")

        print(f"\n[Kronos Optimization] {self.symbol} - {trials} trials")

        sampler = optuna.samplers.TPESampler(seed=43)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        def objective(trial):
            temperature = trial.suggest_float("temperature", 0.10, 0.30)
            top_p = trial.suggest_float("top_p", 0.70, 0.90)
            top_k = trial.suggest_categorical("top_k", [0, 16, 20, 24, 28, 32])
            sample_count = trial.suggest_categorical("sample_count", [128, 160, 192, 224, 256, 288, 320])
            max_context = trial.suggest_categorical("max_context", [192, 224, 256])
            clip = trial.suggest_float("clip", 1.2, 2.5)

            config = {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "sample_count": sample_count,
                "max_context": max_context,
                "clip": clip,
            }

            result = self.evaluate_kronos_config(config, self.val_indices)

            print(
                f"  Trial {trial.number}: pct_return_mae={result['pct_return_mae']:.6f}, "
                f"temp={temperature:.3f}, top_p={top_p:.2f}"
            )

            return result["pct_return_mae"]  # Optimize for return MAE!

        study.optimize(objective, n_trials=trials, show_progress_bar=False)

        best_config = {
            "temperature": study.best_params["temperature"],
            "top_p": study.best_params["top_p"],
            "top_k": study.best_params["top_k"],
            "sample_count": study.best_params["sample_count"],
            "max_context": study.best_params["max_context"],
            "clip": study.best_params["clip"],
        }

        # Evaluate on test set
        val_result = self.evaluate_kronos_config(best_config, self.val_indices)
        test_result = self.evaluate_kronos_config(best_config, self.test_indices)

        print(f"\n  Best config: {best_config}")
        print(f"  Val pct_return_mae: {val_result['pct_return_mae']:.6f}")
        print(f"  Test pct_return_mae: {test_result['pct_return_mae']:.6f}")

        return best_config, val_result, test_result


def main():
    parser = argparse.ArgumentParser(description="Optimize per-stock configurations")
    parser.add_argument("--symbol", type=str, required=True, help="Symbol to optimize")
    parser.add_argument("--model", choices=["toto", "kronos", "both"], default="both", help="Model to optimize")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials per model")
    args = parser.parse_args()

    print(f"Optimizing {args.symbol} with {args.trials} trials per model")
    print(f"Optimization metric: pct_return_mae ⭐")

    optimizer = StockOptimizer(args.symbol)

    results = {}

    if args.model in ["toto", "both"]:
        try:
            toto_config, toto_val, toto_test = optimizer.optimize_toto_focused(trials=args.trials)
            results["toto"] = {
                "config": toto_config,
                "validation": toto_val,
                "test": toto_test,
            }

            # Save
            output_path = OUTPUT_DIR / "toto" / f"{args.symbol}.json"
            output_path.parent.mkdir(exist_ok=True)
            with output_path.open("w") as f:
                json.dump({
                    "symbol": args.symbol,
                    "model": "toto",
                    "config": toto_config,
                    "validation": toto_val,
                    "test": toto_test,
                }, f, indent=2)
            print(f"\nSaved Toto config to: {output_path}")
        except Exception as e:
            print(f"[ERROR] Toto optimization failed: {e}")

    if args.model in ["kronos", "both"]:
        try:
            kronos_config, kronos_val, kronos_test = optimizer.optimize_kronos_focused(trials=args.trials)
            results["kronos"] = {
                "config": kronos_config,
                "validation": kronos_val,
                "test": kronos_test,
            }

            # Save
            output_path = OUTPUT_DIR / "kronos" / f"{args.symbol}.json"
            output_path.parent.mkdir(exist_ok=True)
            with output_path.open("w") as f:
                json.dump({
                    "symbol": args.symbol,
                    "model": "kronos",
                    "config": kronos_config,
                    "validation": kronos_val,
                    "test": kronos_test,
                }, f, indent=2)
            print(f"\nSaved Kronos config to: {output_path}")
        except Exception as e:
            print(f"[ERROR] Kronos optimization failed: {e}")

    # Compare and recommend
    if len(results) == 2:
        toto_mae = results["toto"]["validation"]["pct_return_mae"]
        kronos_mae = results["kronos"]["validation"]["pct_return_mae"]

        print(f"\n{'='*60}")
        print("RECOMMENDATION")
        print(f"{'='*60}")
        print(f"Toto pct_return_mae:   {toto_mae:.6f}")
        print(f"Kronos pct_return_mae: {kronos_mae:.6f}")

        if toto_mae < kronos_mae:
            improvement = ((kronos_mae - toto_mae) / kronos_mae * 100)
            print(f"\n✅ Use Toto ({improvement:.2f}% better)")
        else:
            improvement = ((toto_mae - kronos_mae) / toto_mae * 100)
            print(f"\n✅ Use Kronos ({improvement:.2f}% better)")


if __name__ == "__main__":
    main()
