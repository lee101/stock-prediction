#!/usr/bin/env python3
"""
Targeted comparison and optimization for Toto vs Kronos models.

This script:
1. Loads existing best configs for both models
2. Evaluates them side-by-side on the same validation data
3. Uses pct_return_mae as primary metric (not price_mae)
4. Tests ensemble/hybrid approaches
5. Generates actionable recommendations per stock pair
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

from src.models.kronos_wrapper import KronosForecastingWrapper
from src.models.toto_wrapper import TotoPipeline
from src.models.toto_aggregation import aggregate_with_spec

# Configuration
VAL_WINDOW = 20
DATA_DIR = Path("trainingdata")


class ModelEvaluator:
    """Evaluator for comparing Toto and Kronos models."""

    def __init__(self):
        self.kronos_wrappers: Dict[str, KronosForecastingWrapper] = {}
        self.toto_pipeline: Optional[TotoPipeline] = None

    def get_kronos_wrapper(self, max_context: int, clip: float) -> KronosForecastingWrapper:
        """Get or create Kronos wrapper with caching."""
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
        """Get or create Toto pipeline (singleton)."""
        if self.toto_pipeline is None:
            device_map = "cuda" if torch.cuda.is_available() else "cpu"
            self.toto_pipeline = TotoPipeline.from_pretrained(
                model_id="Datadog/Toto-Open-Base-1.0",
                device_map=device_map,
                compile_model=False,
                torch_compile=False,
            )
        return self.toto_pipeline

    def evaluate_kronos(
        self,
        df: pd.DataFrame,
        config: dict,
        val_indices: range,
    ) -> dict:
        """Evaluate Kronos configuration."""
        wrapper = self.get_kronos_wrapper(config["max_context"], config["clip"])

        preds = []
        returns = []
        actual_returns = []
        actual_prices = []
        total_latency = 0.0

        for idx in val_indices:
            sub_df = df.iloc[: idx + 1].copy()
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
            actual_price = float(df["close"].iloc[idx])
            prev_price = float(df["close"].iloc[idx - 1])
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
            "predictions": preds,
            "predicted_returns": returns,
            "actual_returns": actual_returns,
        }

    def evaluate_toto(
        self,
        df: pd.DataFrame,
        config: dict,
        val_indices: range,
    ) -> dict:
        """Evaluate Toto configuration."""
        pipeline = self.get_toto_pipeline()
        prices = df["close"].to_numpy(dtype=np.float64)

        preds = []
        returns = []
        actual_returns = []
        actual_prices = []
        total_latency = 0.0

        for idx in val_indices:
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
            "predictions": preds,
            "predicted_returns": returns,
            "actual_returns": actual_returns,
        }

    def evaluate_ensemble(
        self,
        kronos_result: dict,
        toto_result: dict,
        weight_kronos: float = 0.5,
    ) -> dict:
        """Evaluate ensemble of Kronos and Toto predictions."""
        # Ensure same length
        min_len = min(len(kronos_result["predicted_returns"]), len(toto_result["predicted_returns"]))

        ensemble_returns = []
        for i in range(min_len):
            ensemble_return = (
                weight_kronos * kronos_result["predicted_returns"][i]
                + (1 - weight_kronos) * toto_result["predicted_returns"][i]
            )
            ensemble_returns.append(ensemble_return)

        actual_returns = kronos_result["actual_returns"][:min_len]
        pct_return_mae = mean_absolute_error(actual_returns, ensemble_returns)

        return {
            "pct_return_mae": pct_return_mae,
            "ensemble_returns": ensemble_returns,
            "weight_kronos": weight_kronos,
        }


def load_config(config_path: Path) -> Optional[dict]:
    """Load configuration from JSON file."""
    if not config_path.exists():
        return None
    with config_path.open("r") as f:
        return json.load(f)


def compare_symbol(
    symbol: str,
    evaluator: ModelEvaluator,
    test_ensemble: bool = True,
) -> dict:
    """Compare Kronos vs Toto for a single symbol."""
    print(f"\n{'='*60}")
    print(f"Comparing models for {symbol}")
    print(f"{'='*60}")

    # Load data
    data_path = DATA_DIR / f"{symbol}.csv"
    if not data_path.exists():
        print(f"[WARN] Data not found for {symbol}")
        return {}

    df = pd.read_csv(data_path)
    df = df.sort_values("timestamp").reset_index(drop=True)

    if len(df) < VAL_WINDOW + 128:
        print(f"[WARN] Not enough data for {symbol}")
        return {}

    val_start = len(df) - VAL_WINDOW
    val_indices = range(val_start, len(df))

    # Load configurations
    kronos_config_path = Path("hyperparams_extended") / "kronos" / f"{symbol}.json"
    toto_config_path = Path("hyperparams_extended") / "toto" / f"{symbol}.json"

    kronos_data = load_config(kronos_config_path)
    toto_data = load_config(toto_config_path)

    results = {"symbol": symbol}

    # Evaluate Kronos
    if kronos_data:
        print(f"\n[Kronos] Config: {kronos_data['config']['name']}")
        try:
            kronos_result = evaluator.evaluate_kronos(df, kronos_data["config"], val_indices)
            results["kronos"] = kronos_result
            print(f"  price_mae: {kronos_result['price_mae']:.4f}")
            print(f"  pct_return_mae: {kronos_result['pct_return_mae']:.6f} ⭐")
            print(f"  latency: {kronos_result['latency_s']:.2f}s")
        except Exception as e:
            print(f"[ERROR] Kronos evaluation failed: {e}")

    # Evaluate Toto
    if toto_data:
        print(f"\n[Toto] Config: {toto_data['config']['name']}")
        try:
            toto_result = evaluator.evaluate_toto(df, toto_data["config"], val_indices)
            results["toto"] = toto_result
            print(f"  price_mae: {toto_result['price_mae']:.4f}")
            print(f"  pct_return_mae: {toto_result['pct_return_mae']:.6f} ⭐")
            print(f"  latency: {toto_result['latency_s']:.2f}s")
        except Exception as e:
            print(f"[ERROR] Toto evaluation failed: {e}")

    # Test ensemble
    if test_ensemble and "kronos" in results and "toto" in results:
        print(f"\n[Ensemble] Testing weighted combinations...")
        best_ensemble = None
        best_mae = float("inf")

        for weight in [0.3, 0.5, 0.7]:
            ensemble_result = evaluator.evaluate_ensemble(
                results["kronos"], results["toto"], weight
            )
            print(f"  weight_kronos={weight:.1f}: pct_return_mae={ensemble_result['pct_return_mae']:.6f}")

            if ensemble_result["pct_return_mae"] < best_mae:
                best_mae = ensemble_result["pct_return_mae"]
                best_ensemble = ensemble_result

        results["ensemble"] = best_ensemble
        print(f"  Best: weight={best_ensemble['weight_kronos']:.1f}, mae={best_mae:.6f} ⭐")

    # Recommendation
    print(f"\n[Recommendation]")
    models = []
    if "kronos" in results:
        models.append(("Kronos", results["kronos"]["pct_return_mae"]))
    if "toto" in results:
        models.append(("Toto", results["toto"]["pct_return_mae"]))
    if "ensemble" in results:
        models.append((f"Ensemble({results['ensemble']['weight_kronos']:.1f})", results["ensemble"]["pct_return_mae"]))

    if models:
        best_model = min(models, key=lambda x: x[1])
        results["recommendation"] = best_model[0]
        results["best_pct_return_mae"] = best_model[1]

        print(f"  Best model: {best_model[0]} (pct_return_mae: {best_model[1]:.6f})")

        # Show improvement over current selection
        current_best = load_config(Path("hyperparams/best") / f"{symbol}.json")
        if current_best:
            current_val_mae = current_best.get("validation", {}).get("pct_return_mae", 0)
            improvement = ((current_val_mae - best_model[1]) / current_val_mae * 100) if current_val_mae else 0
            print(f"  Current best: {current_best['model']} (pct_return_mae: {current_val_mae:.6f})")
            if improvement > 0:
                print(f"  Improvement: {improvement:.2f}% better! ✅")
            else:
                print(f"  Change: {improvement:.2f}% (regression ❌)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare and optimize Toto vs Kronos")
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="Symbols to evaluate (default: AAPL NVDA SPY)",
    )
    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Skip ensemble evaluation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comparison_results.json",
        help="Output file for results",
    )
    args = parser.parse_args()

    symbols = args.symbols or ["AAPL", "NVDA", "SPY", "AMD", "META", "TSLA"]

    print(f"Comparing models for symbols: {', '.join(symbols)}")
    print(f"Primary metric: pct_return_mae ⭐")

    evaluator = ModelEvaluator()
    all_results = []

    for symbol in symbols:
        try:
            result = compare_symbol(symbol, evaluator, test_ensemble=not args.no_ensemble)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"[ERROR] Failed on {symbol}: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    recommendations = {}
    for result in all_results:
        if "recommendation" in result:
            rec = result["recommendation"]
            recommendations[rec] = recommendations.get(rec, 0) + 1
            print(f"{result['symbol']:10s} -> {rec:15s} (mae: {result['best_pct_return_mae']:.6f})")

    print(f"\nModel Selection Counts:")
    for model, count in sorted(recommendations.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model:15s}: {count}")

    # Save results
    output_path = Path(args.output)
    with output_path.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
