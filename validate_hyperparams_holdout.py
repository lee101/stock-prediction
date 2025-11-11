#!/usr/bin/env python3
"""
Validate hyperparameters on 7-day holdout test data for retrained models.
Tests inference-time hyperparameters for optimal MAE on recent data.
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Import model wrappers
sys.path.insert(0, str(Path(__file__).parent))
from src.models.kronos_wrapper import KronosForecastingWrapper
from src.models.toto_wrapper import TotoPipeline


class HyperparamValidator:
    """Validates hyperparameters on recent holdout data."""

    def __init__(
        self,
        holdout_days: int = 7,
        prediction_length: int = 64
    ):
        self.holdout_days = holdout_days
        self.prediction_length = prediction_length
        self.results = {}

    def load_stock_data(self, stock: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and split stock data into train/holdout sets."""
        data_path = Path(f"trainingdata/{stock}.csv")

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        df = pd.read_csv(data_path)

        # Ensure we have a timestamp column
        if "timestamp" not in df.columns and "date" in df.columns:
            df["timestamp"] = pd.to_datetime(df["date"])
        elif "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        else:
            # Assume rows are chronological
            df["timestamp"] = pd.date_range(
                end=datetime.now(),
                periods=len(df),
                freq="1D"
            )

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Split: last N days for holdout, rest for context
        holdout_start_idx = len(df) - self.holdout_days
        train_df = df.iloc[:holdout_start_idx].copy()
        holdout_df = df.iloc[holdout_start_idx:].copy()

        return train_df, holdout_df

    def test_kronos_hyperparams(
        self,
        stock: str,
        model_path: Optional[str] = None
    ) -> Dict:
        """Test various Kronos hyperparameters on holdout data."""
        print(f"\nTesting Kronos hyperparameters for {stock}...")

        train_df, holdout_df = self.load_stock_data(stock)

        # Hyperparameter grid
        param_grid = {
            "num_samples": [5, 10, 20, 50],
            "temperature": [0.5, 0.7, 1.0, 1.2],
            "top_k": [20, 50, None],
            "top_p": [0.8, 0.9, 0.95, None],
        }

        best_mae = float("inf")
        best_params = {}
        results = []

        # Use model from kronostraining if available (future enhancement)
        if model_path is None:
            adapter_path = Path(
                f"kronostraining/artifacts/stock_specific/{stock}/adapters/{stock}/adapter.pt"
            )
            if adapter_path.exists():
                model_path = str(adapter_path)

        if model_path is not None:
            print(
                "  Note: custom Kronos model paths are not yet supported by KronosForecastingWrapper; "
                "falling back to base weights."
            )

        try:
            # Initialize model
            device = "cuda:0" if self._has_cuda() else "cpu"
            if not device.startswith("cuda"):
                raise RuntimeError("KronosForecastingWrapper currently requires a CUDA-capable device.")

            wrapper = KronosForecastingWrapper(
                model_name="NeoQuasar/Kronos-small",
                tokenizer_name="NeoQuasar/Kronos-Tokenizer-base",
                device=device,
            )

            # Test combinations (sample subset for speed)
            import itertools
            param_combinations = list(itertools.product(
                param_grid["num_samples"][:2],  # Test 2 values
                param_grid["temperature"][:3],  # Test 3 values
                param_grid["top_k"][:2],  # Test 2 values
                param_grid["top_p"][:2],  # Test 2 values
            ))

            for num_samples, temp, top_k, top_p in tqdm(
                param_combinations[:20],  # Limit to 20 combinations
                desc=f"{stock} Kronos"
            ):
                try:
                    # Prepare contextual window
                    context_length = min(512, len(train_df))
                    context_df = train_df.tail(context_length).copy()
                    timestamp_col = "timestamp" if "timestamp" in context_df.columns else "timestamps"
                    if timestamp_col not in context_df.columns:
                        context_df[timestamp_col] = pd.date_range(
                            end=pd.Timestamp.utcnow(), periods=len(context_df), freq="1D"
                        )

                    forecast = wrapper.predict_series(
                        data=context_df,
                        timestamp_col=timestamp_col,
                        columns=["close"],
                        pred_len=self.prediction_length,
                        lookback=context_length,
                        temperature=temp,
                        top_k=top_k if top_k is not None else None,
                        top_p=top_p if top_p is not None else None,
                        sample_count=num_samples,
                    )

                    kronos_close = forecast["close"].absolute
                    holdout_actual = holdout_df["close"].to_numpy(dtype=np.float64)
                    pred_length = min(len(kronos_close), len(holdout_actual))
                    if pred_length == 0:
                        raise ValueError("Kronos forecast returned no samples")

                    mae = np.mean(np.abs(kronos_close[:pred_length] - holdout_actual[:pred_length]))
                    denom = np.mean(np.abs(holdout_actual[:pred_length])) or 1.0
                    mae_pct = (mae / denom) * 100

                    result = {
                        "num_samples": num_samples,
                        "temperature": temp,
                        "top_k": top_k,
                        "top_p": top_p,
                        "mae": float(mae),
                        "mae_pct": float(mae_pct),
                    }
                    results.append(result)

                    if mae < best_mae:
                        best_mae = mae
                        best_params = result.copy()

                except Exception as e:
                    print(f"  Error with params {num_samples}/{temp}/{top_k}/{top_p}: {e}")
                    continue

        except Exception as e:
            print(f"  Failed to initialize Kronos for {stock}: {e}")
            return {
                "stock": stock,
                "model": "kronos",
                "status": "failed",
                "error": str(e)
            }

        return {
            "stock": stock,
            "model": "kronos",
            "status": "success",
            "best_params": best_params,
            "best_mae": float(best_mae),
            "all_results": results[:10],  # Top 10
        }

    def test_toto_hyperparams(
        self,
        stock: str,
        model_path: Optional[str] = None
    ) -> Dict:
        """Test various Toto hyperparameters on holdout data."""
        print(f"\nTesting Toto hyperparameters for {stock}...")

        train_df, holdout_df = self.load_stock_data(stock)

        # Hyperparameter grid (Toto has different params)
        param_grid = {
            "temperature": [0.7, 1.0, 1.3],
            "num_samples": [256, 512],
        }

        best_mae = float("inf")
        best_params = {}
        results = []

        # Use model from tototraining if available
        if model_path is None:
            model_dir = Path(f"tototraining/stock_models/{stock}/{stock}_model")
            if model_dir.exists():
                model_path = str(model_dir)

        try:
            # Initialize model
            device = "cuda" if self._has_cuda() else "cpu"
            pipeline = TotoPipeline.from_pretrained(
                model_path or "Datadog/Toto-Open-Base-1.0",
                device_map=device,
            )

            # Test combinations
            import itertools
            param_combinations = list(itertools.product(
                param_grid["temperature"],
                param_grid["num_samples"],
            ))

            for temp, sample_count in tqdm(
                param_combinations,
                desc=f"{stock} Toto"
            ):
                try:
                    context_length = min(1024, len(train_df))
                    context_series = train_df.tail(context_length)["close"].to_numpy(dtype=np.float32)
                    forecasts = pipeline.predict(
                        context=context_series,
                        prediction_length=self.prediction_length,
                        num_samples=sample_count,
                        temperature=temp,
                    )
                    if not forecasts:
                        raise RuntimeError("Toto pipeline returned no forecasts")

                    samples = forecasts[0].numpy()
                    sample_array = np.asarray(samples, dtype=np.float64)
                    if sample_array.ndim == 1:
                        preds = sample_array[: self.prediction_length]
                    else:
                        if sample_array.shape[-1] != self.prediction_length:
                            sample_array = sample_array.reshape(sample_array.shape[0], -1)
                        preds = sample_array.mean(axis=0)

                    holdout_actual = holdout_df["close"].to_numpy(dtype=np.float64)
                    pred_length = min(len(preds), len(holdout_actual))
                    if pred_length == 0:
                        raise ValueError("Toto forecast returned insufficient horizon length")

                    mae = np.mean(np.abs(preds[:pred_length] - holdout_actual[:pred_length]))
                    denom = np.mean(np.abs(holdout_actual[:pred_length])) or 1.0
                    mae_pct = (mae / denom) * 100

                    result = {
                        "temperature": temp,
                        "num_samples": sample_count,
                        "mae": float(mae),
                        "mae_pct": float(mae_pct),
                    }
                    results.append(result)

                    if mae < best_mae:
                        best_mae = mae
                        best_params = result.copy()

                except Exception as e:
                    print(f"  Error with params temp={temp}, samples={sample_count}: {e}")
                    continue

        except Exception as e:
            print(f"  Failed to initialize Toto for {stock}: {e}")
            return {
                "stock": stock,
                "model": "toto",
                "status": "failed",
                "error": str(e)
            }

        return {
            "stock": stock,
            "model": "toto",
            "status": "success",
            "best_params": best_params,
            "best_mae": float(best_mae),
            "all_results": results,
        }

    def validate_stock(
        self,
        stock: str,
        model_type: str = "both"
    ) -> Dict:
        """Validate hyperparameters for a stock."""
        results = {
            "stock": stock,
            "timestamp": datetime.now().isoformat(),
            "holdout_days": self.holdout_days,
            "prediction_length": self.prediction_length,
        }

        if model_type in ["kronos", "both"]:
            results["kronos"] = self.test_kronos_hyperparams(stock)

        if model_type in ["toto", "both"]:
            results["toto"] = self.test_toto_hyperparams(stock)

        return results

    def validate_all(
        self,
        stocks: List[str],
        model_type: str = "both"
    ) -> Dict:
        """Validate hyperparameters for all stocks."""
        all_results = []

        for stock in stocks:
            print(f"\n{'='*60}")
            print(f"Validating {stock}")
            print(f"{'='*60}")

            result = self.validate_stock(stock, model_type)
            all_results.append(result)

        # Save results
        output_path = Path("hyperparameter_validation_results.json")
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\n{'='*60}")
        print("Validation Summary")
        print(f"{'='*60}\n")

        # Summarize results
        kronos_improvements = []
        toto_improvements = []

        for result in all_results:
            if "kronos" in result and result["kronos"]["status"] == "success":
                kronos_improvements.append(result["kronos"]["best_mae"])

            if "toto" in result and result["toto"]["status"] == "success":
                toto_improvements.append(result["toto"]["best_mae"])

        if kronos_improvements:
            print(f"Kronos: {len(kronos_improvements)} stocks validated")
            print(f"  Mean MAE: {np.mean(kronos_improvements):.4f}")
            print(f"  Median MAE: {np.median(kronos_improvements):.4f}")

        if toto_improvements:
            print(f"Toto: {len(toto_improvements)} stocks validated")
            print(f"  Mean MAE: {np.mean(toto_improvements):.4f}")
            print(f"  Median MAE: {np.median(toto_improvements):.4f}")

        print(f"\nResults saved to: {output_path}")

        return {
            "results": all_results,
            "output_path": str(output_path),
        }

    @staticmethod
    def _has_cuda() -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Validate hyperparameters on holdout data"
    )
    parser.add_argument(
        "--stocks",
        nargs="+",
        help="Specific stocks to validate"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all available stocks"
    )
    parser.add_argument(
        "--holdout-days",
        type=int,
        default=7,
        help="Number of days to use for holdout validation"
    )
    parser.add_argument(
        "--prediction-length",
        type=int,
        default=64,
        help="Prediction horizon length"
    )
    parser.add_argument(
        "--model-type",
        choices=["kronos", "toto", "both"],
        default="both",
        help="Which model type to validate"
    )

    args = parser.parse_args()

    # Determine which stocks to validate
    if args.stocks:
        stocks = args.stocks
    elif args.all:
        # Get all stocks from training data
        data_dir = Path("trainingdata")
        stocks = [f.stem for f in data_dir.glob("*.csv")]
    else:
        # Default to priority stocks
        stocks = [
            "SPY", "QQQ", "MSFT", "AAPL", "GOOG",
            "NVDA", "AMD", "META", "TSLA", "BTCUSD"
        ]

    print(f"Validating stocks: {', '.join(stocks)}")
    print(f"Holdout days: {args.holdout_days}")
    print(f"Prediction length: {args.prediction_length}")
    print(f"Model type: {args.model_type}\n")

    validator = HyperparamValidator(
        holdout_days=args.holdout_days,
        prediction_length=args.prediction_length
    )

    validator.validate_all(stocks, args.model_type)


if __name__ == "__main__":
    main()
