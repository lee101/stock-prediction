#!/usr/bin/env python3
"""Hyperparameter optimization for Chronos2 on basis points (bp) formatted data.

This tuner optimizes Chronos2 hyperparameters specifically for predicting
daily basis point changes, which have different statistical properties
than raw price series.

Key differences from price-based tuning:
1. BP series are stationary (mean-reverting around 0)
2. Smaller magnitude values (typically -500 to +500 bps)
3. Shorter prediction horizons (1-5 days vs 64+ for prices)
4. Different context length requirements

Usage:
    python hyperparam_chronos_bp.py --symbols SPY QQQ BTCUSD
    python hyperparam_chronos_bp.py --all --quick
"""

import argparse
import json
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm


# Default hyperparameter grid for BP forecasting
# Note: Chronos2Pipeline uses predict_df which doesn't have num_samples/temperature
# Instead we optimize context_length and prediction_length
DEFAULT_PARAM_GRID = {
    # Context lengths - shorter for BP (more recent data matters more)
    "context_length": [64, 128, 256, 512, 1024],
    # Prediction horizons - typically 1-5 days for BP
    "prediction_length": [1, 3, 5],
}

# Quick mode grid for faster iteration
QUICK_PARAM_GRID = {
    "context_length": [128, 256, 512],
    "prediction_length": [1, 3],
}

DATA_DIR = Path("trainingdatabp")
RESULTS_DIR = Path("hyperparam_bp_results")


class Chronos2BPTuner:
    """Hyperparameter tuner for Chronos2 on BP-formatted data.

    Uses the Chronos2Pipeline.predict_df API which is the pandas-first interface
    for Chronos2 forecasting.
    """

    def __init__(
        self,
        holdout_days: int = 10,
        device: str = "cuda",
    ):
        self.holdout_days = holdout_days
        self.device = device
        self.model = None
        self.results: List[Dict] = []

    def load_data(self, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and split BP data for a symbol."""
        data_path = DATA_DIR / f"{symbol}.csv"

        if not data_path.exists():
            raise FileNotFoundError(f"BP data not found: {data_path}")

        df = pd.read_csv(data_path)

        # Ensure timestamp is datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Split: train and holdout
        split_idx = len(df) - self.holdout_days
        train_df = df.iloc[:split_idx].copy()
        holdout_df = df.iloc[split_idx:].copy()

        return train_df, holdout_df

    def _load_model(self):
        """Load Chronos2 model (lazy loading)."""
        if self.model is not None:
            return

        try:
            from chronos import Chronos2Pipeline

            logger.info("Loading Chronos2 model...")
            self.model = Chronos2Pipeline.from_pretrained(
                "amazon/chronos-2",
                device_map=self.device,
            )
            logger.info("Chronos2 model loaded")

        except ImportError as e:
            raise RuntimeError(f"Chronos2 not available: {e}")

    def predict_bp(
        self,
        context_df: pd.DataFrame,
        symbol: str,
        prediction_length: int,
        context_length: int,
    ) -> pd.DataFrame:
        """Make BP predictions using Chronos2Pipeline.predict_df."""
        self._load_model()

        # Prepare context dataframe in the format Chronos2 expects
        # Add symbol column for id
        df = context_df.copy()
        df["symbol"] = symbol

        # Trim to context_length
        if len(df) > context_length:
            df = df.iloc[-context_length:].copy()

        # Call predict_df
        predictions = self.model.predict_df(
            df,
            id_column="symbol",
            timestamp_column="timestamp",
            target="close_bps",  # Predict the close_bps column
            prediction_length=prediction_length,
            quantile_levels=[0.1, 0.5, 0.9],
        )

        return predictions

    def evaluate_params(
        self,
        symbol: str,
        context_length: int,
        prediction_length: int,
    ) -> Dict:
        """Evaluate a set of hyperparameters on holdout data."""
        train_df, holdout_df = self.load_data(symbol)

        if len(train_df) < context_length:
            return {
                "status": "error",
                "error": f"Insufficient data: {len(train_df)} < {context_length}",
            }

        # Get actual holdout values
        actual = holdout_df["close_bps"].values[:prediction_length].astype(np.float32)

        if len(actual) < prediction_length:
            return {
                "status": "error",
                "error": f"Insufficient holdout: {len(actual)} < {prediction_length}",
            }

        try:
            # Make prediction
            pred_df = self.predict_bp(
                context_df=train_df,
                symbol=symbol,
                prediction_length=prediction_length,
                context_length=context_length,
            )

            # Extract median (0.5 quantile) predictions
            if "0.5" in pred_df.columns:
                predicted = pred_df["0.5"].values[:prediction_length].astype(np.float32)
            else:
                # Fall back to first numeric column
                numeric_cols = pred_df.select_dtypes(include=[np.number]).columns
                predicted = pred_df[numeric_cols[0]].values[:prediction_length].astype(np.float32)

            # Compute metrics
            mae = float(np.mean(np.abs(predicted - actual)))
            rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
            # Direction accuracy (did we predict the sign correctly?)
            direction_acc = float(np.mean(np.sign(predicted) == np.sign(actual)))
            # Correlation
            if len(predicted) > 1:
                corr = float(np.corrcoef(predicted, actual)[0, 1])
                if np.isnan(corr):
                    corr = 0.0
            else:
                corr = 0.0

            return {
                "status": "success",
                "mae_bps": mae,
                "rmse_bps": rmse,
                "direction_accuracy": direction_acc,
                "correlation": corr,
                "predicted": predicted.tolist(),
                "actual": actual.tolist(),
            }

        except Exception as e:
            logger.warning(f"Evaluation error for {symbol}: {e}")
            return {
                "status": "error",
                "error": str(e),
            }

    def grid_search(
        self,
        symbols: List[str],
        param_grid: Dict[str, List],
    ) -> Dict:
        """Run grid search over hyperparameters."""
        logger.info(f"Starting grid search for {len(symbols)} symbols")

        # Generate all combinations
        keys = list(param_grid.keys())
        combinations = list(product(*[param_grid[k] for k in keys]))
        logger.info(f"Testing {len(combinations)} parameter combinations")

        all_results = []

        for params in tqdm(combinations, desc="Hyperparams"):
            param_dict = dict(zip(keys, params))

            # Test on all symbols
            symbol_results = []
            for symbol in symbols:
                try:
                    result = self.evaluate_params(
                        symbol=symbol,
                        context_length=param_dict["context_length"],
                        prediction_length=param_dict["prediction_length"],
                    )
                    result["symbol"] = symbol
                    result.update(param_dict)
                    symbol_results.append(result)
                except Exception as e:
                    logger.warning(f"Error for {symbol} with {param_dict}: {e}")

            if symbol_results:
                # Compute aggregate metrics
                successful = [r for r in symbol_results if r["status"] == "success"]
                if successful:
                    avg_mae = np.mean([r["mae_bps"] for r in successful])
                    avg_rmse = np.mean([r["rmse_bps"] for r in successful])
                    avg_dir_acc = np.mean([r["direction_accuracy"] for r in successful])

                    result = {
                        **param_dict,
                        "avg_mae_bps": float(avg_mae),
                        "avg_rmse_bps": float(avg_rmse),
                        "avg_direction_accuracy": float(avg_dir_acc),
                        "num_symbols_tested": len(successful),
                        "symbol_results": symbol_results,
                    }
                    all_results.append(result)

        # Sort by MAE
        all_results.sort(key=lambda x: x.get("avg_mae_bps", float("inf")))

        return {
            "results": all_results,
            "best": all_results[0] if all_results else None,
            "timestamp": datetime.now().isoformat(),
        }

    def unload_model(self):
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass


def main():
    parser = argparse.ArgumentParser(description="Chronos2 BP hyperparameter tuning")
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Symbols to tune on"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Use all available BP data"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode with smaller grid"
    )
    parser.add_argument(
        "--holdout-days",
        type=int,
        default=10,
        help="Days to hold out for validation"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device (cuda or cpu)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "latest_results.json",
        help="Output file for results"
    )

    args = parser.parse_args()

    # Create output directory
    RESULTS_DIR.mkdir(exist_ok=True)

    # Determine symbols
    if args.symbols:
        symbols = args.symbols
    elif args.all:
        if not DATA_DIR.exists():
            logger.error(f"BP data directory not found: {DATA_DIR}")
            logger.info("Run 'python create_trainingdatabp.py' first to create BP data")
            sys.exit(1)
        symbols = [f.stem for f in DATA_DIR.glob("*.csv")]
    else:
        # Default to key symbols
        symbols = [
            "SPY", "QQQ", "AAPL", "MSFT", "NVDA",
            "BTCUSD", "ETHUSD", "UNIUSD",
        ]

    # Filter to available data
    available = []
    for sym in symbols:
        if (DATA_DIR / f"{sym}.csv").exists():
            available.append(sym)
        else:
            logger.warning(f"No BP data for {sym}, skipping")

    if not available:
        logger.error("No symbols with BP data available")
        logger.info("Run 'python create_trainingdatabp.py' first")
        sys.exit(1)

    logger.info(f"Tuning on {len(available)} symbols: {available}")

    # Select param grid
    param_grid = QUICK_PARAM_GRID if args.quick else DEFAULT_PARAM_GRID
    logger.info(f"Using {'quick' if args.quick else 'full'} parameter grid")

    # Run tuning
    tuner = Chronos2BPTuner(
        holdout_days=args.holdout_days,
        device=args.device,
    )

    try:
        results = tuner.grid_search(available, param_grid)

        # Save results
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")

        # Print summary
        print("\n" + "=" * 70)
        print("HYPERPARAMETER TUNING COMPLETE")
        print("=" * 70)

        if results["best"]:
            best = results["best"]
            print(f"\nBest parameters:")
            print(f"  context_length: {best['context_length']}")
            print(f"  prediction_length: {best['prediction_length']}")
            print(f"\nPerformance:")
            print(f"  Avg MAE: {best['avg_mae_bps']:.2f} bps")
            print(f"  Avg RMSE: {best['avg_rmse_bps']:.2f} bps")
            print(f"  Direction accuracy: {best['avg_direction_accuracy']:.1%}")
            print(f"  Symbols tested: {best['num_symbols_tested']}")

        print(f"\nTop 5 configurations by MAE:")
        for i, result in enumerate(results["results"][:5], 1):
            print(f"  {i}. ctx={result['context_length']}, "
                  f"pred={result['prediction_length']} "
                  f"-> MAE={result['avg_mae_bps']:.2f} bps, "
                  f"Dir={result['avg_direction_accuracy']:.1%}")

    finally:
        tuner.unload_model()


if __name__ == "__main__":
    main()
