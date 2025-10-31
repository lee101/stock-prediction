#!/usr/bin/env python3
"""
Comprehensive stock-specific model retraining from checkpoints.
Trains both Kronos and Toto models per stock pair with extended epochs.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Priority stocks for quick iteration
PRIORITY_STOCKS = [
    "SPY", "QQQ", "MSFT", "AAPL", "GOOG",
    "NVDA", "AMD", "META", "TSLA", "BTCUSD", "ETHUSD"
]

# All available stocks
ALL_STOCKS = [
    "AAPL", "ADBE", "ADSK", "AMD", "AMZN", "BTCUSD",
    "COIN", "COUR", "CRWD", "ETHUSD", "GOOG", "GOOGL",
    "INTC", "LCID", "META", "MSFT", "NET", "NVDA",
    "QQQ", "QUBT", "SPY", "TSLA", "U", "UNIUSD"
]


class StockRetrainer:
    """Manages retraining for both Kronos and Toto models."""

    def __init__(
        self,
        stocks: List[str],
        extended_epochs: bool = True,
        resume_from_checkpoint: bool = True
    ):
        self.stocks = stocks
        self.extended_epochs = extended_epochs
        self.resume_from_checkpoint = resume_from_checkpoint
        self.results = []

        # Load baseline to determine training difficulty
        self.baseline = self._load_baseline()

    def _load_baseline(self) -> Dict:
        """Load baseline results to inform training strategy."""
        baseline_path = Path("tototraining/baseline_results.json")
        if baseline_path.exists():
            with open(baseline_path) as f:
                return json.load(f)
        return {}

    def _get_difficulty_level(self, stock: str) -> str:
        """Determine stock difficulty based on baseline MAE."""
        if stock not in self.baseline:
            return "medium"

        h64_pct = self.baseline[stock].get("h64_pct", 10.0)

        if h64_pct < 8.0:
            return "easy"
        elif h64_pct < 15.0:
            return "medium"
        else:
            return "hard"

    def _get_kronos_config(self, stock: str, difficulty: str) -> Dict:
        """Get Kronos training configuration based on difficulty."""
        base_config = {
            "data_dir": f"trainingdata/{stock}.csv",
            "output_dir": f"kronostraining/artifacts/stock_specific/{stock}",
            "adapter": "lora",
            "adapter_name": stock,
        }

        # Sample count determines context length
        sample_count = self.baseline.get(stock, {}).get("count", 1000)

        if sample_count < 500:
            lookback, horizon, epochs = 256, 16, 15
        elif sample_count < 1000:
            lookback, horizon, epochs = 512, 32, 20
        elif sample_count < 1500:
            lookback, horizon, epochs = 768, 48, 25
        else:
            lookback, horizon, epochs = 1024, 64, 30

        # Adjust for difficulty
        if difficulty == "hard":
            epochs += 10
            adapter_r = 16
            lr = 1e-4
        elif difficulty == "easy":
            adapter_r = 8
            lr = 3e-4
        else:  # medium
            adapter_r = 12
            lr = 2e-4

        if self.extended_epochs:
            epochs = int(epochs * 1.5)

        base_config.update({
            "lookback": lookback,
            "horizon": horizon,
            "epochs": epochs,
            "adapter_r": adapter_r,
            "adapter_alpha": adapter_r * 2,
            "lr": lr,
            "batch_size": 4,
            "validation_days": 30,
        })

        return base_config

    def _get_toto_config(self, stock: str, difficulty: str) -> Dict:
        """Get Toto training configuration based on difficulty."""
        sample_count = self.baseline.get(stock, {}).get("count", 1000)

        if sample_count < 500:
            context_length, pred_length, epochs = 256, 16, 10
        elif sample_count < 1000:
            context_length, pred_length, epochs = 512, 32, 15
        elif sample_count < 1500:
            context_length, pred_length, epochs = 768, 48, 20
        else:
            context_length, pred_length, epochs = 1024, 64, 25

        # Adjust for difficulty
        if difficulty == "hard":
            epochs += 10
            adapter_r = 16
            loss_type = "heteroscedastic"
            lr = 1e-4
        elif difficulty == "easy":
            adapter_r = 8
            loss_type = "huber"
            lr = 3e-4
        else:
            adapter_r = 12
            loss_type = "heteroscedastic"
            lr = 2e-4

        if self.extended_epochs:
            epochs = int(epochs * 1.5)

        return {
            "stock": stock,
            "context_length": context_length,
            "prediction_length": pred_length,
            "epochs": epochs,
            "adapter_r": adapter_r,
            "loss_type": loss_type,
            "lr": lr,
            "batch_size": 4,
        }

    def train_kronos(self, stock: str) -> bool:
        """Train Kronos model for a specific stock."""
        print(f"\n{'='*60}")
        print(f"Training Kronos for {stock}")
        print(f"{'='*60}\n")

        difficulty = self._get_difficulty_level(stock)
        config = self._get_kronos_config(stock, difficulty)

        # Build command
        cmd = [
            "uv", "run", "python", "-m", "kronostraining.run_training"
        ]

        for key, value in config.items():
            if key == "data_dir":
                cmd.extend(["--data-dir", str(value)])
            elif key == "output_dir":
                cmd.extend(["--output-dir", str(value)])
            elif isinstance(value, bool):
                if value:
                    cmd.append(f"--{key.replace('_', '-')}")
            else:
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])

        print(f"Command: {' '.join(cmd)}")
        print(f"Difficulty: {difficulty}")
        print(f"Config: {json.dumps(config, indent=2)}\n")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error training Kronos for {stock}:")
            print(e.stderr)
            return False

    def train_toto(self, stock: str) -> bool:
        """Train Toto model for a specific stock."""
        print(f"\n{'='*60}")
        print(f"Training Toto for {stock}")
        print(f"{'='*60}\n")

        difficulty = self._get_difficulty_level(stock)
        config = self._get_toto_config(stock, difficulty)

        # Use the toto_retrain_wrapper with custom config
        cmd = [
            "uv", "run", "python",
            "tototraining/toto_retrain_wrapper.py",
            "--stocks", stock
        ]

        print(f"Command: {' '.join(cmd)}")
        print(f"Difficulty: {difficulty}")
        print(f"Config: {json.dumps(config, indent=2)}\n")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error training Toto for {stock}:")
            print(e.stderr)
            return False

    def retrain_all(self, model_type: str = "both") -> Dict:
        """Retrain all stocks for specified model type(s)."""
        results = {
            "stocks_processed": [],
            "kronos_successes": [],
            "kronos_failures": [],
            "toto_successes": [],
            "toto_failures": [],
        }

        for stock in self.stocks:
            print(f"\n{'#'*60}")
            print(f"# Processing {stock}")
            print(f"{'#'*60}\n")

            results["stocks_processed"].append(stock)

            # Train Kronos
            if model_type in ["kronos", "both"]:
                success = self.train_kronos(stock)
                if success:
                    results["kronos_successes"].append(stock)
                else:
                    results["kronos_failures"].append(stock)

            # Train Toto
            if model_type in ["toto", "both"]:
                success = self.train_toto(stock)
                if success:
                    results["toto_successes"].append(stock)
                else:
                    results["toto_failures"].append(stock)

        # Save results
        results_path = Path("retraining_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*60}")
        print("Retraining Summary")
        print(f"{'='*60}\n")
        print(f"Stocks processed: {len(results['stocks_processed'])}")
        print(f"Kronos successes: {len(results['kronos_successes'])}")
        print(f"Kronos failures: {len(results['kronos_failures'])}")
        print(f"Toto successes: {len(results['toto_successes'])}")
        print(f"Toto failures: {len(results['toto_failures'])}")
        print(f"\nResults saved to: {results_path}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Retrain stock-specific models with extended training"
    )
    parser.add_argument(
        "--stocks",
        nargs="+",
        help="Specific stocks to train (default: priority stocks)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all available stocks"
    )
    parser.add_argument(
        "--priority-only",
        action="store_true",
        help="Train only priority stocks"
    )
    parser.add_argument(
        "--model-type",
        choices=["kronos", "toto", "both"],
        default="both",
        help="Which model type to train"
    )
    parser.add_argument(
        "--no-extended-epochs",
        action="store_true",
        help="Don't use extended epoch counts"
    )

    args = parser.parse_args()

    # Determine which stocks to train
    if args.stocks:
        stocks = args.stocks
    elif args.all:
        stocks = ALL_STOCKS
    elif args.priority_only:
        stocks = PRIORITY_STOCKS
    else:
        stocks = PRIORITY_STOCKS

    print(f"Stocks to train: {', '.join(stocks)}")
    print(f"Model type: {args.model_type}")
    print(f"Extended epochs: {not args.no_extended_epochs}\n")

    retrainer = StockRetrainer(
        stocks=stocks,
        extended_epochs=not args.no_extended_epochs,
        resume_from_checkpoint=True
    )

    results = retrainer.retrain_all(model_type=args.model_type)

    if results["kronos_failures"] or results["toto_failures"]:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
