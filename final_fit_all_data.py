#!/usr/bin/env python3
"""
Final fitting stage: Train on ALL data including latest with no validation split.

This is the final step after hyperparameter search - take the best config
and train on 100% of data to get the most up-to-date model.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

def load_best_config(results_file: str = "improvement_results.jsonl") -> dict:
    """Load the best performing config from experiment results."""
    results_path = Path(results_file)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file {results_file} not found")

    best_result = None
    best_score = float('-inf')

    with open(results_path) as f:
        for line in f:
            result = json.loads(line)
            # Prioritize Sortino, but require positive PnL
            if result['pnl'] > 0 and result['sortino'] > best_score:
                best_score = result['sortino']
                best_result = result

    if best_result is None:
        raise ValueError("No profitable configs found in results")

    return best_result['config']

def train_final_model(config: dict, output_name: str = "final_model"):
    """Train final model on all data with no validation split."""
    print("="*80)
    print("FINAL FIT - Training on ALL data (no validation)")
    print("="*80)
    print(f"\nBest config:\n{json.dumps(config, indent=2)}\n")

    # Build training command with validation_days=0
    cmd = [
        ".venv313/bin/python", "-m", "neuraldailytraining.train",
        "--data-root", "trainingdata/train",
        "--forecast-cache", "strategytraining/forecast_cache",
        "--epochs", "100",  # More epochs since we're not overfitting on validation
        "--validation-days", "0",  # KEY: No validation split!
        "--run-name", f"final_{output_name}",
    ]

    # Add config params
    for key, value in config.items():
        # Convert python naming to CLI naming
        cli_key = key.replace('_', '-')
        cmd.extend([f"--{cli_key}", str(value)])

    print(f"Running: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            env={"PYTHONPATH": "."},
        )
        print("\n✅ Final model training complete!")
        print(f"Checkpoint saved to: neuraldailytraining/checkpoints/final_{output_name}/")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e}")
        return 1

def main():
    parser = argparse.ArgumentParser(description="Train final model on all data")
    parser.add_argument("--results-file", default="improvement_results.jsonl",
                       help="Path to experiment results")
    parser.add_argument("--output-name", default="model",
                       help="Name for final model checkpoint")
    parser.add_argument("--config-file", help="Optional: Load config from JSON file instead of results")
    args = parser.parse_args()

    if args.config_file:
        with open(args.config_file) as f:
            config = json.load(f)
    else:
        config = load_best_config(args.results_file)

    return train_final_model(config, args.output_name)

if __name__ == "__main__":
    sys.exit(main())
