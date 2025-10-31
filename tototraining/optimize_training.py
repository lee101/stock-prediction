#!/usr/bin/env python3
"""
Comprehensive Training Optimization Script
Systematically tests different hyperparameters to improve MAE across stock pairs
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import itertools
import subprocess
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class OptimizationConfig:
    """Configuration for hyperparameter optimization"""

    # Learning rates to test
    LEARNING_RATES = [1e-4, 3e-4, 5e-4, 1e-3]

    # Loss functions to test
    LOSS_FUNCTIONS = ["huber", "heteroscedastic", "mse"]

    # Context lengths to test
    CONTEXT_LENGTHS = [2048, 4096, 8192]

    # Prediction lengths to test
    PREDICTION_LENGTHS = [32, 64, 128]

    # Batch sizes
    BATCH_SIZES = [2, 4, 8]

    # Epochs
    EPOCHS = [5, 10, 15]

    # Huber delta values (for huber loss)
    HUBER_DELTAS = [0.01, 0.05, 0.1]

    # Weight decay values
    WEIGHT_DECAYS = [1e-3, 1e-2, 5e-2]

    # Gradient clip values
    GRAD_CLIPS = [0.5, 1.0, 2.0]

    # LoRA configs
    LORA_RANKS = [4, 8, 16]
    LORA_ALPHAS = [8.0, 16.0, 32.0]


class TrainingExperiment:
    """Manages a single training experiment"""

    def __init__(self, config: Dict, experiment_id: str):
        self.config = config
        self.experiment_id = experiment_id
        self.results = {}

    def run(self, train_data: Path, val_data: Path = None) -> Dict:
        """Run training experiment with given config"""

        # Build command
        cmd = [
            "python", "tototraining/train.py",
            "--train-root", str(train_data),
            "--context-length", str(self.config.get("context_length", 4096)),
            "--prediction-length", str(self.config.get("prediction_length", 64)),
            "--batch-size", str(self.config.get("batch_size", 2)),
            "--epochs", str(self.config.get("epochs", 10)),
            "--learning-rate", str(self.config.get("learning_rate", 3e-4)),
            "--loss", self.config.get("loss", "huber"),
            "--weight-decay", str(self.config.get("weight_decay", 1e-2)),
            "--clip-grad", str(self.config.get("grad_clip", 1.0)),
            "--output-dir", f"tototraining/checkpoints/opt/{self.experiment_id}",
            "--checkpoint-name", f"model_{self.experiment_id}",
        ]

        if val_data:
            cmd.extend(["--val-root", str(val_data)])

        if self.config.get("loss") == "huber":
            cmd.extend(["--huber-delta", str(self.config.get("huber_delta", 0.01))])

        if self.config.get("use_lora", False):
            cmd.extend([
                "--adapter", "lora",
                "--adapter-r", str(self.config.get("lora_rank", 8)),
                "--adapter-alpha", str(self.config.get("lora_alpha", 16.0)),
            ])

        # Run training
        print(f"\n{'='*80}")
        print(f"Running experiment: {self.experiment_id}")
        print(f"Config: {json.dumps(self.config, indent=2)}")
        print(f"{'='*80}\n")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

            # Parse output for metrics
            output = result.stdout + result.stderr
            self.results = self._parse_metrics(output)
            self.results["success"] = result.returncode == 0
            self.results["config"] = self.config

        except subprocess.TimeoutExpired:
            print(f"Experiment {self.experiment_id} timed out!")
            self.results = {"success": False, "error": "timeout", "config": self.config}
        except Exception as e:
            print(f"Experiment {self.experiment_id} failed: {e}")
            self.results = {"success": False, "error": str(e), "config": self.config}

        return self.results

    def _parse_metrics(self, output: str) -> Dict:
        """Parse metrics from training output"""
        metrics = {}

        # Look for validation metrics in output
        for line in output.split('\n'):
            if 'val_loss=' in line:
                try:
                    val_loss = float(line.split('val_loss=')[1].split()[0])
                    metrics['val_loss'] = val_loss
                except:
                    pass
            if 'val_mape=' in line:
                try:
                    val_mape = float(line.split('val_mape=')[1].split('%')[0])
                    metrics['val_mape'] = val_mape
                except:
                    pass
            if 'train_loss=' in line:
                try:
                    train_loss = float(line.split('train_loss=')[1].split()[0])
                    metrics['train_loss'] = train_loss
                except:
                    pass

        return metrics


class OptimizationRunner:
    """Runs multiple experiments and tracks results"""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("tototraining/optimization_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []

    def run_grid_search(self,
                       param_grid: Dict[str, List],
                       train_data: Path,
                       val_data: Path = None,
                       max_experiments: int = None):
        """Run grid search over hyperparameter space"""

        # Generate all combinations
        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        combinations = list(itertools.product(*values))

        if max_experiments:
            combinations = combinations[:max_experiments]

        print(f"Running {len(combinations)} experiments...")

        for i, combo in enumerate(combinations):
            config = dict(zip(keys, combo))
            experiment_id = f"exp_{i:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            experiment = TrainingExperiment(config, experiment_id)
            results = experiment.run(train_data, val_data)

            self.results.append(results)
            self._save_results()

        return self.results

    def run_random_search(self,
                         param_grid: Dict[str, List],
                         train_data: Path,
                         val_data: Path = None,
                         n_experiments: int = 20):
        """Run random search over hyperparameter space"""

        import random

        print(f"Running {n_experiments} random experiments...")

        for i in range(n_experiments):
            config = {k: random.choice(v) for k, v in param_grid.items()}
            experiment_id = f"random_{i:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            experiment = TrainingExperiment(config, experiment_id)
            results = experiment.run(train_data, val_data)

            self.results.append(results)
            self._save_results()

        return self.results

    def _save_results(self):
        """Save current results to JSON"""
        results_file = self.output_dir / "optimization_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        # Also create a summary CSV
        self._create_summary_csv()

    def _create_summary_csv(self):
        """Create CSV summary of results"""
        summary_data = []

        for result in self.results:
            if result.get("success"):
                row = {
                    **result.get("config", {}),
                    "val_loss": result.get("val_loss"),
                    "val_mape": result.get("val_mape"),
                    "train_loss": result.get("train_loss"),
                }
                summary_data.append(row)

        if summary_data:
            df = pd.DataFrame(summary_data)
            # Sort by validation loss
            if "val_loss" in df.columns:
                df = df.sort_values("val_loss")
            df.to_csv(self.output_dir / "summary.csv", index=False)

            print("\n" + "="*80)
            print("TOP 5 CONFIGURATIONS:")
            print("="*80)
            print(df.head(5).to_string())
            print("\n")

    def get_best_config(self) -> Dict:
        """Get best performing configuration"""
        successful_results = [r for r in self.results if r.get("success")]
        if not successful_results:
            return {}

        best = min(successful_results,
                  key=lambda x: x.get("val_loss", float('inf')))
        return best.get("config", {})


def quick_optimization():
    """Quick optimization with a focused parameter grid"""

    param_grid = {
        "learning_rate": [1e-4, 3e-4, 5e-4],
        "loss": ["huber", "heteroscedastic"],
        "context_length": [4096],
        "prediction_length": [64],
        "batch_size": [4],
        "epochs": [5],
        "huber_delta": [0.01, 0.05],
        "weight_decay": [1e-2],
        "grad_clip": [1.0],
    }

    train_data = Path("trainingdata")

    runner = OptimizationRunner()
    results = runner.run_random_search(
        param_grid,
        train_data,
        val_data=None,
        n_experiments=10
    )

    print("\n" + "="*80)
    print("BEST CONFIGURATION:")
    print("="*80)
    best_config = runner.get_best_config()
    print(json.dumps(best_config, indent=2))

    return results


def comprehensive_optimization():
    """Comprehensive optimization with full parameter grid"""

    param_grid = {
        "learning_rate": OptimizationConfig.LEARNING_RATES,
        "loss": OptimizationConfig.LOSS_FUNCTIONS,
        "context_length": OptimizationConfig.CONTEXT_LENGTHS,
        "prediction_length": OptimizationConfig.PREDICTION_LENGTHS,
        "batch_size": OptimizationConfig.BATCH_SIZES,
        "epochs": OptimizationConfig.EPOCHS,
        "huber_delta": OptimizationConfig.HUBER_DELTAS,
        "weight_decay": OptimizationConfig.WEIGHT_DECAYS,
        "grad_clip": OptimizationConfig.GRAD_CLIPS,
    }

    train_data = Path("trainingdata")

    runner = OptimizationRunner()
    results = runner.run_random_search(
        param_grid,
        train_data,
        val_data=None,
        n_experiments=50  # Run 50 random experiments
    )

    print("\n" + "="*80)
    print("BEST CONFIGURATION:")
    print("="*80)
    best_config = runner.get_best_config()
    print(json.dumps(best_config, indent=2))

    return results


def test_single_stock(stock_file: str):
    """Test optimization on a single stock"""

    param_grid = {
        "learning_rate": [1e-4, 3e-4],
        "loss": ["huber", "heteroscedastic"],
        "context_length": [4096],
        "prediction_length": [64],
        "batch_size": [4],
        "epochs": [5],
        "huber_delta": [0.01],
        "weight_decay": [1e-2],
        "grad_clip": [1.0],
    }

    train_data = Path(f"trainingdata/{stock_file}")

    runner = OptimizationRunner(
        output_dir=Path(f"tototraining/optimization_results/{stock_file.replace('.csv', '')}")
    )
    results = runner.run_grid_search(
        param_grid,
        train_data,
        val_data=None,
        max_experiments=8
    )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimize training hyperparameters")
    parser.add_argument("--mode", choices=["quick", "comprehensive", "single"],
                       default="quick",
                       help="Optimization mode")
    parser.add_argument("--stock", type=str, help="Stock file for single-stock mode")

    args = parser.parse_args()

    if args.mode == "quick":
        quick_optimization()
    elif args.mode == "comprehensive":
        comprehensive_optimization()
    elif args.mode == "single":
        if not args.stock:
            print("Error: --stock required for single mode")
            sys.exit(1)
        test_single_stock(args.stock)
