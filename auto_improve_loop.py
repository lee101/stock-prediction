#!/usr/bin/env python3
"""
Automated hyperparameter search loop for neural daily trading model.
Trains multiple configurations, runs market simulations, and tracks best performers.
"""
import json
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import itertools

from refresh_daily_inputs import refresh_daily_inputs

@dataclass
class TrainingConfig:
    """Configuration for a single training run"""
    transformer_dim: int = 256
    transformer_heads: int = 8
    transformer_layers: int = 4
    sequence_length: int = 256
    learning_rate: float = 0.0001
    batch_size: int = 32
    dropout: float = 0.1
    price_offset_pct: float = 0.025
    max_trade_qty: float = 3.0
    risk_threshold: float = 1.0

    def to_cli_args(self) -> List[str]:
        """Convert config to CLI arguments"""
        return [
            "--transformer-dim", str(self.transformer_dim),
            "--transformer-heads", str(self.transformer_heads),
            "--transformer-layers", str(self.transformer_layers),
            "--sequence-length", str(self.sequence_length),
            "--learning-rate", str(self.learning_rate),
            "--batch-size", str(self.batch_size),
            "--dropout", str(self.dropout),
            "--price-offset-pct", str(self.price_offset_pct),
            "--max-trade-qty", str(self.max_trade_qty),
            "--risk-threshold", str(self.risk_threshold),
        ]

@dataclass
class ExperimentResult:
    """Results from a training + simulation run"""
    config: TrainingConfig
    checkpoint_path: str
    final_equity: float
    pnl: float
    sortino: float
    training_time: float
    simulation_date: str

class AutoImprovementLoop:
    def __init__(
        self,
        results_file: str = "improvement_results.jsonl",
        best_checkpoint_dir: str = "best_neuraldaily_checkpoints",
    ):
        self.results_file = Path(results_file)
        self.best_checkpoint_dir = Path(best_checkpoint_dir)
        self.best_checkpoint_dir.mkdir(exist_ok=True)
        self.best_pnl = float('-inf')
        self.best_sortino = float('-inf')

    def generate_configs(self) -> List[TrainingConfig]:
        """Generate grid of hyperparameter configurations to try"""
        configs = []

        # Grid search parameters
        transformer_dims = [256, 384, 512]
        transformer_heads = [8, 12, 16]
        transformer_layers = [4, 6, 8]
        sequence_lengths = [128, 256, 384]
        learning_rates = [0.00005, 0.0001, 0.0002]
        dropouts = [0.05, 0.1, 0.15]

        # Generate all combinations (warning: this can be large!)
        # For initial run, let's do a smaller subset
        for dim, heads, layers in itertools.product(transformer_dims, transformer_heads, transformer_layers):
            # Only try compatible combinations
            if dim % heads != 0:
                continue
            configs.append(TrainingConfig(
                transformer_dim=dim,
                transformer_heads=heads,
                transformer_layers=layers,
            ))

        # Add some learning rate variations
        for lr in learning_rates:
            configs.append(TrainingConfig(learning_rate=lr))

        # Add dropout variations
        for dropout in dropouts:
            configs.append(TrainingConfig(dropout=dropout))

        # Add sequence length variations
        for seq_len in sequence_lengths:
            configs.append(TrainingConfig(sequence_length=seq_len))

        print(f"Generated {len(configs)} configurations to test")
        return configs

    def train_model(self, config: TrainingConfig) -> Optional[str]:
        """Train a model with given config and return checkpoint path"""
        print(f"\n{'='*80}")
        print(f"Training with config: {asdict(config)}")
        print(f"{'='*80}\n")

        start_time = time.time()

        # Build training command
        cmd = [
            "python", "-m", "neuraldailytraining.train",
            "--data-root", "trainingdata/train",
            "--forecast-cache", "strategytraining/forecast_cache",
            "--epochs", "50",
            "--validation-days", "40",
        ]
        cmd.extend(config.to_cli_args())

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout
            )
            training_time = time.time() - start_time

            # Extract checkpoint path from output
            # Assume it outputs something like "Saved checkpoint: path/to/checkpoint"
            for line in result.stdout.split('\n'):
                if 'Saved checkpoint' in line or 'checkpoint_path' in line:
                    # Extract path
                    checkpoint_path = line.split()[-1]
                    return checkpoint_path

            # If no explicit path, look for most recent checkpoint
            checkpoints_dir = Path("neuraldailytraining/checkpoints")
            if checkpoints_dir.exists():
                latest = max(checkpoints_dir.glob("neuraldaily_*/epoch_*.pt"),
                           key=lambda p: p.stat().st_mtime)
                return str(latest)

        except subprocess.TimeoutExpired:
            print(f"Training timed out after 2 hours")
            return None
        except subprocess.CalledProcessError as e:
            print(f"Training failed: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            return None

        return None

    def run_simulation(self, checkpoint_path: str, days: int = 10) -> Optional[Dict]:
        """Run market simulation and return results"""
        print(f"\nRunning simulation for checkpoint: {checkpoint_path}")

        cmd = [
            "python", "neuraldailymarketsimulator/simulator.py",
            "--checkpoint", checkpoint_path,
            "--days", str(days),
            "--start-date", "2025-10-05",  # Most recent data available
        ]

        try:
            result = subprocess.run(
                ["env", "PYTHONPATH=."] + cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=600,
            )

            # Parse simulation output
            final_equity = pnl = sortino = None
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Final Equity' in line:
                    try:
                        final_equity = float(line.split(':')[-1].strip())
                    except ValueError:
                        pass
                elif 'Net PnL' in line:
                    try:
                        pnl = float(line.split(':')[-1].strip())
                    except ValueError:
                        pass
                elif 'Sortino Ratio' in line:
                    try:
                        sortino = float(line.split(':')[-1].strip())
                    except ValueError:
                        pass

            missing_metrics = [
                name for name, value in (
                    ('Final Equity', final_equity),
                    ('Net PnL', pnl),
                    ('Sortino Ratio', sortino),
                ) if value is None
            ]
            if missing_metrics:
                print(
                    "Simulation did not report metrics: "
                    f"{', '.join(missing_metrics)}\n"
                    f"Last 20 lines of stdout:\n" +
                    "\n".join(lines[-20:])
                )
                return None

            return {
                'final_equity': final_equity,
                'pnl': pnl,
                'sortino': sortino,
            }

        except Exception as e:
            print(f"Simulation failed: {e}")
            return None

    def save_result(self, result: ExperimentResult):
        """Append result to results file"""
        with open(self.results_file, 'a') as f:
            result_dict = {
                'config': asdict(result.config),
                'checkpoint_path': result.checkpoint_path,
                'final_equity': result.final_equity,
                'pnl': result.pnl,
                'sortino': result.sortino,
                'training_time': result.training_time,
                'simulation_date': result.simulation_date,
            }
            f.write(json.dumps(result_dict) + '\n')

    def run(self, max_experiments: Optional[int] = None):
        """Run the automated improvement loop"""
        print("Refreshing training data and forecast caches before experiments...")
        refresh_daily_inputs()

        configs = self.generate_configs()

        if max_experiments:
            configs = configs[:max_experiments]

        for i, config in enumerate(configs, 1):
            print(f"\n\n{'#'*80}")
            print(f"# Experiment {i}/{len(configs)}")
            print(f"{'#'*80}\n")

            start_time = time.time()

            # Train model
            checkpoint = self.train_model(config)
            if not checkpoint:
                print(f"Skipping experiment {i} due to training failure")
                continue

            training_time = time.time() - start_time

            # Run simulation
            sim_results = self.run_simulation(checkpoint)
            if not sim_results:
                print(f"Skipping experiment {i} due to simulation failure")
                continue

            # Create and save result
            result = ExperimentResult(
                config=config,
                checkpoint_path=checkpoint,
                final_equity=sim_results['final_equity'],
                pnl=sim_results['pnl'],
                sortino=sim_results['sortino'],
                training_time=training_time,
                simulation_date=datetime.now().isoformat(),
            )

            self.save_result(result)

            # Track best performers
            if result.pnl > self.best_pnl:
                self.best_pnl = result.pnl
                print(f"\n🎉 NEW BEST PNL: {result.pnl:.4f} (Sortino: {result.sortino:.4f})")
                # Copy checkpoint to best directory
                import shutil
                best_path = self.best_checkpoint_dir / f"best_pnl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                shutil.copy(checkpoint, best_path)

            if result.sortino > self.best_sortino and result.pnl > 0:
                self.best_sortino = result.sortino
                print(f"\n🎉 NEW BEST SORTINO: {result.sortino:.4f} (PnL: {result.pnl:.4f})")
                import shutil
                best_path = self.best_checkpoint_dir / f"best_sortino_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                shutil.copy(checkpoint, best_path)

            print(f"\n📊 Experiment {i} Summary:")
            print(f"   PnL: {result.pnl:.4f}")
            print(f"   Sortino: {result.sortino:.4f}")
            print(f"   Final Equity: {result.final_equity:.4f}")
            print(f"   Training Time: {result.training_time:.1f}s")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run automated hyperparameter search")
    parser.add_argument("--max-experiments", type=int, help="Max number of experiments to run")
    parser.add_argument("--results-file", default="improvement_results.jsonl")
    args = parser.parse_args()

    loop = AutoImprovementLoop(results_file=args.results_file)
    loop.run(max_experiments=args.max_experiments)
