#!/usr/bin/env python3
"""
Toto Stock-Specific Retraining Wrapper

Trains optimized, stock-specific Toto models for comparison with Kronos.
Creates models and configs compatible with the existing test_kronos_vs_toto.py framework.
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class StockConfig:
    """Configuration for training a stock-specific model"""
    symbol: str
    num_samples: int
    context_length: int
    prediction_length: int
    batch_size: int
    epochs: int
    learning_rate: float
    loss: str
    huber_delta: float = 0.01
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: float = 16.0
    weight_decay: float = 0.01
    grad_clip: float = 1.0


class TotoRetrainer:
    """Manages training of stock-specific Toto models"""

    def __init__(self, output_root: Path = None, hyperparam_root: Path = None):
        self.output_root = output_root or Path("tototraining/stock_models")
        self.hyperparam_root = hyperparam_root or Path("hyperparams/toto")
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.hyperparam_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_optimal_config_for_stock(
        symbol: str,
        num_samples: int,
        baseline_mae_pct: float
    ) -> StockConfig:
        """Generate optimal training configuration based on stock characteristics"""

        # Determine context and prediction lengths based on sample size
        if num_samples <= 500:
            context = 256
            pred = 16
            epochs = 15
            batch = 2
        elif num_samples <= 1000:
            context = 512
            pred = 32
            epochs = 12
            batch = 4
        elif num_samples <= 1500:
            context = 768
            pred = 48
            epochs = 10
            batch = 4
        else:  # 1500+
            context = 1024
            pred = 64
            epochs = 10
            batch = 4

        # Adjust hyperparameters based on baseline difficulty
        if baseline_mae_pct < 10:  # Easy stocks
            learning_rate = 3e-4
            loss = "huber"
            lora_rank = 8
        elif baseline_mae_pct < 20:  # Medium difficulty
            learning_rate = 5e-4
            loss = "heteroscedastic"
            lora_rank = 12
        else:  # Hard stocks
            learning_rate = 5e-4
            loss = "heteroscedastic"
            lora_rank = 16

        return StockConfig(
            symbol=symbol,
            num_samples=num_samples,
            context_length=context,
            prediction_length=pred,
            batch_size=batch,
            epochs=epochs,
            learning_rate=learning_rate,
            loss=loss,
            lora_rank=lora_rank,
        )

    def train_stock_model(
        self,
        config: StockConfig,
        train_file: Path,
        val_file: Optional[Path] = None
    ) -> Tuple[bool, Dict, Path]:
        """Train a single stock-specific model"""

        print(f"\n{'='*100}")
        print(f"Training {config.symbol} Model")
        print(f"{'='*100}")
        print(f"  Samples: {config.num_samples}")
        print(f"  Context: {config.context_length}, Prediction: {config.prediction_length}")
        print(f"  Loss: {config.loss}, LR: {config.learning_rate}")
        print(f"  LoRA Rank: {config.lora_rank}, Epochs: {config.epochs}")
        print(f"{'='*100}\n")

        # Output directory for this model
        model_dir = self.output_root / config.symbol
        model_dir.mkdir(parents=True, exist_ok=True)

        # Build training command
        cmd = [
            "uv", "run", "python", "tototraining/train.py",
            "--train-root", str(train_file),
            "--context-length", str(config.context_length),
            "--prediction-length", str(config.prediction_length),
            "--batch-size", str(config.batch_size),
            "--epochs", str(config.epochs),
            "--learning-rate", str(config.learning_rate),
            "--loss", config.loss,
            "--weight-decay", str(config.weight_decay),
            "--clip-grad", str(config.grad_clip),
            "--precision", "bf16",
            "--output-dir", str(model_dir),
            "--checkpoint-name", f"{config.symbol}_model",
            "--log-interval", "20",
        ]

        if val_file:
            cmd.extend(["--val-root", str(val_file)])
        else:
            cmd.extend(["--val-root", str(train_file)])  # Use train as validation

        if config.loss == "huber":
            cmd.extend(["--huber-delta", str(config.huber_delta)])

        if config.use_lora:
            cmd.extend([
                "--adapter", "lora",
                "--adapter-r", str(config.lora_rank),
                "--adapter-alpha", str(config.lora_alpha),
                "--freeze-backbone",
                "--adapter-name", config.symbol,
            ])

        # Save config
        config_file = model_dir / "training_config.json"
        with open(config_file, 'w') as f:
            json.dump(asdict(config), f, indent=2)

        print(f"Command: {' '.join(cmd)}\n")

        # Run training
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                cwd=PROJECT_ROOT
            )

            # Save output
            output_file = model_dir / "training_output.txt"
            with open(output_file, 'w') as f:
                f.write(result.stdout)
                f.write("\n" + "="*80 + "\n")
                f.write(result.stderr)

            # Parse metrics
            metrics = self._parse_training_output(result.stdout + result.stderr)
            metrics["success"] = result.returncode == 0
            metrics["timestamp"] = datetime.now().isoformat()

            # Save metrics
            metrics_file = model_dir / "training_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)

            if metrics["success"]:
                print(f"✅ {config.symbol} training completed successfully!")
                if "final_val_mape" in metrics:
                    print(f"   Final Val MAPE: {metrics['final_val_mape']:.2f}%")
                if "final_val_loss" in metrics:
                    print(f"   Final Val Loss: {metrics['final_val_loss']:.6f}")
            else:
                print(f"❌ {config.symbol} training failed!")

            return metrics["success"], metrics, model_dir

        except subprocess.TimeoutExpired:
            print(f"⏱️  {config.symbol} training timed out!")
            return False, {"error": "timeout"}, model_dir
        except Exception as e:
            print(f"❌ {config.symbol} training error: {e}")
            return False, {"error": str(e)}, model_dir

    def _parse_training_output(self, output: str) -> Dict:
        """Parse metrics from training output"""
        metrics = {
            "train_losses": [],
            "val_losses": [],
            "val_mapes": [],
        }

        for line in output.split('\n'):
            if 'train_loss=' in line and 'Epoch' in line:
                try:
                    train_loss = float(line.split('train_loss=')[1].split()[0])
                    metrics["train_losses"].append(train_loss)
                except:
                    pass

            if 'val_loss=' in line:
                try:
                    val_loss = float(line.split('val_loss=')[1].split()[0])
                    metrics["val_losses"].append(val_loss)
                except:
                    pass

            if 'val_mape=' in line:
                try:
                    val_mape = float(line.split('val_mape=')[1].split('%')[0])
                    metrics["val_mapes"].append(val_mape)
                except:
                    pass

            if 'Best validation loss' in line:
                try:
                    best_val = float(line.split('Best validation loss')[1].split()[0])
                    best_epoch = int(line.split('epoch')[1].split('.')[0].strip())
                    metrics["best_val_loss"] = best_val
                    metrics["best_epoch"] = best_epoch
                except:
                    pass

        # Compute final metrics
        if metrics["val_losses"]:
            metrics["final_val_loss"] = metrics["val_losses"][-1]
            metrics["min_val_loss"] = min(metrics["val_losses"])

        if metrics["val_mapes"]:
            metrics["final_val_mape"] = metrics["val_mapes"][-1]
            metrics["min_val_mape"] = min(metrics["val_mapes"])

        if metrics["train_losses"]:
            metrics["final_train_loss"] = metrics["train_losses"][-1]

        return metrics

    def create_hyperparam_config(
        self,
        symbol: str,
        config: StockConfig,
        metrics: Dict,
        model_path: Path
    ) -> Path:
        """Create hyperparameter config compatible with test_kronos_vs_toto.py"""

        hyperparam_config = {
            "config": {
                "name": f"toto_{symbol}_retrained",
                "num_samples": 256,  # For inference sampling
                "aggregate": "mean",
                "samples_per_batch": 128,
                "model_path": str(model_path),
                "context_length": config.context_length,
                "prediction_length": config.prediction_length,
            },
            "training": {
                "learning_rate": config.learning_rate,
                "loss": config.loss,
                "epochs": config.epochs,
                "lora_rank": config.lora_rank,
                "batch_size": config.batch_size,
            },
            "validation": {
                "loss": metrics.get("final_val_loss"),
                "price_mae": None,  # Will be computed during comparison
                "pct_return_mae": None,
            },
            "test": {
                "loss": None,
                "price_mae": None,
                "pct_return_mae": None,
            },
            "metadata": {
                "source": "toto_retrain_wrapper",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "baseline_mae_pct": None,  # Can be filled in later
            },
        }

        # Save to hyperparams directory
        config_file = self.hyperparam_root / f"{symbol}.json"
        with open(config_file, 'w') as f:
            json.dump(hyperparam_config, f, indent=2)

        print(f"   Saved hyperparam config: {config_file}")
        return config_file

    def load_baseline_results(self) -> Dict[str, Dict]:
        """Load baseline results from evaluation"""
        baseline_file = Path("tototraining/baseline_results.json")

        if not baseline_file.exists():
            print(f"Warning: Baseline results not found at {baseline_file}")
            return {}

        with open(baseline_file, 'r') as f:
            return json.load(f)

    def train_all_stocks(
        self,
        data_dir: Path = Path("trainingdata"),
        stocks: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """Train models for all stocks or specified subset"""

        # Load baseline to get optimal configs
        baseline = self.load_baseline_results()

        if not baseline:
            print("ERROR: Run baseline_eval_simple.py first to get baseline metrics!")
            return {}

        # Get list of stocks to train
        if stocks is None:
            stocks = list(baseline.keys())

        print(f"\n{'='*100}")
        print(f"TRAINING {len(stocks)} STOCK-SPECIFIC TOTO MODELS")
        print(f"{'='*100}\n")

        results = {}

        for symbol in stocks:
            # Get baseline info
            stock_baseline = baseline.get(symbol, {})
            num_samples = stock_baseline.get("count", 1000)
            baseline_mae_pct = stock_baseline.get("h64_pct", 15.0)

            # Get optimal config
            config = self.get_optimal_config_for_stock(
                symbol,
                num_samples,
                baseline_mae_pct
            )

            # Train file
            train_file = data_dir / f"{symbol}.csv"
            if not train_file.exists():
                print(f"⚠️  Skipping {symbol}: {train_file} not found")
                continue

            # Train model
            success, metrics, model_dir = self.train_stock_model(config, train_file)

            if success:
                # Create hyperparam config for comparison
                hyperparam_file = self.create_hyperparam_config(
                    symbol, config, metrics, model_dir
                )

                # Store results
                results[symbol] = {
                    "success": True,
                    "config": asdict(config),
                    "metrics": metrics,
                    "model_dir": str(model_dir),
                    "hyperparam_file": str(hyperparam_file),
                    "baseline_mae_pct": baseline_mae_pct,
                }

                # Compare to baseline
                if "final_val_mape" in metrics:
                    improvement = ((baseline_mae_pct - metrics["final_val_mape"])
                                  / baseline_mae_pct * 100)
                    results[symbol]["improvement_pct"] = improvement
                    print(f"   📊 Baseline: {baseline_mae_pct:.2f}% → Model: {metrics['final_val_mape']:.2f}%")
                    print(f"   {'✅' if improvement > 0 else '❌'} Improvement: {improvement:+.1f}%\n")
            else:
                results[symbol] = {
                    "success": False,
                    "error": metrics.get("error", "unknown"),
                }

        # Save overall results
        summary_file = self.output_root / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*100}")
        print("TRAINING SUMMARY")
        print(f"{'='*100}")
        successful = sum(1 for r in results.values() if r.get("success"))
        print(f"Successful: {successful}/{len(results)}")
        print(f"Results saved to: {summary_file}")
        print(f"{'='*100}\n")

        return results


def train_priority_stocks(retrainer: TotoRetrainer):
    """Train high-priority stocks first (easy + high data)"""

    priority_stocks = [
        # Easy stocks (good for validation)
        "SPY", "MSFT", "AAPL", "QQQ", "GOOG",
        # High data stocks (good for training)
        "NVDA", "AMD", "META", "TSLA",
        # Crypto (interesting comparisons)
        "BTCUSD", "ETHUSD",
    ]

    print("Training priority stocks...")
    return retrainer.train_all_stocks(stocks=priority_stocks)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Retrain Toto models for stock-specific optimization")
    parser.add_argument("--stocks", nargs="+", help="Specific stocks to train (default: all)")
    parser.add_argument("--priority-only", action="store_true",
                       help="Train only priority stocks")
    parser.add_argument("--output-dir", type=Path, default=Path("tototraining/stock_models"),
                       help="Output directory for trained models")
    parser.add_argument("--hyperparam-dir", type=Path, default=Path("hyperparams/toto"),
                       help="Directory for hyperparameter configs")

    args = parser.parse_args()

    # Create retrainer
    retrainer = TotoRetrainer(
        output_root=args.output_dir,
        hyperparam_root=args.hyperparam_dir
    )

    # Train models
    if args.priority_only:
        results = train_priority_stocks(retrainer)
    else:
        results = retrainer.train_all_stocks(stocks=args.stocks)

    # Print summary
    print("\n" + "="*100)
    print("NEXT STEPS")
    print("="*100)
    print("1. Models saved to:", args.output_dir)
    print("2. Hyperparameter configs saved to:", args.hyperparam_dir)
    print("3. Run comparison:")
    print(f"   python test_kronos_vs_toto.py --symbol [STOCK] --forecast-horizon 64")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()
