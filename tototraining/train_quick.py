#!/usr/bin/env python3
"""
Quick training script for rapid iteration on stock prediction
Optimized for the RTX 3090 with sensible defaults
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def create_quick_config(
    stock: str,
    learning_rate: float = 3e-4,
    loss: str = "huber",
    epochs: int = 10,
    batch_size: int = 4,
    context_length: int = 4096,
    prediction_length: int = 64,
    use_lora: bool = True,
):
    """Create a training configuration"""
    return {
        "stock": stock,
        "learning_rate": learning_rate,
        "loss": loss,
        "epochs": epochs,
        "batch_size": batch_size,
        "context_length": context_length,
        "prediction_length": prediction_length,
        "use_lora": use_lora,
        "huber_delta": 0.01 if loss == "huber" else None,
        "weight_decay": 1e-2,
        "grad_clip": 1.0,
        "precision": "bf16",
        "compile": True,
    }


def run_training(config: dict, output_dir: Path = None):
    """Run a training experiment with given config"""

    import subprocess

    stock = config["stock"]
    train_file = Path(f"trainingdata/{stock}.csv")

    if not train_file.exists():
        print(f"Error: {train_file} not found!")
        return None

    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"tototraining/checkpoints/quick/{stock}_{timestamp}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_file = output_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    # Build training command
    cmd = [
        "uv", "run", "python", "tototraining/train.py",
        "--train-root", str(train_file),
        "--val-root", str(train_file),  # Use same for quick validation
        "--context-length", str(config["context_length"]),
        "--prediction-length", str(config["prediction_length"]),
        "--batch-size", str(config["batch_size"]),
        "--epochs", str(config["epochs"]),
        "--learning-rate", str(config["learning_rate"]),
        "--loss", config["loss"],
        "--weight-decay", str(config["weight_decay"]),
        "--clip-grad", str(config["grad_clip"]),
        "--precision", config.get("precision", "bf16"),
        "--output-dir", str(output_dir),
        "--checkpoint-name", f"{stock}_model",
        "--log-interval", "20",
    ]

    if config.get("huber_delta"):
        cmd.extend(["--huber-delta", str(config["huber_delta"])])

    if config.get("use_lora", False):
        cmd.extend([
            "--adapter", "lora",
            "--adapter-r", "8",
            "--adapter-alpha", "16.0",
            "--freeze-backbone",
        ])

    if not config.get("compile", True):
        cmd.extend(["--compile", "false"])

    print("\n" + "="*100)
    print(f"TRAINING: {stock}")
    print("="*100)
    print(f"Config: {json.dumps(config, indent=2)}")
    print(f"Output: {output_dir}")
    print(f"Command: {' '.join(str(c) for c in cmd)}")
    print("="*100 + "\n")

    # Run training
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

        # Save output
        output_file = output_dir / "training_output.txt"
        with open(output_file, 'w') as f:
            f.write(result.stdout)
            f.write("\n" + "="*80 + "\n")
            f.write(result.stderr)

        # Parse results
        metrics = parse_training_output(result.stdout + result.stderr)

        # Save metrics
        metrics_file = output_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        print("\n" + "="*100)
        print("TRAINING COMPLETE")
        print("="*100)
        print(f"Final Val Loss: {metrics.get('final_val_loss', 'N/A')}")
        print(f"Final Val MAPE: {metrics.get('final_val_mape', 'N/A')}")
        print(f"Best Epoch: {metrics.get('best_epoch', 'N/A')}")
        print("="*100 + "\n")

        return metrics

    except subprocess.TimeoutExpired:
        print("Training timed out!")
        return {"error": "timeout"}
    except Exception as e:
        print(f"Training failed: {e}")
        return {"error": str(e)}


def parse_training_output(output: str) -> dict:
    """Parse key metrics from training output"""
    metrics = {
        "train_losses": [],
        "val_losses": [],
        "val_mapes": [],
    }

    for line in output.split('\n'):
        # Parse epoch results
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
                best_epoch = int(line.split('epoch')[1].split('.')[0])
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


def load_baseline(stock: str) -> dict:
    """Load baseline metrics for comparison"""
    baseline_file = Path("tototraining/baseline_results.json")

    if not baseline_file.exists():
        return {}

    with open(baseline_file, 'r') as f:
        baselines = json.load(f)

    return baselines.get(stock, {})


def compare_to_baseline(stock: str, val_mape: float):
    """Compare results to baseline"""
    baseline = load_baseline(stock)

    if not baseline:
        print(f"No baseline found for {stock}")
        return

    baseline_mape = baseline.get("h64_pct", 0)

    print("\n" + "="*100)
    print("BASELINE COMPARISON")
    print("="*100)
    print(f"Stock: {stock}")
    print(f"Naive Baseline MAE%: {baseline_mape:.2f}%")
    print(f"Model Val MAPE: {val_mape:.2f}%")

    if val_mape < baseline_mape:
        improvement = ((baseline_mape - val_mape) / baseline_mape) * 100
        print(f"✅ IMPROVED by {improvement:.1f}% relative!")
    else:
        degradation = ((val_mape - baseline_mape) / baseline_mape) * 100
        print(f"❌ WORSE by {degradation:.1f}% relative")

    print("="*100 + "\n")


def run_experiment_grid(stocks: list, configs: list):
    """Run grid of experiments across stocks and configs"""

    all_results = []

    for stock in stocks:
        for config_params in configs:
            config = create_quick_config(stock, **config_params)

            print(f"\n\n{'#'*100}")
            print(f"# Experiment: {stock} - {config_params}")
            print(f"{'#'*100}\n")

            metrics = run_training(config)

            if metrics and "final_val_mape" in metrics:
                compare_to_baseline(stock, metrics["final_val_mape"])

            all_results.append({
                "stock": stock,
                "config": config_params,
                "metrics": metrics,
            })

    # Save all results
    results_file = Path("tototraining/experiment_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*100)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*100)
    print(f"Results saved to: {results_file}")
    print("="*100 + "\n")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Quick training experiments")
    parser.add_argument("--stock", type=str, default="SPY",
                       help="Stock symbol to train on")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--loss", type=str, default="huber",
                       choices=["huber", "mse", "heteroscedastic", "quantile"],
                       help="Loss function")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--no-lora", action="store_true",
                       help="Disable LoRA (train full model)")
    parser.add_argument("--grid", action="store_true",
                       help="Run grid search on easy stocks")

    args = parser.parse_args()

    if args.grid:
        # Run grid search on easy stocks
        stocks = ["SPY", "MSFT", "AAPL"]
        configs = [
            {"learning_rate": 1e-4, "loss": "huber", "epochs": 8},
            {"learning_rate": 3e-4, "loss": "huber", "epochs": 8},
            {"learning_rate": 5e-4, "loss": "huber", "epochs": 8},
            {"learning_rate": 3e-4, "loss": "heteroscedastic", "epochs": 8},
            {"learning_rate": 3e-4, "loss": "mse", "epochs": 8},
        ]
        run_experiment_grid(stocks, configs)

    else:
        # Run single experiment
        config = create_quick_config(
            stock=args.stock,
            learning_rate=args.lr,
            loss=args.loss,
            epochs=args.epochs,
            batch_size=args.batch_size,
            use_lora=not args.no_lora,
        )

        metrics = run_training(config)

        if metrics and "final_val_mape" in metrics:
            compare_to_baseline(args.stock, metrics["final_val_mape"])


if __name__ == "__main__":
    main()
