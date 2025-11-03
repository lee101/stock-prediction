#!/usr/bin/env python3
"""
Train Toto models on priority stocks with proper sequence lengths.
Uses configurations that ensure (context_length + prediction_length) is divisible by patch_size (64).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Priority stocks for quick iteration
PRIORITY_STOCKS = [
    "SPY", "QQQ", "MSFT", "AAPL", "GOOG",
    "NVDA", "AMD", "META", "TSLA", "BTCUSD", "ETHUSD"
]

# Valid configurations (total must be divisible by 64)
# NOTE: actual_time_steps = (context - 32) + prediction due to internal overlap
# So to get divisible-by-64 totals, we need adjusted context values
VALID_CONFIGS = {
    "tiny": {"context": 96, "prediction": 64},  # 96-32+64 = 128 total
    "small": {"context": 160, "prediction": 64},  # 160-32+64 = 192 total
    "medium": {"context": 288, "prediction": 64},  # 288-32+64 = 320 total
    "large": {"context": 416, "prediction": 64},  # 416-32+64 = 448 total
    "xlarge": {"context": 480, "prediction": 64},  # 480-32+64 = 512 total
}


def get_stock_sample_count(stock: str) -> int:
    """Get number of samples for a stock from baseline"""
    baseline_file = Path("tototraining/baseline_results.json")

    if not baseline_file.exists():
        return 1000  # Default

    with open(baseline_file, 'r') as f:
        baselines = json.load(f)

    return baselines.get(stock, {}).get("count", 1000)


def select_config_for_stock(stock: str) -> dict:
    """Select appropriate config based on data size"""
    count = get_stock_sample_count(stock)

    if count < 500:
        return VALID_CONFIGS["tiny"]  # Use tiny for small datasets
    elif count < 1000:
        return VALID_CONFIGS["small"]
    elif count < 1500:
        return VALID_CONFIGS["medium"]
    else:
        return VALID_CONFIGS["large"]


def train_stock(
    stock: str,
    epochs: int = 15,
    learning_rate: float = 3e-4,
    loss: str = "huber",
    batch_size: int = 4,
    lora_rank: int = 8,
    config_size: str = None,
    use_lora: bool = False,  # Disable LoRA by default to avoid device issues
):
    """Train a single stock"""

    # Select config
    if config_size:
        config = VALID_CONFIGS[config_size]
    else:
        config = select_config_for_stock(stock)

    context_length = config["context"]
    prediction_length = config["prediction"]
    # NOTE: Model applies (context - 32 + prediction) internally
    actual_time_steps = (context_length - 32) + prediction_length

    print(f"\n{'='*100}")
    print(f"Training {stock}")
    print(f"{'='*100}")
    print(f"Context: {context_length}, Prediction: {prediction_length}")
    print(f"Actual time steps (model will see): {actual_time_steps}")
    print(f"Time steps/64 = {actual_time_steps/64} (must be integer)")
    print(f"Epochs: {epochs}, LR: {learning_rate}, Loss: {loss}")
    print(f"LoRA Rank: {lora_rank}, Batch Size: {batch_size}")
    print(f"{'='*100}\n")

    # Verify actual time steps is divisible by 64
    assert actual_time_steps % 64 == 0, f"Actual time steps {actual_time_steps} not divisible by 64!"

    # Check if training data exists
    train_file = Path(f"trainingdata/{stock}.csv")
    if not train_file.exists():
        print(f"❌ Training data not found: {train_file}")
        return {"status": "error", "error": "missing_data"}

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"tototraining/priority_models/{stock}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    # Use stride = actual_time_steps to avoid partial windows
    stride = actual_time_steps  # This ensures non-overlapping windows that are all complete

    cmd = [
        "uv", "run", "python", "tototraining/train.py",
        "--train-root", str(train_file),
        "--val-root", str(train_file),
        "--context-length", str(context_length),
        "--prediction-length", str(prediction_length),
        "--stride", str(stride),  # Set stride to avoid partial windows
        "--batch-size", str(batch_size),
        "--epochs", str(epochs),
        "--learning-rate", str(learning_rate),
        "--loss", loss,
        "--weight-decay", "0.01",
        "--clip-grad", "1.0",
        "--precision", "bf16",
        "--output-dir", str(output_dir),
        "--checkpoint-name", f"{stock}_model",
        "--log-interval", "10",
        "--compile", "false",  # Disable compile to avoid device issues
    ]

    # Add LoRA if requested
    if use_lora:
        cmd.extend([
            "--adapter", "lora",
            "--adapter-r", str(lora_rank),
            "--adapter-alpha", str(lora_rank * 2),
            "--freeze-backbone",
        ])

    if loss == "huber":
        cmd.extend(["--huber-delta", "0.01"])

    # Save config
    config_data = {
        "stock": stock,
        "context_length": context_length,
        "prediction_length": prediction_length,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "loss": loss,
        "batch_size": batch_size,
        "lora_rank": lora_rank,
        "timestamp": timestamp,
    }

    with open(output_dir / "config.json", 'w') as f:
        json.dump(config_data, f, indent=2)

    # Run training
    try:
        print(f"Running: {' '.join(str(c) for c in cmd)}\n")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        # Save output
        with open(output_dir / "training_output.txt", 'w') as f:
            f.write(result.stdout)
            f.write("\n" + "="*80 + "\n")
            f.write(result.stderr)

        # Check for success
        if result.returncode == 0:
            print(f"✅ {stock} training completed successfully!")
            return {"status": "success", "output_dir": str(output_dir)}
        else:
            print(f"❌ {stock} training failed with return code {result.returncode}")
            # Print last 50 lines of error
            error_lines = (result.stdout + result.stderr).split('\n')
            print("\nLast 50 lines of output:")
            print('\n'.join(error_lines[-50:]))
            return {"status": "error", "error": "training_failed", "returncode": result.returncode}

    except subprocess.TimeoutExpired:
        print(f"❌ {stock} training timed out after 1 hour!")
        return {"status": "error", "error": "timeout"}
    except Exception as e:
        print(f"❌ {stock} training failed with exception: {e}")
        return {"status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Train Toto models on priority stocks")
    parser.add_argument("--stocks", nargs="+", default=PRIORITY_STOCKS,
                       help="Stocks to train (default: all priority stocks)")
    parser.add_argument("--epochs", type=int, default=15,
                       help="Number of epochs (default: 15)")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate (default: 3e-4)")
    parser.add_argument("--loss", type=str, default="huber",
                       choices=["huber", "mse", "heteroscedastic"],
                       help="Loss function (default: huber)")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size (default: 4)")
    parser.add_argument("--lora-rank", type=int, default=8,
                       help="LoRA rank (default: 8)")
    parser.add_argument("--use-lora", action="store_true",
                       help="Enable LoRA (disabled by default due to device issues)")
    parser.add_argument("--config-size", type=str, choices=list(VALID_CONFIGS.keys()),
                       help="Force specific config size (default: auto-select based on data)")

    args = parser.parse_args()

    print("\n" + "="*100)
    print("TOTO PRIORITY STOCK TRAINING")
    print("="*100)
    print(f"Stocks to train: {', '.join(args.stocks)}")
    print(f"Config: {args.epochs} epochs, LR={args.lr}, loss={args.loss}")
    print(f"LoRA rank: {args.lora_rank}, Batch size: {args.batch_size}")
    print("="*100 + "\n")

    results = {}

    for stock in args.stocks:
        result = train_stock(
            stock=stock,
            epochs=args.epochs,
            learning_rate=args.lr,
            loss=args.loss,
            batch_size=args.batch_size,
            lora_rank=args.lora_rank,
            config_size=args.config_size,
            use_lora=args.use_lora,
        )
        results[stock] = result

    # Save summary
    summary_file = Path("tototraining/priority_training_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*100)
    print("TRAINING SUMMARY")
    print("="*100)

    successes = [s for s, r in results.items() if r.get("status") == "success"]
    failures = [s for s, r in results.items() if r.get("status") != "success"]

    print(f"✅ Successful: {len(successes)}/{len(results)}")
    if successes:
        print(f"   {', '.join(successes)}")

    print(f"❌ Failed: {len(failures)}/{len(results)}")
    if failures:
        print(f"   {', '.join(failures)}")

    print(f"\nResults saved to: {summary_file}")
    print("="*100 + "\n")

    return results


if __name__ == "__main__":
    main()
