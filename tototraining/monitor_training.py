#!/usr/bin/env python3
"""
Monitor ongoing training progress
"""

import time
from pathlib import Path
import json


def monitor_latest_experiment(checkpoints_dir: Path = Path("tototraining/checkpoints/quick")):
    """Monitor the latest experiment"""

    if not checkpoints_dir.exists():
        print(f"Checkpoints directory not found: {checkpoints_dir}")
        return

    # Find latest experiment
    experiments = sorted(checkpoints_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)

    if not experiments:
        print("No experiments found")
        return

    latest = experiments[0]
    print(f"Monitoring: {latest.name}")
    print("="*80)

    # Monitor files
    config_file = latest / "config.json"
    metrics_file = latest / "metrics.json"
    output_file = latest / "training_output.txt"

    # Show config
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        print("\nConfiguration:")
        for key, value in config.items():
            print(f"  {key}: {value}")

    print("\n" + "="*80)
    print("Waiting for training output...")
    print("="*80 + "\n")

    # Tail the output file
    last_size = 0
    while True:
        if output_file.exists():
            current_size = output_file.stat().st_size

            if current_size > last_size:
                with open(output_file, 'r') as f:
                    f.seek(last_size)
                    new_content = f.read()
                    print(new_content, end='')
                last_size = current_size

            # Check if training is done
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)

                if metrics.get("final_val_mape") or metrics.get("final_val_loss"):
                    print("\n" + "="*80)
                    print("Training Complete!")
                    print("="*80)
                    print(f"Final Val Loss: {metrics.get('final_val_loss', 'N/A')}")
                    print(f"Final Val MAPE: {metrics.get('final_val_mape', 'N/A')}")
                    print(f"Best Epoch: {metrics.get('best_epoch', 'N/A')}")
                    print("="*80)
                    break

        time.sleep(2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("tototraining/checkpoints/quick"))

    args = parser.parse_args()

    try:
        monitor_latest_experiment(args.dir)
    except KeyboardInterrupt:
        print("\nMonitoring stopped")
