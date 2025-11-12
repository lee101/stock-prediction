#!/usr/bin/env python3
"""
Best Model Selector for Inference

Automatically selects the best trained model across Toto, Kronos, and Chronos2
based on validation pct_mae for use in forecasting/trading.

Usage:
    # Get best model overall:
    python select_best_model.py

    # Get best Toto model:
    python select_best_model.py --model toto

    # Get top 3 models:
    python select_best_model.py --top-k 3

    # Interactive selection:
    python select_best_model.py --interactive
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List
from tabulate import tabulate

from hparams_tracker import HyperparamTracker, HyperparamRun


def display_model_info(run: HyperparamRun, rank: Optional[int] = None):
    """Display detailed model information"""
    prefix = f"#{rank} " if rank else ""

    print(f"\n{prefix}{'=' * 70}")
    print(f"{run.model_name.upper()} - {run.run_id}")
    print("=" * 70)

    print(f"\n📊 METRICS:")
    metrics_to_show = [
        ("Val pct_MAE", "val_pct_mae"),
        ("Test pct_MAE", "test_pct_mae"),
        ("Val R²", "val_r2"),
        ("Test R²", "test_r2"),
        ("Val Price MAE", "val_price_mae"),
        ("Test Price MAE", "test_price_mae"),
    ]

    for label, key in metrics_to_show:
        value = run.metrics.get(key)
        if value is not None:
            if "mae" in key.lower():
                print(f"  {label:20s}: {value:.4f}")
            elif "r2" in key.lower():
                print(f"  {label:20s}: {value:.4f}")
            else:
                print(f"  {label:20s}: {value:.6f}")

    print(f"\n⚙️  KEY HYPERPARAMETERS:")
    # Show most important hyperparams
    important_params = [
        "learning_rate", "context_length", "prediction_length",
        "batch_size", "epochs", "max_epochs", "loss", "loss_type"
    ]
    for param in important_params:
        if param in run.hyperparams:
            print(f"  {param:20s}: {run.hyperparams[param]}")

    print(f"\n📁 CHECKPOINT:")
    print(f"  {run.checkpoint_path or 'N/A'}")

    if run.notes:
        print(f"\n📝 NOTES:")
        print(f"  {run.notes}")

    print()


def select_best_model_cli(args):
    """Command-line model selection"""
    tracker = HyperparamTracker(args.tracker_db)

    if not tracker.runs:
        print("❌ No trained models found in tracker database.")
        print(f"   Database: {args.tracker_db}")
        print("\n💡 Run some training first:")
        print("   python run_sweep.py --model toto --mode priority --max-runs 3")
        return None

    print("=" * 80)
    print("MODEL SELECTOR - Finding Best Trained Models")
    print("=" * 80)
    print(f"Database: {args.tracker_db}")
    print(f"Total runs: {len(tracker.runs)}")
    print()

    # Get best models
    if args.top_k > 1:
        print(f"🏆 TOP {args.top_k} MODELS (by {args.metric}):\n")
        best_models = tracker.get_top_k_models(
            k=args.top_k,
            metric=args.metric,
            model_name=args.model,
            minimize=args.minimize
        )

        if not best_models:
            print(f"❌ No models found with metric '{args.metric}'")
            return None

        for i, run in enumerate(best_models, 1):
            display_model_info(run, rank=i)

        return best_models[0] if best_models else None

    else:
        print(f"🏆 BEST MODEL (by {args.metric}):\n")
        best = tracker.get_best_model(
            metric=args.metric,
            model_name=args.model,
            minimize=args.minimize,
            require_checkpoint=True
        )

        if not best:
            print(f"❌ No models found with metric '{args.metric}'")
            return None

        display_model_info(best)
        return best


def select_best_model_interactive(args):
    """Interactive model selection"""
    tracker = HyperparamTracker(args.tracker_db)

    if not tracker.runs:
        print("❌ No trained models found.")
        return None

    print("=" * 80)
    print("INTERACTIVE MODEL SELECTOR")
    print("=" * 80)

    # Filter by model type
    print("\n1️⃣  Select model type:")
    print("  1. Toto")
    print("  2. Kronos")
    print("  3. Chronos2")
    print("  4. All models")

    choice = input("\nChoice [1-4]: ").strip()
    model_filter = None
    if choice == "1":
        model_filter = "toto"
    elif choice == "2":
        model_filter = "kronos"
    elif choice == "3":
        model_filter = "chronos2"

    # Get candidates
    runs = tracker.get_runs(model_name=model_filter)
    runs = [r for r in runs if r.checkpoint_path is not None]

    if not runs:
        print("❌ No models with checkpoints found.")
        return None

    # Sort by val_pct_mae
    runs = sorted(runs, key=lambda r: r.metrics.get("val_pct_mae", float('inf')))
    runs = runs[:10]  # Show top 10

    print(f"\n2️⃣  Select from top {len(runs)} models:")
    print()

    # Display table
    table_data = []
    for i, run in enumerate(runs, 1):
        table_data.append([
            i,
            run.model_name,
            run.run_id[:20] + "...",
            f"{run.metrics.get('val_pct_mae', float('inf')):.4f}",
            f"{run.metrics.get('test_pct_mae', float('inf')):.4f}",
            f"{run.metrics.get('val_r2', float('-inf')):.2f}",
        ])

    headers = ["#", "Model", "Run ID", "Val MAE", "Test MAE", "Val R²"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    choice = input(f"\nSelect model [1-{len(runs)}] or 'q' to quit: ").strip()

    if choice.lower() == 'q':
        return None

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(runs):
            selected = runs[idx]
            display_model_info(selected)
            return selected
        else:
            print("❌ Invalid choice")
            return None
    except ValueError:
        print("❌ Invalid input")
        return None


def export_best_model_path(run: HyperparamRun, output_file: str = ".best_model_path"):
    """Export best model path to file for easy loading"""
    if run and run.checkpoint_path:
        with open(output_file, 'w') as f:
            f.write(run.checkpoint_path)
        print(f"\n✅ Best model path exported to: {output_file}")
        print(f"   Load with: model_path = open('{output_file}').read().strip()")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        choices=["toto", "kronos", "chronos2"],
        help="Filter by model type"
    )
    parser.add_argument(
        "--metric",
        default="val_pct_mae",
        help="Metric to optimize (default: val_pct_mae)"
    )
    parser.add_argument(
        "--minimize",
        action="store_true",
        default=True,
        help="Minimize metric (default: True for MAE)"
    )
    parser.add_argument(
        "--maximize",
        dest="minimize",
        action="store_false",
        help="Maximize metric (use for R2, accuracy, etc.)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Show top K models"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive selection mode"
    )
    parser.add_argument(
        "--tracker-db",
        default="hyperparams/sweep_results.json",
        help="Path to tracker database"
    )
    parser.add_argument(
        "--export-path",
        action="store_true",
        help="Export best model path to .best_model_path file"
    )

    args = parser.parse_args()

    # Select model
    if args.interactive:
        best_run = select_best_model_interactive(args)
    else:
        best_run = select_best_model_cli(args)

    # Export if requested
    if best_run and args.export_path:
        export_best_model_path(best_run)

    return 0 if best_run else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
        sys.exit(1)
