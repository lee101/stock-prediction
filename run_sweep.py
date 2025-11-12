#!/usr/bin/env python3
"""
Automated Hyperparameter Sweep Runner

Runs systematic hyperparameter sweeps across Toto, Kronos, and Chronos2,
logging all results to the unified tracker for easy comparison.

Usage:
    # Run priority configs for Toto:
    python run_sweep.py --model toto --mode priority --max-runs 3

    # Run full sweep for Kronos:
    python run_sweep.py --model kronos --mode full --max-runs 20

    # Run quick test sweep:
    python run_sweep.py --model toto --mode quick --max-runs 5
"""

import argparse
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from hparams_tracker import HyperparamTracker
from hyperparams.sweep_configs import (
    generate_sweep_configs,
    get_priority_configs,
    TOTO_SWEEP_GRID,
    TOTO_QUICK_SWEEP,
    KRONOS_SWEEP_GRID,
    KRONOS_QUICK_SWEEP,
    CHRONOS2_SWEEP_GRID,
    CHRONOS2_QUICK_SWEEP,
    format_config_for_cli,
)


def run_toto_training(config: Dict[str, Any], tracker: HyperparamTracker) -> str:
    """Run Toto training with given config and log results"""

    # Build command
    cmd = [
        ".venv/bin/python",
        "tototraining/run_gpu_training.py",
    ]

    # Add config as CLI args
    cli_args = format_config_for_cli(config, "toto")
    cmd.extend(cli_args)

    # Add run name
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_name = f"sweep_toto_{timestamp}"
    cmd.extend(["--run-name", run_name])

    print(f"\n🚀 Starting Toto training: {run_name}")
    print(f"   Config: {config}")
    print(f"   Command: {' '.join(cmd)}")

    # Run training
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout
        )
        training_time = time.time() - start_time

        # Parse metrics from output
        metrics = parse_toto_metrics(result.stdout)

        # Find checkpoint
        checkpoint_pattern = f"tototraining/checkpoints/gpu_run/*{run_name}*/best/rank1_*.pt"
        checkpoints = list(Path(".").glob(checkpoint_pattern))
        checkpoint_path = str(checkpoints[0]) if checkpoints else None

        # Log to tracker
        run_id = tracker.log_run(
            model_name="toto",
            hyperparams=config,
            metrics=metrics,
            checkpoint_path=checkpoint_path,
            training_time_seconds=training_time,
            notes=f"Automated sweep run"
        )

        print(f"✅ Training completed: {run_id}")
        print(f"   Val pct_MAE: {metrics.get('val_pct_mae', 'N/A'):.4f}")
        print(f"   Test pct_MAE: {metrics.get('test_pct_mae', 'N/A'):.4f}")

        return run_id

    except subprocess.TimeoutExpired:
        print(f"❌ Training timed out after 2 hours")
        return None
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None


def parse_toto_metrics(output: str) -> Dict[str, float]:
    """Parse metrics from Toto training output"""
    metrics = {}

    # Look for FINAL_VAL_METRICS and FINAL_TEST_METRICS lines
    for line in output.split('\n'):
        if 'FINAL_VAL_METRICS' in line:
            # Extract JSON-like dict
            try:
                import ast
                dict_str = line.split('FINAL_VAL_METRICS')[1].strip()
                val_metrics = ast.literal_eval(dict_str)
                metrics['val_pct_mae'] = val_metrics.get('pct_mae', float('inf'))
                metrics['val_price_mae'] = val_metrics.get('price_mae', float('inf'))
                metrics['val_r2'] = val_metrics.get('pct_r2', float('-inf'))
                metrics['val_naive_mae'] = val_metrics.get('naive_mae', float('inf'))
            except Exception:
                pass

        if 'FINAL_TEST_METRICS' in line:
            try:
                import ast
                dict_str = line.split('FINAL_TEST_METRICS')[1].strip()
                test_metrics = ast.literal_eval(dict_str)
                metrics['test_pct_mae'] = test_metrics.get('pct_mae', float('inf'))
                metrics['test_price_mae'] = test_metrics.get('price_mae', float('inf'))
                metrics['test_r2'] = test_metrics.get('pct_r2', float('-inf'))
                metrics['test_naive_mae'] = test_metrics.get('naive_mae', float('inf'))
            except Exception:
                pass

    return metrics


def run_kronos_training(config: Dict[str, Any], tracker: HyperparamTracker) -> str:
    """Run Kronos training - placeholder for now"""
    print(f"⚠️  Kronos training not yet implemented")
    return None


def run_chronos2_training(config: Dict[str, Any], tracker: HyperparamTracker) -> str:
    """Run Chronos2 training - placeholder for now"""
    print(f"⚠️  Chronos2 training not yet implemented")
    return None


def run_sweep(
    model_name: str,
    mode: str,
    max_runs: int,
    tracker: HyperparamTracker
):
    """Run hyperparameter sweep"""

    print("=" * 80)
    print(f"HYPERPARAMETER SWEEP: {model_name.upper()}")
    print(f"Mode: {mode}")
    print(f"Max runs: {max_runs}")
    print("=" * 80)

    # Get configs
    if mode == "priority":
        configs = get_priority_configs(model_name)
    elif mode == "quick":
        if model_name == "toto":
            configs = generate_sweep_configs(TOTO_QUICK_SWEEP, max_configs=max_runs)
        elif model_name == "kronos":
            configs = generate_sweep_configs(KRONOS_QUICK_SWEEP, max_configs=max_runs)
        elif model_name == "chronos2":
            configs = generate_sweep_configs(CHRONOS2_QUICK_SWEEP, max_configs=max_runs)
    elif mode == "full":
        if model_name == "toto":
            configs = generate_sweep_configs(TOTO_SWEEP_GRID, max_configs=max_runs, random_sample=True)
        elif model_name == "kronos":
            configs = generate_sweep_configs(KRONOS_SWEEP_GRID, max_configs=max_runs, random_sample=True)
        elif model_name == "chronos2":
            configs = generate_sweep_configs(CHRONOS2_SWEEP_GRID, max_configs=max_runs, random_sample=True)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    configs = configs[:max_runs]

    print(f"\n📋 Running {len(configs)} configurations...")

    # Run each config
    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n{'=' * 80}")
        print(f"RUN {i}/{len(configs)}")
        print(f"{'=' * 80}")

        if model_name == "toto":
            run_id = run_toto_training(config, tracker)
        elif model_name == "kronos":
            run_id = run_kronos_training(config, tracker)
        elif model_name == "chronos2":
            run_id = run_chronos2_training(config, tracker)
        else:
            run_id = None

        if run_id:
            results.append(run_id)

        # Small delay between runs
        time.sleep(5)

    # Generate summary report
    print(f"\n{'=' * 80}")
    print(f"SWEEP COMPLETE")
    print(f"{'=' * 80}")
    print(f"Completed {len(results)}/{len(configs)} runs")

    # Show best result
    best = tracker.get_best_model(metric="val_pct_mae", model_name=model_name)
    if best:
        print(f"\n🏆 BEST MODEL:")
        print(f"   Run ID: {best.run_id}")
        print(f"   Val pct_MAE: {best.metrics.get('val_pct_mae', 'N/A'):.4f}")
        print(f"   Test pct_MAE: {best.metrics.get('test_pct_mae', 'N/A'):.4f}")
        print(f"   Checkpoint: {best.checkpoint_path}")

    # Generate report
    report_path = f"hyperparams/sweep_report_{model_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md"
    tracker.generate_report(report_path)
    print(f"\n📄 Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["toto", "kronos", "chronos2"],
        required=True,
        help="Model to sweep"
    )
    parser.add_argument(
        "--mode",
        choices=["priority", "quick", "full"],
        default="priority",
        help="Sweep mode"
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=5,
        help="Maximum number of runs"
    )
    parser.add_argument(
        "--tracker-db",
        default="hyperparams/sweep_results.json",
        help="Path to tracker database"
    )

    args = parser.parse_args()

    # Initialize tracker
    tracker = HyperparamTracker(args.tracker_db)

    # Run sweep
    run_sweep(
        model_name=args.model,
        mode=args.mode,
        max_runs=args.max_runs,
        tracker=tracker
    )


if __name__ == "__main__":
    main()
