#!/usr/bin/env python3
"""
Analyze and compare training results across experiments
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List


def load_baseline() -> Dict:
    """Load baseline results"""
    baseline_file = Path("tototraining/baseline_results.json")

    if not baseline_file.exists():
        return {}

    with open(baseline_file, 'r') as f:
        return json.load(f)


def find_all_experiments(checkpoints_dir: Path = Path("tototraining/checkpoints/quick")) -> List[Path]:
    """Find all experiment directories"""
    if not checkpoints_dir.exists():
        return []

    experiments = []
    for exp_dir in checkpoints_dir.iterdir():
        if exp_dir.is_dir() and (exp_dir / "metrics.json").exists():
            experiments.append(exp_dir)

    return sorted(experiments)


def load_experiment(exp_dir: Path) -> Dict:
    """Load experiment results"""
    result = {
        "dir": str(exp_dir),
        "name": exp_dir.name,
    }

    # Load config
    config_file = exp_dir / "config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            result["config"] = json.load(f)

    # Load metrics
    metrics_file = exp_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            result["metrics"] = json.load(f)

    return result


def compare_experiments(experiments: List[Dict], baseline: Dict):
    """Compare experiments and show best results"""

    print("\n" + "="*120)
    print("EXPERIMENT COMPARISON")
    print("="*120)

    # Group by stock
    by_stock = defaultdict(list)
    for exp in experiments:
        stock = exp.get("config", {}).get("stock", "unknown")
        by_stock[stock].append(exp)

    # For each stock, show results
    for stock, stock_exps in sorted(by_stock.items()):
        print(f"\nStock: {stock}")
        print("-" * 120)

        baseline_mape = baseline.get(stock, {}).get("h64_pct", 0)
        print(f"Baseline MAE%: {baseline_mape:.2f}%")

        # Show each experiment
        print(f"\n{'Experiment':<40} | {'Loss':<15} | {'LR':<10} | {'Val MAPE':<12} | {'Improvement':<15} | {'Status':<10}")
        print("-" * 120)

        for exp in stock_exps:
            name = exp["name"]
            config = exp.get("config", {})
            metrics = exp.get("metrics", {})

            loss = config.get("loss", "?")
            lr = config.get("learning_rate", 0)
            val_mape = metrics.get("final_val_mape", metrics.get("min_val_mape", float('inf')))

            if val_mape == float('inf'):
                status = "FAILED"
                improvement = "-"
            elif baseline_mape > 0:
                improvement_pct = ((baseline_mape - val_mape) / baseline_mape) * 100
                if val_mape < baseline_mape:
                    status = "✅ BETTER"
                    improvement = f"{improvement_pct:+.1f}%"
                else:
                    status = "❌ WORSE"
                    improvement = f"{improvement_pct:+.1f}%"
            else:
                status = "?"
                improvement = "?"

            print(f"{name[:40]:<40} | {loss:<15} | {lr:<10.2e} | {val_mape:<12.2f} | {improvement:<15} | {status:<10}")

        print("-" * 120)

    print("=" * 120 + "\n")


def find_best_configs(experiments: List[Dict], baseline: Dict) -> Dict:
    """Find best configuration for each stock"""

    best_by_stock = {}

    # Group by stock
    by_stock = defaultdict(list)
    for exp in experiments:
        stock = exp.get("config", {}).get("stock", "unknown")
        by_stock[stock].append(exp)

    # Find best for each stock
    for stock, stock_exps in by_stock.items():
        valid_exps = [
            exp for exp in stock_exps
            if "final_val_mape" in exp.get("metrics", {}) or "min_val_mape" in exp.get("metrics", {})
        ]

        if not valid_exps:
            continue

        # Find experiment with lowest val_mape
        best = min(
            valid_exps,
            key=lambda x: x.get("metrics", {}).get("final_val_mape", x.get("metrics", {}).get("min_val_mape", float('inf')))
        )

        best_by_stock[stock] = {
            "config": best.get("config", {}),
            "metrics": best.get("metrics", {}),
            "experiment": best["name"],
        }

    return best_by_stock


def print_best_configs(best_configs: Dict, baseline: Dict):
    """Print best configurations summary"""

    print("\n" + "="*120)
    print("BEST CONFIGURATIONS PER STOCK")
    print("="*120)

    print(f"\n{'Stock':<10} | {'Best MAPE':<12} | {'Baseline':<12} | {'Improvement':<15} | {'Loss':<15} | {'LR':<10} | {'Experiment':<30}")
    print("-" * 120)

    for stock, best in sorted(best_configs.items()):
        val_mape = best["metrics"].get("final_val_mape", best["metrics"].get("min_val_mape", 0))
        baseline_mape = baseline.get(stock, {}).get("h64_pct", 0)

        if baseline_mape > 0:
            improvement = ((baseline_mape - val_mape) / baseline_mape) * 100
            improvement_str = f"{improvement:+.1f}%"
        else:
            improvement_str = "?"

        loss = best["config"].get("loss", "?")
        lr = best["config"].get("learning_rate", 0)
        exp_name = best["experiment"]

        print(f"{stock:<10} | {val_mape:<12.2f} | {baseline_mape:<12.2f} | {improvement_str:<15} | {loss:<15} | {lr:<10.2e} | {exp_name:<30}")

    print("=" * 120 + "\n")


def save_best_configs(best_configs: Dict, output_file: Path = Path("tototraining/best_configs.json")):
    """Save best configurations"""
    with open(output_file, 'w') as f:
        json.dump(best_configs, f, indent=2)

    print(f"Best configurations saved to: {output_file}\n")


def print_summary_stats(experiments: List[Dict], baseline: Dict):
    """Print summary statistics"""

    print("\n" + "="*120)
    print("SUMMARY STATISTICS")
    print("="*120)

    total = len(experiments)
    successful = sum(1 for exp in experiments if "final_val_mape" in exp.get("metrics", {}) or "min_val_mape" in exp.get("metrics", {}))
    failed = total - successful

    print(f"\nTotal experiments: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if successful > 0:
        # Count improvements
        better_than_baseline = 0
        total_improvement = 0

        for exp in experiments:
            stock = exp.get("config", {}).get("stock", "")
            val_mape = exp.get("metrics", {}).get("final_val_mape", exp.get("metrics", {}).get("min_val_mape", float('inf')))
            baseline_mape = baseline.get(stock, {}).get("h64_pct", 0)

            if val_mape < float('inf') and baseline_mape > 0:
                if val_mape < baseline_mape:
                    better_than_baseline += 1
                    improvement = ((baseline_mape - val_mape) / baseline_mape) * 100
                    total_improvement += improvement

        print(f"\nBetter than baseline: {better_than_baseline}/{successful} ({better_than_baseline/successful*100:.1f}%)")

        if better_than_baseline > 0:
            print(f"Average improvement: {total_improvement/better_than_baseline:.1f}%")

    print("=" * 120 + "\n")


def main():
    # Load baseline
    baseline = load_baseline()

    if not baseline:
        print("No baseline results found. Run baseline_eval_simple.py first.")
        return

    # Find all experiments
    experiments_raw = find_all_experiments()

    if not experiments_raw:
        print("No experiments found in tototraining/checkpoints/quick/")
        return

    # Load experiment details
    experiments = [load_experiment(exp_dir) for exp_dir in experiments_raw]

    # Print comparison
    compare_experiments(experiments, baseline)

    # Find and print best configs
    best_configs = find_best_configs(experiments, baseline)
    print_best_configs(best_configs, baseline)

    # Save best configs
    save_best_configs(best_configs)

    # Print summary
    print_summary_stats(experiments, baseline)


if __name__ == "__main__":
    main()
