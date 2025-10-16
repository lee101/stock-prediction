#!/usr/bin/env python3
"""
Compare one-batch training + simulated PnL between:
- training (PPO over DailyTradingEnv, synthetic data)
- hftraining (HF transformer, profit-tracker simulation)

Prints a concise comparison so we can see which makes more money.
"""

import json
from typing import Dict, Any

from training.single_batch_example import run_single_batch_training
from hftraining.single_batch_hf import run_single_batch_hf


def main() -> None:
    print("=== Running classic training single batch ===")
    _, _, _, classic_metrics = run_single_batch_training()

    print("\n=== Running HF training single batch ===")
    hf_metrics = run_single_batch_hf()

    # Normalize/align key fields for comparison
    classic_summary = {
        'engine': 'classic_env_ppo',
        'total_return': float(classic_metrics.get('total_return', 0.0)),
        'sharpe_ratio': float(classic_metrics.get('sharpe_ratio', 0.0)),
        'max_drawdown': float(classic_metrics.get('max_drawdown', 0.0)),
        'win_rate': float(classic_metrics.get('win_rate', 0.0)),
        'num_trades': int(classic_metrics.get('num_trades', 0)),
    }

    hf_summary = {
        'engine': 'hf_transformer',
        'total_return': float(hf_metrics.get('total_return', 0.0)),
        'sharpe_ratio': float(hf_metrics.get('sharpe_ratio', 0.0)),
        'max_drawdown': float(hf_metrics.get('max_drawdown', 0.0)),
        'win_rate': float(hf_metrics.get('win_rate', 0.0)),
        'num_trades': int(hf_metrics.get('total_trades', 0)),
    }

    print("\n=== Single-batch PnL Comparison ===")
    print(json.dumps({'classic': classic_summary, 'hf': hf_summary}, indent=2))

    # Determine which "made more money" by total_return
    winner = max((classic_summary, hf_summary), key=lambda m: m['total_return'])
    print(f"\nWinner by total_return: {winner['engine']} ({winner['total_return']:.2%})")


if __name__ == '__main__':
    main()

