"""Run multiple pretrain-finetune experiments sequentially."""
from __future__ import annotations
import json
import time
from pathlib import Path

from experiments.pretrain_finetune.run import run_experiment, RESULTS_DIR


EXPERIMENTS = [
    # Cross-target: pretrain each symbol on the other 2 targets
    {"pool": "cross_target", "pretrain_epochs": 15, "finetune_epochs": 20, "finetune_lr": 1e-4},
    # Cross-target with more finetune epochs
    {"pool": "cross_target", "pretrain_epochs": 10, "finetune_epochs": 30, "finetune_lr": 1e-4},
    # Cross-target with lower LR
    {"pool": "cross_target", "pretrain_epochs": 15, "finetune_epochs": 20, "finetune_lr": 5e-5},
]


def main():
    all_results = []
    for i, exp in enumerate(EXPERIMENTS):
        print(f"\n{'#'*60}")
        print(f"Experiment {i+1}/{len(EXPERIMENTS)}: {exp}")
        print(f"{'#'*60}")
        try:
            result = run_experiment(**exp)
            all_results.append(result)
            ret = result.get("selector", {}).get("total_return", "N/A")
            print(f"=> Selector return: {ret}")
        except Exception as e:
            print(f"Experiment failed: {e}")
            all_results.append({"error": str(e), **exp})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    summary_path = RESULTS_DIR / f"batch_summary_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nBatch summary: {summary_path}")

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"  BASELINE: ft30 direct = 2478x")
    print(f"  highvol pt15+ft20 lr=1e-4 = 55x (BAD)")
    for r in all_results:
        pool = r.get("pool", "?")
        pe = r.get("pretrain_epochs", "?")
        fe = r.get("finetune_epochs", "?")
        lr = r.get("finetune_lr", "?")
        sel = r.get("selector", {})
        ret = sel.get("total_return", "ERR")
        sortino = sel.get("sortino", "ERR")
        print(f"  {pool} pt={pe} ft={fe} lr={lr}: return={ret} sortino={sortino}")


if __name__ == "__main__":
    main()
