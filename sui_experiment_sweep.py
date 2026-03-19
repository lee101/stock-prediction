#!/usr/bin/env python3
"""Sweep experiments for SUI: longer training + bitbank-style + optimizations."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List

from loguru import logger

EXPERIMENTS = [
    # Longer training with current architecture
    {"name": "longer_20ep", "type": "longer", "epochs": 20},
    {"name": "longer_40ep", "type": "longer", "epochs": 40},
    {"name": "longer_60ep", "type": "longer", "epochs": 60},

    # Bitbank-style experiments
    {"name": "bitbank_base", "type": "bitbank", "epochs": 30, "hidden_dim": 256, "n_layers": 4, "aggressiveness": 0.8},
    {"name": "bitbank_deep", "type": "bitbank", "epochs": 30, "hidden_dim": 256, "n_layers": 6, "aggressiveness": 0.8},
    {"name": "bitbank_wide", "type": "bitbank", "epochs": 30, "hidden_dim": 512, "n_layers": 4, "aggressiveness": 0.8},
    {"name": "bitbank_aggr_low", "type": "bitbank", "epochs": 30, "hidden_dim": 256, "n_layers": 4, "aggressiveness": 0.6},
    {"name": "bitbank_aggr_high", "type": "bitbank", "epochs": 30, "hidden_dim": 256, "n_layers": 4, "aggressiveness": 0.9},
    {"name": "bitbank_long", "type": "bitbank", "epochs": 60, "hidden_dim": 256, "n_layers": 4, "aggressiveness": 0.8},
]


def run_experiment(exp: dict) -> dict:
    run_id = time.strftime("%Y%m%d_%H%M%S")

    if exp["type"] == "longer":
        cmd = [
            sys.executable, "-m", "longertrainsui.run_experiment",
            "--epochs", str(exp["epochs"]),
            "--run-name", f"{exp['name']}_{run_id}",
        ]
    else:  # bitbank
        cmd = [
            sys.executable, "-m", "bitbankstylelongsuitrain.run_experiment",
            "--epochs", str(exp["epochs"]),
            "--hidden-dim", str(exp.get("hidden_dim", 256)),
            "--n-layers", str(exp.get("n_layers", 4)),
            "--aggressiveness", str(exp.get("aggressiveness", 0.8)),
            "--run-name", f"{exp['name']}_{run_id}",
        ]

    logger.info(f"Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)

    # Parse output for results
    output = proc.stdout + proc.stderr
    result = {"name": exp["name"], "config": exp, "output": output[-2000:]}

    # Try to extract metrics from output
    for line in output.split("\n"):
        if "7d Return:" in line:
            try:
                result["return"] = float(line.split(":")[1].strip().split()[0])
            except:
                pass
        if "Sortino:" in line and "Val" not in line:
            try:
                result["sortino"] = float(line.split(":")[1].strip())
            except:
                pass

    return result


def main():
    results: List[dict] = []

    # Run baseline comparison first
    logger.info("Running baseline (10 epoch) for comparison...")
    baseline = run_experiment({"name": "baseline_10ep", "type": "longer", "epochs": 10})
    results.append(baseline)

    # Run all experiments
    for exp in EXPERIMENTS:
        logger.info(f"Running experiment: {exp['name']}")
        try:
            result = run_experiment(exp)
            results.append(result)
            logger.info(f"  Result: return={result.get('return', 'N/A')}, sortino={result.get('sortino', 'N/A')}")
        except Exception as e:
            logger.error(f"  Failed: {e}")
            results.append({"name": exp["name"], "error": str(e)})

    # Summary
    print("\n" + "="*80)
    print("SUI Experiment Sweep Results")
    print("="*80)

    sorted_results = sorted(
        [r for r in results if "return" in r],
        key=lambda x: x.get("return", -999),
        reverse=True
    )

    for r in sorted_results:
        print(f"{r['name']:25s}: return={r.get('return', 'N/A'):+.4f}, sortino={r.get('sortino', 'N/A'):8.2f}")

    # Save
    output = Path("reports/sui_sweep_results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"Saved sweep results to {output}")


if __name__ == "__main__":
    main()
