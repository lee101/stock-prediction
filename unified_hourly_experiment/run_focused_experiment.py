#!/usr/bin/env python3
"""Focused stock subset experiments: train on best-performing symbols only."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from loguru import logger

CHECKPOINT_ROOT = Path("unified_hourly_experiment/checkpoints")

# Top performers from per-symbol analysis
FOCUSED_4 = "NVDA,GOOG,EBAY,PLTR"
# Extended set: add other large-cap tech
FOCUSED_8 = "NVDA,MSFT,META,GOOG,EBAY,PLTR,NET,BKNG"
# Longable only (no short)
LONGABLE = "NVDA,MSFT,META,GOOG,NET,PLTR,DBX"
# Shortable only
SHORTABLE = "YELP,EBAY,TRIP,MTCH,KIND,ANGI,Z,EXPE,BKNG,NWSA,NYT"
# All 18
ALL = "NVDA,MSFT,META,GOOG,NET,PLTR,NYT,YELP,DBX,TRIP,KIND,EBAY,MTCH,ANGI,Z,EXPE,BKNG,NWSA"


def train_and_sweep(name: str, symbols: str, epochs: int = 200, hidden: int = 512,
                    layers: int = 4, heads: int = 8, lr: float = 1e-4, seq_len: int = 32):
    logger.info(f"=== {name}: {symbols} ({epochs}ep) ===")

    train_cmd = [
        sys.executable, "unified_hourly_experiment/train_unified_policy.py",
        "--symbols", symbols, "--crypto-symbols", "",
        "--epochs", str(epochs), "--hidden-dim", str(hidden),
        "--num-layers", str(layers), "--num-heads", str(heads),
        "--batch-size", "64", "--lr", str(lr),
        "--sequence-length", str(seq_len), "--checkpoint-name", name,
    ]
    result = subprocess.run(train_cmd, capture_output=True, text=True, timeout=14400)

    val_sortino = None
    for line in (result.stdout + result.stderr).split("\n"):
        if "Val Sortino:" in line:
            try: val_sortino = float(line.split("Val Sortino:")[1].split(",")[0].strip())
            except: pass

    logger.info(f"  Train: val_sortino={val_sortino}, exit={result.returncode}")

    bt_results = []
    for min_edge in [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01]:
        bt_cmd = [
            sys.executable, "unified_hourly_experiment/backtest_unified.py",
            "--checkpoint-dir", str(CHECKPOINT_ROOT / name),
            "--symbols", symbols,
            "--min-edge", str(min_edge),
        ]
        bt_result = subprocess.run(bt_cmd, capture_output=True, text=True, timeout=600)
        bt_return = bt_sortino = None
        for line in (bt_result.stdout + bt_result.stderr).split("\n"):
            if "Total return:" in line:
                try: bt_return = float(line.split(":")[1].replace("%", "").strip())
                except: pass
            if "sortino:" in line.lower():
                try: bt_sortino = float(line.split(":")[-1].strip())
                except: pass
        bt_results.append({"min_edge": min_edge, "return": bt_return, "sortino": bt_sortino})
        if bt_return is not None:
            logger.info(f"  BT min_edge={min_edge}: ret={bt_return:.2f}%, sortino={bt_sortino}")

    return {
        "name": name, "symbols": symbols, "epochs": epochs,
        "val_sortino": val_sortino, "backtests": bt_results,
        "train_success": result.returncode == 0,
    }


CONFIGS = [
    ("focused_4_200ep", FOCUSED_4, 200),
    ("focused_4_400ep", FOCUSED_4, 400),
    ("focused_8_200ep", FOCUSED_8, 200),
    ("longable_200ep", LONGABLE, 200),
    ("shortable_200ep", SHORTABLE, 200),
    ("all18_400ep", ALL, 400),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", help="Config names to run")
    args = parser.parse_args()

    configs = CONFIGS
    if args.configs:
        configs = [c for c in configs if c[0] in args.configs]

    logger.info(f"Running {len(configs)} focused experiments")
    results = []
    for name, symbols, epochs in configs:
        try:
            res = train_and_sweep(name, symbols, epochs=epochs)
            results.append(res)
        except Exception as e:
            logger.error(f"Failed {name}: {e}")
            results.append({"name": name, "error": str(e)})

        with open("unified_hourly_experiment/focused_experiment_results.json", "w") as f:
            json.dump(results, f, indent=2)

    logger.info("=" * 60)
    for r in results:
        if "error" in r:
            logger.error(f"  {r['name']}: FAILED")
        else:
            best = max(r.get("backtests", [{}]), key=lambda x: x.get("sortino") or 0, default={})
            logger.info(f"  {r['name']}: val_s={r.get('val_sortino')}, bt_ret={best.get('return')}%, bt_s={best.get('sortino')}")


if __name__ == "__main__":
    main()
