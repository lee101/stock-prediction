#!/usr/bin/env python3
"""Neural Architecture Search for stock trading policy."""
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from loguru import logger
import json

STOCKS = "NVDA,MSFT,META,GOOG,NET,PLTR,NYT,YELP,DBX,TRIP,KIND,EBAY,MTCH,ANGI,Z,EXPE,BKNG,NWSA"
CHECKPOINT_ROOT = Path("unified_hourly_experiment/checkpoints")

CONFIGS = [
    {"epochs": 200, "hidden": 128, "layers": 3, "heads": 4, "name": "nas_128h_3L"},
    {"epochs": 200, "hidden": 256, "layers": 3, "heads": 4, "name": "nas_256h_3L"},
    {"epochs": 200, "hidden": 256, "layers": 4, "heads": 4, "name": "nas_256h_4L"},
    {"epochs": 200, "hidden": 256, "layers": 6, "heads": 4, "name": "nas_256h_6L"},
    {"epochs": 200, "hidden": 512, "layers": 3, "heads": 8, "name": "nas_512h_3L"},
    {"epochs": 200, "hidden": 512, "layers": 4, "heads": 8, "name": "nas_512h_4L"},
    {"epochs": 300, "hidden": 256, "layers": 4, "heads": 4, "name": "nas_256h_4L_300ep"},
    {"epochs": 200, "hidden": 384, "layers": 4, "heads": 6, "name": "nas_384h_4L"},
]

def train_config(cfg):
    """Train a single config and return metrics."""
    cmd = [
        sys.executable, "unified_hourly_experiment/train_unified_policy.py",
        "--symbols", STOCKS,
        "--crypto-symbols", "",
        "--epochs", str(cfg["epochs"]),
        "--hidden-dim", str(cfg["hidden"]),
        "--num-layers", str(cfg["layers"]),
        "--num-heads", str(cfg["heads"]),
        "--checkpoint-name", cfg["name"],
    ]

    logger.info(f"Training {cfg['name']}: {cfg['epochs']}ep, {cfg['hidden']}h, {cfg['layers']}L")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

    val_sortino = None
    val_return = None
    for line in result.stderr.split("\n"):
        if "Val Sortino:" in line:
            try:
                val_sortino = float(line.split("Val Sortino:")[1].split(",")[0].strip())
            except:
                pass
        if "Val Return:" in line:
            try:
                val_return = float(line.split("Val Return:")[1].split("%")[0].strip())
            except:
                pass

    return {
        "name": cfg["name"],
        "config": cfg,
        "val_sortino": val_sortino,
        "val_return": val_return,
        "success": result.returncode == 0,
    }

def run_backtest(checkpoint_name):
    """Run backtest for a checkpoint."""
    cmd = [
        sys.executable, "unified_hourly_experiment/backtest_unified.py",
        "--checkpoint-dir", str(CHECKPOINT_ROOT / checkpoint_name),
        "--symbols", STOCKS,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    bt_return = None
    bt_sortino = None
    for line in result.stderr.split("\n"):
        if "Total return:" in line:
            try:
                bt_return = float(line.split(":")[1].replace("%", "").strip())
            except:
                pass
        if "sortino:" in line.lower():
            try:
                bt_sortino = float(line.split(":")[1].strip())
            except:
                pass

    return {"bt_return": bt_return, "bt_sortino": bt_sortino}

def main():
    logger.info("=" * 60)
    logger.info("Neural Architecture Search")
    logger.info("=" * 60)

    results = []
    best_sortino = 0
    best_config = None

    for cfg in CONFIGS:
        res = train_config(cfg)
        results.append(res)

        if res["val_sortino"] and res["val_sortino"] > best_sortino:
            best_sortino = res["val_sortino"]
            best_config = cfg["name"]

        logger.info(f"  -> Val Sortino: {res['val_sortino']}, Val Return: {res['val_return']}")

        # Save intermediate results
        with open("unified_hourly_experiment/nas_results.json", "w") as f:
            json.dump(results, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Best: {best_config} with Val Sortino {best_sortino}")
    logger.info("=" * 60)

    # Run backtest on best
    if best_config:
        logger.info(f"Running backtest on {best_config}")
        bt = run_backtest(best_config)
        logger.info(f"Backtest: Return={bt['bt_return']}%, Sortino={bt['bt_sortino']}")

if __name__ == "__main__":
    main()
