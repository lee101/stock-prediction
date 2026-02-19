#!/usr/bin/env python3
"""Full stock-only training experiment with architecture sweep + backtest."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from loguru import logger

STOCKS = "NVDA,MSFT,META,GOOG,NET,PLTR,NYT,YELP,DBX,TRIP,KIND,EBAY,MTCH,ANGI,Z,EXPE,BKNG,NWSA"
CHECKPOINT_ROOT = Path("unified_hourly_experiment/checkpoints")


def train_and_backtest(cfg: dict, stocks: str = STOCKS) -> dict:
    name = cfg["name"]
    logger.info(f"=== Training {name} ===")

    train_cmd = [
        sys.executable, "unified_hourly_experiment/train_unified_policy.py",
        "--symbols", stocks,
        "--crypto-symbols", "",
        "--epochs", str(cfg["epochs"]),
        "--hidden-dim", str(cfg["hidden"]),
        "--num-layers", str(cfg["layers"]),
        "--num-heads", str(cfg["heads"]),
        "--batch-size", str(cfg.get("batch_size", 64)),
        "--lr", str(cfg.get("lr", 1e-4)),
        "--sequence-length", str(cfg.get("seq_len", 32)),
        "--checkpoint-name", name,
    ]

    result = subprocess.run(train_cmd, capture_output=True, text=True, timeout=14400)

    val_sortino = None
    val_return = None
    for line in (result.stdout + result.stderr).split("\n"):
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
        if "val_sortino" in line.lower() and val_sortino is None:
            try:
                val_sortino = float(line.split("val_sortino")[1].split(",")[0].strip().strip(":= "))
            except:
                pass

    logger.info(f"  Train done: val_sortino={val_sortino}, val_return={val_return}")

    bt_results = []
    for min_edge in [0.0, 0.001, 0.005, 0.01, 0.012]:
        bt_cmd = [
            sys.executable, "unified_hourly_experiment/backtest_unified.py",
            "--checkpoint-dir", str(CHECKPOINT_ROOT / name),
            "--symbols", stocks,
            "--min-edge", str(min_edge),
        ]
        bt_result = subprocess.run(bt_cmd, capture_output=True, text=True, timeout=600)
        bt_return = None
        bt_sortino = None
        for line in (bt_result.stdout + bt_result.stderr).split("\n"):
            if "Total return:" in line:
                try:
                    bt_return = float(line.split(":")[1].replace("%", "").strip())
                except:
                    pass
            if "sortino:" in line.lower():
                try:
                    bt_sortino = float(line.split(":")[-1].strip())
                except:
                    pass
        bt_results.append({
            "min_edge": min_edge,
            "return": bt_return,
            "sortino": bt_sortino,
        })
        if bt_return is not None:
            logger.info(f"  Backtest min_edge={min_edge}: return={bt_return:.2f}%, sortino={bt_sortino}")

    return {
        "name": name,
        "config": cfg,
        "val_sortino": val_sortino,
        "val_return": val_return,
        "backtests": bt_results,
        "train_success": result.returncode == 0,
    }


CONFIGS = [
    {"epochs": 200, "hidden": 512, "layers": 4, "heads": 8, "name": "exp_512h_4L_baseline"},
    {"epochs": 400, "hidden": 512, "layers": 4, "heads": 8, "name": "exp_512h_4L_400ep"},
    {"epochs": 200, "hidden": 768, "layers": 4, "heads": 8, "name": "exp_768h_4L_stockonly"},
    {"epochs": 300, "hidden": 512, "layers": 4, "heads": 8, "lr": 5e-5, "name": "exp_512h_4L_lr5e5"},
    {"epochs": 200, "hidden": 512, "layers": 4, "heads": 8, "seq_len": 64, "name": "exp_512h_4L_seq64"},
    {"epochs": 200, "hidden": 512, "layers": 6, "heads": 8, "name": "exp_512h_6L"},
    {"epochs": 200, "hidden": 1024, "layers": 4, "heads": 16, "name": "exp_1024h_4L"},
    {"epochs": 200, "hidden": 1024, "layers": 6, "heads": 16, "name": "exp_1024h_6L"},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", help="Config names to run (default: all)")
    parser.add_argument("--stocks", default=STOCKS)
    args = parser.parse_args()

    if args.configs:
        configs = [c for c in CONFIGS if c["name"] in args.configs]
    else:
        configs = CONFIGS

    logger.info(f"Running {len(configs)} experiments")
    results = []

    for cfg in configs:
        try:
            res = train_and_backtest(cfg, args.stocks)
            results.append(res)
        except Exception as e:
            logger.error(f"Failed {cfg['name']}: {e}")
            results.append({"name": cfg["name"], "error": str(e)})

        out_path = Path("unified_hourly_experiment/experiment_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    logger.info("=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    for r in results:
        if "error" in r:
            logger.error(f"  {r['name']}: FAILED - {r['error']}")
        else:
            best_bt = max(r.get("backtests", [{}]), key=lambda x: x.get("sortino") or 0, default={})
            logger.info(f"  {r['name']}: val_sortino={r.get('val_sortino')}, bt_return={best_bt.get('return')}%, bt_sortino={best_bt.get('sortino')}")


if __name__ == "__main__":
    main()
