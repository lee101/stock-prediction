#!/usr/bin/env python3
"""Sweep training hyperparameters for portfolio policy to find best sortino."""
import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

CONFIGS = [
    # Baseline (deployed): return_weight=0.08, smooth=0, fill_temp=5e-4, val_days=30
    # Sweep return_weight
    {"name": "rw0.02", "return_weight": 0.02, "smoothness": 0.0, "fill_temp": 5e-4, "epochs": 15, "optimizer": "adamw"},
    {"name": "rw0.15", "return_weight": 0.15, "smoothness": 0.0, "fill_temp": 5e-4, "epochs": 15, "optimizer": "adamw"},
    {"name": "rw0.30", "return_weight": 0.30, "smoothness": 0.0, "fill_temp": 5e-4, "epochs": 15, "optimizer": "adamw"},
    # Sweep smoothness
    {"name": "smooth0.01", "return_weight": 0.08, "smoothness": 0.01, "fill_temp": 5e-4, "epochs": 15, "optimizer": "adamw"},
    {"name": "smooth0.05", "return_weight": 0.08, "smoothness": 0.05, "fill_temp": 5e-4, "epochs": 15, "optimizer": "adamw"},
    # Sweep fill_temperature
    {"name": "filltemp1e-3", "return_weight": 0.08, "smoothness": 0.0, "fill_temp": 1e-3, "epochs": 15, "optimizer": "adamw"},
    {"name": "filltemp1e-4", "return_weight": 0.08, "smoothness": 0.0, "fill_temp": 1e-4, "epochs": 15, "optimizer": "adamw"},
    # Combined best-guess
    {"name": "rw0.15_smooth0.01", "return_weight": 0.15, "smoothness": 0.01, "fill_temp": 5e-4, "epochs": 15, "optimizer": "adamw"},
    {"name": "rw0.02_smooth0.05", "return_weight": 0.02, "smoothness": 0.05, "fill_temp": 5e-4, "epochs": 15, "optimizer": "adamw"},
    # Lower LR (slower convergence, maybe less overfitting)
    {"name": "lr1e-4_rw0.08", "return_weight": 0.08, "smoothness": 0.0, "fill_temp": 5e-4, "epochs": 20, "optimizer": "adamw", "lr": 1e-4},
    # Longer validation
    {"name": "val60d_rw0.08", "return_weight": 0.08, "smoothness": 0.0, "fill_temp": 5e-4, "epochs": 15, "optimizer": "adamw", "val_days": 60},
]

RESULTS_FILE = Path("unified_hourly_experiment/hparam_sweep_results.json")


def run_config(cfg):
    name = cfg["name"]
    epochs = cfg["epochs"]
    rw = cfg["return_weight"]
    smooth = cfg["smoothness"]
    ft = cfg["fill_temp"]
    opt = cfg["optimizer"]
    lr = cfg.get("lr", 3e-4)
    val_days = cfg.get("val_days", 30)

    run_name = f"sweep_{name}"

    cmd = [
        sys.executable, "unified_hourly_experiment/train_portfolio_policy.py",
        "--epochs", str(epochs),
        "--batch-size", "16",
        "--sequence-length", "512",
        "--hidden-dim", "512",
        "--num-layers", "6",
        "--num-heads", "8",
        "--num-outputs", "6",
        "--forecast-horizons", "1",
        "--run-name", run_name,
        "--validation-days", str(val_days),
        "--maker-fee", "0.001",
        "--return-weight", str(rw),
        "--smoothness-penalty", str(smooth),
        "--fill-temperature", str(ft),
        "--optimizer", opt,
        "--learning-rate", str(lr),
        "--seed", "1337",
        "--no-compile",
    ]

    print(f"\n{'='*60}")
    print(f"Config: {name}")
    print(f"  rw={rw} smooth={smooth} fill_temp={ft} opt={opt} lr={lr} val_days={val_days}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        output = result.stdout + result.stderr

        # Parse best results from training meta
        meta_path = Path(f"unified_hourly_experiment/checkpoints/{run_name}/training_meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            history = meta.get("history", [])
            if history:
                best = max(history, key=lambda h: h.get("val_sortino", 0) or 0)
                return {
                    "name": name,
                    "best_epoch": best["epoch"],
                    "val_sortino": best.get("val_sortino"),
                    "val_return": best.get("val_return"),
                    "train_sortino": best.get("train_sortino"),
                    "config": cfg,
                    "status": "ok",
                }

        # Fallback: parse from stdout
        for line in output.split("\n"):
            if "val_sortino" in line.lower():
                print(f"  {line.strip()}")

        return {"name": name, "status": "no_meta", "output_tail": output[-500:]}

    except subprocess.TimeoutExpired:
        return {"name": name, "status": "timeout"}
    except Exception as e:
        return {"name": name, "status": "error", "error": str(e)}


def main():
    results = []
    if RESULTS_FILE.exists():
        results = json.loads(RESULTS_FILE.read_text())
    done_names = {r["name"] for r in results}

    for cfg in CONFIGS:
        if cfg["name"] in done_names:
            print(f"Skipping {cfg['name']} (already done)")
            continue

        r = run_config(cfg)
        results.append(r)
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_FILE.write_text(json.dumps(results, indent=2))

        if r.get("val_sortino"):
            print(f"  -> ep{r['best_epoch']}: sortino={r['val_sortino']:.2f} return={r['val_return']:.4f}")
        else:
            print(f"  -> {r.get('status', 'unknown')}")

    # Summary
    print("\n" + "=" * 70)
    print(f"{'Name':<25} {'Epoch':>5} {'Sortino':>8} {'Return':>8}")
    print("-" * 50)
    for r in sorted(results, key=lambda x: x.get("val_sortino", 0) or 0, reverse=True):
        if r.get("val_sortino"):
            print(f"{r['name']:<25} {r['best_epoch']:>5} {r['val_sortino']:>8.2f} {r['val_return']:>8.4f}")
        else:
            print(f"{r['name']:<25} {'':>5} {r.get('status', 'N/A'):>8}")


if __name__ == "__main__":
    main()
