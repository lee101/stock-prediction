#!/usr/bin/env python3
"""Sweep best SAP configs across ALL Binance hourly pairs.

Trains each symbol with the top configs found so far, saves results
to a combined leaderboard CSV.

Usage:
    python -m sharpnessadjustedproximalpolicy2.sweep_all_pairs
    python -m sharpnessadjustedproximalpolicy2.sweep_all_pairs --epochs 15 --top-k 3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .config import SAPConfig
from .sweep import build_configs, run_single_experiment

# Best configs from sweeps so far
TOP_CONFIGS = [
    {"name": "baseline_wd01", "sam_mode": "none", "weight_decay": 0.1},
    {"name": "periodic_wd01", "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.1},
    {"name": "periodic_wd005", "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.05},
    {"name": "baseline_adamw", "sam_mode": "none"},
]


def get_trainable_symbols():
    """Find all symbols with hourly data AND at least h1+h24 forecast cache."""
    data_root = Path("trainingdatahourly") / "crypto"
    cache_root = Path("binanceneural") / "forecast_cache"
    symbols = []
    for csv_path in sorted(data_root.glob("*USD.csv")):
        sym = csv_path.stem
        has_h1 = (cache_root / "h1" / f"{sym}.parquet").exists()
        has_h24 = (cache_root / "h24" / f"{sym}.parquet").exists()
        if has_h1 and has_h24:
            symbols.append(sym)
    return symbols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--top-k", type=int, default=4, help="how many configs to test per symbol")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--configs", type=str, default=None, help="comma-sep config names")
    args = parser.parse_args()

    symbols = args.symbols or get_trainable_symbols()
    configs = TOP_CONFIGS[:args.top_k]
    if args.configs:
        names = set(args.configs.split(","))
        configs = [c for c in TOP_CONFIGS if c["name"] in names]

    print(f"Symbols ({len(symbols)}): {', '.join(symbols)}", flush=True)
    print(f"Configs ({len(configs)}): {', '.join(c['name'] for c in configs)}", flush=True)
    print(f"Epochs: {args.epochs}", flush=True)
    print(f"Total runs: {len(symbols) * len(configs)}", flush=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_json = Path(f"sharpnessadjustedproximalpolicy2/allpairs_results_{timestamp}.json")
    output_csv = Path(f"sharpnessadjustedproximalpolicy2/allpairs_leaderboard_{timestamp}.csv")
    output_json.parent.mkdir(parents=True, exist_ok=True)

    all_results = []
    overrides = {"epochs": args.epochs}

    for si, symbol in enumerate(symbols):
        for ci, config in enumerate(configs):
            idx = si * len(configs) + ci + 1
            total = len(symbols) * len(configs)
            print(f"\n[{idx}/{total}] {config['name']} / {symbol}", flush=True)

            result = run_single_experiment(config, symbol, overrides)
            all_results.append(result)

            # save incrementally
            output_json.write_text(json.dumps(all_results, indent=2))

            # write CSV leaderboard
            with open(output_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["symbol", "config", "best_epoch", "val_sortino", "val_return", "val_score", "sharpness_ema", "step_scale", "error"])
                for r in sorted(all_results, key=lambda x: x.get("best_val_sortino", -999), reverse=True):
                    writer.writerow([
                        r.get("symbol", ""),
                        r.get("name", ""),
                        r.get("best_epoch", ""),
                        f"{r.get('best_val_sortino', 0):.3f}" if not r.get("error") else "",
                        f"{r.get('best_val_return', 0):.4f}" if not r.get("error") else "",
                        f"{r.get('best_val_score', 0):.3f}" if not r.get("error") else "",
                        f"{r.get('final_sharpness_ema', 0):.1f}" if not r.get("error") else "",
                        f"{r.get('final_step_scale', r.get('final_lr_scale', 1)):.2f}" if not r.get("error") else "",
                        r.get("error", ""),
                    ])

    # final summary
    print(f"\n{'='*90}", flush=True)
    print(f"ALL PAIRS SWEEP COMPLETE: {len(all_results)} runs", flush=True)
    print(f"Results: {output_json}", flush=True)
    print(f"Leaderboard: {output_csv}", flush=True)

    # per-symbol best
    print(f"\n{'Symbol':<12} {'Best Config':<25} {'Sort':>8} {'Ret':>10} {'Ep':>4}", flush=True)
    print("-" * 65, flush=True)
    by_symbol = {}
    for r in all_results:
        if r.get("error"):
            continue
        sym = r["symbol"]
        if sym not in by_symbol or r.get("best_val_sortino", -999) > by_symbol[sym].get("best_val_sortino", -999):
            by_symbol[sym] = r
    for sym in sorted(by_symbol.keys()):
        r = by_symbol[sym]
        print(f"{sym:<12} {r['name']:<25} {r['best_val_sortino']:>8.3f} {r['best_val_return']:>10.4f} {r['best_epoch']:>4d}", flush=True)

    # SAM vs baseline summary
    sam_wins = 0
    total_pairs = 0
    for sym in by_symbol:
        sym_results = [r for r in all_results if r["symbol"] == sym and not r.get("error")]
        baseline = [r for r in sym_results if r.get("sam_mode") == "none"]
        sam = [r for r in sym_results if r.get("sam_mode") != "none"]
        if baseline and sam:
            total_pairs += 1
            best_baseline = max(r.get("best_val_sortino", -999) for r in baseline)
            best_sam = max(r.get("best_val_sortino", -999) for r in sam)
            if best_sam > best_baseline:
                sam_wins += 1

    if total_pairs > 0:
        print(f"\nSAM wins: {sam_wins}/{total_pairs} symbols ({100*sam_wins/total_pairs:.0f}%)", flush=True)


if __name__ == "__main__":
    main()
