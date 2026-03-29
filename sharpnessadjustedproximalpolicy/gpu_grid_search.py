#!/usr/bin/env python3
"""Fast GPU grid search over SAPP hyperparameters.

Trains many configs in parallel (sequential on single GPU, batched where possible).
Uses continuous_val_eval for honest checkpoint selection.
Evaluates full-period marketsim for the final score.

Usage:
    python -m sharpnessadjustedproximalpolicy.gpu_grid_search --symbols DOTUSD BTCUSD --budget 3600
    python -m sharpnessadjustedproximalpolicy.gpu_grid_search --full-universe --budget 7200
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from binanceneural.config import DatasetConfig, TrainingConfig
from binanceneural.data import BinanceHourlyDataModule

from .config import DEFAULT_TRAINING_OVERRIDES, SAPConfig
from .evaluate_marketsim import chunk_based_eval, summarize_windows
from .trainer import SAPTrainer

# Hyperparameter grid -- all combinations tested
GRID = {
    "learning_rate": [1e-4, 3e-4, 5e-4],
    "weight_decay": [0.04, 0.1, 0.15],
    "sam_mode": ["none", "periodic"],
    "spectral_reg_weight": [0.0, 1e-3],
    "epochs": [10, 15, 20],
}

# Compact grid for fast runs
GRID_FAST = {
    "learning_rate": [3e-4, 5e-4],
    "weight_decay": [0.1],
    "sam_mode": ["periodic"],
    "spectral_reg_weight": [0.0, 1e-3],
    "epochs": [12],
}

TOP_SYMBOLS = [
    "DOTUSD", "ARBUSD", "APTUSD", "NEARUSD", "INJUSD",
    "ALGOUSD", "ADAUSD", "OPUSD", "SEIUSD", "ICPUSD",
    "DOGEUSD", "FILUSD", "ATOMUSD", "BTCUSD", "ETHUSD",
]


def grid_configs(grid: dict) -> list[dict]:
    """Generate all combinations from grid."""
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    configs = []
    for combo in itertools.product(*vals):
        cfg = dict(zip(keys, combo))
        name_parts = []
        for k, v in cfg.items():
            if k == "epochs":
                name_parts.append(f"ep{v}")
            elif k == "learning_rate":
                name_parts.append(f"lr{v:.0e}")
            elif k == "weight_decay":
                name_parts.append(f"wd{v}")
            elif k == "sam_mode":
                name_parts.append(v[:4])
            elif k == "spectral_reg_weight" and v > 0:
                name_parts.append(f"spec{v:.0e}")
        cfg["name"] = "_".join(name_parts)
        configs.append(cfg)
    return configs


def run_one(config: dict, symbol: str, contval: bool = True) -> dict:
    """Train one config on one symbol, return marketsim result."""
    name = config["name"]
    epochs = config.pop("epochs", 15)

    tc_kwargs = dict(DEFAULT_TRAINING_OVERRIDES)
    tc_kwargs["epochs"] = epochs
    tc_kwargs["use_compile"] = False

    sap_kwargs = {"continuous_val_eval": contval}
    tc_fields = set(TrainingConfig.__dataclass_fields__.keys())
    sap_fields = set(SAPConfig.__dataclass_fields__.keys())

    for k, v in config.items():
        if k == "name":
            continue
        if k in tc_fields:
            tc_kwargs[k] = v
        elif k in sap_fields:
            sap_kwargs[k] = v

    horizons = []
    cache_root = Path("binanceneural/forecast_cache")
    for h in (1, 24):
        if (cache_root / f"h{h}" / f"{symbol}.parquet").exists():
            horizons.append(h)
    if not horizons:
        return {"name": name, "symbol": symbol, "error": "no cache"}

    ds = DatasetConfig(
        symbol=symbol,
        data_root=Path("trainingdatahourly/crypto"),
        forecast_cache_root=cache_root,
        forecast_horizons=tuple(horizons),
        sequence_length=tc_kwargs.get("sequence_length", 72),
        validation_days=70,
        cache_only=True,
    )
    tc_kwargs["dataset"] = ds
    tc_kwargs["run_name"] = f"grid_{name}_{symbol}_{time.strftime('%Y%m%d_%H%M%S')}"
    tc_kwargs["checkpoint_root"] = Path("sharpnessadjustedproximalpolicy/checkpoints")

    tc = TrainingConfig(**tc_kwargs)
    sc = SAPConfig(**sap_kwargs)

    try:
        dm = BinanceHourlyDataModule(ds)
    except Exception as e:
        return {"name": name, "symbol": symbol, "error": str(e)}

    trainer = SAPTrainer(tc, sc, dm)
    t0 = time.time()
    try:
        artifacts, history = trainer.train()
    except Exception as e:
        traceback.print_exc()
        return {"name": name, "symbol": symbol, "error": str(e)}

    wall = time.time() - t0

    # Get best checkpoint by continuous marketsim sortino if available
    best = max(history, key=lambda h: h.ms_sortino if h.ms_sortino != 0.0 else h.val_score)
    best_ep = best.epoch
    best_sort = best.ms_sortino if best.ms_sortino != 0.0 else best.val_sortino

    # Run full-period marketsim eval
    ckpt_path = trainer.checkpoint_dir / f"epoch_{best_ep:03d}.pt"
    ms_ret = ms_sort = ms_dd = 0.0
    ms_trades = 0
    if ckpt_path.exists():
        try:
            results = chunk_based_eval(ckpt_path, symbol, lag=2, n_windows=0)
            if results:
                r = results[0]
                ms_ret = r.total_return_pct
                ms_sort = r.sortino
                ms_dd = r.max_drawdown_pct
                ms_trades = r.num_trades
        except Exception as e:
            pass

    config["epochs"] = epochs  # restore

    return {
        "name": name, "symbol": symbol,
        "best_epoch": best_ep,
        "val_sortino": round(best_sort, 3),
        "ms_return": round(ms_ret, 2),
        "ms_sortino": round(ms_sort, 3),
        "ms_dd": round(ms_dd, 2),
        "ms_trades": ms_trades,
        "wall_s": round(wall, 0),
        "error": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--n-symbols", type=int, default=5)
    parser.add_argument("--budget", type=int, default=3600, help="Time budget seconds")
    parser.add_argument("--fast", action="store_true", help="Use compact grid")
    parser.add_argument("--no-contval", action="store_true")
    args = parser.parse_args()

    symbols = args.symbols or TOP_SYMBOLS[:args.n_symbols]
    grid = GRID_FAST if args.fast else GRID
    configs = grid_configs(grid)

    total = len(configs) * len(symbols)
    print(f"Grid search: {len(configs)} configs x {len(symbols)} symbols = {total} runs", flush=True)
    print(f"Configs: {[c['name'] for c in configs[:5]]}...", flush=True)
    print(f"Budget: {args.budget}s", flush=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    results_path = Path(f"sharpnessadjustedproximalpolicy/grid_results_{ts}.json")
    lb_path = Path(f"sharpnessadjustedproximalpolicy/grid_leaderboard_{ts}.csv")

    results = []
    t_start = time.time()

    for sym in symbols:
        print(f"\n{'='*60}\n{sym}\n{'='*60}", flush=True)
        for cfg in configs:
            if time.time() - t_start > args.budget:
                print(f"Budget exhausted ({args.budget}s)", flush=True)
                break

            print(f"  {cfg['name']}...", end="", flush=True)
            r = run_one(dict(cfg), sym, contval=not args.no_contval)
            results.append(r)

            if r.get("error"):
                print(f" ERROR: {r['error'][:50]}", flush=True)
            else:
                print(f" ms_sort={r['ms_sortino']:.2f} ms_ret={r['ms_return']:+.1f}% ({r['wall_s']:.0f}s)", flush=True)

            results_path.write_text(json.dumps(results, indent=2))

        if time.time() - t_start > args.budget:
            break

    # Build leaderboard: best per symbol by ms_sortino
    ok = [r for r in results if not r.get("error")]
    best_per_sym = {}
    for r in ok:
        sym = r["symbol"]
        if sym not in best_per_sym or r["ms_sortino"] > best_per_sym[sym]["ms_sortino"]:
            best_per_sym[sym] = r

    ranked = sorted(best_per_sym.values(), key=lambda x: x["ms_sortino"], reverse=True)
    if ranked:
        fields = ["symbol", "name", "best_epoch", "val_sortino", "ms_return", "ms_sortino", "ms_dd", "ms_trades", "wall_s"]
        with open(lb_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in ranked:
                w.writerow(r)

    print(f"\n{'Symbol':<12} {'Config':<30} {'MS Sort':>8} {'MS Ret':>8} {'MS DD':>8} {'Trades':>7}", flush=True)
    print("-" * 80, flush=True)
    for r in ranked:
        print(f"{r['symbol']:<12} {r['name']:<30} {r['ms_sortino']:>8.2f} {r['ms_return']:>7.1f}% {r['ms_dd']:>7.1f}% {r['ms_trades']:>7}", flush=True)

    # Also show full leaderboard sorted by ms_sortino
    all_ranked = sorted(ok, key=lambda x: x.get("ms_sortino", -999), reverse=True)
    print(f"\n--- Full Grid ({len(ok)} runs) ---", flush=True)
    for r in all_ranked[:20]:
        print(f"  {r['symbol']:<10} {r['name']:<30} ms_sort={r['ms_sortino']:>6.2f} ms_ret={r['ms_return']:>+7.1f}% ep={r['best_epoch']}", flush=True)


if __name__ == "__main__":
    main()
