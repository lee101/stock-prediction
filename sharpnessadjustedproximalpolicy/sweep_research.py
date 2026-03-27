#!/usr/bin/env python3
"""Research sweep: run round 5 configs on top symbols.

Usage:
    python -m sharpnessadjustedproximalpolicy.sweep_research --epochs 15
    python -m sharpnessadjustedproximalpolicy.sweep_research --symbols DOGEUSD BTCUSD --configs spectral
    python -m sharpnessadjustedproximalpolicy.sweep_research --category all --epochs 20
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from binanceneural.config import DatasetConfig, TrainingConfig
from binanceneural.data import BinanceHourlyDataModule

from .config import DEFAULT_TRAINING_OVERRIDES, SAPConfig
from .research_configs import (
    ALL_R5_CONFIGS,
    CHAMPION_TUNE_CONFIGS,
    COMBO_CONFIGS,
    MULTIPERIOD_CONFIGS,
    SPECTRAL_CONFIGS,
)
from .trainer import SAPTrainer

CATEGORY_MAP = {
    "spectral": SPECTRAL_CONFIGS,
    "multiperiod": MULTIPERIOD_CONFIGS,
    "combo": COMBO_CONFIGS,
    "champion": CHAMPION_TUNE_CONFIGS,
    "all": ALL_R5_CONFIGS,
}

# Top symbols from scaled leaderboard (sorted by val sortino)
TOP_SYMBOLS = [
    "DOTUSD", "APTUSD", "ARBUSD", "INJUSD", "ALGOUSD",
    "ICPUSD", "DOGEUSD", "ADAUSD", "FILUSD", "ATOMUSD",
    "LINKUSD", "BCHUSD", "ETHUSD", "BTCUSD", "BNBUSD",
]


def ensure_cache(symbol: str, horizons: tuple[int, ...] = (1, 24)) -> tuple[int, ...]:
    cache_root = Path("binanceneural") / "forecast_cache"
    available = []
    for h in horizons:
        if (cache_root / f"h{h}" / f"{symbol}.parquet").exists():
            available.append(h)
    return tuple(available)


def run_experiment(config: dict, symbol: str, epochs: int) -> dict:
    name = config["name"]
    print(f"\n  {name} x {symbol}...", end="", flush=True)

    tc_kwargs = dict(DEFAULT_TRAINING_OVERRIDES)
    tc_kwargs["epochs"] = epochs
    tc_kwargs["use_compile"] = False

    sap_kwargs = {}
    tc_fields = set(TrainingConfig.__dataclass_fields__.keys())
    sap_fields = set(SAPConfig.__dataclass_fields__.keys())

    for k, v in config.items():
        if k == "name":
            continue
        if k in tc_fields:
            tc_kwargs[k] = v
        elif k in sap_fields:
            sap_kwargs[k] = v

    horizons = ensure_cache(symbol)
    if len(horizons) < 1:
        return {"name": name, "symbol": symbol, "error": "no cache"}

    ds = DatasetConfig(
        symbol=symbol,
        data_root=Path("trainingdatahourly") / "crypto",
        forecast_cache_root=Path("binanceneural") / "forecast_cache",
        forecast_horizons=horizons,
        sequence_length=tc_kwargs.get("sequence_length", 72),
        validation_days=70,
        cache_only=True,
    )
    tc_kwargs["dataset"] = ds
    tc_kwargs["run_name"] = f"sap_{name}_{symbol}_{time.strftime('%Y%m%d_%H%M%S')}"
    tc_kwargs["checkpoint_root"] = Path("sharpnessadjustedproximalpolicy") / "checkpoints"

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

    best = max(history, key=lambda h: h.val_score)
    wall = time.time() - t0
    sharp_ema = history[-1].sharpness_ema if history else 0
    wd_scale = history[-1].lr_scale if history else 1.0

    print(f" sort={best.val_sortino:.1f} ret={best.val_return:.3f} ep={best.epoch} ({wall:.0f}s)", flush=True)

    return {
        "name": name, "symbol": symbol,
        "best_epoch": best.epoch,
        "best_val_sortino": round(best.val_sortino, 3),
        "best_val_return": round(best.val_return, 4),
        "best_val_score": round(best.val_score, 3),
        "sharpness_ema": round(sharp_ema, 1),
        "wd_scale": round(wd_scale, 2),
        "wall_s": round(wall, 0),
        "checkpoint_dir": str(trainer.checkpoint_dir),
        "error": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--n-symbols", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--category", default="all", choices=list(CATEGORY_MAP.keys()))
    parser.add_argument("--configs", type=str, default=None)
    args = parser.parse_args()

    symbols = args.symbols or TOP_SYMBOLS[:args.n_symbols]
    configs = CATEGORY_MAP[args.category]
    if args.configs:
        names = set(args.configs.split(","))
        configs = [c for c in configs if c["name"] in names]

    print(f"Research sweep R5: {len(configs)} configs x {len(symbols)} symbols = {len(configs)*len(symbols)} runs", flush=True)
    print(f"Symbols: {', '.join(symbols)}", flush=True)
    print(f"Configs: {', '.join(c['name'] for c in configs)}", flush=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    results_path = Path(f"sharpnessadjustedproximalpolicy/research_results_{ts}.json")
    leaderboard_path = Path(f"sharpnessadjustedproximalpolicy/research_leaderboard_{ts}.csv")

    results = []
    for sym in symbols:
        print(f"\n{'='*60}\n{sym}\n{'='*60}", flush=True)
        for config in configs:
            r = run_experiment(config, sym, args.epochs)
            results.append(r)
            results_path.write_text(json.dumps(results, indent=2))

    # Leaderboard CSV
    ok = [r for r in results if not r.get("error")]
    ok.sort(key=lambda r: r.get("best_val_sortino", -999), reverse=True)
    if ok:
        fields = ["symbol", "name", "best_epoch", "best_val_sortino", "best_val_return",
                  "sharpness_ema", "wd_scale", "wall_s", "error"]
        with open(leaderboard_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in ok:
                w.writerow(r)
        print(f"\nLeaderboard: {leaderboard_path}", flush=True)

    # Summary table
    print(f"\n{'Symbol':<10} {'Config':<30} {'Sort':>8} {'Ret':>8} {'Ep':>3} {'Sharp':>6} {'WDS':>5}", flush=True)
    print("-" * 75, flush=True)
    for r in ok[:30]:
        print(f"{r['symbol']:<10} {r['name']:<30} {r['best_val_sortino']:>8.1f} {r['best_val_return']:>8.3f} {r['best_epoch']:>3} {r.get('sharpness_ema',0):>6.0f} {r.get('wd_scale',1):>5.2f}", flush=True)

    errors = [r for r in results if r.get("error")]
    if errors:
        print(f"\nErrors ({len(errors)}):", flush=True)
        for r in errors:
            print(f"  {r['symbol']} {r['name']}: {r['error'][:60]}", flush=True)


if __name__ == "__main__":
    main()
