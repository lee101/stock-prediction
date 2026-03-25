#!/usr/bin/env python3
"""Multi-symbol joint training sweep.

Train on multiple symbols simultaneously (shared model), evaluate per-symbol.
This should regularize better than single-symbol training.

Usage:
    python -m sharpnessadjustedproximalpolicy.sweep_multisym --epochs 15
    python -m sharpnessadjustedproximalpolicy.sweep_multisym --symbols DOGEUSD BTCUSD ETHUSD --epochs 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from binanceneural.config import DatasetConfig, TrainingConfig
from binanceneural.data import MultiSymbolDataModule

from .config import DEFAULT_TRAINING_OVERRIDES, SAPConfig
from .trainer import SAPTrainer


def get_cached_symbols(min_horizons: int = 2) -> list[str]:
    cache_root = Path("binanceneural") / "forecast_cache"
    data_root = Path("trainingdatahourly") / "crypto"
    symbols = []
    for csv_path in sorted(data_root.glob("*USD.csv")):
        sym = csv_path.stem
        n_h = sum(1 for h in (1, 4, 12, 24) if (cache_root / f"h{h}" / f"{sym}.parquet").exists())
        if n_h >= min_horizons:
            symbols.append(sym)
    return symbols


MULTI_CONFIGS = [
    {"name": "multi_baseline_wd01", "sam_mode": "none", "weight_decay": 0.1},
    {"name": "multi_periodic_wd01", "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.1},
    {"name": "multi_baseline_wd004", "sam_mode": "none", "weight_decay": 0.04},
    {"name": "multi_periodic_wd004", "sam_mode": "periodic", "rho": 0.05, "weight_decay": 0.04},
]


def run_multi_experiment(config: dict, symbols: list[str], target_symbol: str, epochs: int) -> dict:
    name = config["name"]
    print(f"\n{'='*60}", flush=True)
    print(f"Multi-sym: {name} | Train: {','.join(symbols)} | Eval: {target_symbol}", flush=True)
    print(f"{'='*60}", flush=True)

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

    tc_kwargs["run_name"] = f"sap_{name}_{target_symbol}_{time.strftime('%Y%m%d_%H%M%S')}"
    tc_kwargs["checkpoint_root"] = Path("sharpnessadjustedproximalpolicy") / "checkpoints"

    cache_root = Path("binanceneural") / "forecast_cache"
    horizons = []
    for h in (1, 4, 12, 24):
        if all((cache_root / f"h{h}" / f"{s}.parquet").exists() for s in symbols):
            horizons.append(h)
    if len(horizons) < 2:
        # fallback to per-symbol minimum
        horizons = [h for h in (1, 24) if all((cache_root / f"h{h}" / f"{s}.parquet").exists() for s in symbols)]
    if not horizons:
        return {"name": name, "symbol": target_symbol, "error": "no common horizons"}

    ds = DatasetConfig(
        symbol=target_symbol,
        data_root=Path("trainingdatahourly") / "crypto",
        forecast_cache_root=cache_root,
        forecast_horizons=tuple(horizons),
        sequence_length=tc_kwargs.get("sequence_length", 72),
        validation_days=70,
        cache_only=True,
    )
    tc_kwargs["dataset"] = ds
    tc = TrainingConfig(**tc_kwargs)
    sc = SAPConfig(**sap_kwargs)

    try:
        dm = MultiSymbolDataModule(symbols, ds)
    except Exception as e:
        print(f"Data loading failed: {e}", flush=True)
        return {"name": name, "symbol": target_symbol, "error": str(e)}

    trainer = SAPTrainer(tc, sc, dm)
    t0 = time.time()
    try:
        artifacts, history = trainer.train()
    except Exception as e:
        print(f"Training failed: {e}", flush=True)
        traceback.print_exc()
        return {"name": name, "symbol": target_symbol, "error": str(e)}

    best = max(history, key=lambda h: h.val_score)
    return {
        "name": name, "symbol": target_symbol,
        "train_symbols": symbols,
        "best_epoch": best.epoch,
        "best_val_sortino": best.val_sortino,
        "best_val_return": best.val_return,
        "best_val_score": best.val_score,
        "final_sharpness_ema": history[-1].sharpness_ema,
        "total_epochs": len(history),
        "wall_time_s": time.time() - t0,
        "checkpoint_dir": str(trainer.checkpoint_dir),
        "sam_mode": sc.sam_mode,
        "n_train_symbols": len(symbols),
        "error": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--target", default="DOGEUSD")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--configs", type=str, default=None)
    args = parser.parse_args()

    symbols = args.symbols or get_cached_symbols()
    if not symbols:
        print("No cached symbols found")
        sys.exit(1)
    if args.target not in symbols:
        symbols.append(args.target)

    configs = MULTI_CONFIGS
    if args.configs:
        names = set(args.configs.split(","))
        configs = [c for c in MULTI_CONFIGS if c["name"] in names]

    print(f"Training symbols ({len(symbols)}): {', '.join(symbols)}", flush=True)
    print(f"Target: {args.target}", flush=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(f"sharpnessadjustedproximalpolicy/multisym_results_{timestamp}.json")

    results = []
    for config in configs:
        r = run_multi_experiment(config, symbols, args.target, args.epochs)
        results.append(r)
        out.write_text(json.dumps(results, indent=2))
        if not r.get("error"):
            print(f"  Sort={r['best_val_sortino']:.1f} Ret={r['best_val_return']:.3f} @ ep{r['best_epoch']}", flush=True)

    print(f"\n{'Config':<30} {'Sort':>8} {'Ret':>8} {'Ep':>3}", flush=True)
    print("-" * 55, flush=True)
    for r in sorted(results, key=lambda x: x.get("best_val_sortino", -999), reverse=True):
        if r.get("error"):
            print(f"{r['name']:<30} ERROR: {r['error'][:40]}", flush=True)
        else:
            print(f"{r['name']:<30} {r['best_val_sortino']:>8.1f} {r['best_val_return']:>8.3f} {r['best_epoch']:>3}", flush=True)


if __name__ == "__main__":
    main()
