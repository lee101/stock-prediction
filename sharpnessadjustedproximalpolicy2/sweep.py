#!/usr/bin/env python3
"""Sweep runner for sharpness-adjusted proximal policy experiments.

Usage:
    python -m sharpnessadjustedproximalpolicy2.sweep --symbol DOGEUSD
    python -m sharpnessadjustedproximalpolicy2.sweep --symbol DOGEUSD --experiments periodic_rho005,baseline_adamw
    python -m sharpnessadjustedproximalpolicy2.sweep --symbol DOGEUSD --all-symbols
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from binanceneural.config import DatasetConfig, TrainingConfig
from binanceneural.data import BinanceHourlyDataModule, MultiSymbolDataModule

from .config import DEFAULT_TRAINING_OVERRIDES, EXPERIMENTS, SAPConfig, SYMBOLS_CORE
from .trainer import SAPTrainer


def build_configs(exp: dict, symbol: str, base_overrides: dict | None = None) -> tuple[TrainingConfig, SAPConfig]:
    """Build TrainingConfig + SAPConfig from experiment dict."""
    tc_kwargs = dict(DEFAULT_TRAINING_OVERRIDES)
    sap_kwargs = {}
    tc_fields = set(TrainingConfig.__dataclass_fields__.keys())
    sap_fields = set(SAPConfig.__dataclass_fields__.keys())

    for k, v in exp.items():
        if k == "name":
            continue
        if k in tc_fields:
            tc_kwargs[k] = v
        elif k in sap_fields:
            sap_kwargs[k] = v

    if base_overrides:
        for k, v in base_overrides.items():
            if k in tc_fields:
                tc_kwargs[k] = v
            elif k in sap_fields:
                sap_kwargs[k] = v

    tc_kwargs["run_name"] = f"sap_{exp['name']}_{symbol}_{time.strftime('%Y%m%d_%H%M%S')}"
    tc_kwargs["checkpoint_root"] = Path("sharpnessadjustedproximalpolicy2") / "checkpoints"

    cache_root = Path("binanceneural") / "forecast_cache"
    available_horizons = []
    for h in (1, 4, 12, 24):
        if (cache_root / f"h{h}" / f"{symbol}.parquet").exists():
            available_horizons.append(h)
    if not available_horizons:
        available_horizons = [1, 4, 24]

    ds = DatasetConfig(
        symbol=symbol,
        data_root=Path("trainingdatahourly") / "crypto",
        forecast_cache_root=cache_root,
        forecast_horizons=tuple(available_horizons),
        sequence_length=tc_kwargs.get("sequence_length", 72),
        validation_days=70,
        cache_only=True,
        bar_shift_range=tc_kwargs.get("bar_shift_range", 0),
    )
    tc_kwargs["dataset"] = ds

    tc = TrainingConfig(**tc_kwargs)
    sc = SAPConfig(**sap_kwargs)
    return tc, sc


def run_single_experiment(
    exp: dict,
    symbol: str,
    base_overrides: dict | None = None,
) -> dict:
    """Run one experiment, return results dict."""
    name = exp["name"]
    print(f"\n{'='*60}")
    print(f"Experiment: {name} | Symbol: {symbol}")
    print(f"{'='*60}")

    tc, sc = build_configs(exp, symbol, base_overrides)
    try:
        dm = BinanceHourlyDataModule(tc.dataset)
    except Exception as e:
        print(f"Data loading failed for {symbol}: {e}")
        return {"name": name, "symbol": symbol, "error": str(e)}

    trainer = SAPTrainer(tc, sc, dm)
    t0 = time.time()
    try:
        artifacts, history = trainer.train()
    except Exception as e:
        print(f"Training failed: {e}")
        traceback.print_exc()
        return {"name": name, "symbol": symbol, "error": str(e)}

    elapsed = time.time() - t0
    best_ep = max(history, key=lambda h: h.val_score)
    result = {
        "name": name,
        "symbol": symbol,
        "best_epoch": best_ep.epoch,
        "best_val_sortino": best_ep.val_sortino,
        "best_val_return": best_ep.val_return,
        "best_val_score": best_ep.val_score,
        "final_sharpness_ema": history[-1].sharpness_ema if history else 0,
        "final_step_scale": history[-1].step_scale if history else 1,
        "final_lr_scale": history[-1].lr_scale if history else 1,
        "total_epochs": len(history),
        "wall_time_s": elapsed,
        "checkpoint_dir": str(trainer.checkpoint_dir),
        "sam_mode": sc.sam_mode,
        "rho": sc.rho,
        "error": None,
    }
    print(f"Result: Sort={best_ep.val_sortino:.3f} Ret={best_ep.val_return:.4f} @ ep{best_ep.epoch}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="DOGEUSD")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--experiments", type=str, default=None, help="comma-sep experiment names")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    args = parser.parse_args()

    symbols = args.symbols or [args.symbol]
    exps = EXPERIMENTS
    if args.experiments:
        names = set(args.experiments.split(","))
        exps = [e for e in EXPERIMENTS if e["name"] in names]
        if not exps:
            print(f"No matching experiments found for: {names}")
            sys.exit(1)

    overrides = {}
    if args.epochs:
        overrides["epochs"] = args.epochs
    if args.batch_size:
        overrides["batch_size"] = args.batch_size
    if args.lr:
        overrides["learning_rate"] = args.lr
    if args.max_train_batches is not None:
        overrides["max_train_batches"] = args.max_train_batches
    if args.max_val_batches is not None:
        overrides["max_val_batches"] = args.max_val_batches

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output or f"sharpnessadjustedproximalpolicy2/sweep_results_{timestamp}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = []
    for symbol in symbols:
        for exp in exps:
            result = run_single_experiment(exp, symbol, overrides)
            all_results.append(result)
            output_path.write_text(json.dumps(all_results, indent=2))
            print(f"[{len(all_results)}/{len(symbols)*len(exps)}] Saved to {output_path}")

    print(f"\nSweep complete: {len(all_results)} experiments")
    print(f"Results: {output_path}")

    # Print summary table
    print(f"\n{'Name':<30} {'Symbol':<10} {'Sort':>8} {'Ret':>10} {'Ep':>4} {'Sharp':>8} {'Step':>6}")
    print("-" * 86)
    for r in sorted(all_results, key=lambda x: x.get("best_val_sortino", -999), reverse=True):
        if r.get("error"):
            print(f"{r['name']:<30} {r['symbol']:<10} ERROR: {r['error'][:40]}")
        else:
            print(
                f"{r['name']:<30} {r['symbol']:<10} "
                f"{r['best_val_sortino']:>8.3f} {r['best_val_return']:>10.4f} "
                f"{r['best_epoch']:>4d} {r.get('final_sharpness_ema', 0):>8.3f} "
                f"{r.get('final_step_scale', r.get('final_lr_scale', 1)):>6.2f}"
            )


if __name__ == "__main__":
    main()
