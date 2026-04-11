#!/usr/bin/env python3
"""Per-symbol Binance training sweep with decision_lag=2 and new asymmetry features.

Trains per-symbol models (BTC, ETH, SOL) using Chronos2 forecasts at
available horizons, validating with binary fills at lag=2 (minimax).
Sweeps weight_decay, return_weight, and loss_type.
"""
from __future__ import annotations
import json
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from binanceneural.config import TrainingConfig, DatasetConfig
from binanceneural.data import BinanceHourlyDataModule
from binanceneural.trainer import BinanceHourlyTrainer


SYMBOLS = ["BTCUSD", "ETHUSD", "SOLUSD"]
HORIZONS_BY_SYMBOL = {
    "BTCUSD": (1, 24),
    "ETHUSD": (1, 24),
    "SOLUSD": (1, 24),
    "DOGEUSD": (1, 4, 12, 24),
    "LINKUSD": (1, 24),
    "AAVEUSD": (1, 24),
}

SWEEP_GRID = {
    "weight_decay": [1e-4, 4e-4, 1e-3, 4e-3],
    "return_weight": [0.05, 0.08, 0.15],
    "loss_type": ["sortino", "sortino_dd"],
}


def run_one(symbol: str, wd: float, rw: float, lt: str, seed: int = 42) -> dict:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{symbol}_lag2_wd{wd}_rw{rw}_{lt}_s{seed}_{ts}"
    horizons = HORIZONS_BY_SYMBOL.get(symbol, (1, 24))

    dataset_cfg = DatasetConfig(
        symbol=symbol,
        data_root=Path("trainingdatahourly/crypto"),
        forecast_cache_root=Path("binanceneural/forecast_cache"),
        sequence_length=72,
        validation_days=70,
        forecast_horizons=horizons,
        cache_only=True,
    )

    config = TrainingConfig(
        epochs=40,
        batch_size=16,
        sequence_length=72,
        learning_rate=3e-4,
        weight_decay=wd,
        transformer_dim=256,
        transformer_heads=8,
        transformer_layers=4,
        model_arch="classic",
        decision_lag_bars=2,
        decision_lag_range="1,2",
        validation_lag_aggregation="minimax",
        validation_use_binary_fills=True,
        loss_type=lt,
        return_weight=rw,
        maker_fee=0.001,
        fill_buffer_pct=0.0005,
        fill_temperature=0.1,
        max_leverage=1.0,
        margin_annual_rate=0.0625,
        use_causal_attention=True,
        use_qk_norm=True,
        lr_schedule="cosine",
        lr_min_ratio=0.01,
        run_name=run_name,
    )

    print(f"\n{'='*60}")
    print(f"Training {symbol} | wd={wd} rw={rw} loss={lt} seed={seed}")
    print(f"Run: {run_name}")
    print(f"{'='*60}")

    try:
        data = BinanceHourlyDataModule(dataset_cfg)
        trainer = BinanceHourlyTrainer(config, data)
        t0 = time.time()
        artifacts = trainer.train()
        elapsed = time.time() - t0

        result = {
            "symbol": symbol,
            "run_name": run_name,
            "weight_decay": wd,
            "return_weight": rw,
            "loss_type": lt,
            "seed": seed,
            "elapsed_s": round(elapsed, 1),
            "best_epoch": getattr(artifacts, "best_epoch", None),
            "best_val_sortino": getattr(artifacts, "best_val_sortino", None),
            "best_val_return": getattr(artifacts, "best_val_return", None),
            "best_checkpoint": str(getattr(artifacts, "best_checkpoint", "")),
        }
        print(f"Result: sort={result.get('best_val_sortino')} ret={result.get('best_val_return')} ep={result.get('best_epoch')}")
        return result

    except Exception as e:
        print(f"FAILED: {e}")
        return {
            "symbol": symbol,
            "run_name": run_name,
            "error": str(e),
            "weight_decay": wd,
            "return_weight": rw,
            "loss_type": lt,
        }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default=",".join(SYMBOLS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Reduced grid for fast test")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    results = []
    outdir = Path("analysis/binance_lag2_sweep")
    outdir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        grid = [
            (1e-4, 0.08, "sortino"),
            (4e-4, 0.08, "sortino"),
            (1e-3, 0.08, "sortino"),
        ]
    else:
        grid = list(product(
            SWEEP_GRID["weight_decay"],
            SWEEP_GRID["return_weight"],
            SWEEP_GRID["loss_type"],
        ))

    print(f"Sweep: {len(symbols)} symbols x {len(grid)} configs = {len(symbols)*len(grid)} runs")

    for sym in symbols:
        for wd, rw, lt in grid:
            result = run_one(sym, wd, rw, lt, seed=args.seed)
            results.append(result)
            outpath = outdir / f"sweep_results_{datetime.now().strftime('%Y%m%d')}.json"
            with open(outpath, "w") as f:
                json.dump(results, f, indent=2)

    # Print leaderboard
    print(f"\n{'='*80}")
    print("LEADERBOARD")
    print(f"{'='*80}")
    valid = [r for r in results if "error" not in r and r.get("best_val_sortino") is not None]
    valid.sort(key=lambda x: x.get("best_val_sortino", float("-inf")), reverse=True)
    for i, r in enumerate(valid[:20]):
        print(f"{i+1:3d}. {r['symbol']:8s} wd={r['weight_decay']:<8.4f} rw={r['return_weight']:<5.2f} "
              f"loss={r['loss_type']:<12s} sort={r['best_val_sortino']:+8.3f} "
              f"ret={r['best_val_return']:+8.4f} ep={r['best_epoch']}")


if __name__ == "__main__":
    main()
