#!/usr/bin/env python3
"""Train 6-output portfolio policy with directional constraints and short selling."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from binanceneural.config import DatasetConfig, TrainingConfig
from binanceneural.data import MultiSymbolDataModule
from binanceneural.trainer import BinanceHourlyTrainer

LONG_ONLY = {"NVDA", "MSFT", "META", "GOOG", "NET", "PLTR", "DBX", "TSLA", "AAPL"}
SHORT_ONLY = {"YELP", "EBAY", "TRIP", "MTCH", "KIND", "ANGI", "Z", "EXPE", "BKNG", "NWSA", "NYT"}

ALL_SYMBOLS = sorted(LONG_ONLY | SHORT_ONLY)

STOCK_PERIODS_PER_YEAR = 252.0 * 6.5  # trading hours per year


def build_directional_constraints(symbols):
    constraints = {}
    for s in symbols:
        if s in LONG_ONLY:
            constraints[s] = (1.0, 0.0)
        elif s in SHORT_ONLY:
            constraints[s] = (0.0, 1.0)
        else:
            constraints[s] = (1.0, 1.0)
    return constraints


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default=",".join(ALL_SYMBOLS))
    p.add_argument("--data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    p.add_argument("--cache-root", type=Path, default=Path("unified_hourly_experiment/forecast_cache"))
    p.add_argument("--checkpoint-root", type=Path, default=Path("unified_hourly_experiment/checkpoints"))
    p.add_argument("--run-name", default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--sequence-length", type=int, default=512)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-outputs", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--return-weight", type=float, default=0.08)
    p.add_argument("--smoothness-penalty", type=float, default=0.0)
    p.add_argument("--maker-fee", type=float, default=0.001)
    p.add_argument("--fill-temperature", type=float, default=5e-4)
    p.add_argument("--validation-days", type=int, default=30)
    p.add_argument("--forecast-horizons", default="1")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--model-arch", default="nano")
    p.add_argument("--optimizer", default="muon")
    p.add_argument("--max-hold-hours", type=float, default=24.0)
    p.add_argument("--preload", type=Path, default=None)
    p.add_argument("--no-compile", action="store_true")
    args = p.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    horizons = tuple(int(h) for h in args.forecast_horizons.split(","))

    print(f"Training portfolio policy: {len(symbols)} symbols, {args.num_outputs} outputs")
    print(f"Long-only: {[s for s in symbols if s in LONG_ONLY]}")
    print(f"Short-only: {[s for s in symbols if s in SHORT_ONLY]}")

    constraints = build_directional_constraints(symbols)

    ds_cfg = DatasetConfig(
        symbol=symbols[0],
        data_root=args.data_root,
        forecast_cache_root=args.cache_root,
        forecast_horizons=horizons,
        sequence_length=args.sequence_length,
        validation_days=args.validation_days,
        cache_only=True,
        min_history_hours=args.sequence_length + args.validation_days * 24 + 48,
    )

    print("Loading data for all symbols...")
    loaded_symbols = []
    for s in symbols:
        try:
            from binanceneural.data import BinanceHourlyDataModule
            test_cfg = DatasetConfig(
                symbol=s,
                data_root=args.data_root,
                forecast_cache_root=args.cache_root,
                forecast_horizons=horizons,
                sequence_length=args.sequence_length,
                validation_days=args.validation_days,
                cache_only=True,
                min_history_hours=args.sequence_length + args.validation_days * 24 + 48,
            )
            BinanceHourlyDataModule(test_cfg)
            loaded_symbols.append(s)
        except Exception as e:
            print(f"  Skipping {s}: {e}")

    if not loaded_symbols:
        print("ERROR: No symbols loaded successfully")
        sys.exit(1)

    symbols = loaded_symbols
    constraints = build_directional_constraints(symbols)
    ds_cfg.symbol = symbols[0]

    data_module = MultiSymbolDataModule(
        symbols=symbols,
        config=ds_cfg,
        directional_constraints=constraints,
    )
    print(f"Train samples: {len(data_module.train_dataset)}, Val samples: {len(data_module.val_dataset)}")
    print(f"Features: {len(data_module.feature_columns)}")

    run_name = args.run_name or f"portfolio_{args.hidden_dim}h_{args.num_layers}L_{args.num_outputs}out"

    train_cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate,
        return_weight=args.return_weight,
        smoothness_penalty=args.smoothness_penalty,
        maker_fee=args.maker_fee,
        fill_temperature=args.fill_temperature,
        transformer_dim=args.hidden_dim,
        transformer_layers=args.num_layers,
        transformer_heads=args.num_heads,
        model_arch=args.model_arch,
        optimizer_name=args.optimizer,
        num_outputs=args.num_outputs,
        max_hold_hours=args.max_hold_hours,
        seed=args.seed,
        use_compile=not args.no_compile,
        use_tf32=True,
        use_flash_attention=True,
        periods_per_year=STOCK_PERIODS_PER_YEAR,
        checkpoint_root=args.checkpoint_root,
        run_name=run_name,
        preload_checkpoint_path=args.preload,
        validation_use_binary_fills=True,
    )

    trainer = BinanceHourlyTrainer(train_cfg, data_module)
    artifacts = trainer.train()

    print(f"\nBest checkpoint: {artifacts.best_checkpoint}")
    if artifacts.history:
        best = max(artifacts.history, key=lambda h: h.val_sortino or 0)
        print(f"Best epoch {best.epoch}: val_sortino={best.val_sortino:.4f} val_return={best.val_return:.4f}")

    meta = {
        "symbols": symbols,
        "long_only": sorted(s for s in symbols if s in LONG_ONLY),
        "short_only": sorted(s for s in symbols if s in SHORT_ONLY),
        "num_outputs": args.num_outputs,
        "max_hold_hours": args.max_hold_hours,
        "feature_columns": list(data_module.feature_columns),
        "best_checkpoint": str(artifacts.best_checkpoint) if artifacts.best_checkpoint else None,
        "history": [
            {
                "epoch": h.epoch,
                "train_sortino": h.train_sortino,
                "train_return": h.train_return,
                "val_sortino": h.val_sortino,
                "val_return": h.val_return,
            }
            for h in artifacts.history
        ],
    }
    meta_path = trainer.checkpoint_dir / "training_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
