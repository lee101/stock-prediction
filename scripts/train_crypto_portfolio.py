#!/usr/bin/env python3
"""Multi-symbol crypto RL training using BinanceHourlyTrainer + MultiSymbolDataModule.

Trains a single policy on concatenated data from multiple crypto pairs.
Key insight from prior experiments: training on more symbols + blacklisting bad ones
beats training on fewer symbols directly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from binanceneural.config import DatasetConfig, TrainingConfig
from binanceneural.data import MultiSymbolDataModule
from binanceneural.trainer import BinanceHourlyTrainer


def main():
    p = argparse.ArgumentParser(description="Train crypto portfolio RL policy")
    p.add_argument("--symbols", type=str, default="BTCUSD,ETHUSD,SOLUSD,DOGEUSD,AAVEUSD,LINKUSD")
    p.add_argument("--target-symbol", type=str, default=None,
                   help="Primary symbol for validation (default: first in --symbols)")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.04)
    p.add_argument("--transformer-dim", type=int, default=384)
    p.add_argument("--transformer-layers", type=int, default=6)
    p.add_argument("--transformer-heads", type=int, default=6)
    p.add_argument("--sequence-length", type=int, default=48)
    p.add_argument("--lr-schedule", type=str, default="cosine")
    p.add_argument("--lr-min-ratio", type=float, default=0.01)
    p.add_argument("--maker-fee", type=float, default=0.001)
    p.add_argument("--margin-rate", type=float, default=0.0625)
    p.add_argument("--max-leverage", type=float, default=5.0)
    p.add_argument("--fill-buffer", type=float, default=0.0005)
    p.add_argument("--return-weight", type=float, default=0.10)
    p.add_argument("--loss-type", type=str, default="sortino")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--data-root", type=Path, default=REPO / "trainingdatahourlybinance")
    p.add_argument("--forecast-cache", type=Path, default=REPO / "binanceneural" / "forecast_cache")
    p.add_argument("--checkpoint-root", type=Path, default=REPO / "binanceneural" / "checkpoints")
    p.add_argument("--validation-days", type=int, default=70)
    p.add_argument("--forecast-horizons", type=str, default="1")
    p.add_argument("--decision-lag", type=int, default=1)
    p.add_argument("--model-arch", type=str, default="classic")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--feature-noise", type=float, default=0.0)
    args = p.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    target = args.target_symbol or symbols[0]
    horizons = tuple(int(h) for h in args.forecast_horizons.split(","))

    print(f"symbols: {symbols}")
    print(f"target (validation): {target}")

    ds_cfg = DatasetConfig(
        symbol=target,
        data_root=args.data_root,
        forecast_cache_root=args.forecast_cache,
        forecast_horizons=horizons,
        sequence_length=args.sequence_length,
        validation_days=args.validation_days,
        cache_only=True,
    )

    data = MultiSymbolDataModule(symbols, ds_cfg)

    tc = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        transformer_dim=args.transformer_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_dropout=args.dropout,
        model_arch=args.model_arch,
        lr_schedule=args.lr_schedule,
        lr_min_ratio=args.lr_min_ratio,
        maker_fee=args.maker_fee,
        margin_annual_rate=args.margin_rate,
        max_leverage=args.max_leverage,
        fill_buffer_pct=args.fill_buffer,
        return_weight=args.return_weight,
        loss_type=args.loss_type,
        use_compile=not args.no_compile,
        checkpoint_root=Path(args.checkpoint_root),
        run_name=args.run_name,
        seed=args.seed,
        decision_lag_bars=args.decision_lag,
        feature_noise_std=args.feature_noise,
    )

    trainer = BinanceHourlyTrainer(tc, data)
    artifacts = trainer.train()
    print(f"best checkpoint: {artifacts.best_checkpoint}")


if __name__ == "__main__":
    main()
