#!/usr/bin/env python3
"""Deep binanceneural training sweep for BTC/ETH/SOL.

Sweeps transformer_dim, weight_decay, learning_rate with early stopping.
Logs to WandB and saves CSV leaderboard.
"""
from __future__ import annotations

import argparse
import csv
import gc
import itertools
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from binanceneural.config import DatasetConfig, TrainingConfig
from binanceneural.data import BinanceHourlyDataModule
from binanceneural.trainer import BinanceHourlyTrainer

SYMBOLS = ["BTCUSD", "ETHUSD", "SOLUSD"]
DIMS = [256, 384, 512]
WEIGHT_DECAYS = [0.01, 0.03, 0.04, 0.05]
LEARNING_RATES = [1e-4, 3e-4, 5e-4]
EPOCHS = 80
EARLY_STOP_PATIENCE = 15
RESULTS_DIR = REPO / "binanceneural" / "deep_sweep_results"


def run_single(
    symbol: str,
    dim: int,
    wd: float,
    lr: float,
    *,
    epochs: int = EPOCHS,
    patience: int = EARLY_STOP_PATIENCE,
    dry_run: bool = False,
    wandb_project: str | None = "binanceneural-sweep",
) -> dict:
    tag = f"{symbol}_d{dim}_wd{wd}_lr{lr}"
    print(f"\n{'='*60}\n{tag}\n{'='*60}")

    ds_cfg = DatasetConfig(
        symbol=symbol,
        sequence_length=72,
        validation_days=70,
        cache_only=True,
    )
    try:
        dm = BinanceHourlyDataModule(ds_cfg)
    except Exception as e:
        print(f"SKIP {tag}: {e}")
        return {"symbol": symbol, "dim": dim, "wd": wd, "lr": lr, "status": "data_error", "error": str(e)}

    tc = TrainingConfig(
        epochs=2 if dry_run else epochs,
        batch_size=16,
        sequence_length=72,
        learning_rate=lr,
        weight_decay=wd,
        transformer_dim=dim,
        transformer_layers=4,
        transformer_heads=8,
        lr_schedule="cosine",
        lr_min_ratio=0.01,
        warmup_steps=100,
        return_weight=0.10,
        maker_fee=0.001,
        fill_temperature=0.1,
        fill_buffer_pct=0.0005,
        decision_lag_bars=1,
        loss_type="sortino",
        use_compile=False,
        seed=1337,
        run_name=tag,
        wandb_project=wandb_project,
        dry_train_steps=2 if dry_run else None,
        checkpoint_root=RESULTS_DIR / "checkpoints",
    )

    trainer = BinanceHourlyTrainer(tc, dm)
    artifacts = trainer.train()

    history = artifacts.history
    best_val_sortino = float("-inf")
    best_epoch = 0
    no_improve = 0
    for entry in history:
        vs = entry.val_sortino if entry.val_sortino is not None else float("-inf")
        if vs > best_val_sortino:
            best_val_sortino = vs
            best_epoch = entry.epoch
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience and not dry_run:
            print(f"Early stop at epoch {entry.epoch} (best={best_epoch}, val_sortino={best_val_sortino:.4f})")
            break

    best_entry = history[best_epoch - 1] if best_epoch > 0 and best_epoch <= len(history) else history[-1]

    result = {
        "symbol": symbol,
        "dim": dim,
        "wd": wd,
        "lr": lr,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "val_sortino": best_entry.val_sortino,
        "val_return": best_entry.val_return,
        "val_loss": best_entry.val_loss,
        "train_sortino": best_entry.train_sortino,
        "train_return": best_entry.train_return,
        "train_loss": best_entry.train_loss,
        "status": "ok",
    }
    print(f"RESULT {tag}: ep={best_epoch} val_sort={result['val_sortino']:.4f} val_ret={result['val_return']:.4f}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description="Deep binanceneural sweep")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--dims", nargs="+", type=int, default=DIMS)
    parser.add_argument("--wds", nargs="+", type=float, default=WEIGHT_DECAYS)
    parser.add_argument("--lrs", nargs="+", type=float, default=LEARNING_RATES)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wandb-project", default="binanceneural-sweep")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "leaderboard.csv"
    wandb_proj = None if args.no_wandb else args.wandb_project

    combos = list(itertools.product(args.symbols, args.dims, args.wds, args.lrs))
    print(f"Total configs: {len(combos)}")

    results = []
    fieldnames = [
        "symbol", "dim", "wd", "lr", "best_epoch", "epochs_run",
        "val_sortino", "val_return", "val_loss",
        "train_sortino", "train_return", "train_loss", "status",
    ]

    for i, (symbol, dim, wd, lr) in enumerate(combos):
        print(f"\n[{i+1}/{len(combos)}]")
        result = run_single(
            symbol, dim, wd, lr,
            epochs=args.epochs,
            patience=args.patience,
            dry_run=args.dry_run,
            wandb_project=wandb_proj,
        )
        results.append(result)

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in sorted(results, key=lambda x: x.get("val_sortino") or -999, reverse=True):
                writer.writerow(r)

    print(f"\nLeaderboard: {csv_path}")
    ok = [r for r in results if r["status"] == "ok"]
    if ok:
        best = max(ok, key=lambda x: x.get("val_sortino") or -999)
        print(f"Best: {best['symbol']} d={best['dim']} wd={best['wd']} lr={best['lr']} "
              f"ep={best['best_epoch']} val_sort={best['val_sortino']:.4f}")


if __name__ == "__main__":
    main()
