#!/usr/bin/env python3
"""Hyperparameter sweep for neural work-steal policy.

Trains PerSymbolWorkStealPolicy with multiple architecture, training,
and loss configurations. Evaluates on multi-window holdout.
Outputs CSV with config + val_sortino + test_sortino columns.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from binance_worksteal.model import PerSymbolWorkStealPolicy
from binance_worksteal.data import (
    build_datasets,
    build_sequential_datasets,
    build_dataloader,
    FEATURE_NAMES,
)
from binance_worksteal.train_neural import (
    train_epoch,
    train_epoch_multistep,
    eval_epoch,
    eval_epoch_multistep,
    neural_backtest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SWEEP_CONFIGS = [
    # Architecture: small
    {"name": "arch_h128_t2_c2", "hidden_dim": 128, "num_temporal_layers": 2, "num_cross_layers": 2, "num_heads": 4},
    # Architecture: medium
    {"name": "arch_h256_t2_c2", "hidden_dim": 256, "num_temporal_layers": 2, "num_cross_layers": 2, "num_heads": 4},
    # Architecture: deep
    {"name": "arch_h256_t4_c2", "hidden_dim": 256, "num_temporal_layers": 4, "num_cross_layers": 2, "num_heads": 8},
    # Training: short rollout
    {"name": "train_r5", "lr": 1e-4, "wd": 0.01, "rollout_len": 5, "multistep": True},
    # Training: long rollout
    {"name": "train_r10", "lr": 1e-4, "wd": 0.01, "rollout_len": 10, "multistep": True},
    # Training: higher lr + wd
    {"name": "train_lr5e4_wd03", "lr": 5e-4, "wd": 0.03, "rollout_len": 10, "multistep": True},
    # Loss: sortino (default)
    {"name": "loss_sortino", "loss_type": "sortino"},
    # Loss: calmar
    {"name": "loss_calmar", "loss_type": "calmar"},
    # Loss: sortino_dd
    {"name": "loss_sortino_dd", "loss_type": "sortino_dd"},
]

DEFAULTS = {
    "hidden_dim": 256,
    "num_temporal_layers": 2,
    "num_cross_layers": 2,
    "num_heads": 4,
    "lr": 1e-4,
    "wd": 0.01,
    "dropout": 0.1,
    "seq_len": 30,
    "rollout_len": 10,
    "multistep": False,
    "loss_type": "sortino",
    "return_weight": 0.05,
    "maker_fee": 0.001,
    "initial_cash": 10000.0,
    "temperature": 0.02,
    "max_positions": 5,
    "max_hold_days": 14,
    "grad_clip": 1.0,
    "batch_size": 32,
    "cosine_lr": False,
    "seed": 1337,
}


def merge_config(override):
    cfg = dict(DEFAULTS)
    for k, v in override.items():
        if k != "name":
            cfg[k] = v
    cfg["name"] = override.get("name", "unnamed")
    return cfg


def build_model(cfg, n_features, n_symbols, device):
    model = PerSymbolWorkStealPolicy(
        n_features=n_features,
        n_symbols=n_symbols,
        hidden_dim=cfg["hidden_dim"],
        num_temporal_layers=cfg["num_temporal_layers"],
        num_cross_layers=cfg["num_cross_layers"],
        num_heads=cfg["num_heads"],
        seq_len=cfg["seq_len"],
        dropout=cfg["dropout"],
    ).to(device)
    return model


def train_and_eval(cfg, data_dir, symbols, epochs, device, ckpt_base_dir,
                   test_days=60, val_days=30):
    torch.manual_seed(cfg["seed"])
    name = cfg["name"]
    logger.info("=== Config: %s ===", name)

    sim_config = {
        "maker_fee": cfg["maker_fee"],
        "initial_cash": cfg["initial_cash"],
        "temperature": cfg["temperature"],
        "max_positions": cfg["max_positions"],
        "max_hold_days": cfg["max_hold_days"],
        "loss_type": cfg["loss_type"],
        "return_weight": cfg["return_weight"],
        "grad_clip": cfg["grad_clip"],
    }

    if cfg["multistep"]:
        train_ds, _, _, loaded = build_sequential_datasets(
            data_dir=data_dir, symbols=symbols, seq_len=cfg["seq_len"],
            rollout_len=cfg["rollout_len"], test_days=test_days, val_days=val_days,
        )
        _, val_ds, test_ds, _ = build_datasets(
            data_dir=data_dir, symbols=symbols, seq_len=cfg["seq_len"],
            test_days=test_days, val_days=val_days,
        )
        train_loader = build_dataloader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
        val_loader = build_dataloader(val_ds, batch_size=cfg["batch_size"], shuffle=False)
        test_loader = build_dataloader(test_ds, batch_size=cfg["batch_size"], shuffle=False)
        train_fn = train_epoch_multistep
    else:
        train_ds, val_ds, test_ds, loaded = build_datasets(
            data_dir=data_dir, symbols=symbols, seq_len=cfg["seq_len"],
            test_days=test_days, val_days=val_days,
        )
        train_loader = build_dataloader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
        val_loader = build_dataloader(val_ds, batch_size=cfg["batch_size"], shuffle=False)
        test_loader = build_dataloader(test_ds, batch_size=cfg["batch_size"], shuffle=False)
        train_fn = train_epoch

    n_features = len(FEATURE_NAMES)
    n_symbols = len(loaded)
    logger.info("Loaded %d symbols, train=%d val=%d test=%d",
                n_symbols, len(train_ds), len(val_ds), len(test_ds))

    model = build_model(cfg, n_features, n_symbols, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model params: %d", n_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    scheduler = None
    if cfg["cosine_lr"]:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_dir = ckpt_base_dir / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_sortino = float("-inf")
    best_epoch = 0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_fn(model, train_loader, optimizer, device, sim_config)
        val_metrics = eval_epoch(model, val_loader, device, sim_config)
        elapsed = time.time() - t0

        if scheduler is not None:
            scheduler.step()

        logger.info(
            "[%s] ep%d/%d %.1fs | tr_loss=%.4f tr_ret=%.5f | val_loss=%.4f val_ret=%.5f val_sort=%.4f",
            name, epoch, epochs, elapsed,
            train_metrics["loss"], train_metrics["mean_return"],
            val_metrics["loss"], val_metrics["mean_return"], val_metrics["sortino"],
        )

        entry = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_return": train_metrics["mean_return"],
            "val_loss": val_metrics["loss"],
            "val_return": val_metrics["mean_return"],
            "val_sortino": val_metrics["sortino"],
        }
        history.append(entry)

        ckpt_data = {
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": val_metrics,
            "config": {
                "n_features": n_features, "n_symbols": n_symbols,
                "hidden_dim": cfg["hidden_dim"],
                "num_temporal_layers": cfg["num_temporal_layers"],
                "num_cross_layers": cfg["num_cross_layers"],
                "num_heads": cfg["num_heads"],
                "seq_len": cfg["seq_len"],
                "dropout": cfg["dropout"],
                "symbols": loaded,
                "model_type": "persymbol",
                "num_layers": cfg["num_temporal_layers"] + cfg["num_cross_layers"],
            },
        }

        if val_metrics["sortino"] > best_val_sortino:
            best_val_sortino = val_metrics["sortino"]
            best_epoch = epoch
            torch.save(ckpt_data, ckpt_dir / "best.pt")

    # Test eval using best checkpoint
    best_ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["state_dict"])
    test_metrics = eval_epoch(model, test_loader, device, sim_config)
    logger.info("[%s] TEST: loss=%.4f ret=%.5f sort=%.4f (best_ep=%d)",
                name, test_metrics["loss"], test_metrics["mean_return"],
                test_metrics["sortino"], best_epoch)

    # Backtest eval
    bt_metrics = {}
    try:
        eq_df, trades, bt_metrics = neural_backtest(
            model, data_dir, symbols=loaded, seq_len=cfg["seq_len"],
            test_days=test_days, maker_fee=cfg["maker_fee"],
            initial_cash=cfg["initial_cash"], max_positions=cfg["max_positions"],
            max_hold_days=cfg["max_hold_days"], device=device,
        )
        if bt_metrics:
            logger.info("[%s] BACKTEST: ret=%.2f%% sort=%.2f dd=%.2f%%",
                        name, bt_metrics.get("total_return_pct", 0),
                        bt_metrics.get("sortino", 0),
                        bt_metrics.get("max_drawdown_pct", 0))
    except Exception as e:
        logger.warning("[%s] Backtest failed: %s", name, e)

    # Save training meta
    meta = {
        "name": name, "config": cfg, "history": history,
        "best_epoch": best_epoch, "best_val_sortino": best_val_sortino,
        "test_metrics": test_metrics, "backtest_metrics": bt_metrics,
        "n_params": n_params, "symbols": loaded,
    }
    with open(ckpt_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return {
        "name": name,
        "best_epoch": best_epoch,
        "val_sortino": best_val_sortino,
        "test_sortino": test_metrics["sortino"],
        "test_return": test_metrics["mean_return"],
        "test_loss": test_metrics["loss"],
        "bt_return_pct": bt_metrics.get("total_return_pct", 0),
        "bt_sortino": bt_metrics.get("sortino", 0),
        "bt_max_dd_pct": bt_metrics.get("max_drawdown_pct", 0),
        "bt_win_rate": bt_metrics.get("win_rate", 0),
        "n_params": n_params,
        "hidden_dim": cfg["hidden_dim"],
        "num_temporal_layers": cfg["num_temporal_layers"],
        "num_cross_layers": cfg["num_cross_layers"],
        "num_heads": cfg["num_heads"],
        "lr": cfg["lr"],
        "wd": cfg["wd"],
        "loss_type": cfg["loss_type"],
        "rollout_len": cfg["rollout_len"],
        "multistep": cfg["multistep"],
    }


def main():
    parser = argparse.ArgumentParser(description="Neural work-steal hyperparameter sweep")
    parser.add_argument("--data-dir", default="trainingdata/train")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--configs", type=int, default=0, help="Limit to first N configs (0=all)")
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--val-days", type=int, default=30)
    parser.add_argument("--output", default="binance_worksteal/neural_sweep_results.csv")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("binance_worksteal/checkpoints/sweep"))
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    configs = [merge_config(c) for c in SWEEP_CONFIGS]
    if args.configs > 0:
        configs = configs[:args.configs]

    for c in configs:
        c["seed"] = args.seed

    logger.info("Running sweep: %d configs x %d epochs", len(configs), args.epochs)

    results = []
    t0 = time.time()

    for i, cfg in enumerate(configs):
        logger.info("--- Sweep %d/%d: %s ---", i + 1, len(configs), cfg["name"])
        try:
            result = train_and_eval(
                cfg, args.data_dir, args.symbols, args.epochs, device,
                args.checkpoint_dir, test_days=args.test_days, val_days=args.val_days,
            )
            results.append(result)
        except Exception as e:
            logger.error("Config %s FAILED: %s", cfg["name"], e)
            import traceback
            traceback.print_exc()
            continue

    elapsed = time.time() - t0
    logger.info("Sweep complete: %d/%d configs in %.1fs", len(results), len(configs), elapsed)

    if not results:
        logger.error("No results")
        return 1

    # Sort by test sortino
    results.sort(key=lambda r: r["test_sortino"], reverse=True)

    # Write CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info("Results saved to %s", output_path)

    # Print summary
    print(f"\n{'='*80}")
    print(f"NEURAL SWEEP RESULTS ({len(results)} configs, {args.epochs} epochs)")
    print(f"{'='*80}")
    print(f"{'Rank':<5} {'Name':<25} {'ValSort':>8} {'TestSort':>8} {'BtRet%':>8} {'BtSort':>8} {'BtDD%':>8} {'Ep':>4} {'Params':>8}")
    print("-" * 80)
    for i, r in enumerate(results):
        print(f"{i+1:<5} {r['name']:<25} {r['val_sortino']:>8.3f} {r['test_sortino']:>8.3f} "
              f"{r['bt_return_pct']:>8.2f} {r['bt_sortino']:>8.2f} {r['bt_max_dd_pct']:>8.2f} "
              f"{r['best_epoch']:>4} {r['n_params']:>8}")
    print(f"\nTop config: {results[0]['name']} (test_sortino={results[0]['test_sortino']:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
