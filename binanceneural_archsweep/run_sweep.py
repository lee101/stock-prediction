#!/usr/bin/env python3
"""Architecture sweep: train many configs, robustly evaluate, produce leaderboard.

Usage:
    python -m binanceneural_archsweep.run_sweep --symbols BTCUSD,ETHUSD,SOLUSD --quick
    python -m binanceneural_archsweep.run_sweep --symbols BTCUSD --configs nano_medium,mamba_medium
    python -m binanceneural_archsweep.run_sweep --all
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import torch

os.environ.setdefault("TORCH_NO_COMPILE", "1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def train_config(
    config_name: str,
    config_overrides: dict,
    symbol: str,
    checkpoint_root: Path,
) -> Path | None:
    from binanceneural.config import DatasetConfig, TrainingConfig
    from binanceneural.data import BinanceHourlyDataModule
    from binanceneural.trainer import BinanceHourlyTrainer

    run_name = f"archsweep_{config_name}_{symbol}_{time.strftime('%Y%m%d_%H%M%S')}"

    ds_overrides = config_overrides.pop("dataset", {})
    ds_cfg = DatasetConfig(symbol=symbol, **ds_overrides)

    train_cfg = TrainingConfig(
        **{k: v for k, v in config_overrides.items() if hasattr(TrainingConfig, k)},
        run_name=run_name,
        checkpoint_root=checkpoint_root,
    )
    train_cfg.dataset = ds_cfg

    data = BinanceHourlyDataModule(ds_cfg)

    trainer = BinanceHourlyTrainer(train_cfg, data)
    artifacts = trainer.train()

    if artifacts.best_checkpoint:
        logger.info("[%s/%s] best checkpoint: %s", config_name, symbol, artifacts.best_checkpoint)
    return artifacts.best_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default="BTCUSD,ETHUSD,SOLUSD")
    p.add_argument("--configs", default=None, help="Comma-sep config names (default: all)")
    p.add_argument("--quick", action="store_true", help="Use QUICK_CONFIGS subset")
    p.add_argument("--all", action="store_true", help="Run ALL_CONFIGS")
    p.add_argument("--checkpoint-root", default="binanceneural/checkpoints")
    p.add_argument("--leaderboard", default="binanceneural_archsweep/leaderboard.csv")
    p.add_argument("--eval-only", default=None, help="Skip training, eval this checkpoint dir")
    p.add_argument("--skip-eval", action="store_true", help="Train only, skip eval")
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--robust-eval-quick", action="store_true", help="Fewer eval scenarios")
    return p.parse_args()


def main():
    args = parse_args()
    from .sweep_configs import ALL_CONFIGS, QUICK_CONFIGS
    from .robust_eval import robust_evaluate, evaluate_multi_symbol

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    checkpoint_root = Path(args.checkpoint_root)
    leaderboard_path = Path(args.leaderboard)

    if args.configs:
        config_names = [c.strip() for c in args.configs.split(",")]
        configs = {k: ALL_CONFIGS[k] for k in config_names if k in ALL_CONFIGS}
    elif args.quick:
        configs = QUICK_CONFIGS
    else:
        configs = ALL_CONFIGS

    logger.info("Sweep: %d configs x %d symbols", len(configs), len(symbols))
    logger.info("Configs: %s", list(configs.keys()))
    logger.info("Symbols: %s", symbols)

    eval_kwargs = {}
    if args.robust_eval_quick:
        eval_kwargs = dict(
            fee_scenarios=[0.001, 0.0015],
            fill_buffers=[5.0, 10.0],
            lags=[1],
            intensities=[0.8, 1.0, 1.2],
            cash_levels=[10000.0],
            val_periods=[30, 90],
        )

    results_rows = []

    for config_name, config_overrides in configs.items():
        for symbol in symbols:
            logger.info("=== %s / %s ===", config_name, symbol)
            t0 = time.time()

            try:
                cfg_copy = {k: (v.copy() if isinstance(v, dict) else v) for k, v in config_overrides.items()}

                if args.eval_only:
                    ckpt_dir = Path(args.eval_only)
                    ckpts = sorted(ckpt_dir.glob("*.pt"))
                    best_ckpt = ckpts[-1] if ckpts else None
                else:
                    best_ckpt = train_config(config_name, cfg_copy, symbol, checkpoint_root)

                train_time = time.time() - t0

                if best_ckpt and not args.skip_eval:
                    eval_result = robust_evaluate(
                        config_name, best_ckpt, symbol=symbol,
                        sequence_length=config_overrides.get("sequence_length", 96),
                        horizon=args.horizon,
                        **eval_kwargs,
                    )
                    row = {
                        "config": config_name,
                        "symbol": symbol,
                        "robust_score": f"{eval_result.robust_score:.6f}",
                        "mean_return": f"{eval_result.mean_return:.4f}",
                        "mean_sortino": f"{eval_result.mean_sortino:.2f}",
                        "worst_return": f"{eval_result.worst_return:.4f}",
                        "p25_return": f"{eval_result.p25_return:.4f}",
                        "p25_sortino": f"{eval_result.p25_sortino:.2f}",
                        "profitable_pct": f"{eval_result.profitable_pct:.1f}",
                        "best_intensity": f"{eval_result.best_intensity:.1f}",
                        "train_seconds": f"{train_time:.0f}",
                        "eval_seconds": f"{eval_result.eval_seconds:.0f}",
                        "checkpoint": str(best_ckpt),
                        "arch": config_overrides.get("model_arch", "nano"),
                        "dim": config_overrides.get("transformer_dim", 256),
                        "layers": config_overrides.get("transformer_layers", 4),
                        "heads": config_overrides.get("transformer_heads", 8),
                    }
                    results_rows.append(row)

                    logger.info(
                        "[%s/%s] robust=%.4f mean_ret=%.4f sortino=%.2f profitable=%.1f%% (%.0fs train, %.0fs eval)",
                        config_name, symbol, eval_result.robust_score, eval_result.mean_return,
                        eval_result.mean_sortino, eval_result.profitable_pct, train_time, eval_result.eval_seconds,
                    )

                    # Write leaderboard incrementally
                    _write_leaderboard(leaderboard_path, results_rows)
                elif args.skip_eval and best_ckpt:
                    row = {
                        "config": config_name, "symbol": symbol,
                        "checkpoint": str(best_ckpt),
                        "train_seconds": f"{train_time:.0f}",
                        "arch": config_overrides.get("model_arch", "nano"),
                    }
                    results_rows.append(row)
                    _write_leaderboard(leaderboard_path, results_rows)

            except Exception as e:
                logger.error("[%s/%s] FAILED: %s", config_name, symbol, e)
                traceback.print_exc()
                results_rows.append({
                    "config": config_name, "symbol": symbol,
                    "error": str(e)[:200],
                })
                _write_leaderboard(leaderboard_path, results_rows)

    logger.info("Sweep complete. Leaderboard: %s", leaderboard_path)
    _print_leaderboard(results_rows)


def _write_leaderboard(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    all_keys = []
    for r in rows:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        w.writerows(rows)


def _print_leaderboard(rows: list[dict]):
    sorted_rows = sorted(rows, key=lambda r: float(r.get("robust_score", 0)), reverse=True)
    print("\n" + "=" * 100)
    print("ARCHITECTURE SWEEP LEADERBOARD")
    print("=" * 100)
    print(f"{'Rank':>4} {'Config':<30} {'Symbol':<8} {'Robust':>8} {'MeanRet':>8} {'Sortino':>8} {'P25Ret':>8} {'Prof%':>6} {'BestInt':>7}")
    print("-" * 100)
    for i, r in enumerate(sorted_rows, 1):
        if "error" in r:
            print(f"{i:4d} {r.get('config','?'):<30} {r.get('symbol','?'):<8} ERROR: {r.get('error','')[:50]}")
        else:
            print(f"{i:4d} {r.get('config','?'):<30} {r.get('symbol','?'):<8} "
                  f"{r.get('robust_score','0'):>8} {r.get('mean_return','0'):>8} "
                  f"{r.get('mean_sortino','0'):>8} {r.get('p25_return','0'):>8} "
                  f"{r.get('profitable_pct','0'):>6} {r.get('best_intensity','1'):>7}")
    print("=" * 100)


if __name__ == "__main__":
    main()
