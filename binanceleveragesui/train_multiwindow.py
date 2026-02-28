#!/usr/bin/env python3
"""Train DOGE with multi-window robustness loss."""
from __future__ import annotations
import argparse, json, sys
from dataclasses import asdict
from pathlib import Path
import torch
import pandas as pd
from loguru import logger

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from binanceneural.config import TrainingConfig
from binanceneural.trainer import BinanceHourlyTrainer
from binanceneural.inference import generate_actions_from_frame
from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import load_policy_checkpoint
from binanceleveragesui.run_leverage_sweep import LeverageConfig, simulate_with_margin_cost

DATA_ROOT = REPO / "trainingdatahourlybinance"
CHECKPOINT_ROOT = REPO / "binanceleveragesui" / "checkpoints"


def evaluate_checkpoint(ckpt_path, dm, symbol="DOGEUSD"):
    model, norm, fcols, meta = load_policy_checkpoint(str(ckpt_path), device="cuda")
    seq = meta.get("sequence_length", 72)
    frame = dm.full_frame.copy()
    actions = generate_actions_from_frame(
        model=model, frame=frame, feature_columns=fcols,
        normalizer=norm, sequence_length=seq, horizon=1,
    )
    last_ts = frame.timestamp.max()
    windows = [7, 14, 30, 60, 90, 180]
    results = {}
    for wd in windows:
        st = last_ts - pd.Timedelta(days=wd)
        wb = frame[frame.timestamp >= st].copy()
        wa = actions[actions.timestamp >= st].copy()
        cfg = LeverageConfig(
            symbol=symbol, max_leverage=1.0, maker_fee=0.001,
            initial_cash=10000.0, fill_buffer_pct=0.0,
            decision_lag_bars=1, intensity_scale=5.0, max_hold_bars=6,
        )
        m = simulate_with_margin_cost(wb, wa, cfg)
        results[wd] = {"ret": m["total_return"]*100, "sort": m["sortino"], "dd": m["max_drawdown"]*100}
    del model; torch.cuda.empty_cache()
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="DOGEUSD")
    p.add_argument("--rw", type=float, default=0.30)
    p.add_argument("--wd", type=float, default=0.03)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--loss-type", default="multiwindow_dd")
    p.add_argument("--dd-penalty", type=float, default=2.0)
    p.add_argument("--multiwindow-fractions", default="0.25,0.5,0.75,1.0")
    p.add_argument("--aggregation", default="minimax")
    p.add_argument("--feature-noise", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--lag", type=int, default=1)
    p.add_argument("--name", default=None)
    args = p.parse_args()

    name = args.name or f"{args.symbol}_mw_{args.loss_type}_rw{int(args.rw*100):02d}"
    logger.info(f"=== Training {name} loss={args.loss_type} agg={args.aggregation} ===")

    dm = ChronosSolDataModule(
        symbol=args.symbol, data_root=DATA_ROOT,
        forecast_cache_root=REPO / "binanceneural/forecast_cache",
        forecast_horizons=(1,), context_hours=512,
        quantile_levels=(0.1, 0.5, 0.9), batch_size=32,
        model_id="amazon/chronos-t5-small", sequence_length=72,
        split_config=SplitConfig(val_days=30, test_days=30),
        cache_only=True, max_history_days=365,
    )

    ckpt_root = CHECKPOINT_ROOT / name
    tc = TrainingConfig(
        epochs=args.epochs, batch_size=16, sequence_length=72,
        learning_rate=args.lr, weight_decay=args.wd,
        return_weight=args.rw, seed=args.seed,
        loss_type=args.loss_type,
        dd_penalty=args.dd_penalty,
        multiwindow_fractions=args.multiwindow_fractions,
        multiwindow_aggregation=args.aggregation,
        decision_lag_bars=args.lag,
        feature_noise_std=args.feature_noise,
        transformer_dim=256, transformer_layers=4, transformer_heads=8,
        maker_fee=0.001, fill_temperature=0.1,
        checkpoint_root=ckpt_root,
        log_dir=Path("tensorboard_logs/binanceleveragesui_mw"),
        use_compile=False,
    )

    trainer = BinanceHourlyTrainer(tc, dm)
    artifacts = trainer.train()

    logger.info("\n=== Evaluating all epochs ===")
    epoch_files = sorted(ckpt_root.rglob("epoch_*.pt"))
    all_eval = {}
    for ep_path in epoch_files:
        ep_name = ep_path.stem
        try:
            results = evaluate_checkpoint(ep_path, dm, args.symbol)
            min_sort = min(r["sort"] for r in results.values())
            avg_ret = sum(r["ret"] for r in results.values()) / len(results)
            all_eval[ep_name] = {"results": results, "min_sort": min_sort, "avg_ret": avg_ret}
            logger.info(f"  {ep_name}: min_sort={min_sort:.2f} avg_ret={avg_ret:+.1f}%  " +
                " ".join(f"{w}d:{r['ret']:+.1f}%({r['sort']:.1f})" for w, r in sorted(results.items())))
        except Exception as e:
            logger.warning(f"  {ep_name}: FAILED {e}")

    best_ep = max(all_eval, key=lambda k: all_eval[k]["min_sort"]) if all_eval else None
    if best_ep:
        logger.info(f"\nBest by min_sort: {best_ep} = {all_eval[best_ep]['min_sort']:.2f}")

    out = REPO / "binanceleveragesui" / f"mw_results_{name}.json"
    with open(out, "w") as f:
        json.dump({"config": asdict(tc), "epochs": {k: v for k, v in all_eval.items()}}, f, indent=2, default=str)
    logger.info(f"Saved: {out}")


if __name__ == "__main__":
    main()
