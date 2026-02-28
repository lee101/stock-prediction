#!/usr/bin/env python3
"""Sweep training configs focused on generalization.

Tests: loss types, LR schedules, regularization, model size.
Trains on one symbol, evaluates on ALL symbols to measure transfer.
"""
from __future__ import annotations
import argparse, gc, json, sys, time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
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
FORECAST_CACHE = REPO / "binanceneural/forecast_cache"
CKPT_ROOT = REPO / "binanceleveragesui/checkpoints"
RESULTS_DIR = REPO / "binanceleveragesui/sweep_results"

# symbols with good forecast caches
EVAL_SYMBOLS = ["DOGEUSD", "BTCUSD", "ETHUSD", "SOLUSD", "LINKUSD",
                "LTCUSD", "UNIUSD", "AAVEUSD", "AVAXUSD", "ARBUSDT", "OPUSDT", "SUIUSDT"]

CONFIGS = {
    # baseline: current best config
    "baseline": dict(
        epochs=20, batch_size=16, sequence_length=72, learning_rate=1e-4,
        weight_decay=0.03, return_weight=0.10, transformer_dim=256,
        transformer_layers=4, transformer_heads=8, fill_temperature=0.1,
        fill_buffer_pct=0.0005, loss_type="sortino", lr_schedule="none",
        feature_noise_std=0.0, transformer_dropout=0.1,
    ),
    # calmar loss: penalizes drawdown explicitly
    "calmar_loss": dict(
        epochs=20, batch_size=16, sequence_length=72, learning_rate=1e-4,
        weight_decay=0.03, return_weight=0.10, transformer_dim=256,
        transformer_layers=4, transformer_heads=8, fill_temperature=0.1,
        fill_buffer_pct=0.0005, loss_type="calmar", lr_schedule="none",
        feature_noise_std=0.0, transformer_dropout=0.1,
    ),
    # sortino_dd loss
    "sortino_dd": dict(
        epochs=20, batch_size=16, sequence_length=72, learning_rate=1e-4,
        weight_decay=0.03, return_weight=0.10, transformer_dim=256,
        transformer_layers=4, transformer_heads=8, fill_temperature=0.1,
        fill_buffer_pct=0.0005, loss_type="sortino_dd", dd_penalty=2.0,
        lr_schedule="none", feature_noise_std=0.0, transformer_dropout=0.1,
    ),
    # cosine LR schedule
    "cosine_lr": dict(
        epochs=20, batch_size=16, sequence_length=72, learning_rate=1e-4,
        weight_decay=0.03, return_weight=0.10, transformer_dim=256,
        transformer_layers=4, transformer_heads=8, fill_temperature=0.1,
        fill_buffer_pct=0.0005, loss_type="sortino", lr_schedule="cosine",
        lr_warmdown_ratio=0.3, lr_min_ratio=0.01,
        feature_noise_std=0.0, transformer_dropout=0.1,
    ),
    # feature noise regularization
    "fnoise_02": dict(
        epochs=20, batch_size=16, sequence_length=72, learning_rate=1e-4,
        weight_decay=0.03, return_weight=0.10, transformer_dim=256,
        transformer_layers=4, transformer_heads=8, fill_temperature=0.1,
        fill_buffer_pct=0.0005, loss_type="sortino", lr_schedule="none",
        feature_noise_std=0.02, transformer_dropout=0.1,
    ),
    # heavy regularization: more dropout + wd + fnoise
    "heavy_reg": dict(
        epochs=20, batch_size=16, sequence_length=72, learning_rate=1e-4,
        weight_decay=0.08, return_weight=0.10, transformer_dim=256,
        transformer_layers=4, transformer_heads=8, fill_temperature=0.1,
        fill_buffer_pct=0.0005, loss_type="sortino", lr_schedule="none",
        feature_noise_std=0.03, transformer_dropout=0.2,
    ),
    # smaller model: less capacity = less overfitting
    "small_model": dict(
        epochs=20, batch_size=16, sequence_length=72, learning_rate=1e-4,
        weight_decay=0.03, return_weight=0.10, transformer_dim=128,
        transformer_layers=2, transformer_heads=4, fill_temperature=0.1,
        fill_buffer_pct=0.0005, loss_type="sortino", lr_schedule="none",
        feature_noise_std=0.0, transformer_dropout=0.1,
    ),
    # lower rw: less aggressive return optimization
    "low_rw": dict(
        epochs=20, batch_size=16, sequence_length=72, learning_rate=1e-4,
        weight_decay=0.03, return_weight=0.03, transformer_dim=256,
        transformer_layers=4, transformer_heads=8, fill_temperature=0.1,
        fill_buffer_pct=0.0005, loss_type="sortino", lr_schedule="none",
        feature_noise_std=0.0, transformer_dropout=0.1,
    ),
    # combined: calmar + cosine + fnoise + heavy wd
    "combined": dict(
        epochs=20, batch_size=16, sequence_length=72, learning_rate=1e-4,
        weight_decay=0.05, return_weight=0.10, transformer_dim=256,
        transformer_layers=4, transformer_heads=8, fill_temperature=0.1,
        fill_buffer_pct=0.0005, loss_type="calmar", lr_schedule="cosine",
        lr_warmdown_ratio=0.3, lr_min_ratio=0.01,
        feature_noise_std=0.02, transformer_dropout=0.15,
    ),
    # === Architecture experiments ===
    # memory tokens: 8 learnable global memory slots
    "mem8": dict(
        epochs=20, batch_size=16, sequence_length=72, learning_rate=1e-4,
        weight_decay=0.03, return_weight=0.10, transformer_dim=256,
        transformer_layers=4, transformer_heads=8, fill_temperature=0.1,
        fill_buffer_pct=0.0005, loss_type="sortino", lr_schedule="none",
        model_arch="nano", num_memory_tokens=8,
        feature_noise_std=0.0, transformer_dropout=0.1,
    ),
    # dilated attention: heads at strides 1,4,24,72 for multi-scale
    "dilated_1_4_24_72": dict(
        epochs=20, batch_size=16, sequence_length=72, learning_rate=1e-4,
        weight_decay=0.03, return_weight=0.10, transformer_dim=256,
        transformer_layers=4, transformer_heads=8, fill_temperature=0.1,
        fill_buffer_pct=0.0005, loss_type="sortino", lr_schedule="none",
        model_arch="nano", dilated_strides="1,4,24,72",
        feature_noise_std=0.0, transformer_dropout=0.1,
    ),
    # memory + dilated combined
    "mem8_dilated": dict(
        epochs=20, batch_size=16, sequence_length=72, learning_rate=1e-4,
        weight_decay=0.03, return_weight=0.10, transformer_dim=256,
        transformer_layers=4, transformer_heads=8, fill_temperature=0.1,
        fill_buffer_pct=0.0005, loss_type="sortino", lr_schedule="none",
        model_arch="nano", num_memory_tokens=8, dilated_strides="1,4,24,72",
        feature_noise_std=0.0, transformer_dropout=0.1,
    ),
    # memory + dilated + regularization (best generalization candidate)
    "mem8_dilated_reg": dict(
        epochs=20, batch_size=16, sequence_length=72, learning_rate=1e-4,
        weight_decay=0.05, return_weight=0.10, transformer_dim=256,
        transformer_layers=4, transformer_heads=8, fill_temperature=0.1,
        fill_buffer_pct=0.0005, loss_type="sortino", lr_schedule="cosine",
        lr_warmdown_ratio=0.3, lr_min_ratio=0.01,
        model_arch="nano", num_memory_tokens=8, dilated_strides="1,4,24,72",
        feature_noise_std=0.02, transformer_dropout=0.15,
    ),
    # longer sequence with dilated attention to capture more context
    "long_seq_dilated": dict(
        epochs=20, batch_size=8, sequence_length=168, learning_rate=1e-4,
        weight_decay=0.03, return_weight=0.10, transformer_dim=256,
        transformer_layers=4, transformer_heads=8, fill_temperature=0.1,
        fill_buffer_pct=0.0005, loss_type="sortino", lr_schedule="none",
        model_arch="nano", num_memory_tokens=4, dilated_strides="1,4,24,72",
        attention_window=72,
        feature_noise_std=0.0, transformer_dropout=0.1,
    ),
    # nano baseline (no memory/dilated) for fair comparison
    "nano_baseline": dict(
        epochs=20, batch_size=16, sequence_length=72, learning_rate=1e-4,
        weight_decay=0.03, return_weight=0.10, transformer_dim=256,
        transformer_layers=4, transformer_heads=8, fill_temperature=0.1,
        fill_buffer_pct=0.0005, loss_type="sortino", lr_schedule="none",
        model_arch="nano",
        feature_noise_std=0.0, transformer_dropout=0.1,
    ),
    # === Round 2: building on mem8_dilated winner ===
    # value embeddings every 2 layers
    "value_embed": dict(
        epochs=20, batch_size=16, sequence_length=72, learning_rate=1e-4,
        weight_decay=0.03, return_weight=0.10, transformer_dim=256,
        transformer_layers=4, transformer_heads=8, fill_temperature=0.1,
        fill_buffer_pct=0.0005, loss_type="sortino", lr_schedule="none",
        model_arch="nano", num_memory_tokens=8, dilated_strides="1,4,24,72",
        use_value_embedding=True, value_embedding_every=2, value_embedding_scale=0.1,
        feature_noise_std=0.0, transformer_dropout=0.1,
    ),
    # learnable residual scalars (skip connections)
    "residual_scalars": dict(
        epochs=20, batch_size=16, sequence_length=72, learning_rate=1e-4,
        weight_decay=0.03, return_weight=0.10, transformer_dim=256,
        transformer_layers=4, transformer_heads=8, fill_temperature=0.1,
        fill_buffer_pct=0.0005, loss_type="sortino", lr_schedule="none",
        model_arch="nano", num_memory_tokens=8, dilated_strides="1,4,24,72",
        use_residual_scalars=True, residual_scale_init=1.0, skip_scale_init=0.0,
        feature_noise_std=0.0, transformer_dropout=0.1,
    ),
    # RoPE base=72 for hourly periodicity (1 day = 24 positions)
    "rope72": dict(
        epochs=20, batch_size=16, sequence_length=72, learning_rate=1e-4,
        weight_decay=0.03, return_weight=0.10, transformer_dim=256,
        transformer_layers=4, transformer_heads=8, fill_temperature=0.1,
        fill_buffer_pct=0.0005, loss_type="sortino", lr_schedule="none",
        model_arch="nano", num_memory_tokens=8, dilated_strides="1,4,24,72",
        rope_base=72.0,
        feature_noise_std=0.0, transformer_dropout=0.1,
    ),
    # deeper: 6 layers
    "deeper_6L": dict(
        epochs=20, batch_size=16, sequence_length=72, learning_rate=1e-4,
        weight_decay=0.03, return_weight=0.10, transformer_dim=256,
        transformer_layers=6, transformer_heads=8, fill_temperature=0.1,
        fill_buffer_pct=0.0005, loss_type="sortino", lr_schedule="none",
        model_arch="nano", num_memory_tokens=8, dilated_strides="1,4,24,72",
        feature_noise_std=0.0, transformer_dropout=0.1,
    ),
    # 16 memory tokens
    "mem16_dilated": dict(
        epochs=20, batch_size=16, sequence_length=72, learning_rate=1e-4,
        weight_decay=0.03, return_weight=0.10, transformer_dim=256,
        transformer_layers=4, transformer_heads=8, fill_temperature=0.1,
        fill_buffer_pct=0.0005, loss_type="sortino", lr_schedule="none",
        model_arch="nano", num_memory_tokens=16, dilated_strides="1,4,24,72",
        feature_noise_std=0.0, transformer_dropout=0.1,
    ),
    # wider FFN (8x ratio instead of 4x)
    "wider_mlp8": dict(
        epochs=20, batch_size=16, sequence_length=72, learning_rate=1e-4,
        weight_decay=0.03, return_weight=0.10, transformer_dim=256,
        transformer_layers=4, transformer_heads=8, fill_temperature=0.1,
        fill_buffer_pct=0.0005, loss_type="sortino", lr_schedule="none",
        model_arch="nano", num_memory_tokens=8, dilated_strides="1,4,24,72",
        mlp_ratio=8.0,
        feature_noise_std=0.0, transformer_dropout=0.1,
    ),
    # finer dilated strides: 1,2,6,24
    "fine_strides": dict(
        epochs=20, batch_size=16, sequence_length=72, learning_rate=1e-4,
        weight_decay=0.03, return_weight=0.10, transformer_dim=256,
        transformer_layers=4, transformer_heads=8, fill_temperature=0.1,
        fill_buffer_pct=0.0005, loss_type="sortino", lr_schedule="none",
        model_arch="nano", num_memory_tokens=8, dilated_strides="1,2,6,24",
        feature_noise_std=0.0, transformer_dropout=0.1,
    ),
    # kitchen sink: value_embed + residual_scalars + rope72 + 6L
    "kitchen_sink": dict(
        epochs=20, batch_size=16, sequence_length=72, learning_rate=1e-4,
        weight_decay=0.03, return_weight=0.10, transformer_dim=256,
        transformer_layers=6, transformer_heads=8, fill_temperature=0.1,
        fill_buffer_pct=0.0005, loss_type="sortino", lr_schedule="none",
        model_arch="nano", num_memory_tokens=8, dilated_strides="1,4,24,72",
        use_value_embedding=True, value_embedding_every=2, value_embedding_scale=0.1,
        use_residual_scalars=True, residual_scale_init=1.0, skip_scale_init=0.0,
        rope_base=72.0,
        feature_noise_std=0.0, transformer_dropout=0.1,
    ),
}


def train_config(train_symbol: str, config_name: str, overrides: dict) -> Path:
    tag = f"{train_symbol}_gen_{config_name}"
    ckpt_root = CKPT_ROOT / tag
    logger.info(f"\n{'='*60}\nTraining: {tag}\n{'='*60}")

    dm = ChronosSolDataModule(
        symbol=train_symbol,
        data_root=DATA_ROOT,
        forecast_cache_root=FORECAST_CACHE,
        forecast_horizons=(1,),
        context_hours=512,
        quantile_levels=(0.1, 0.5, 0.9),
        batch_size=32,
        model_id="amazon/chronos-t5-small",
        sequence_length=overrides.get("sequence_length", 72),
        split_config=SplitConfig(val_days=30, test_days=30),
        cache_only=True,
        max_history_days=365,
    )

    tc = TrainingConfig(
        seed=1337,
        maker_fee=0.001,
        checkpoint_root=ckpt_root,
        log_dir=Path("tensorboard_logs/gen_sweep"),
        use_compile=False,
        decision_lag_bars=1,
        **overrides,
    )

    trainer = BinanceHourlyTrainer(tc, dm)
    trainer.train()
    gc.collect()
    torch.cuda.empty_cache()
    return ckpt_root


def eval_checkpoint(ckpt_path: Path, eval_symbol: str) -> dict:
    """Evaluate a checkpoint on a given symbol."""
    try:
        model, normalizer, feature_columns, meta = load_policy_checkpoint(ckpt_path, device="cuda")
        seq_len = meta.get("sequence_length", 72)

        dm = ChronosSolDataModule(
            symbol=eval_symbol,
            data_root=DATA_ROOT,
            forecast_cache_root=FORECAST_CACHE,
            forecast_horizons=(1,),
            context_hours=512,
            quantile_levels=(0.1, 0.5, 0.9),
            batch_size=32,
            model_id="amazon/chronos-t5-small",
            sequence_length=seq_len,
            split_config=SplitConfig(val_days=30, test_days=30),
            cache_only=True,
            max_history_days=365,
        )

        actions = generate_actions_from_frame(
            model=model, frame=dm.test_frame, feature_columns=feature_columns,
            normalizer=normalizer, sequence_length=seq_len, horizon=1,
        )
        bars = dm.test_frame[["timestamp", "symbol", "open", "high", "low", "close"]].copy()

        cfg = LeverageConfig(
            symbol=eval_symbol, max_leverage=1.0, can_short=False,
            maker_fee=0.001, margin_hourly_rate=0.0, initial_cash=10000.0,
            fill_buffer_pct=0.0005, decision_lag_bars=1, min_edge=0.0,
            max_hold_bars=6, intensity_scale=5.0,
        )
        r = simulate_with_margin_cost(bars, actions, cfg)
        dd = abs(r["max_drawdown"]) if r["max_drawdown"] != 0 else 1e-6
        r["calmar"] = r["total_return"] / dd
        r["composite"] = r["sortino"] * (1 - min(dd, 1.0))
        return r
    except Exception as e:
        logger.warning(f"  eval {eval_symbol} failed: {e}")
        return {"sortino": -999, "total_return": 0, "max_drawdown": 0, "num_trades": 0,
                "calmar": 0, "composite": -999}


def sweep_epochs_cross_symbol(ckpt_root: Path, eval_symbols: List[str]) -> List[dict]:
    """Evaluate all epochs across all eval symbols."""
    results = []
    # find checkpoint dirs
    ckpt_dirs = sorted(ckpt_root.glob("binanceneural_*/"))
    if not ckpt_dirs:
        return results

    latest_dir = ckpt_dirs[-1]
    epoch_files = sorted(latest_dir.glob("epoch_*.pt"))

    # sample epochs: 1,2,3,5,8,10,15,20 for efficiency
    sample_epochs = [1, 2, 3, 5, 8, 10, 15, 20]
    for ep_file in epoch_files:
        ep_num = int(ep_file.stem.split("_")[1])
        if ep_num not in sample_epochs:
            continue

        for sym in eval_symbols:
            r = eval_checkpoint(ep_file, sym)
            r["epoch"] = ep_num
            r["eval_symbol"] = sym
            r["ckpt"] = str(ep_file)
            results.append(r)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-symbol", type=str, default="DOGEUSD")
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-sep config names (default: all)")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--eval-symbols", type=str, default=None)
    args = parser.parse_args()

    configs = args.configs.split(",") if args.configs else list(CONFIGS.keys())
    eval_syms = args.eval_symbols.split(",") if args.eval_symbols else EVAL_SYMBOLS

    all_results = {}
    for cfg_name in configs:
        if cfg_name not in CONFIGS:
            logger.warning(f"Unknown config: {cfg_name}")
            continue

        overrides = CONFIGS[cfg_name]

        # train
        if not args.skip_train:
            ckpt_root = train_config(args.train_symbol, cfg_name, overrides)
        else:
            ckpt_root = CKPT_ROOT / f"{args.train_symbol}_gen_{cfg_name}"
            if not ckpt_root.exists():
                logger.warning(f"No checkpoints for {cfg_name}")
                continue

        # eval across symbols
        logger.info(f"\nEvaluating {cfg_name} across {len(eval_syms)} symbols...")
        evals = sweep_epochs_cross_symbol(ckpt_root, eval_syms)
        all_results[cfg_name] = evals

        # print per-config summary
        if evals:
            # best epoch by mean sortino across symbols
            epoch_scores = {}
            for r in evals:
                ep = r["epoch"]
                if ep not in epoch_scores:
                    epoch_scores[ep] = []
                epoch_scores[ep].append(r["sortino"])
            best_ep = max(epoch_scores, key=lambda e: np.mean(epoch_scores[e]))
            mean_sort = np.mean(epoch_scores[best_ep])
            pos_count = sum(1 for s in epoch_scores[best_ep] if s > 0)

            logger.info(f"  {cfg_name} best_ep={best_ep}: mean_sort={mean_sort:.2f}, "
                        f"positive={pos_count}/{len(epoch_scores[best_ep])}")

            # per-symbol at best epoch
            for r in sorted([e for e in evals if e["epoch"] == best_ep],
                            key=lambda x: x["sortino"], reverse=True):
                sym = r["eval_symbol"]
                logger.info(f"    {sym:<12} Sort={r['sortino']:>7.2f} Ret={r['total_return']*100:>6.1f}% "
                            f"DD={r['max_drawdown']*100:>6.1f}% Trades={r['num_trades']}")

        gc.collect()
        torch.cuda.empty_cache()

    # final summary
    print(f"\n{'Config':<16} {'BestEp':>6} {'MeanSort':>9} {'Pos/Total':>10} {'TrainSort':>10} {'TrainRet':>9}")
    print("-" * 65)
    for cfg_name, evals in all_results.items():
        if not evals:
            continue
        epoch_scores = {}
        train_scores = {}
        for r in evals:
            ep = r["epoch"]
            if ep not in epoch_scores:
                epoch_scores[ep] = []
                train_scores[ep] = None
            epoch_scores[ep].append(r["sortino"])
            if r["eval_symbol"] == args.train_symbol:
                train_scores[ep] = r

        best_ep = max(epoch_scores, key=lambda e: np.mean(epoch_scores[e]))
        mean_sort = np.mean(epoch_scores[best_ep])
        pos = sum(1 for s in epoch_scores[best_ep] if s > 0)
        total = len(epoch_scores[best_ep])
        ts = train_scores.get(best_ep)
        ts_sort = ts["sortino"] if ts else 0
        ts_ret = ts["total_return"] * 100 if ts else 0

        print(f"{cfg_name:<16} {best_ep:>6} {mean_sort:>9.2f} {pos:>4}/{total:<5} "
              f"{ts_sort:>10.2f} {ts_ret:>8.1f}%")

    out = RESULTS_DIR / "generalization_sweep_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2, default=str))
    logger.info(f"\nResults: {out}")


if __name__ == "__main__":
    main()
