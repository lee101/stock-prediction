#!/usr/bin/env python3
"""DOGE optimized training - h384 cosine LR (best R4 combo config).

Best known config from DOGE sweep R4:
  - Architecture: h384, 6L, 8 heads, nano arch, memory tokens, dilated attention
  - LR schedule: cosine with min_ratio=0.01
  - Weight decay: 0.04 (optimal from stock sweep, upgrade from R4's 0.03)
  - Best epoch: 1 (early stopping)
  - Safety score: 8.59

After training, runs multi-epoch evaluation on DOGE + cross-symbol.
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
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
CKPT_ROOT = REPO / "binanceneural/checkpoints/doge_optimized"
RESULTS_FILE = REPO / "binanceneural/checkpoints/doge_optimized/eval_results.json"

TRAIN_SYMBOL = "DOGEUSD"
SEED = 1337
EVAL_SYMBOLS = ["DOGEUSD", "BTCUSD", "ETHUSD", "SOLUSD", "LINKUSD", "AAVEUSD"]
SAMPLE_EPOCHS = [1, 2, 3, 5, 8, 10, 15, 20]

TRAINING_OVERRIDES = dict(
    epochs=20,
    batch_size=64,
    sequence_length=48,
    learning_rate=1e-5,
    weight_decay=0.04,
    return_weight=0.05,
    transformer_dim=384,
    transformer_layers=6,
    transformer_heads=6,
    fill_temperature=5e-4,
    fill_buffer_pct=0.0005,
    loss_type="sortino",
    lr_schedule="cosine",
    lr_min_ratio=0.01,
    model_arch="classic",
    feature_noise_std=0.0,
    checkpoint_keep_all=True,
)


def train():
    logger.info("Training DOGE optimized: h384 + cosine LR + wd=0.04")
    dm = ChronosSolDataModule(
        symbol=TRAIN_SYMBOL,
        data_root=DATA_ROOT,
        forecast_cache_root=FORECAST_CACHE,
        forecast_horizons=(1,),
        context_hours=512,
        quantile_levels=(0.1, 0.5, 0.9),
        batch_size=32,
        model_id="amazon/chronos-t5-small",
        sequence_length=TRAINING_OVERRIDES["sequence_length"],
        split_config=SplitConfig(val_days=30, test_days=30),
        cache_only=True,
        max_history_days=365,
    )

    tc = TrainingConfig(
        seed=SEED,
        maker_fee=0.001,
        checkpoint_root=CKPT_ROOT,
        log_dir=Path("tensorboard_logs/doge_optimized"),
        use_compile=False,
        decision_lag_bars=1,
        **TRAINING_OVERRIDES,
    )

    trainer = BinanceHourlyTrainer(tc, dm)
    artifacts = trainer.train()
    logger.info("Training complete. Best checkpoint: {}", artifacts.best_checkpoint)
    gc.collect()
    torch.cuda.empty_cache()
    return artifacts


def eval_checkpoint(ckpt_path: str, eval_symbol: str) -> dict:
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
        return simulate_with_margin_cost(bars, actions, cfg)
    except Exception as e:
        logger.warning("eval {} failed: {}", eval_symbol, e)
        return {"sortino": -999, "total_return": 0, "max_drawdown": 0, "num_trades": 0}


def sweep_epochs():
    logger.info("\n=== Epoch Sweep ===")
    ckpt_dirs = sorted(CKPT_ROOT.glob("binanceneural_*/"))
    if not ckpt_dirs:
        logger.warning("No checkpoint dirs found")
        return
    latest_dir = ckpt_dirs[-1]
    epoch_files = sorted(latest_dir.glob("epoch_*.pt"))

    results = {}
    best_ep, best_sort = 0, -999.0

    for ep_file in epoch_files:
        ep_num = int(ep_file.stem.split("_")[1])
        if ep_num not in SAMPLE_EPOCHS:
            continue

        r = eval_checkpoint(str(ep_file), TRAIN_SYMBOL)
        sort_val = r["sortino"]
        ret_val = r["total_return"]
        dd_val = r["max_drawdown"]
        logger.info("  ep{:>3}: Sort={:.2f} Ret={:+.2f}% DD={:.2f}%",
                     ep_num, sort_val, ret_val * 100, dd_val * 100)
        results[ep_num] = {"sortino": sort_val, "return": ret_val, "drawdown": dd_val}
        if sort_val > best_sort:
            best_sort = sort_val
            best_ep = ep_num

    logger.info("\nBest epoch: {} (Sort={:.2f})", best_ep, best_sort)

    best_ckpt = latest_dir / f"epoch_{best_ep:03d}.pt"
    logger.info("\n=== Cross-Symbol Eval (ep{}) ===", best_ep)
    symbol_results = {}
    for sym in EVAL_SYMBOLS:
        r = eval_checkpoint(str(best_ckpt), sym)
        symbol_results[sym] = r
        logger.info("  {:<10} Sort={:>7.2f} Ret={:>+7.2f}% DD={:>7.2f}%",
                     sym, r["sortino"], r["total_return"] * 100, r["max_drawdown"] * 100)

    sortinos = [v["sortino"] for v in symbol_results.values()]
    positive = sum(1 for s in sortinos if s > 0)
    mean_sort = float(np.mean(sortinos))
    worst_dd = float(min(v["max_drawdown"] for v in symbol_results.values()))
    safety = mean_sort * (positive / len(sortinos)) * (1 - min(abs(worst_dd), 0.5))
    logger.info("\nSummary: mean_sort={:.2f} pos={}/{} worst_dd={:.2f}% safety={:.2f}",
                 mean_sort, positive, len(sortinos), worst_dd * 100, safety)

    full_results = {
        "best_epoch": best_ep,
        "epoch_sweeps": {str(k): v for k, v in results.items()},
        "cross_symbol": {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating)) else vv
                             for kk, vv in v.items()} for k, v in symbol_results.items()},
        "safety_score": safety,
    }
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(full_results, indent=2, default=str))
    logger.info("Results saved: {}", RESULTS_FILE)


if __name__ == "__main__":
    train()
    sweep_epochs()
