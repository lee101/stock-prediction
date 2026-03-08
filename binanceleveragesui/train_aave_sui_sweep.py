#!/usr/bin/env python3
"""Train + eval sweep for AAVE and SUI (0-fee FDUSD pair)."""
from __future__ import annotations
import argparse, gc, json, sys
from pathlib import Path
from typing import Dict

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
CKPT_ROOT = REPO / "binanceleveragesui/checkpoints"
RESULTS_FILE = REPO / "binanceleveragesui/aave_sui_sweep_results.json"
SEED = 1337
MARGIN_HOURLY_RATE = 0.0000025457
SAMPLE_EPOCHS = [1, 2, 3, 5, 8, 10, 15, 20]

EVAL_WINDOWS = [
    (30, 3, "3d"), (30, 7, "7d"), (30, 14, "14d"),
    (30, 30, "30d"), (30, 60, "60d"), (30, 90, "90d"),
]
LEVERAGE_LEVELS = [1.0, 1.5, 2.0, 3.0]

# winning DOGE config as base
BASE = dict(
    epochs=20, batch_size=16, sequence_length=72, learning_rate=1e-4,
    weight_decay=0.03, return_weight=0.10, transformer_dim=384,
    transformer_layers=6, transformer_heads=8, fill_temperature=0.1,
    fill_buffer_pct=0.0005, loss_type="sortino", lr_schedule="cosine",
    lr_min_ratio=0.01, model_arch="nano", num_memory_tokens=8,
    dilated_strides="1,2,6,24", feature_noise_std=0.0,
    transformer_dropout=0.1,
)

CONFIGS = {
    # --- AAVE configs (10bps fee, standard margin) ---
    "AAVE_h384_cosine": {
        "symbol": "AAVEUSD", "data_symbol": "AAVEUSD",
        "maker_fee": 0.001, **BASE,
    },
    "AAVE_h384_cosine_rw05": {
        "symbol": "AAVEUSD", "data_symbol": "AAVEUSD",
        "maker_fee": 0.001, **BASE, "return_weight": 0.05,
    },
    "AAVE_h384_cosine_rw20": {
        "symbol": "AAVEUSD", "data_symbol": "AAVEUSD",
        "maker_fee": 0.001, **BASE, "return_weight": 0.20,
    },
    "AAVE_h384_cosine_wd05": {
        "symbol": "AAVEUSD", "data_symbol": "AAVEUSD",
        "maker_fee": 0.001, **BASE, "weight_decay": 0.05,
    },
    "AAVE_h512_cosine": {
        "symbol": "AAVEUSD", "data_symbol": "AAVEUSD",
        "maker_fee": 0.001, **BASE, "transformer_dim": 512,
    },
    # --- SUI configs (0 fee for FDUSD pair) ---
    "SUI_h384_cosine_0fee": {
        "symbol": "SUIUSDT", "data_symbol": "SUIUSDT",
        "maker_fee": 0.0, **BASE,
    },
    "SUI_h384_cosine_0fee_rw05": {
        "symbol": "SUIUSDT", "data_symbol": "SUIUSDT",
        "maker_fee": 0.0, **BASE, "return_weight": 0.05,
    },
    "SUI_h384_cosine_0fee_rw20": {
        "symbol": "SUIUSDT", "data_symbol": "SUIUSDT",
        "maker_fee": 0.0, **BASE, "return_weight": 0.20,
    },
    "SUI_h384_cosine_0fee_wd05": {
        "symbol": "SUIUSDT", "data_symbol": "SUIUSDT",
        "maker_fee": 0.0, **BASE, "weight_decay": 0.05,
    },
    "SUI_h384_cosine_10bps": {
        "symbol": "SUIUSDT", "data_symbol": "SUIUSDT",
        "maker_fee": 0.001, **BASE,
    },
}


def load_results() -> dict:
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return {}


def save_results(results: dict):
    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))


def eval_checkpoint(ckpt_path, eval_symbol, maker_fee=0.001, max_leverage=1.0,
                    val_days=30, test_days=30) -> dict:
    try:
        model, normalizer, feature_columns, meta = load_policy_checkpoint(ckpt_path, device="cuda")
        seq_len = meta.get("sequence_length", 72)
        dm = ChronosSolDataModule(
            symbol=eval_symbol, data_root=DATA_ROOT,
            forecast_cache_root=FORECAST_CACHE, forecast_horizons=(1,),
            context_hours=512, quantile_levels=(0.1, 0.5, 0.9),
            batch_size=32, model_id="amazon/chronos-t5-small",
            sequence_length=seq_len,
            split_config=SplitConfig(val_days=val_days, test_days=test_days),
            cache_only=True, max_history_days=365,
        )
        actions = generate_actions_from_frame(
            model=model, frame=dm.test_frame, feature_columns=feature_columns,
            normalizer=normalizer, sequence_length=seq_len, horizon=1,
        )
        bars = dm.test_frame[["timestamp", "symbol", "open", "high", "low", "close"]].copy()
        margin_rate = MARGIN_HOURLY_RATE if max_leverage > 1.0 else 0.0
        cfg = LeverageConfig(
            symbol=eval_symbol, max_leverage=max_leverage, can_short=False,
            maker_fee=maker_fee, margin_hourly_rate=margin_rate,
            initial_cash=10000.0, fill_buffer_pct=0.0005,
            decision_lag_bars=1, min_edge=0.0, max_hold_bars=6,
            intensity_scale=5.0,
        )
        r = simulate_with_margin_cost(bars, actions, cfg)
        return r
    except Exception as e:
        logger.warning(f"eval {eval_symbol} failed: {e}")
        return {"sortino": -999, "total_return": 0, "max_drawdown": 0, "num_trades": 0}


def train_and_eval(cfg_name: str, cfg: dict, results: dict):
    phase = results.setdefault("configs", {})
    if cfg_name in phase:
        logger.info(f"skip {cfg_name} (done)")
        return

    symbol = cfg.pop("symbol")
    data_symbol = cfg.pop("data_symbol")
    fee = cfg.pop("maker_fee")

    tag = f"{data_symbol}_sweep_{cfg_name}"
    ckpt_root = CKPT_ROOT / tag

    # train if needed
    existing = list(ckpt_root.rglob("epoch_*.pt"))
    if len(existing) < 8:
        logger.info(f"\n{'='*60}\nTraining: {cfg_name} ({data_symbol}, fee={fee})\n{'='*60}")
        dm = ChronosSolDataModule(
            symbol=data_symbol, data_root=DATA_ROOT,
            forecast_cache_root=FORECAST_CACHE, forecast_horizons=(1,),
            context_hours=512, quantile_levels=(0.1, 0.5, 0.9),
            batch_size=32, model_id="amazon/chronos-t5-small",
            sequence_length=cfg.get("sequence_length", 72),
            split_config=SplitConfig(val_days=30, test_days=30),
            cache_only=True, max_history_days=365,
        )
        tc = TrainingConfig(
            seed=SEED, maker_fee=fee, checkpoint_root=ckpt_root,
            log_dir=Path("tensorboard_logs/aave_sui_sweep"),
            use_compile=False, decision_lag_bars=1,
            **cfg,
        )
        trainer = BinanceHourlyTrainer(tc, dm)
        trainer.train()
        gc.collect()
        torch.cuda.empty_cache()

    # find best epoch on primary symbol (30d holdout)
    ckpt_dirs = sorted(ckpt_root.glob("binanceneural_*/"))
    if not ckpt_dirs:
        logger.warning(f"no checkpoints for {cfg_name}")
        return
    latest_dir = ckpt_dirs[-1]

    best_ep, best_sort = 0, -999
    for ep_file in sorted(latest_dir.glob("epoch_*.pt")):
        ep_num = int(ep_file.stem.split("_")[1])
        if ep_num not in SAMPLE_EPOCHS:
            continue
        r = eval_checkpoint(ep_file, data_symbol, maker_fee=fee)
        s = r["sortino"]
        logger.info(f"  ep{ep_num}: sort={s:.1f} ret={r['total_return']*100:.1f}% dd={r['max_drawdown']*100:.1f}%")
        if s > best_sort:
            best_sort = s
            best_ep = ep_num

    best_ckpt = latest_dir / f"epoch_{best_ep:03d}.pt"
    logger.info(f"  BEST: ep{best_ep} sort={best_sort:.1f}")

    # multi-window eval
    window_results = {}
    for val_d, test_d, label in EVAL_WINDOWS:
        lev_results = {}
        for lev in LEVERAGE_LEVELS:
            r = eval_checkpoint(best_ckpt, data_symbol, maker_fee=fee,
                                max_leverage=lev, val_days=val_d, test_days=test_d)
            lev_results[f"{lev}x"] = {
                "sortino": round(r["sortino"], 2),
                "return": round(r["total_return"] * 100, 2),
                "max_dd": round(r["max_drawdown"] * 100, 2),
                "trades": r["num_trades"],
            }
        window_results[label] = lev_results
        d = lev_results["1.0x"]
        d2 = lev_results["2.0x"]
        logger.info(f"  {label}: Sort={d['sortino']:.1f} Ret={d['return']:+.1f}% DD={d['max_dd']:.1f}% | "
                    f"2x: Sort={d2['sortino']:.1f} Ret={d2['return']:+.1f}% DD={d2['max_dd']:.1f}%")

    # cross-symbol eval (30d holdout, 1x)
    cross_symbols = ["DOGEUSD", "BTCUSD", "ETHUSD", "SOLUSD", "LINKUSD", "AAVEUSD"]
    cross_results = {}
    for sym in cross_symbols:
        r = eval_checkpoint(best_ckpt, sym, maker_fee=fee)
        cross_results[sym] = round(r["sortino"], 2)

    positive = sum(1 for s in cross_results.values() if s > 0)
    mean_sort = np.mean(list(cross_results.values()))
    min_sort = min(cross_results.values())

    summary = {
        "train_symbol": data_symbol,
        "maker_fee": fee,
        "best_epoch": best_ep,
        "primary_sortino": best_sort,
        "windows": window_results,
        "cross_symbol": cross_results,
        "mean_sortino": round(float(mean_sort), 2),
        "positive_symbols": f"{positive}/{len(cross_results)}",
        "safety_score": round(float(mean_sort) * (positive / len(cross_results)) * (1 - min(max(abs(v) for v in [r.get("max_dd", 0) for r in [lev_results.get("1.0x", {}) for lev_results in window_results.values()] if r]) / 100, 0.5)), 2),
    }

    phase[cfg_name] = summary
    save_results(results)
    logger.info(f"  {cfg_name}: primary_sort={best_sort:.1f} mean={mean_sort:.1f} pos={positive}/{len(cross_results)}")


def print_summary(results: dict):
    configs = results.get("configs", {})
    if not configs:
        return
    logger.info(f"\n{'='*90}\nAAVE/SUI SWEEP RESULTS\n{'='*90}")
    logger.info(f"{'Name':<30} {'Sym':<10} {'Fee':>5} {'Ep':>3} {'Sort':>7} {'Mean':>7} {'Pos':>5}")
    logger.info("-" * 80)
    rows = sorted(configs.items(), key=lambda x: x[1].get("primary_sortino", 0), reverse=True)
    for name, s in rows:
        logger.info(f"{name:<30} {s['train_symbol']:<10} {s['maker_fee']:>5.3f} {s['best_epoch']:>3} "
                    f"{s['primary_sortino']:>7.1f} {s['mean_sortino']:>7.1f} {s['positive_symbols']:>5}")

    # window comparison table for top models
    logger.info(f"\n{'='*90}\nMULTI-WINDOW COMPARISON (1x leverage)\n{'='*90}")
    for name, s in rows[:5]:
        logger.info(f"\n  {name} (ep{s['best_epoch']}, fee={s['maker_fee']}):")
        logger.info(f"  {'Win':<6} {'Sort':>7} {'Ret':>8} {'DD':>7}")
        for w_label, w_data in s.get("windows", {}).items():
            d = w_data.get("1.0x", {})
            logger.info(f"  {w_label:<6} {d.get('sortino',0):>7.1f} {d.get('return',0):>+7.1f}% {d.get('max_dd',0):>6.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, default=None, help="comma-sep config names to run")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    filter_names = set(args.configs.split(",")) if args.configs else None

    results = load_results()

    if not args.eval_only:
        for cfg_name, cfg in CONFIGS.items():
            if filter_names and cfg_name not in filter_names:
                continue
            train_and_eval(cfg_name, dict(cfg), results)

    print_summary(results)


if __name__ == "__main__":
    main()
