#!/usr/bin/env python3
"""Round 4: Combine best Round 3 findings.

R3 winners:
- h384 (wider): safety=7.49, 5/6 pos, DD=-11.8%
- cosine_fn02: DOGE sort=49.5, DD=-13.7%, 4/6 pos
- 8L_fn02: lowest DD=-9.9%, 5/6 pos
- smooth01: lowest DD=-10.7% but 3/6 pos

Try combinations + per-symbol training on top assets.
"""
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
RESULTS_FILE = REPO / "binanceleveragesui/round4_combo_results.json"
SEED = 1337
MARGIN_HOURLY_RATE = 0.0000025457
SAMPLE_EPOCHS = [1, 2, 3, 5, 8, 10, 15, 20]
EVAL_SYMBOLS = ["DOGEUSD", "BTCUSD", "ETHUSD", "SOLUSD", "LINKUSD", "AAVEUSD"]

BASE_6L = dict(
    epochs=20, batch_size=16, sequence_length=72, learning_rate=1e-4,
    weight_decay=0.03, return_weight=0.10, transformer_dim=256,
    transformer_layers=6, transformer_heads=8, fill_temperature=0.1,
    fill_buffer_pct=0.0005, loss_type="sortino", lr_schedule="none",
    model_arch="nano", num_memory_tokens=8, dilated_strides="1,2,6,24",
    feature_noise_std=0.0, transformer_dropout=0.1,
)

CONFIGS = {
    # combo: h384 + cosine LR
    "R4_h384_cosine": {**BASE_6L, "transformer_dim": 384, "lr_schedule": "cosine", "lr_min_ratio": 0.01},
    # combo: h384 + cosine + fnoise
    "R4_h384_cosine_fn02": {**BASE_6L, "transformer_dim": 384, "lr_schedule": "cosine", "lr_min_ratio": 0.01, "feature_noise_std": 0.02},
    # combo: h384 + smooth005
    "R4_h384_smooth005": {**BASE_6L, "transformer_dim": 384, "smoothness_penalty": 0.005},
    # combo: h384 + cosine + smooth005
    "R4_h384_cosine_smooth005": {**BASE_6L, "transformer_dim": 384, "lr_schedule": "cosine", "lr_min_ratio": 0.01, "smoothness_penalty": 0.005},
    # combo: 8L + h384 (wider+deeper)
    "R4_8L_h384": {**BASE_6L, "transformer_layers": 8, "transformer_dim": 384},
    # combo: 8L + h384 + cosine
    "R4_8L_h384_cosine": {**BASE_6L, "transformer_layers": 8, "transformer_dim": 384, "lr_schedule": "cosine", "lr_min_ratio": 0.01},
    # combo: 8L + cosine + fn02
    "R4_8L_cosine_fn02": {**BASE_6L, "transformer_layers": 8, "lr_schedule": "cosine", "lr_min_ratio": 0.01, "feature_noise_std": 0.02},
    # h384 + sortino_dd2 (mild DD penalty)
    "R4_h384_dd2": {**BASE_6L, "transformer_dim": 384, "loss_type": "sortino_dd", "dd_penalty": 2.0},
    # h384 + calmar (DD-focused loss)
    "R4_h384_calmar": {**BASE_6L, "transformer_dim": 384, "loss_type": "calmar"},
    # kitchen sink: h384 + cosine + smooth + fn02
    "R4_kitchen": {**BASE_6L, "transformer_dim": 384, "lr_schedule": "cosine", "lr_min_ratio": 0.01, "smoothness_penalty": 0.005, "feature_noise_std": 0.02},
}

# per-symbol training configs (train on target symbol, not just DOGE)
PER_SYMBOL_CONFIGS = {
    "BTCUSD": {**BASE_6L, "transformer_dim": 384, "lr_schedule": "cosine", "lr_min_ratio": 0.01},
    "ETHUSD": {**BASE_6L, "transformer_dim": 384, "lr_schedule": "cosine", "lr_min_ratio": 0.01},
    "SOLUSD": {**BASE_6L, "transformer_dim": 384, "lr_schedule": "cosine", "lr_min_ratio": 0.01},
}


def load_results() -> dict:
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return {}


def save_results(results: dict):
    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))


def eval_single_window(ckpt_path, eval_symbol, **cfg_overrides) -> dict:
    try:
        model, normalizer, feature_columns, meta = load_policy_checkpoint(ckpt_path, device="cuda")
        seq_len = meta.get("sequence_length", 72)
        dm = ChronosSolDataModule(
            symbol=eval_symbol, data_root=DATA_ROOT,
            forecast_cache_root=FORECAST_CACHE, forecast_horizons=(1,),
            context_hours=512, quantile_levels=(0.1, 0.5, 0.9),
            batch_size=32, model_id="amazon/chronos-t5-small",
            sequence_length=seq_len,
            split_config=SplitConfig(val_days=30, test_days=30),
            cache_only=True, max_history_days=365,
        )
        actions = generate_actions_from_frame(
            model=model, frame=dm.test_frame, feature_columns=feature_columns,
            normalizer=normalizer, sequence_length=seq_len, horizon=1,
        )
        bars = dm.test_frame[["timestamp", "symbol", "open", "high", "low", "close"]].copy()
        cfg = LeverageConfig(
            symbol=eval_symbol, max_leverage=1.0, can_short=False,
            maker_fee=0.001, margin_hourly_rate=0.0, initial_cash=10000.0,
            fill_buffer_pct=0.0013, decision_lag_bars=1, min_edge=0.0,
            max_hold_bars=6, intensity_scale=5.0,
        )
        for k, v in cfg_overrides.items():
            setattr(cfg, k, v)
        r = simulate_with_margin_cost(bars, actions, cfg)
        r["eval_symbol"] = eval_symbol
        return r
    except Exception as e:
        logger.warning(f"eval {eval_symbol} failed: {e}")
        return {"sortino": -999, "total_return": 0, "max_drawdown": 0, "num_trades": 0, "eval_symbol": eval_symbol}


def train_config(config_name: str, train_symbol: str, overrides: dict) -> Path:
    tag = f"{train_symbol}_r4_{config_name}"
    ckpt_root = CKPT_ROOT / tag
    logger.info(f"\n{'='*60}\nTraining: {tag}\n{'='*60}")
    dm = ChronosSolDataModule(
        symbol=train_symbol, data_root=DATA_ROOT,
        forecast_cache_root=FORECAST_CACHE, forecast_horizons=(1,),
        context_hours=512, quantile_levels=(0.1, 0.5, 0.9),
        batch_size=32, model_id="amazon/chronos-t5-small",
        sequence_length=overrides.get("sequence_length", 72),
        split_config=SplitConfig(val_days=30, test_days=30),
        cache_only=True, max_history_days=365,
    )
    tc = TrainingConfig(
        seed=SEED, maker_fee=0.001, checkpoint_root=ckpt_root,
        log_dir=Path("tensorboard_logs/r4_combo"),
        use_compile=False, decision_lag_bars=1,
        **overrides,
    )
    trainer = BinanceHourlyTrainer(tc, dm)
    trainer.train()
    gc.collect()
    torch.cuda.empty_cache()
    return ckpt_root


def eval_config(cfg_name: str, train_symbol: str, results: dict):
    phase = results.setdefault("configs", {})
    key = f"{train_symbol}_{cfg_name}"
    if key in phase:
        logger.info(f"Skipping {key} (done)")
        return

    ckpt_root = CKPT_ROOT / f"{train_symbol}_r4_{cfg_name}"
    ckpt_dirs = sorted(ckpt_root.glob("binanceneural_*/"))
    if not ckpt_dirs:
        return
    latest_dir = ckpt_dirs[-1]

    best_ep, best_sort = 0, -999
    for ep_file in sorted(latest_dir.glob("epoch_*.pt")):
        ep_num = int(ep_file.stem.split("_")[1])
        if ep_num not in SAMPLE_EPOCHS:
            continue
        r = eval_single_window(ep_file, train_symbol)
        if r["sortino"] > best_sort:
            best_sort = r["sortino"]
            best_ep = ep_num

    best_ckpt = latest_dir / f"epoch_{best_ep:03d}.pt"
    logger.info(f"  {key} best_ep={best_ep} ({train_symbol} sort={best_sort:.1f}), multi-symbol eval...")

    symbol_results = {}
    for sym in EVAL_SYMBOLS:
        r = eval_single_window(best_ckpt, sym)
        symbol_results[sym] = {
            "sortino": r["sortino"],
            "total_return": r.get("total_return", 0),
            "max_drawdown": r.get("max_drawdown", 0),
            "num_trades": r.get("num_trades", 0),
        }

    sortinos = [v["sortino"] for v in symbol_results.values()]
    drawdowns = [abs(v["max_drawdown"]) for v in symbol_results.values()]
    positive = sum(1 for s in sortinos if s > 0)

    summary = {
        "config_name": cfg_name,
        "train_symbol": train_symbol,
        "best_epoch": best_ep,
        "primary_sortino": symbol_results.get(train_symbol, {}).get("sortino", 0),
        "mean_sortino": float(np.mean(sortinos)),
        "min_sortino": float(np.min(sortinos)),
        "worst_drawdown": float(-np.max(drawdowns)),
        "positive": positive,
        "total": len(sortinos),
        "symbols": symbol_results,
        "safety_score": float(np.mean(sortinos)) * (positive / len(sortinos)) * (1 - min(np.max(drawdowns), 0.5)),
    }

    phase[key] = summary
    save_results(results)
    logger.info(f"  {key}: primary={summary['primary_sortino']:.1f} mean={summary['mean_sortino']:.1f} "
                f"min={summary['min_sortino']:.1f} worst_dd={summary['worst_drawdown']*100:.1f}% "
                f"pos={positive}/{len(sortinos)} safety={summary['safety_score']:.2f}")


def run_combo_sweep(results: dict, filter_names=None):
    # phase 1: DOGE combo configs
    for cfg_name, overrides in CONFIGS.items():
        if filter_names and cfg_name not in filter_names:
            continue
        key = f"DOGEUSD_{cfg_name}"
        if key in results.get("configs", {}):
            logger.info(f"Skipping {key} (done)")
            continue
        ckpt_root = CKPT_ROOT / f"DOGEUSD_r4_{cfg_name}"
        existing = list(ckpt_root.rglob("epoch_*.pt"))
        if len(existing) < 8:
            train_config(cfg_name, "DOGEUSD", overrides)
        eval_config(cfg_name, "DOGEUSD", results)

    # phase 2: per-symbol training with best config (h384+cosine)
    for sym, overrides in PER_SYMBOL_CONFIGS.items():
        cfg_name = "h384_cosine"
        if filter_names and f"{sym}_{cfg_name}" not in filter_names:
            continue
        key = f"{sym}_{cfg_name}"
        if key in results.get("configs", {}):
            logger.info(f"Skipping {key} (done)")
            continue
        ckpt_root = CKPT_ROOT / f"{sym}_r4_{cfg_name}"
        existing = list(ckpt_root.rglob("epoch_*.pt"))
        if len(existing) < 8:
            train_config(cfg_name, sym, overrides)
        eval_config(cfg_name, sym, results)

    # leverage sweep on top-3
    logger.info("\n=== Leverage Sweep on Top Configs ===")
    phase = results.get("configs", {})
    sorted_configs = sorted(phase.items(), key=lambda x: x[1].get("safety_score", 0), reverse=True)
    leverage = results.setdefault("leverage", {})

    for key, summary in sorted_configs[:3]:
        if key in leverage:
            continue
        train_sym = summary["train_symbol"]
        cfg_name = summary["config_name"]
        best_ep = summary["best_epoch"]
        ckpt_root = CKPT_ROOT / f"{train_sym}_r4_{cfg_name}"
        ckpt_dirs = sorted(ckpt_root.glob("binanceneural_*/"))
        if not ckpt_dirs:
            continue
        best_ckpt = ckpt_dirs[-1] / f"epoch_{best_ep:03d}.pt"

        lev_results = []
        for lev in [1.0, 1.5, 2.0, 3.0, 5.0]:
            r = eval_single_window(
                best_ckpt, train_sym,
                max_leverage=lev,
                margin_hourly_rate=MARGIN_HOURLY_RATE if lev > 1.0 else 0.0,
            )
            lev_results.append({
                "leverage": lev, "sortino": r["sortino"],
                "total_return": r.get("total_return", 0),
                "max_drawdown": r.get("max_drawdown", 0),
            })
            logger.info(f"  {key} lev={lev}x: Sort={r['sortino']:.2f} Ret={r['total_return']*100:.1f}% DD={r['max_drawdown']*100:.1f}%")

        leverage[key] = lev_results
        save_results(results)


def print_summary(results: dict):
    logger.info(f"\n{'='*90}\nROUND 4 COMBINATION RESULTS\n{'='*90}")
    configs = results.get("configs", {})
    rows = []
    for name, s in configs.items():
        if isinstance(s, dict) and "safety_score" in s:
            rows.append((name, s["train_symbol"], s["best_epoch"], s["primary_sortino"],
                         s["mean_sortino"], s["min_sortino"], s["worst_drawdown"]*100,
                         s["positive"], s["total"], s["safety_score"]))
    rows.sort(key=lambda x: x[-1], reverse=True)
    logger.info(f"{'Name':<35} {'Sym':<8} {'Ep':>3} {'Prim':>7} {'Mean':>7} {'Min':>7} {'WstDD%':>7} {'Pos':>5} {'Safety':>7}")
    logger.info("-" * 100)
    for name, sym, ep, ps, ms, mins, dd, p, t, ss in rows:
        logger.info(f"{name:<35} {sym:<8} {ep:>3} {ps:>7.1f} {ms:>7.1f} {mins:>7.1f} {dd:>7.1f} {p}/{t} {ss:>7.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, default=None)
    args = parser.parse_args()
    filter_names = set(args.configs.split(",")) if args.configs else None

    results = load_results()
    run_combo_sweep(results, filter_names)
    print_summary(results)


if __name__ == "__main__":
    main()
