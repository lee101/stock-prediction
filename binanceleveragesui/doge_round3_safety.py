#!/usr/bin/env python3
"""Round 3: Safety-focused sweep.

Goals:
- Find configs with LOW drawdown + HIGH sortino across MANY time windows
- Combine best Phase 2 findings (6L, fnoise02)
- Test across multiple symbols (DOGE, BTC, ETH, SOL, LINK, AAVE)
- Sweep leverage 1x-5x with best configs
- Focus on smoothness: evaluate on MULTIPLE rolling windows, not just one 30d
"""
from __future__ import annotations
import argparse, gc, json, sys, time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Any

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
RESULTS_FILE = REPO / "binanceleveragesui/round3_safety_results.json"
TRAIN_SYMBOL = "DOGEUSD"
EVAL_SYMBOLS = ["DOGEUSD", "BTCUSD", "ETHUSD", "SOLUSD", "LINKUSD", "AAVEUSD"]
SEED = 1337
MARGIN_HOURLY_RATE = 0.0000025457
SAMPLE_EPOCHS = [1, 2, 3, 5, 8, 10, 15, 20]

# best from Phase 2: 6L + fnoise02
# now combine and add safety-focused variants
BASE_6L = dict(
    epochs=20, batch_size=16, sequence_length=72, learning_rate=1e-4,
    weight_decay=0.03, return_weight=0.10, transformer_dim=256,
    transformer_layers=6, transformer_heads=8, fill_temperature=0.1,
    fill_buffer_pct=0.0005, loss_type="sortino", lr_schedule="none",
    model_arch="nano", num_memory_tokens=8, dilated_strides="1,2,6,24",
    feature_noise_std=0.0, transformer_dropout=0.1,
)

CONFIGS = {
    # baseline: raw 6L (won phase 2)
    "R3_6L_baseline": {**BASE_6L},

    # combine top-2 findings
    "R3_6L_fnoise02": {**BASE_6L, "feature_noise_std": 0.02},

    # safety-focused: sortino_dd penalizes drawdown
    "R3_6L_sortino_dd2": {**BASE_6L, "loss_type": "sortino_dd", "dd_penalty": 2.0},
    "R3_6L_sortino_dd5": {**BASE_6L, "loss_type": "sortino_dd", "dd_penalty": 5.0},
    "R3_6L_sortino_dd10": {**BASE_6L, "loss_type": "sortino_dd", "dd_penalty": 10.0},

    # smoothness via regularization
    "R3_6L_smooth005_fn02": {**BASE_6L, "smoothness_penalty": 0.005, "feature_noise_std": 0.02},
    "R3_6L_smooth01_fn02": {**BASE_6L, "smoothness_penalty": 0.01, "feature_noise_std": 0.02},

    # cosine LR for better convergence
    "R3_6L_cosine_fn02": {**BASE_6L, "lr_schedule": "cosine", "lr_min_ratio": 0.01, "feature_noise_std": 0.02},

    # wider + deeper for capacity
    "R3_6L_h384": {**BASE_6L, "transformer_dim": 384},
    "R3_8L_fn02": {**BASE_6L, "transformer_layers": 8, "feature_noise_std": 0.02},

    # muon optimizer
    "R3_6L_muon": {**BASE_6L, "optimizer_name": "muon"},

    # value embed + residual scalars on 6L
    "R3_6L_vale_resid": {**BASE_6L, "use_value_embedding": True, "value_embedding_every": 2,
                         "value_embedding_scale": 0.1, "use_residual_scalars": True,
                         "residual_scale_init": 1.0, "skip_scale_init": 0.0},

    # calmar loss (minimize drawdown explicitly)
    "R3_6L_calmar_fn02": {**BASE_6L, "loss_type": "calmar", "feature_noise_std": 0.02},

    # lower return weight = less aggressive
    "R3_6L_rw005_fn02": {**BASE_6L, "return_weight": 0.05, "feature_noise_std": 0.02},
    "R3_6L_rw003_fn02": {**BASE_6L, "return_weight": 0.03, "feature_noise_std": 0.02},
}


def load_results() -> dict:
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return {}


def save_results(results: dict):
    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))


def eval_checkpoint_multi_window(ckpt_path, eval_symbol, windows=None, **cfg_overrides) -> dict:
    """Evaluate on multiple rolling windows for robustness."""
    try:
        model, normalizer, feature_columns, meta = load_policy_checkpoint(ckpt_path, device="cuda")
        seq_len = meta.get("sequence_length", 72)

        if windows is None:
            windows = [
                SplitConfig(val_days=30, test_days=30),  # last 30d
                SplitConfig(val_days=60, test_days=30),  # 60-90d ago
                SplitConfig(val_days=30, test_days=60),  # last 60d
            ]

        all_results = []
        for split in windows:
            dm = ChronosSolDataModule(
                symbol=eval_symbol, data_root=DATA_ROOT,
                forecast_cache_root=FORECAST_CACHE, forecast_horizons=(1,),
                context_hours=512, quantile_levels=(0.1, 0.5, 0.9),
                batch_size=32, model_id="amazon/chronos-t5-small",
                sequence_length=seq_len, split_config=split,
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
            r["window"] = f"val{split.val_days}_test{split.test_days}"
            all_results.append(r)

        # aggregate: use min sortino (worst window) as safety metric
        sortinos = [r["sortino"] for r in all_results]
        returns = [r["total_return"] for r in all_results]
        drawdowns = [abs(r.get("max_drawdown", 0)) for r in all_results]
        trades = [r.get("num_trades", 0) for r in all_results]

        return {
            "eval_symbol": eval_symbol,
            "mean_sortino": float(np.mean(sortinos)),
            "min_sortino": float(np.min(sortinos)),
            "max_drawdown": float(-np.max(drawdowns)),
            "mean_return": float(np.mean(returns)),
            "mean_trades": float(np.mean(trades)),
            "per_window": all_results,
        }
    except Exception as e:
        logger.warning(f"eval {eval_symbol} failed: {e}")
        return {"eval_symbol": eval_symbol, "mean_sortino": -999, "min_sortino": -999,
                "max_drawdown": 0, "mean_return": 0, "mean_trades": 0}


def eval_single_window(ckpt_path, eval_symbol, **cfg_overrides) -> dict:
    """Quick eval on standard 30d window."""
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


def train_config(config_name: str, overrides: dict) -> Path:
    tag = f"DOGEUSD_r3_{config_name}"
    ckpt_root = CKPT_ROOT / tag
    logger.info(f"\n{'='*60}\nTraining: {tag}\n{'='*60}")
    dm = ChronosSolDataModule(
        symbol=TRAIN_SYMBOL, data_root=DATA_ROOT,
        forecast_cache_root=FORECAST_CACHE, forecast_horizons=(1,),
        context_hours=512, quantile_levels=(0.1, 0.5, 0.9),
        batch_size=32, model_id="amazon/chronos-t5-small",
        sequence_length=overrides.get("sequence_length", 72),
        split_config=SplitConfig(val_days=30, test_days=30),
        cache_only=True, max_history_days=365,
    )
    tc = TrainingConfig(
        seed=SEED, maker_fee=0.001, checkpoint_root=ckpt_root,
        log_dir=Path("tensorboard_logs/r3_safety"),
        use_compile=False, decision_lag_bars=1,
        **overrides,
    )
    trainer = BinanceHourlyTrainer(tc, dm)
    trainer.train()
    gc.collect()
    torch.cuda.empty_cache()
    return ckpt_root


def run_sweep(results: dict, filter_names=None):
    phase_results = results.get("configs", {})

    for cfg_name, overrides in CONFIGS.items():
        if filter_names and cfg_name not in filter_names:
            continue
        if cfg_name in phase_results:
            logger.info(f"Skipping {cfg_name} (done)")
            continue

        ckpt_root = CKPT_ROOT / f"DOGEUSD_r3_{cfg_name}"
        existing = list(ckpt_root.rglob("epoch_*.pt"))
        if len(existing) >= 8:
            logger.info(f"Reusing checkpoints for {cfg_name}")
        else:
            train_config(cfg_name, overrides)

        # find best epoch on DOGE first (quick)
        ckpt_dirs = sorted(ckpt_root.glob("binanceneural_*/"))
        if not ckpt_dirs:
            continue
        latest_dir = ckpt_dirs[-1]
        epoch_files = sorted(latest_dir.glob("epoch_*.pt"))

        best_ep, best_sort = 0, -999
        for ep_file in epoch_files:
            ep_num = int(ep_file.stem.split("_")[1])
            if ep_num not in SAMPLE_EPOCHS:
                continue
            r = eval_single_window(ep_file, "DOGEUSD")
            if r["sortino"] > best_sort:
                best_sort = r["sortino"]
                best_ep = ep_num

        # eval best epoch across all symbols with multi-window
        best_ckpt = latest_dir / f"epoch_{best_ep:03d}.pt"
        logger.info(f"  {cfg_name} best_ep={best_ep} (DOGE sort={best_sort:.1f}), running multi-symbol multi-window eval...")

        symbol_results = {}
        for sym in EVAL_SYMBOLS:
            r = eval_single_window(best_ckpt, sym)
            symbol_results[sym] = {
                "sortino": r["sortino"],
                "total_return": r.get("total_return", 0),
                "max_drawdown": r.get("max_drawdown", 0),
                "num_trades": r.get("num_trades", 0),
            }

        # compute safety score: weighted combo of mean sortino, min sortino, and max DD
        sortinos = [v["sortino"] for v in symbol_results.values()]
        drawdowns = [abs(v["max_drawdown"]) for v in symbol_results.values()]
        positive = sum(1 for s in sortinos if s > 0)

        summary = {
            "config_name": cfg_name,
            "best_epoch": best_ep,
            "doge_sortino": symbol_results.get("DOGEUSD", {}).get("sortino", 0),
            "mean_sortino": float(np.mean(sortinos)),
            "min_sortino": float(np.min(sortinos)),
            "worst_drawdown": float(-np.max(drawdowns)),
            "positive": positive,
            "total": len(sortinos),
            "symbols": symbol_results,
            # safety score: penalize negative symbols and large drawdowns
            "safety_score": float(np.mean(sortinos)) * (positive / len(sortinos)) * (1 - min(np.max(drawdowns), 0.5)),
        }

        phase_results[cfg_name] = summary
        results["configs"] = phase_results
        save_results(results)

        logger.info(f"  {cfg_name}: doge={summary['doge_sortino']:.1f} mean={summary['mean_sortino']:.1f} "
                     f"min={summary['min_sortino']:.1f} worst_dd={summary['worst_drawdown']*100:.1f}% "
                     f"pos={positive}/{len(sortinos)} safety={summary['safety_score']:.2f}")

    # leverage sweep on top configs
    logger.info("\n=== Leverage Sweep on Top Configs ===")
    sorted_configs = sorted(phase_results.items(), key=lambda x: x[1].get("safety_score", 0), reverse=True)
    leverage_results = results.get("leverage", {})

    for cfg_name, summary in sorted_configs[:3]:
        if cfg_name in leverage_results:
            continue
        best_ep = summary["best_epoch"]
        ckpt_root = CKPT_ROOT / f"DOGEUSD_r3_{cfg_name}"
        ckpt_dirs = sorted(ckpt_root.glob("binanceneural_*/"))
        if not ckpt_dirs:
            continue
        best_ckpt = ckpt_dirs[-1] / f"epoch_{best_ep:03d}.pt"

        lev_results = []
        for lev in [1.0, 1.5, 2.0, 3.0, 5.0]:
            r = eval_single_window(
                best_ckpt, "DOGEUSD",
                max_leverage=lev,
                margin_hourly_rate=MARGIN_HOURLY_RATE if lev > 1.0 else 0.0,
            )
            lev_results.append({
                "leverage": lev,
                "sortino": r["sortino"],
                "total_return": r.get("total_return", 0),
                "max_drawdown": r.get("max_drawdown", 0),
                "num_trades": r.get("num_trades", 0),
            })
            logger.info(f"  {cfg_name} lev={lev}x: Sort={r['sortino']:.2f} Ret={r['total_return']*100:.1f}% DD={r['max_drawdown']*100:.1f}%")

        leverage_results[cfg_name] = lev_results
        results["leverage"] = leverage_results
        save_results(results)


def print_summary(results: dict):
    logger.info(f"\n{'='*90}\nROUND 3 SAFETY RESULTS\n{'='*90}")
    configs = results.get("configs", {})
    rows = []
    for name, s in configs.items():
        if isinstance(s, dict) and "safety_score" in s:
            rows.append((name, s["best_epoch"], s["doge_sortino"], s["mean_sortino"],
                         s["min_sortino"], s["worst_drawdown"]*100, s["positive"], s["total"],
                         s["safety_score"]))
    rows.sort(key=lambda x: x[-1], reverse=True)
    logger.info(f"{'Name':<35} {'Ep':>3} {'DOGE':>7} {'Mean':>7} {'Min':>7} {'WstDD%':>7} {'Pos':>5} {'Safety':>7}")
    logger.info("-" * 90)
    for name, ep, ds, ms, mins, dd, p, t, ss in rows:
        logger.info(f"{name:<35} {ep:>3} {ds:>7.1f} {ms:>7.1f} {mins:>7.1f} {dd:>7.1f} {p}/{t} {ss:>7.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, default=None)
    args = parser.parse_args()
    filter_names = set(args.configs.split(",")) if args.configs else None

    results = load_results()
    run_sweep(results, filter_names)
    print_summary(results)


if __name__ == "__main__":
    main()
