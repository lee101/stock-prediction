#!/usr/bin/env python3
"""DOGE trading improvement loop.

Phase 1: Eval-only sweeps (leverage, post-training params)
Phase 2: Architecture + hyperparam experiments (retrain)
Phase 3: Cross-asset features (BTC/ETH forecasts)
Phase 4: Combine best from all phases
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
RESULTS_FILE = REPO / "binanceleveragesui/doge_improvement_results.json"
BASELINE_CKPT = CKPT_ROOT / "DOGEUSD_nano_fine_strides_ep5.pt"
TRAIN_SYMBOL = "DOGEUSD"
EVAL_SYMBOLS = ["DOGEUSD", "BTCUSD", "ETHUSD"]
SEED = 1337
MARGIN_HOURLY_RATE = 0.0000025457

BASE = dict(
    epochs=20, batch_size=16, sequence_length=72, learning_rate=1e-4,
    weight_decay=0.03, return_weight=0.10, transformer_dim=256,
    transformer_layers=4, transformer_heads=8, fill_temperature=0.1,
    fill_buffer_pct=0.0005, loss_type="sortino", lr_schedule="none",
    model_arch="nano", num_memory_tokens=8, dilated_strides="1,2,6,24",
    feature_noise_std=0.0, transformer_dropout=0.1,
)

ARCH_CONFIGS = {
    "A1_value_embed": {**BASE, "use_value_embedding": True, "value_embedding_every": 2, "value_embedding_scale": 0.1},
    "A2_residual_scalars": {**BASE, "use_residual_scalars": True, "residual_scale_init": 1.0, "skip_scale_init": 0.0},
    "A3_rope72": {**BASE, "rope_base": 72.0},
    "A4_6L": {**BASE, "transformer_layers": 6},
    "A5_strides_1_3_8_24": {**BASE, "dilated_strides": "1,3,8,24"},
    "A6_strides_1_2_4_12": {**BASE, "dilated_strides": "1,2,4,12"},
    "A7_mlp8": {**BASE, "mlp_ratio": 8.0},
}

HPARAM_CONFIGS = {
    "B1_smooth001": {**BASE, "smoothness_penalty": 0.001},
    "B2_smooth005": {**BASE, "smoothness_penalty": 0.005},
    "B3_fnoise01": {**BASE, "feature_noise_std": 0.01},
    "B4_fnoise02": {**BASE, "feature_noise_std": 0.02},
    "B5_cosine_lr": {**BASE, "lr_schedule": "cosine", "lr_min_ratio": 0.01},
    "B6_rw008": {**BASE, "return_weight": 0.08},
    "B7_rw012": {**BASE, "return_weight": 0.12},
    "B8_rw015": {**BASE, "return_weight": 0.15},
    "B9_wd002": {**BASE, "weight_decay": 0.02},
    "B10_wd005": {**BASE, "weight_decay": 0.05},
    "B11_batch8": {**BASE, "batch_size": 8},
    "B12_batch32": {**BASE, "batch_size": 32},
    "B13_seq48": {**BASE, "sequence_length": 48},
    "B14_seq96": {**BASE, "sequence_length": 96},
    "B15_sortino_dd": {**BASE, "loss_type": "sortino_dd", "dd_penalty": 2.0},
    "B16_calmar": {**BASE, "loss_type": "calmar"},
}

SAMPLE_EPOCHS = [1, 2, 3, 5, 8, 10, 15, 20]


def load_results() -> dict:
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return {}


def save_results(results: dict):
    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))


def eval_checkpoint(ckpt_path, eval_symbol, **cfg_overrides) -> dict:
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
    tag = f"DOGEUSD_loop_{config_name}"
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
        log_dir=Path("tensorboard_logs/doge_loop"),
        use_compile=False, decision_lag_bars=1,
        **overrides,
    )
    trainer = BinanceHourlyTrainer(tc, dm)
    trainer.train()
    gc.collect()
    torch.cuda.empty_cache()
    return ckpt_root


def sweep_epochs(ckpt_root: Path, eval_symbols: List[str], **cfg_overrides) -> List[dict]:
    results = []
    ckpt_dirs = sorted(ckpt_root.glob("binanceneural_*/"))
    if not ckpt_dirs:
        return results
    latest_dir = ckpt_dirs[-1]
    epoch_files = sorted(latest_dir.glob("epoch_*.pt"))
    for ep_file in epoch_files:
        ep_num = int(ep_file.stem.split("_")[1])
        if ep_num not in SAMPLE_EPOCHS:
            continue
        for sym in eval_symbols:
            r = eval_checkpoint(ep_file, sym, **cfg_overrides)
            r["epoch"] = ep_num
            r["ckpt"] = str(ep_file)
            results.append(r)
    return results


def best_epoch_summary(evals: List[dict]) -> dict:
    if not evals:
        return {}
    epoch_scores = {}
    for r in evals:
        ep = r["epoch"]
        if ep not in epoch_scores:
            epoch_scores[ep] = {"sorts": [], "rets": [], "trades": [], "dds": []}
        epoch_scores[ep]["sorts"].append(r["sortino"])
        epoch_scores[ep]["rets"].append(r.get("total_return", 0))
        epoch_scores[ep]["trades"].append(r.get("num_trades", 0))
        epoch_scores[ep]["dds"].append(r.get("max_drawdown", 0))
    best_ep = max(epoch_scores, key=lambda e: np.mean(epoch_scores[e]["sorts"]))
    s = epoch_scores[best_ep]
    return {
        "best_epoch": best_ep,
        "mean_sortino": float(np.mean(s["sorts"])),
        "doge_sortino": float(s["sorts"][0]) if s["sorts"] else 0,
        "positive": sum(1 for x in s["sorts"] if x > 0),
        "total": len(s["sorts"]),
        "mean_return": float(np.mean(s["rets"])),
        "mean_trades": float(np.mean(s["trades"])),
    }


# ── Phase 1: Eval-only sweeps ──────────────────────────────────────────────

def phase1_leverage(results: dict):
    logger.info("\n=== Phase 1a: Leverage Sweep ===")
    leverages = [1.0, 1.5, 2.0, 3.0, 5.0]
    phase_results = []
    for lev in leverages:
        r = eval_checkpoint(
            BASELINE_CKPT, "DOGEUSD",
            max_leverage=lev,
            margin_hourly_rate=MARGIN_HOURLY_RATE if lev > 1.0 else 0.0,
        )
        r["leverage"] = lev
        phase_results.append(r)
        logger.info(f"  lev={lev}x: Sort={r['sortino']:.2f} Ret={r['total_return']*100:.1f}% "
                     f"Trades={r['num_trades']} MaxDD={r['max_drawdown']*100:.1f}%")
    results["phase1_leverage"] = phase_results
    save_results(results)


def phase1_eval_params(results: dict):
    logger.info("\n=== Phase 1b: Post-Training Param Sweeps ===")
    defaults = dict(min_edge=0.0, fill_buffer_pct=0.0013, intensity_scale=5.0, max_hold_bars=6)
    sweeps = {
        "min_edge": [0.0, 0.002, 0.004, 0.006],
        "fill_buffer_pct": [0.0005, 0.001, 0.0013, 0.002],
        "intensity_scale": [3.0, 5.0, 7.0],
        "max_hold_bars": [4, 6, 8, 12],
    }
    independent_results = {}
    for param, values in sweeps.items():
        logger.info(f"\n  Sweeping {param}:")
        param_results = []
        for val in values:
            overrides = {**defaults, param: val}
            r = eval_checkpoint(BASELINE_CKPT, "DOGEUSD", **overrides)
            r["param"] = param
            r["value"] = val
            param_results.append(r)
            logger.info(f"    {param}={val}: Sort={r['sortino']:.2f} Ret={r['total_return']*100:.1f}% "
                         f"Trades={r['num_trades']}")
        independent_results[param] = param_results
        # find top-2
        sorted_r = sorted(param_results, key=lambda x: x["sortino"], reverse=True)
        top2 = [x["value"] for x in sorted_r[:2]]
        independent_results[f"{param}_top2"] = top2
        logger.info(f"    top-2: {top2}")

    # grid top-2 from each dimension
    logger.info("\n  Grid search (top-2 combos):")
    grid_results = []
    for me in independent_results["min_edge_top2"]:
        for fb in independent_results["fill_buffer_pct_top2"]:
            for iscale in independent_results["intensity_scale_top2"]:
                for mh in independent_results["max_hold_bars_top2"]:
                    r = eval_checkpoint(BASELINE_CKPT, "DOGEUSD",
                                        min_edge=me, fill_buffer_pct=fb,
                                        intensity_scale=iscale, max_hold_bars=mh)
                    r["params"] = {"min_edge": me, "fill_buffer_pct": fb,
                                   "intensity_scale": iscale, "max_hold_bars": mh}
                    grid_results.append(r)
    grid_results.sort(key=lambda x: x["sortino"], reverse=True)
    for i, r in enumerate(grid_results[:5]):
        logger.info(f"    #{i+1}: Sort={r['sortino']:.2f} Ret={r['total_return']*100:.1f}% "
                     f"Trades={r['num_trades']} {r['params']}")

    results["phase1_eval_params"] = {
        "independent": {k: v for k, v in independent_results.items() if not k.endswith("_top2")},
        "grid_top5": grid_results[:5],
    }
    save_results(results)


# ── Phase 2: Architecture + Hyperparam experiments ─────────────────────────

def phase2_train_and_eval(results: dict, configs: dict, phase_key: str, filter_names=None):
    phase_results = results.get(phase_key, {})
    for cfg_name, overrides in configs.items():
        if filter_names and cfg_name not in filter_names:
            continue
        if cfg_name in phase_results:
            logger.info(f"Skipping {cfg_name} (already done)")
            continue

        ckpt_root = CKPT_ROOT / f"DOGEUSD_loop_{cfg_name}"
        # check if already trained
        existing = list(ckpt_root.rglob("epoch_*.pt"))
        if len(existing) >= 8:
            logger.info(f"Reusing existing checkpoints for {cfg_name}")
        else:
            train_config(cfg_name, overrides)

        evals = sweep_epochs(ckpt_root, EVAL_SYMBOLS)
        summary = best_epoch_summary(evals)
        summary["config_name"] = cfg_name
        summary["raw"] = evals
        phase_results[cfg_name] = summary
        results[phase_key] = phase_results
        save_results(results)

        if summary:
            logger.info(f"  {cfg_name} best_ep={summary.get('best_epoch')}: "
                         f"mean_sort={summary.get('mean_sortino', 0):.2f} "
                         f"doge_sort={summary.get('doge_sortino', 0):.2f} "
                         f"pos={summary.get('positive', 0)}/{summary.get('total', 0)}")

    return phase_results


# ── Phase 3: Cross-asset features ─────────────────────────────────────────

def phase3_cross_asset(results: dict):
    logger.info("\n=== Phase 3: Cross-Asset Features ===")

    # C1+C2: check if BTC/ETH forecast caches exist
    for sym in ["BTCUSD", "ETHUSD"]:
        cache_path = FORECAST_CACHE / f"h1/{sym}.parquet"
        if not cache_path.exists():
            logger.warning(f"Missing forecast cache: {cache_path}")
            logger.info(f"Run: python chronos2_trainer.py --symbol {sym} --data-root trainingdatahourlybinance")
            logger.info("Then build forecast cache before running phase 3")
            return

    # C3+C4: train with cross-asset features
    # This requires the data pipeline modification - check if it exists
    try:
        dm = ChronosSolDataModule(
            symbol=TRAIN_SYMBOL, data_root=DATA_ROOT,
            forecast_cache_root=FORECAST_CACHE, forecast_horizons=(1,),
            context_hours=512, quantile_levels=(0.1, 0.5, 0.9),
            batch_size=32, model_id="amazon/chronos-t5-small",
            sequence_length=72,
            split_config=SplitConfig(val_days=30, test_days=30),
            cache_only=True, max_history_days=365,
            cross_asset_symbols=["BTCUSD", "ETHUSD"],
        )
    except TypeError:
        logger.warning("ChronosSolDataModule doesn't support cross_asset_symbols yet. "
                        "Need to modify binancechronossolexperiment/data.py first.")
        return

    logger.info(f"Cross-asset features: {len(dm.feature_columns)} features")
    overrides = {**BASE}
    cfg_name = "C1_cross_asset_btc_eth"
    ckpt_root = CKPT_ROOT / f"DOGEUSD_loop_{cfg_name}"

    tc = TrainingConfig(
        seed=SEED, maker_fee=0.001, checkpoint_root=ckpt_root,
        log_dir=Path("tensorboard_logs/doge_loop"),
        use_compile=False, decision_lag_bars=1,
        **overrides,
    )
    trainer = BinanceHourlyTrainer(tc, dm)
    trainer.train()
    gc.collect()
    torch.cuda.empty_cache()

    evals = sweep_epochs(ckpt_root, ["DOGEUSD"])
    summary = best_epoch_summary(evals)
    summary["config_name"] = cfg_name
    results["phase3_cross_asset"] = summary
    save_results(results)
    logger.info(f"  {cfg_name}: doge_sort={summary.get('doge_sortino', 0):.2f}")


# ── Summary ───────────────────────────────────────────────────────────────

def print_summary(results: dict):
    logger.info(f"\n{'='*80}\nSUMMARY\n{'='*80}")

    rows = []
    # baseline
    if "phase1_leverage" in results:
        for r in results["phase1_leverage"]:
            if r.get("leverage") == 1.0:
                rows.append(("BASELINE (1x)", r["sortino"], r["total_return"]*100,
                             r["num_trades"], r["max_drawdown"]*100))

    # leverage
    if "phase1_leverage" in results:
        for r in results["phase1_leverage"]:
            if r.get("leverage", 1.0) != 1.0:
                rows.append((f"lev_{r['leverage']}x", r["sortino"], r["total_return"]*100,
                             r["num_trades"], r["max_drawdown"]*100))

    # eval params grid top-5
    if "phase1_eval_params" in results:
        for r in results["phase1_eval_params"].get("grid_top5", []):
            p = r.get("params", {})
            label = f"me={p.get('min_edge',0)} fb={p.get('fill_buffer_pct',0)} " \
                    f"is={p.get('intensity_scale',0)} mh={p.get('max_hold_bars',0)}"
            rows.append((label, r["sortino"], r["total_return"]*100,
                         r["num_trades"], r["max_drawdown"]*100))

    # phase 2
    for phase_key in ["phase2_arch", "phase2_hparam"]:
        if phase_key in results:
            for cfg_name, summary in results[phase_key].items():
                if isinstance(summary, dict) and "doge_sortino" in summary:
                    rows.append((cfg_name, summary["doge_sortino"],
                                 summary.get("mean_return", 0)*100,
                                 summary.get("mean_trades", 0),
                                 0))

    # sort by sortino desc
    rows.sort(key=lambda x: x[1], reverse=True)
    logger.info(f"{'Name':<45} {'Sort':>8} {'Ret%':>8} {'Trades':>8} {'MaxDD%':>8}")
    logger.info("-" * 80)
    for name, sort, ret, trades, dd in rows[:30]:
        logger.info(f"{name:<45} {sort:>8.2f} {ret:>8.1f} {trades:>8.0f} {dd:>8.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default="all", help="1|2|3|4|all")
    parser.add_argument("--configs", type=str, default=None, help="comma-sep config names")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    filter_names = set(args.configs.split(",")) if args.configs else None
    results = load_results()

    phases = args.phase.split(",") if args.phase != "all" else ["1", "2", "3", "4"]

    if "1" in phases:
        phase1_leverage(results)
        phase1_eval_params(results)

    if "2" in phases:
        logger.info("\n=== Phase 2a: Architecture Experiments ===")
        phase2_train_and_eval(results, ARCH_CONFIGS, "phase2_arch", filter_names)
        logger.info("\n=== Phase 2b: Hyperparam Experiments ===")
        phase2_train_and_eval(results, HPARAM_CONFIGS, "phase2_hparam", filter_names)

    if "3" in phases:
        phase3_cross_asset(results)

    if "4" in phases:
        logger.info("\n=== Phase 4: Combination (manual after reviewing results) ===")

    print_summary(results)


if __name__ == "__main__":
    main()
