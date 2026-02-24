#!/usr/bin/env python3
"""Multi-symbol high-RW sweep with per-epoch eval and min_edge filtering."""
from __future__ import annotations

import json, sys, time
from dataclasses import asdict
from pathlib import Path

import torch
import numpy as np
from loguru import logger

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from binanceneural.config import TrainingConfig
from binanceneural.trainer import BinanceHourlyTrainer
from binanceneural.inference import generate_actions_from_frame
from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import load_policy_checkpoint, _build_policy
from binanceleveragesui.run_leverage_sweep import (
    LeverageConfig, SUI_HOURLY_MARGIN_RATE, MAKER_FEE_10BP,
    simulate_with_margin_cost,
)

DATA_ROOT = REPO / "trainingdatahourlybinance"
CHECKPOINT_ROOT = REPO / "binanceleveragesui" / "checkpoints"

# Symbol -> forecast cache root mapping
SYMBOL_CONFIGS = {
    "SUIUSDT": {
        "forecast_cache": REPO / "binancechronossolexperiment" / "forecast_cache_sui_10bp",
        "horizons": (1,),
    },
    "BTCUSD": {
        "forecast_cache": REPO / "binanceneural" / "forecast_cache",
        "horizons": (1,),
    },
    "ETHUSD": {
        "forecast_cache": REPO / "binanceneural" / "forecast_cache",
        "horizons": (1,),
    },
    "SOLUSD": {
        "forecast_cache": REPO / "binanceneural" / "forecast_cache",
        "horizons": (1,),
    },
    "LINKUSD": {
        "forecast_cache": REPO / "binanceneural" / "forecast_cache",
        "horizons": (1,),
    },
    "DOGEUSD": {
        "forecast_cache": REPO / "binanceneural" / "forecast_cache",
        "horizons": (1,),
    },
    "AVAXUSD": {
        "forecast_cache": REPO / "binanceneural" / "forecast_cache",
        "horizons": (1,),
    },
    "AAVEUSD": {
        "forecast_cache": REPO / "binanceneural" / "forecast_cache",
        "horizons": (1,),
    },
    "LTCUSD": {
        "forecast_cache": REPO / "binanceneural" / "forecast_cache",
        "horizons": (1,),
    },
    "UNIUSD": {
        "forecast_cache": REPO / "binanceneural" / "forecast_cache",
        "horizons": (1,),
    },
}

RW_VALUES = [0.10, 0.30, 0.50]
WD_VALUES = [0.03, 0.05]
EPOCHS = 20
MIN_EDGES = [0.0, 0.002, 0.004, 0.006, 0.008, 0.010]
FILL_BUFFER = 0.0013


def train_model(symbol, rw, wd, epochs=EPOCHS):
    cfg = SYMBOL_CONFIGS[symbol]
    tag = f"{symbol}_rw{int(rw*100):02d}_wd{int(wd*100):02d}"
    logger.info(f"\n=== Training {tag} ===")

    dm = ChronosSolDataModule(
        symbol=symbol,
        data_root=DATA_ROOT,
        forecast_cache_root=cfg["forecast_cache"],
        forecast_horizons=cfg["horizons"],
        context_hours=512,
        quantile_levels=(0.1, 0.5, 0.9),
        batch_size=32,
        model_id="amazon/chronos-t5-small",
        sequence_length=72,
        split_config=SplitConfig(val_days=30, test_days=30),
        cache_only=True,
        max_history_days=365,
    )

    ckpt_root = CHECKPOINT_ROOT / tag
    tc = TrainingConfig(
        epochs=epochs,
        batch_size=16,
        sequence_length=72,
        learning_rate=1e-4,
        weight_decay=wd,
        return_weight=rw,
        seed=1337,
        transformer_dim=256,
        transformer_layers=4,
        transformer_heads=8,
        maker_fee=MAKER_FEE_10BP,
        fill_temperature=0.1,
        fill_buffer_pct=FILL_BUFFER,
        checkpoint_root=ckpt_root,
        log_dir=Path("tensorboard_logs/binanceleveragesui"),
        use_compile=False,
    )

    trainer = BinanceHourlyTrainer(tc, dm)
    artifacts = trainer.train()
    return ckpt_root, dm, cfg["horizons"]


def evaluate_epoch(ckpt_path, dm, horizons, tag):
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = payload.get("state_dict", payload)
    cfg = payload.get("config", {})
    feature_columns = list(dm.feature_columns)
    normalizer = dm.normalizer
    model = _build_policy(sd, cfg, len(feature_columns))
    test_frame = dm.test_frame.copy()
    actions = generate_actions_from_frame(
        model=model, frame=test_frame, feature_columns=feature_columns,
        normalizer=normalizer, sequence_length=72, horizon=horizons[0],
    )
    test_start = dm.test_window_start
    bars = test_frame[test_frame["timestamp"] >= test_start].copy()
    actions_test = actions[actions["timestamp"] >= test_start].copy()

    results = {}
    for min_edge in MIN_EDGES:
        lcfg = LeverageConfig(
            max_leverage=1.0, initial_cash=5000.0,
            decision_lag_bars=1, fill_buffer_pct=FILL_BUFFER,
            margin_hourly_rate=SUI_HOURLY_MARGIN_RATE,
            maker_fee=MAKER_FEE_10BP,
            min_edge=min_edge,
        )
        r = simulate_with_margin_cost(bars, actions_test, lcfg)
        key = f"edge{int(min_edge*1000)}"
        results[key] = {
            "return": r["total_return"],
            "sortino": r["sortino"],
            "trades": r["num_trades"],
        }
    return results


def sweep_epochs(ckpt_root, dm, horizons, tag):
    epoch_results = {}
    best_sort = -999
    best_epoch = -1

    ckpt_files = sorted(ckpt_root.rglob("epoch_*.pt"))
    if not ckpt_files:
        p = ckpt_root / "policy_checkpoint.pt"
        if p.exists():
            ckpt_files = [p]

    for ckpt_path in ckpt_files:
        ep_name = ckpt_path.stem
        try:
            results = evaluate_epoch(ckpt_path, dm, horizons, tag)
        except Exception as e:
            logger.warning(f"  {tag} {ep_name}: FAILED {e}")
            continue

        edge0 = results.get("edge0", {})
        s = edge0.get("sortino", 0)
        r = edge0.get("return", 0)
        t = edge0.get("trades", 0)
        epoch_results[ep_name] = results

        # find best edge for this epoch
        best_edge_sort = max(v.get("sortino", -999) for v in results.values())
        best_edge_key = max(results.keys(), key=lambda k: results[k].get("sortino", -999))

        logger.info(f"  {tag} {ep_name}: edge0 sort={s:.2f} ret={r:+.4f} t={t} | best={best_edge_key} sort={best_edge_sort:.2f}")

        if best_edge_sort > best_sort:
            best_sort = best_edge_sort
            best_epoch = ep_name

    logger.info(f"  {tag} BEST: {best_epoch} sort={best_sort:.2f}")
    return epoch_results, best_epoch, best_sort


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="SUIUSDT",
                        help="Comma-separated symbols")
    parser.add_argument("--rw", type=str, default="0.10,0.30,0.50")
    parser.add_argument("--wd", type=str, default="0.03,0.05")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    rw_vals = [float(x) for x in args.rw.split(",")]
    wd_vals = [float(x) for x in args.wd.split(",")]

    logger.info(f"=== Multi-Symbol High-RW Sweep === symbols={symbols} rw={rw_vals} wd={wd_vals}")
    all_results = []

    for symbol in symbols:
        if symbol not in SYMBOL_CONFIGS:
            logger.warning(f"No config for {symbol}, skipping")
            continue

        csv_path = DATA_ROOT / f"{symbol}.csv"
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            logger.warning(f"No data for {symbol} ({csv_path}), skipping")
            continue

        for rw in rw_vals:
            for wd in wd_vals:
                tag = f"{symbol}_rw{int(rw*100):02d}_wd{int(wd*100):02d}"
                try:
                    ckpt_root, dm, horizons = train_model(symbol, rw, wd, args.epochs)
                    epoch_results, best_epoch, best_sort = sweep_epochs(ckpt_root, dm, horizons, tag)
                    entry = {
                        "tag": tag, "symbol": symbol, "rw": rw, "wd": wd,
                        "best_epoch": best_epoch, "best_sortino": best_sort,
                        "epochs": epoch_results,
                    }
                    all_results.append(entry)
                except Exception as e:
                    logger.error(f"{tag} FAILED: {e}")
                    continue

                out_path = REPO / "binanceleveragesui" / "high_rw_sweep_results.json"
                with open(out_path, "w") as f:
                    json.dump(all_results, f, indent=2, default=str)

    # Summary
    logger.info("\n\n=== SUMMARY (lag=1, buf=13bp) ===")
    logger.info(f"{'Config':>30} | {'Best Epoch':>12} {'Best Sort':>10}")
    logger.info("-" * 60)
    for r in all_results:
        logger.info(f"{r['tag']:>30} | {r['best_epoch']:>12} {r['best_sortino']:>10.2f}")

    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
