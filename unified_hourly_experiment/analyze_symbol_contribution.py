#!/usr/bin/env python3
"""Analyze per-symbol contribution to portfolio returns."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import torch
from loguru import logger

from binanceneural.data import BinanceHourlyDataModule
from binanceneural.config import DatasetConfig, PolicyConfig
from binanceneural.model import build_policy
from binanceneural.inference import generate_actions_from_frame
from unified_hourly_experiment.marketsimulator import PortfolioConfig, run_portfolio_simulation
from src.torch_load_utils import torch_load_compat


def load_model_and_actions(ckpt_dir, epoch, symbols, data_root, cache_root, device):
    with open(ckpt_dir / "config.json") as f:
        config = json.load(f)
    meta_path = ckpt_dir / "training_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            config.update(json.load(f))

    feature_columns = config.get("feature_columns", [])
    policy_cfg = PolicyConfig(
        input_dim=len(feature_columns),
        hidden_dim=config.get("transformer_dim", 128),
        num_heads=config.get("transformer_heads", 4),
        num_layers=config.get("transformer_layers", 3),
        num_outputs=config.get("num_outputs", 4),
        model_arch=config.get("model_arch", "gemma"),
        max_len=config.get("sequence_length", 32),
    )
    model = build_policy(policy_cfg)
    ckpt = torch_load_compat(ckpt_dir / f"epoch_{epoch:03d}.pt", map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)

    horizons = sorted({int(c.split("_h")[1]) for c in feature_columns
                       if "_h" in c and c.split("_h")[1].isdigit()}) or [1, 24]
    all_bars, all_actions = [], []
    for symbol in symbols:
        data_config = DatasetConfig(
            symbol=symbol, data_root=str(data_root),
            forecast_cache_root=str(cache_root),
            forecast_horizons=horizons,
            sequence_length=config.get("sequence_length", 32),
            min_history_hours=100, validation_days=30, cache_only=True,
        )
        try:
            dm = BinanceHourlyDataModule(data_config)
        except Exception as e:
            logger.warning("Skip {}: {}", symbol, e)
            continue
        frame = dm.frame.copy()
        frame["symbol"] = symbol
        all_bars.append(frame)
        actions_df = generate_actions_from_frame(
            model=model, frame=frame, feature_columns=feature_columns,
            normalizer=dm.normalizer, sequence_length=config.get("sequence_length", 32),
            horizon=1, device=device,
        )
        all_actions.append(actions_df)
    return pd.concat(all_bars, ignore_index=True), pd.concat(all_actions, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--symbols", default="NVDA,MSFT,META,GOOG,NET,PLTR,NYT,YELP,DBX,TRIP,KIND,EBAY,MTCH,ANGI,Z,EXPE,BKNG,NWSA")
    parser.add_argument("--data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--cache-root", type=Path, default=Path("unified_hourly_experiment/forecast_cache"))
    parser.add_argument("--initial-cash", type=float, default=10000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    logger.info("Loading model...")
    bars, actions = load_model_and_actions(
        args.checkpoint_dir, args.epoch, symbols,
        args.data_root, args.cache_root, device)

    # Full portfolio baseline
    cfg = PortfolioConfig(initial_cash=args.initial_cash, max_positions=18,
                          min_edge=0.0, max_hold_hours=4, symbols=symbols)
    full = run_portfolio_simulation(bars, actions, cfg, horizon=1)
    logger.info("FULL ({} symbols): ret={:+.2f}% sort={:.2f}",
                len(symbols), full.metrics["total_return"]*100, full.metrics["sortino"])

    # Per-symbol: leave-one-out analysis
    results = []
    for drop in symbols:
        subset = [s for s in symbols if s != drop]
        cfg_sub = PortfolioConfig(initial_cash=args.initial_cash, max_positions=17,
                                  min_edge=0.0, max_hold_hours=4, symbols=subset)
        r = run_portfolio_simulation(bars, actions, cfg_sub, horizon=1)
        delta_ret = r.metrics["total_return"]*100 - full.metrics["total_return"]*100
        delta_sort = r.metrics["sortino"] - full.metrics["sortino"]
        results.append({"symbol": drop, "without_ret": r.metrics["total_return"]*100,
                        "without_sort": r.metrics["sortino"],
                        "delta_ret": delta_ret, "delta_sort": delta_sort})
        logger.info("  -{}: ret={:+.2f}% sort={:.2f} (dRet={:+.2f}% dSort={:+.2f})",
                    drop, r.metrics["total_return"]*100, r.metrics["sortino"], delta_ret, delta_sort)

    # Also per-symbol solo performance
    logger.info("\n=== Solo Symbol Performance ===")
    for sym in symbols:
        cfg_solo = PortfolioConfig(initial_cash=args.initial_cash, max_positions=1,
                                   min_edge=0.0, max_hold_hours=4, symbols=[sym])
        r = run_portfolio_simulation(bars, actions, cfg_solo, horizon=1)
        logger.info("  {}: ret={:+.2f}% sort={:.2f} buys={}",
                    sym, r.metrics["total_return"]*100, r.metrics["sortino"], r.metrics["num_buys"])

    # Sort by impact
    results.sort(key=lambda x: x["delta_sort"])
    logger.info("\n=== Symbols ranked by Sortino impact (drop = improvement means symbol hurts) ===")
    for r in results:
        direction = "HURTS" if r["delta_sort"] > 0 else "HELPS"
        logger.info("  {}: dSort={:+.2f} dRet={:+.2f}% [{}]",
                    r["symbol"], r["delta_sort"], r["delta_ret"], direction)


if __name__ == "__main__":
    main()
