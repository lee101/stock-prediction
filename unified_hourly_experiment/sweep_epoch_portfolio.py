#!/usr/bin/env python3
"""Sweep checkpoint epochs using portfolio mode (multi-position)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import torch
from loguru import logger

from binanceneural.data import BinanceHourlyDataModule
from binanceneural.config import DatasetConfig, PolicyConfig
from binanceneural.model import build_policy
from binanceneural.inference import generate_actions_from_frame
from unified_hourly_experiment.marketsimulator import (
    PortfolioConfig, run_portfolio_simulation,
)
from src.torch_load_utils import torch_load_compat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--symbols", default="NVDA,PLTR,TRIP,EBAY,GOOG,NET,KIND,MTCH,NYT,DBX")
    parser.add_argument("--data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--cache-root", type=Path, default=Path("unified_hourly_experiment/forecast_cache"))
    parser.add_argument("--initial-cash", type=float, default=10000)
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--max-hold-hours", type=int, default=4)
    parser.add_argument("--min-edge", type=float, default=0.0)
    parser.add_argument("--decision-lag-bars", type=int, default=0)
    parser.add_argument("--market-order-entry", action="store_true")
    parser.add_argument("--bar-margin", type=float, default=0.0)
    parser.add_argument("--holdout-days", type=int, default=0,
                        help="Only simulate on last N days (OOS only). 0=all data.")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Run single epoch instead of sweep")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    with open(args.checkpoint_dir / "config.json") as f:
        config = json.load(f)

    feature_columns = config.get("feature_columns", [])
    horizons = sorted({int(c.split("_h")[1]) for c in feature_columns
                       if "_h" in c and c.split("_h")[1].isdigit()}) or [1, 24]

    data_modules = {}
    for symbol in symbols:
        data_config = DatasetConfig(
            symbol=symbol, data_root=str(args.data_root),
            forecast_cache_root=str(args.cache_root),
            forecast_horizons=horizons, sequence_length=config.get("sequence_length", 32),
            min_history_hours=100, validation_days=30, cache_only=True,
        )
        try:
            data_modules[symbol] = BinanceHourlyDataModule(data_config)
        except Exception as e:
            logger.warning("Skip {}: {}", symbol, e)

    normalizer = list(data_modules.values())[0].normalizer

    checkpoints = sorted(args.checkpoint_dir.glob("epoch_*.pt"),
                         key=lambda p: int(p.stem.split("_")[1]))
    if args.epoch is not None:
        checkpoints = [c for c in checkpoints if int(c.stem.split("_")[1]) == args.epoch]
    holdout_suffix = ""
    if args.holdout_days > 0:
        holdout_suffix = f" (OOS {args.holdout_days}d)"
    logger.info("Sweeping {} checkpoints over {} symbols (portfolio mode, pos={}){}",
                len(checkpoints), len(data_modules), args.max_positions, holdout_suffix)

    results = []
    for ckpt_path in checkpoints:
        epoch = int(ckpt_path.stem.split("_")[1])
        ckpt = torch_load_compat(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)

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
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        model.eval().to(device)

        all_bars, all_actions = [], []
        for symbol, dm in data_modules.items():
            frame = dm.frame.copy()
            frame["symbol"] = symbol
            all_bars.append(frame)
            actions_df = generate_actions_from_frame(
                model=model, frame=frame, feature_columns=feature_columns,
                normalizer=normalizer, sequence_length=config.get("sequence_length", 32),
                horizon=1, device=device,
            )
            all_actions.append(actions_df)

        bars = pd.concat(all_bars, ignore_index=True)
        actions = pd.concat(all_actions, ignore_index=True)

        if args.holdout_days > 0:
            cutoff = bars["timestamp"].max() - pd.Timedelta(days=args.holdout_days)
            bars = bars[bars["timestamp"] >= cutoff].reset_index(drop=True)
            actions = actions[actions["timestamp"] >= cutoff].reset_index(drop=True)

        cfg = PortfolioConfig(
            initial_cash=args.initial_cash, max_positions=args.max_positions,
            min_edge=args.min_edge, max_hold_hours=args.max_hold_hours,
            enforce_market_hours=True, close_at_eod=True, symbols=symbols,
            decision_lag_bars=args.decision_lag_bars,
            market_order_entry=args.market_order_entry,
            bar_margin=args.bar_margin,
        )
        r = run_portfolio_simulation(bars, actions, cfg, horizon=1)
        ret = r.metrics["total_return"] * 100
        sort = r.metrics["sortino"]
        buys = r.metrics["num_buys"]
        dd = r.metrics.get("max_drawdown", 0) * 100
        logger.info("Epoch {:3d}: ret={:+7.2f}% sort={:6.2f} dd={:+.1f}% buys={}",
                     epoch, ret, sort, dd, buys)
        results.append({"epoch": epoch, "return": ret, "sortino": sort,
                        "max_drawdown": dd, "buys": buys})

    best_sort = max(results, key=lambda x: x["sortino"])
    best_ret = max(results, key=lambda x: x["return"])
    logger.info("=" * 60)
    logger.info("Best Sortino: Epoch {} ({:.2f}%, sort={:.2f})", best_sort["epoch"], best_sort["return"], best_sort["sortino"])
    logger.info("Best Return:  Epoch {} ({:.2f}%, sort={:.2f})", best_ret["epoch"], best_ret["return"], best_ret["sortino"])

    out_path = args.checkpoint_dir / "epoch_sweep_portfolio.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved to {}", out_path)


if __name__ == "__main__":
    main()
