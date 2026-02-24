#!/usr/bin/env python3
"""Sweep across checkpoint epochs to find optimal early stopping point."""
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
from binanceneural.config import DatasetConfig
from binanceneural.model import build_policy, PolicyConfig
from binanceneural.inference import generate_actions_from_frame
from unified_hourly_experiment.marketsimulator import (
    UnifiedSelectionConfig,
    run_unified_simulation,
)
from src.torch_load_utils import torch_load_compat


def backtest_checkpoint(ckpt_path: Path, config: dict, symbols: list,
                        data_modules: dict, normalizer, device, min_edge: float,
                        initial_cash: float, bar_margin: float = 0.0,
                        max_hold_hours: int = None):
    ckpt = torch_load_compat(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    feature_columns = config.get("feature_columns", [])
    policy_cfg = PolicyConfig(
        input_dim=len(feature_columns),
        hidden_dim=config.get("transformer_dim", 128),
        num_heads=config.get("transformer_heads", 4),
        num_layers=config.get("transformer_layers", 3),
        num_outputs=config.get("num_outputs", 4),
        model_arch="gemma",
        max_len=config.get("sequence_length", 32),
    )
    model = build_policy(policy_cfg)
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)

    all_bars = []
    all_actions = []
    for symbol in data_modules:
        frame = data_modules[symbol].frame.copy()
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

    sim_config = UnifiedSelectionConfig(
        initial_cash=initial_cash, min_edge=min_edge,
        enforce_market_hours=True, close_at_eod=True,
        symbols=symbols, max_leverage_stock=1.0, max_leverage_crypto=1.0,
        bar_margin=bar_margin, max_hold_hours=max_hold_hours,
    )
    result = run_unified_simulation(bars, actions, sim_config, horizon=1)
    ret = (result.equity_curve.iloc[-1] / initial_cash - 1) * 100
    sortino = result.metrics.get("sortino", 0)
    return ret, sortino, len(result.trades)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--symbols", default="NVDA,MSFT,META,GOOG,NET,PLTR,NYT,YELP,DBX,TRIP,KIND,EBAY,MTCH,ANGI,Z,EXPE,BKNG,NWSA")
    parser.add_argument("--data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--cache-root", type=Path, default=Path("unified_hourly_experiment/forecast_cache"))
    parser.add_argument("--initial-cash", type=float, default=10000)
    parser.add_argument("--min-edge", type=float, default=0.001)
    parser.add_argument("--bar-margin", type=float, default=0.0)
    parser.add_argument("--max-hold-hours", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    with open(args.checkpoint_dir / "config.json") as f:
        config = json.load(f)

    feature_cols = config.get("feature_columns", [])
    horizons = sorted({int(c.split("_h")[1]) for c in feature_cols if "_h" in c and c.split("_h")[1].isdigit()}) or [1, 24]

    # Load data once
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
    logger.info("Sweeping {} checkpoints", len(checkpoints))

    results = []
    for ckpt_path in checkpoints:
        epoch = int(ckpt_path.stem.split("_")[1])
        ret, sortino, trades = backtest_checkpoint(
            ckpt_path, config, symbols, data_modules, normalizer,
            device, args.min_edge, args.initial_cash, bar_margin=args.bar_margin,
            max_hold_hours=args.max_hold_hours)
        logger.info("Epoch {:3d}: Return={:+.2f}%, Sortino={:.2f}, Trades={}",
                     epoch, ret, sortino, trades)
        results.append({"epoch": epoch, "return": ret, "sortino": sortino, "trades": trades})

    # Find optimal
    best = max(results, key=lambda x: x["sortino"])
    logger.info("=" * 60)
    logger.info("Best: Epoch {}, Return={:.2f}%, Sortino={:.2f}",
                 best["epoch"], best["return"], best["sortino"])

    out_path = args.checkpoint_dir / "epoch_sweep.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved to {}", out_path)


if __name__ == "__main__":
    main()
