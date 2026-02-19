#!/usr/bin/env python3
"""Portfolio backtest with leverage sweep."""
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


def load_model_and_actions(ckpt_dir: Path, epoch: int, symbols: list,
                           data_root: Path, cache_root: Path, device):
    with open(ckpt_dir / "config.json") as f:
        config = json.load(f)

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
    ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
    ckpt = torch_load_compat(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
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
    parser.add_argument("--min-edge", type=float, default=0.001)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    logger.info("Loading model...")
    bars, actions = load_model_and_actions(
        args.checkpoint_dir, args.epoch, symbols,
        args.data_root, args.cache_root, device)

    results = []
    for max_pos in [10, 18]:
        for hold in [4, 6]:
            for lev in [1.0, 1.5, 2.0]:
                for me in [0.0, 0.001, 0.003, 0.005]:
                    cfg = PortfolioConfig(
                        initial_cash=args.initial_cash,
                        max_positions=max_pos,
                        min_edge=me,
                        max_hold_hours=hold,
                        enforce_market_hours=True,
                        close_at_eod=True,
                        symbols=symbols,
                        max_leverage=lev,
                    )
                    r = run_portfolio_simulation(bars, actions, cfg, horizon=1)
                    ret = r.metrics["total_return"] * 100
                    sort = r.metrics["sortino"]
                    buys = r.metrics["num_buys"]
                    logger.info("pos={:2d} hold={} lev={:.1f} me={:.3f}: ret={:+7.2f}% sort={:6.2f} buys={}",
                                max_pos, hold, lev, me, ret, sort, buys)
                    results.append({
                        "max_pos": max_pos, "hold": hold, "lev": lev, "min_edge": me,
                        "return": ret, "sortino": sort, "buys": buys,
                    })

    # Print sorted by sortino
    results.sort(key=lambda x: x["sortino"], reverse=True)
    logger.info("\n=== Top 10 by Sortino ===")
    for r in results[:10]:
        logger.info("pos={max_pos} hold={hold} lev={lev:.1f} me={min_edge:.3f}: ret={return:+.2f}% sort={sortino:.2f}", **r)

    results.sort(key=lambda x: x["return"], reverse=True)
    logger.info("\n=== Top 10 by Return ===")
    for r in results[:10]:
        logger.info("pos={max_pos} hold={hold} lev={lev:.1f} me={min_edge:.3f}: ret={return:+.2f}% sort={sortino:.2f}", **r)

    with open("unified_hourly_experiment/portfolio_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
