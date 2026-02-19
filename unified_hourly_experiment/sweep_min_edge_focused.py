#!/usr/bin/env python3
"""Sweep min_edge for a given checkpoint + symbol set."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd, torch
from loguru import logger
from binanceneural.data import BinanceHourlyDataModule
from binanceneural.config import DatasetConfig
from binanceneural.model import build_policy, PolicyConfig
from binanceneural.inference import generate_actions_from_frame
from unified_hourly_experiment.marketsimulator import UnifiedSelectionConfig, run_unified_simulation
from src.torch_load_utils import torch_load_compat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--symbols", default="NVDA,GOOG,EBAY,PLTR")
    parser.add_argument("--initial-cash", type=float, default=10000)
    parser.add_argument("--epoch", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    with open(args.checkpoint_dir / "config.json") as f:
        config = json.load(f)
    feature_columns = config.get("feature_columns", [])

    if args.epoch:
        ckpt_path = args.checkpoint_dir / f"epoch_{args.epoch:03d}.pt"
    else:
        ckpts = sorted(args.checkpoint_dir.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        ckpt_path = ckpts[-1]

    ckpt = torch_load_compat(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    policy_cfg = PolicyConfig(
        input_dim=len(feature_columns),
        hidden_dim=config.get("transformer_dim", 128),
        num_heads=config.get("transformer_heads", 4),
        num_layers=config.get("transformer_layers", 3),
        model_arch="gemma", max_len=config.get("sequence_length", 32),
    )
    model = build_policy(policy_cfg)
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)

    data_modules = {}
    all_bars, all_actions = [], []
    for symbol in symbols:
        dc = DatasetConfig(
            symbol=symbol, data_root="trainingdatahourly/stocks",
            forecast_cache_root="unified_hourly_experiment/forecast_cache",
            forecast_horizons=[1, 24], sequence_length=config.get("sequence_length", 32),
            min_history_hours=100, validation_days=30, cache_only=True,
        )
        try:
            dm = BinanceHourlyDataModule(dc)
            data_modules[symbol] = dm
            frame = dm.frame.copy()
            frame["symbol"] = symbol
            all_bars.append(frame)
            actions = generate_actions_from_frame(
                model=model, frame=frame, feature_columns=feature_columns,
                normalizer=dm.normalizer, sequence_length=config.get("sequence_length", 32),
                horizon=1, device=device,
            )
            all_actions.append(actions)
        except Exception as e:
            logger.warning("Skip {}: {}", symbol, e)

    bars = pd.concat(all_bars, ignore_index=True)
    actions_df = pd.concat(all_actions, ignore_index=True)

    logger.info("Checkpoint: {}, Symbols: {}", ckpt_path.name, ",".join(symbols))
    for min_edge in [0.0, 0.0003, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01, 0.02]:
        sim_config = UnifiedSelectionConfig(
            initial_cash=args.initial_cash, min_edge=min_edge,
            enforce_market_hours=True, close_at_eod=True,
            symbols=symbols, max_leverage_stock=1.0, max_leverage_crypto=1.0,
        )
        result = run_unified_simulation(bars, actions_df, sim_config, horizon=1)
        ret = (result.equity_curve.iloc[-1] / args.initial_cash - 1) * 100
        sortino = result.metrics.get("sortino", 0)
        logger.info("min_edge={:.4f}: Return={:.2f}%, Sortino={:.2f}, Trades={}",
                     min_edge, ret, sortino, len(result.trades))


if __name__ == "__main__":
    main()
