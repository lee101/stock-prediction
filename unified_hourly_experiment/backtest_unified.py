#!/usr/bin/env python3
"""Backtest unified policy on stocks with proper market hours."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import torch
from loguru import logger

from binanceneural.data import BinanceHourlyDataModule, FeatureNormalizer
from binanceneural.config import DatasetConfig
from binanceneural.model import build_policy, PolicyConfig
from binanceneural.inference import generate_actions_from_frame
from unified_hourly_experiment.marketsimulator import (
    UnifiedSelectionConfig,
    run_unified_simulation,
)
from src.torch_load_utils import torch_load_compat


def load_model(checkpoint_dir: Path):
    """Load model from checkpoint directory."""
    checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")

    best_ckpt = checkpoints[-1]
    logger.info("Loading checkpoint: {}", best_ckpt.name)

    ckpt = torch_load_compat(best_ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    # Load config.json for feature columns
    config_path = checkpoint_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    feature_columns = config.get("feature_columns", [])

    input_dim = len(feature_columns)
    sequence_length = config.get("sequence_length", 32)
    hidden_dim = config.get("transformer_dim", 128)
    num_heads = config.get("transformer_heads", 4)
    num_layers = config.get("transformer_layers", 3)
    policy_cfg = PolicyConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        model_arch="gemma",
        max_len=sequence_length,
    )
    model = build_policy(policy_cfg)

    # Handle _orig_mod prefix from torch.compile
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, feature_columns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--symbols", default="NVDA,MSFT,META,GOOG,NET,PLTR,NYT,YELP,DBX,TRIP")
    parser.add_argument("--data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--cache-root", type=Path, default=Path("unified_hourly_experiment/forecast_cache"))
    parser.add_argument("--initial-cash", type=float, default=10000)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--min-edge", type=float, default=0.001)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, feature_columns = load_model(args.checkpoint_dir)
    model = model.to(device)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    logger.info("Running backtest for {} symbols", len(symbols))

    # First pass: load all data to compute normalizer
    all_bars = []
    data_modules = {}
    for symbol in symbols:
        data_config = DatasetConfig(
            symbol=symbol,
            data_root=str(args.data_root),
            forecast_cache_root=str(args.cache_root),
            forecast_horizons=[1, 24],
            sequence_length=args.sequence_length,
            min_history_hours=100,
            validation_days=30,
            cache_only=True,
        )
        try:
            data_module = BinanceHourlyDataModule(data_config)
            data_modules[symbol] = data_module
            frame = data_module.frame.copy()
            frame["symbol"] = symbol
            all_bars.append(frame)
        except Exception as e:
            logger.warning("Failed to load {}: {}", symbol, e)

    if not all_bars:
        logger.error("No data loaded")
        return

    # Use first symbol's normalizer (they should be similar)
    first_symbol = list(data_modules.keys())[0]
    normalizer = data_modules[first_symbol].normalizer

    all_actions = []
    for symbol in data_modules:
        frame = data_modules[symbol].frame.copy()
        frame["symbol"] = symbol

        logger.info("Generating actions for {} ({} bars)", symbol, len(frame))
        actions_df = generate_actions_from_frame(
            model=model,
            frame=frame,
            feature_columns=feature_columns,
            normalizer=normalizer,
            sequence_length=args.sequence_length,
            horizon=args.horizon,
            device=device,
        )
        all_actions.append(actions_df)

    if not all_bars:
        logger.error("No data loaded")
        return

    bars = pd.concat(all_bars, ignore_index=True)
    actions = pd.concat(all_actions, ignore_index=True)

    logger.info("Total bars: {} | Total actions: {}", len(bars), len(actions))

    # Run simulation
    sim_config = UnifiedSelectionConfig(
        initial_cash=args.initial_cash,
        min_edge=args.min_edge,
        enforce_market_hours=True,
        close_at_eod=True,
        symbols=symbols,
        max_leverage_stock=1.0,
        max_leverage_crypto=1.0,
    )

    result = run_unified_simulation(bars, actions, sim_config, horizon=args.horizon)

    # Report results
    logger.info("=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)
    logger.info("Initial: ${:.2f}", args.initial_cash)
    logger.info("Final equity: ${:.2f}", result.equity_curve.iloc[-1])
    logger.info("Total return: {:.2f}%", (result.equity_curve.iloc[-1] / args.initial_cash - 1) * 100)
    logger.info("Trades: {}", len(result.trades))

    for metric, value in result.metrics.items():
        logger.info("{}: {:.4f}", metric, value)

    # Save equity curve
    out_path = args.checkpoint_dir / "backtest_equity.csv"
    result.equity_curve.to_csv(out_path)
    logger.info("Equity curve saved to {}", out_path)


if __name__ == "__main__":
    main()
