#!/usr/bin/env python3
"""Sweep checkpoint epochs for crypto using inverse market hours simulator."""
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
from cryptoalpacaexperiment.marketsimulator.crypto_simulator import (
    CryptoPortfolioConfig,
    run_crypto_simulation,
)
from src.torch_load_utils import torch_load_compat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--symbol", default="BTCUSD")
    parser.add_argument("--data-root", type=Path, default=Path("trainingdatahourly/crypto"))
    parser.add_argument("--cache-root", type=Path, default=Path("cryptoalpacaexperiment/forecast_cache"))
    parser.add_argument("--initial-cash", type=float, default=10000)
    parser.add_argument("--max-positions", type=int, default=1)
    parser.add_argument("--max-hold-hours", type=int, default=6)
    parser.add_argument("--min-edge", type=float, default=0.0)
    parser.add_argument("--decision-lag-bars", type=int, default=1)
    parser.add_argument("--bar-margin", type=float, default=0.0005)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--force-close-slippage", type=float, default=0.001)
    parser.add_argument("--skip-market-hours", action="store_true", default=True)
    parser.add_argument("--no-skip-market-hours", dest="skip_market_hours", action="store_false")
    parser.add_argument("--holdout-days", type=int, default=30)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--direction", default="long", choices=["long", "short"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    symbol = args.symbol.upper()

    with open(args.checkpoint_dir / "config.json") as f:
        config = json.load(f)

    feature_columns = config.get("feature_columns", [])
    horizons = config.get("forecast_horizons", None)
    if horizons is None:
        horizons = sorted({int(c.split("_h")[1]) for c in feature_columns
                           if "_h" in c and c.split("_h")[1].isdigit()}) or [1, 6]

    data_config = DatasetConfig(
        symbol=symbol, data_root=str(args.data_root),
        forecast_cache_root=str(args.cache_root),
        forecast_horizons=horizons, sequence_length=config.get("sequence_length", 48),
        min_history_hours=100, validation_days=30, cache_only=True,
    )
    dm = BinanceHourlyDataModule(data_config)

    if "normalizer" in config:
        from binanceneural.data import FeatureNormalizer
        normalizer = FeatureNormalizer.from_dict(config["normalizer"])
    else:
        normalizer = dm.normalizer

    checkpoints = sorted(args.checkpoint_dir.glob("epoch_*.pt"),
                         key=lambda p: int(p.stem.split("_")[1]))
    if args.epoch is not None:
        checkpoints = [c for c in checkpoints if int(c.stem.split("_")[1]) == args.epoch]

    long_only = {symbol} if args.direction == "long" else None
    short_only = {symbol} if args.direction == "short" else None

    holdout_suffix = f" (OOS {args.holdout_days}d)" if args.holdout_days > 0 else ""
    logger.info("Sweeping {} checkpoints for {} ({}, mkt_skip={}){}",
                len(checkpoints), symbol, args.direction, args.skip_market_hours, holdout_suffix)

    results = []
    for ckpt_path in checkpoints:
        epoch = int(ckpt_path.stem.split("_")[1])
        ckpt = torch_load_compat(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)

        pe_key = "pos_encoding.pe"
        max_len = config.get("sequence_length", 48)
        if pe_key in state_dict:
            max_len = max(max_len, state_dict[pe_key].shape[0])

        policy_cfg = PolicyConfig(
            input_dim=len(feature_columns),
            hidden_dim=config.get("transformer_dim", 512),
            num_heads=config.get("transformer_heads", 8),
            num_layers=config.get("transformer_layers", 6),
            num_outputs=config.get("num_outputs", 4),
            model_arch=config.get("model_arch", "gemma"),
            max_len=max_len,
        )
        model = build_policy(policy_cfg)
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        model.eval().to(device)

        frame = dm.frame.copy()
        frame["symbol"] = symbol
        actions_df = generate_actions_from_frame(
            model=model, frame=frame, feature_columns=feature_columns,
            normalizer=normalizer, sequence_length=config.get("sequence_length", 48),
            horizon=1, device=device,
        )

        bars = frame.copy()
        actions = actions_df.copy()

        if args.holdout_days > 0:
            cutoff = bars["timestamp"].max() - pd.Timedelta(days=args.holdout_days)
            bars = bars[bars["timestamp"] >= cutoff].reset_index(drop=True)
            actions = actions[actions["timestamp"] >= cutoff].reset_index(drop=True)

        cfg = CryptoPortfolioConfig(
            initial_cash=args.initial_cash,
            max_positions=args.max_positions,
            min_edge=args.min_edge,
            max_hold_hours=args.max_hold_hours,
            skip_during_market_hours=args.skip_market_hours,
            backout_before_market=args.skip_market_hours,
            symbols=[symbol],
            fee_by_symbol={symbol: args.fee},
            decision_lag_bars=args.decision_lag_bars,
            bar_margin=args.bar_margin,
            force_close_slippage=args.force_close_slippage,
            long_only_symbols=long_only,
            short_only_symbols=short_only,
        )
        r = run_crypto_simulation(bars, actions, cfg, horizon=1)
        ret = r.metrics["total_return"] * 100
        sort = r.metrics["sortino"]
        dd = r.metrics.get("max_drawdown", 0) * 100
        entries = r.metrics["num_entries"]
        targets = r.metrics["target_exits"]
        timeouts = r.metrics["timeout_exits"]
        backouts = r.metrics["market_backouts"]

        logger.info("Epoch {:3d}: ret={:+7.2f}% sort={:6.2f} dd={:+.1f}% entries={} tgt={} to={} bo={}",
                     epoch, ret, sort, dd, entries, targets, timeouts, backouts)
        results.append({
            "epoch": epoch, "return": ret, "sortino": sort, "max_drawdown": dd,
            "entries": entries, "target_exits": targets, "timeout_exits": timeouts,
            "market_backouts": backouts,
        })

    if not results:
        logger.error("No results")
        return

    best_sort = max(results, key=lambda x: x["sortino"])
    best_ret = max(results, key=lambda x: x["return"])
    logger.info("=" * 60)
    logger.info("Best Sortino: Epoch {} ({:+.2f}%, sort={:.2f})", best_sort["epoch"], best_sort["return"], best_sort["sortino"])
    logger.info("Best Return:  Epoch {} ({:+.2f}%, sort={:.2f})", best_ret["epoch"], best_ret["return"], best_ret["sortino"])

    out_path = args.checkpoint_dir / f"sweep_crypto_{symbol}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved to {}", out_path)


if __name__ == "__main__":
    main()
