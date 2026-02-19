#!/usr/bin/env python3
"""Sweep min_buy_amount threshold for a single checkpoint."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd, torch
from loguru import logger
from binanceneural.data import BinanceHourlyDataModule
from binanceneural.config import DatasetConfig, PolicyConfig
from binanceneural.model import build_policy
from binanceneural.inference import generate_actions_from_frame
from unified_hourly_experiment.marketsimulator import PortfolioConfig, run_portfolio_simulation
from src.torch_load_utils import torch_load_compat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--symbols", default="NVDA,PLTR,GOOG,NET,DBX,TRIP,EBAY,KIND,MTCH,NYT")
    parser.add_argument("--data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--cache-root", type=Path, default=Path("unified_hourly_experiment/forecast_cache"))
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--max-hold-hours", type=int, default=4)
    parser.add_argument("--decision-lag-bars", type=int, default=1)
    parser.add_argument("--market-order-entry", action="store_true")
    parser.add_argument("--thresholds", default="0,10,20,30,40,50,60,70,80,90")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    thresholds = [float(t) for t in args.thresholds.split(",")]

    with open(args.config_dir / "config.json") as f:
        config = json.load(f)
    feature_columns = config.get("feature_columns", [])
    horizons = sorted({int(c.split("_h")[1]) for c in feature_columns if "_h" in c and c.split("_h")[1].isdigit()}) or [1]

    data_modules = {}
    for symbol in symbols:
        dc = DatasetConfig(symbol=symbol, data_root=str(args.data_root),
            forecast_cache_root=str(args.cache_root), forecast_horizons=horizons,
            sequence_length=config.get("sequence_length", 32), min_history_hours=100,
            validation_days=30, cache_only=True)
        try:
            data_modules[symbol] = BinanceHourlyDataModule(dc)
        except Exception as e:
            logger.warning("Skip {}: {}", symbol, e)

    normalizer = list(data_modules.values())[0].normalizer
    ckpt = torch_load_compat(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    policy_cfg = PolicyConfig(input_dim=len(feature_columns),
        hidden_dim=config.get("transformer_dim", 128), num_heads=config.get("transformer_heads", 4),
        num_layers=config.get("transformer_layers", 3), num_outputs=config.get("num_outputs", 4),
        model_arch=config.get("model_arch", "gemma"), max_len=config.get("sequence_length", 32))
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
        actions_df = generate_actions_from_frame(model=model, frame=frame,
            feature_columns=feature_columns, normalizer=normalizer,
            sequence_length=config.get("sequence_length", 32), horizon=1, device=device)
        all_actions.append(actions_df)
    bars = pd.concat(all_bars, ignore_index=True)
    actions = pd.concat(all_actions, ignore_index=True)

    for thresh in thresholds:
        cfg = PortfolioConfig(initial_cash=10000, max_positions=args.max_positions,
            min_edge=0.0, max_hold_hours=args.max_hold_hours, enforce_market_hours=True,
            close_at_eod=True, symbols=symbols, decision_lag_bars=args.decision_lag_bars,
            market_order_entry=args.market_order_entry, min_buy_amount=thresh)
        r = run_portfolio_simulation(bars, actions, cfg, horizon=1)
        ret = r.metrics["total_return"] * 100
        sort = r.metrics["sortino"]
        buys = r.metrics["num_buys"]
        logger.info("thresh={:5.1f}: ret={:+8.2f}% sort={:7.2f} buys={}", thresh, ret, sort, buys)

if __name__ == "__main__":
    main()
