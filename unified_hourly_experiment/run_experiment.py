#!/usr/bin/env python3
"""Run unified stock+crypto backtest with proper market hours handling."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Dict
import json

import pandas as pd
from loguru import logger

from binanceneural.inference import generate_actions_from_frame
from binanceneural.model import align_state_dict_input_dim, build_policy, policy_config_from_payload
from src.torch_device_utils import require_cuda as require_cuda_device
from newnanoalpacahourlyexp.config import DatasetConfig
from newnanoalpacahourlyexp.data import AlpacaHourlyDataModule
from unified_hourly_experiment.marketsimulator import UnifiedSelectionConfig, run_unified_simulation
import torch


def _parse_symbols(raw: str) -> List[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def _load_model(checkpoint_path: Path, input_dim: int, sequence_length: int):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)
    state_dict = align_state_dict_input_dim(state_dict, input_dim=input_dim)
    cfg = payload.get("config", {})
    if hasattr(cfg, "__dict__"):
        cfg = cfg.__dict__
    from binanceneural.model import policy_config_from_payload
    policy_cfg = policy_config_from_payload(cfg, input_dim=input_dim, state_dict=state_dict)
    policy_cfg.max_len = max(policy_cfg.max_len, sequence_length)
    model = build_policy(policy_cfg)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--forecast-horizons", default="1,24")
    parser.add_argument("--forecast-cache-root", default=None)
    parser.add_argument("--crypto-data-root", default="trainingdatahourly/crypto")
    parser.add_argument("--stock-data-root", default="trainingdatahourly/stocks")
    parser.add_argument("--eval-days", type=float, default=30)
    parser.add_argument("--initial-cash", type=float, default=10000.0)
    parser.add_argument("--min-edge", type=float, default=0.001)
    parser.add_argument("--risk-weight", type=float, default=0.05)
    parser.add_argument("--edge-mode", default="high_low")
    parser.add_argument("--long-only-symbols", default=None)
    parser.add_argument("--short-only-symbols", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--enable-deferred-orders", action="store_true", default=True)
    args = parser.parse_args()

    device = require_cuda_device("unified backtest", allow_fallback=False)
    symbols = _parse_symbols(args.symbols)
    forecast_horizons = tuple(int(x) for x in args.forecast_horizons.split(",") if x)

    logger.info("Loading model from {}", args.checkpoint)

    # Load data for all symbols
    bars_frames = []
    action_frames = []

    for symbol in symbols:
        data_cfg = DatasetConfig(
            symbols=[symbol],
            forecast_cache_root=Path(args.forecast_cache_root) if args.forecast_cache_root else None,
            crypto_data_root=Path(args.crypto_data_root),
            stock_data_root=Path(args.stock_data_root),
            forecast_horizons=forecast_horizons,
        )

        data_module = AlpacaHourlyDataModule(data_cfg)
        frame = data_module.frame

        # Filter to eval window
        if args.eval_days:
            cutoff = frame["timestamp"].max() - pd.Timedelta(days=args.eval_days)
            frame = frame[frame["timestamp"] >= cutoff]

        if frame.empty:
            logger.warning("No data for {}", symbol)
            continue

        # Generate actions
        input_dim = len([c for c in frame.columns if c not in ["timestamp", "symbol"]])
        model = _load_model(Path(args.checkpoint), input_dim=input_dim, sequence_length=args.sequence_length)
        model = model.to(device)

        actions = generate_actions_from_frame(
            model=model,
            frame=frame,
            sequence_length=args.sequence_length,
            horizon=args.horizon,
        )

        bars_frames.append(frame)
        action_frames.append(actions)

    if not bars_frames:
        logger.error("No data loaded")
        return

    bars = pd.concat(bars_frames, ignore_index=True)
    actions = pd.concat(action_frames, ignore_index=True)

    # Run unified simulation
    config = UnifiedSelectionConfig(
        initial_cash=args.initial_cash,
        min_edge=args.min_edge,
        risk_weight=args.risk_weight,
        edge_mode=args.edge_mode,
        symbols=symbols,
        long_only_symbols=_parse_symbols(args.long_only_symbols) if args.long_only_symbols else None,
        short_only_symbols=_parse_symbols(args.short_only_symbols) if args.short_only_symbols else None,
        enable_deferred_orders=args.enable_deferred_orders,
        decision_lag_bars=1,
    )

    result = run_unified_simulation(bars, actions, config, horizon=args.horizon)

    logger.info("total_return: {:.4f}", result.metrics["total_return"])
    logger.info("sortino: {:.4f}", result.metrics["sortino"])
    logger.info("num_trades: {}", result.metrics["num_trades"])
    logger.info("final_equity: {:.2f}", result.metrics["final_equity"])

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        result.equity_curve.to_csv(out / "equity_curve.csv")
        result.per_hour.to_csv(out / "per_hour.csv", index=False)
        trades_df = pd.DataFrame([
            {
                "timestamp": t.timestamp,
                "symbol": t.symbol,
                "side": t.side,
                "price": t.price,
                "quantity": t.quantity,
                "cash_after": t.cash_after,
                "inventory_after": t.inventory_after,
                "reason": t.reason,
            }
            for t in result.trades
        ])
        trades_df.to_csv(out / "trades.csv", index=False)
        with open(out / "metrics.json", "w") as f:
            json.dump(result.metrics, f, indent=2)
        logger.info("Results saved to {}", out)


if __name__ == "__main__":
    main()
