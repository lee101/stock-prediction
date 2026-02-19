#!/usr/bin/env python3
"""Per-symbol contribution analysis for unified selector backtest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import torch
import numpy as np
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
    checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not checkpoints:
        raise ValueError(f"No checkpoints in {checkpoint_dir}")
    best_ckpt = checkpoints[-1]
    logger.info("Checkpoint: {}", best_ckpt.name)

    ckpt = torch_load_compat(best_ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    with open(checkpoint_dir / "config.json") as f:
        config = json.load(f)
    feature_columns = config.get("feature_columns", [])
    policy_cfg = PolicyConfig(
        input_dim=len(feature_columns),
        hidden_dim=config.get("transformer_dim", 128),
        num_heads=config.get("transformer_heads", 4),
        num_layers=config.get("transformer_layers", 3),
        model_arch="gemma",
        max_len=config.get("sequence_length", 32),
    )
    model = build_policy(policy_cfg)
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, feature_columns, config


def run_backtest(model, feature_columns, symbols, config, device, args):
    all_bars = []
    data_modules = {}
    for symbol in symbols:
        data_config = DatasetConfig(
            symbol=symbol,
            data_root=str(args.data_root),
            forecast_cache_root=str(args.cache_root),
            forecast_horizons=[1, 24],
            sequence_length=config.get("sequence_length", 32),
            min_history_hours=100,
            validation_days=30,
            cache_only=True,
        )
        try:
            dm = BinanceHourlyDataModule(data_config)
            data_modules[symbol] = dm
            frame = dm.frame.copy()
            frame["symbol"] = symbol
            all_bars.append(frame)
        except Exception as e:
            logger.warning("Skip {}: {}", symbol, e)

    if not data_modules:
        return None

    normalizer = list(data_modules.values())[0].normalizer

    all_actions = []
    for symbol in data_modules:
        frame = data_modules[symbol].frame.copy()
        frame["symbol"] = symbol
        actions_df = generate_actions_from_frame(
            model=model,
            frame=frame,
            feature_columns=feature_columns,
            normalizer=normalizer,
            sequence_length=config.get("sequence_length", 32),
            horizon=1,
            device=device,
        )
        all_actions.append(actions_df)

    bars = pd.concat(all_bars, ignore_index=True)
    actions = pd.concat(all_actions, ignore_index=True)

    sim_config = UnifiedSelectionConfig(
        initial_cash=args.initial_cash,
        min_edge=args.min_edge,
        enforce_market_hours=True,
        close_at_eod=True,
        symbols=symbols,
        max_leverage_stock=1.0,
        max_leverage_crypto=1.0,
    )
    return run_unified_simulation(bars, actions, sim_config, horizon=1)


def analyze_trades(result):
    if not result or not result.trades:
        return {}
    trades = result.trades
    roundtrips = []
    pending = {}
    for t in trades:
        if t.side == "buy":
            pending[t.symbol] = t
        elif t.side == "sell" and t.symbol in pending:
            entry = pending.pop(t.symbol)
            pnl_pct = (t.price - entry.price) / entry.price * 100
            roundtrips.append({
                "symbol": t.symbol,
                "entry_time": entry.timestamp,
                "exit_time": t.timestamp,
                "entry_price": entry.price,
                "exit_price": t.price,
                "pnl_pct": pnl_pct,
                "reason": t.reason,
            })

    if not roundtrips:
        return {}

    df = pd.DataFrame(roundtrips)
    stats = {}
    for sym in sorted(df["symbol"].unique()):
        sdf = df[df["symbol"] == sym]
        wins = sdf[sdf["pnl_pct"] > 0]
        losses = sdf[sdf["pnl_pct"] <= 0]
        stats[sym] = {
            "trades": len(sdf),
            "win_rate": len(wins) / len(sdf) * 100 if len(sdf) > 0 else 0,
            "total_pnl_pct": sdf["pnl_pct"].sum(),
            "avg_pnl_pct": sdf["pnl_pct"].mean(),
            "avg_win_pct": wins["pnl_pct"].mean() if len(wins) > 0 else 0,
            "avg_loss_pct": losses["pnl_pct"].mean() if len(losses) > 0 else 0,
            "best_trade_pct": sdf["pnl_pct"].max(),
            "worst_trade_pct": sdf["pnl_pct"].min(),
        }
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--symbols", default="NVDA,MSFT,META,GOOG,NET,PLTR,NYT,YELP,DBX,TRIP,KIND,EBAY,MTCH,ANGI,Z,EXPE,BKNG,NWSA")
    parser.add_argument("--data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--cache-root", type=Path, default=Path("unified_hourly_experiment/forecast_cache"))
    parser.add_argument("--initial-cash", type=float, default=10000)
    parser.add_argument("--min-edge", type=float, default=0.001)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, feature_columns, config = load_model(args.checkpoint_dir)
    model = model.to(device)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    # 1. Full backtest with all symbols
    logger.info("=== Full backtest ({} symbols) ===", len(symbols))
    full_result = run_backtest(model, feature_columns, symbols, config, device, args)
    if not full_result:
        logger.error("Full backtest failed")
        return

    full_return = (full_result.equity_curve.iloc[-1] / args.initial_cash - 1) * 100
    full_sortino = full_result.metrics.get("sortino", 0)
    logger.info("Full: Return={:.2f}%, Sortino={:.2f}, Trades={}", full_return, full_sortino, len(full_result.trades))

    # 2. Per-symbol trade analysis
    logger.info("\n=== Per-Symbol Trade Analysis ===")
    trade_stats = analyze_trades(full_result)
    for sym in sorted(trade_stats, key=lambda s: trade_stats[s]["total_pnl_pct"], reverse=True):
        s = trade_stats[sym]
        logger.info("{:6s}: {:2d} trades, {:.1f}% win, total={:+.2f}%, avg={:+.2f}%, best={:+.2f}%, worst={:+.2f}%",
                     sym, s["trades"], s["win_rate"], s["total_pnl_pct"], s["avg_pnl_pct"], s["best_trade_pct"], s["worst_trade_pct"])

    # 3. Leave-one-out analysis: remove each symbol, measure impact
    logger.info("\n=== Leave-One-Out Analysis ===")
    loo_results = {}
    traded_symbols = list(trade_stats.keys())
    for remove_sym in traded_symbols:
        subset = [s for s in symbols if s != remove_sym]
        result = run_backtest(model, feature_columns, subset, config, device, args)
        if result:
            ret = (result.equity_curve.iloc[-1] / args.initial_cash - 1) * 100
            sortino = result.metrics.get("sortino", 0)
            loo_results[remove_sym] = {"return": ret, "sortino": sortino, "trades": len(result.trades)}
            delta_ret = ret - full_return
            delta_sort = sortino - full_sortino
            logger.info("Without {:6s}: Return={:+.2f}% (delta {:+.2f}%), Sortino={:.2f} (delta {:+.2f})",
                         remove_sym, ret, delta_ret, sortino, delta_sort)

    # 4. Summary: rank symbols by contribution
    logger.info("\n=== Symbol Contribution Ranking (by Sortino impact) ===")
    contributions = []
    for sym in loo_results:
        delta_sortino = full_sortino - loo_results[sym]["sortino"]
        delta_return = full_return - loo_results[sym]["return"]
        contributions.append({"symbol": sym, "sortino_contribution": delta_sortino, "return_contribution": delta_return})
    contributions.sort(key=lambda x: x["sortino_contribution"], reverse=True)

    for c in contributions:
        tag = "KEEP" if c["sortino_contribution"] > 0 else "DROP?"
        logger.info("{:6s}: Sortino contribution={:+.2f}, Return contribution={:+.2f}% [{}]",
                     c["symbol"], c["sortino_contribution"], c["return_contribution"], tag)

    # Save results
    out = {
        "full": {"return": full_return, "sortino": full_sortino, "trades": len(full_result.trades)},
        "per_symbol_trades": trade_stats,
        "leave_one_out": loo_results,
        "contributions": contributions,
    }
    out_path = args.checkpoint_dir / "per_symbol_analysis.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info("Saved to {}", out_path)


if __name__ == "__main__":
    main()
