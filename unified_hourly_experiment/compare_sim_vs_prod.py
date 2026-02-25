#!/usr/bin/env python3
"""Compare portfolio simulator output vs production trades."""
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
from binanceneural.model import build_policy, policy_config_from_payload
from binanceneural.inference import generate_actions_from_frame
from unified_hourly_experiment.marketsimulator import (
    PortfolioConfig, run_portfolio_simulation,
)
from src.torch_load_utils import torch_load_compat

LONG_ONLY = {"NVDA", "MSFT", "META", "GOOG", "NET", "PLTR", "DBX", "TSLA", "AAPL"}
SHORT_ONLY = {"YELP", "EBAY", "TRIP", "MTCH", "ANGI", "Z", "EXPE", "BKNG", "NWSA", "NYT"}


def load_model(checkpoint_dir: Path, epoch: int):
    ckpt_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
    ckpt = torch_load_compat(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    config_path = checkpoint_dir / "config.json"
    meta_path = checkpoint_dir / "training_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            config = json.load(f)
    elif config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = ckpt.get("config", {})

    feature_columns = config.get("feature_columns", [])
    if not feature_columns:
        from binanceneural.data import build_default_feature_columns
        embed_w = state_dict.get("embed.weight")
        if embed_w is not None and embed_w.ndim == 2:
            for h_try in [[1], [1, 24]]:
                fc = build_default_feature_columns(h_try)
                if len(fc) == embed_w.shape[1]:
                    feature_columns = fc
                    break
        if not feature_columns:
            feature_columns = build_default_feature_columns([1])

    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    policy_cfg = policy_config_from_payload(config, input_dim=len(feature_columns), state_dict=state_dict)
    model = build_policy(policy_cfg)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, feature_columns, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("unified_hourly_experiment/checkpoints/top9_lag1_fnoise01"))
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--symbols", default="NVDA,PLTR,GOOG,NET,DBX")
    parser.add_argument("--days", type=int, default=5)
    parser.add_argument("--initial-cash", type=float, default=56068.0)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-hold-hours", type=int, default=6)
    parser.add_argument("--leverage", type=float, default=2.0)
    parser.add_argument("--data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--cache-root", type=Path, default=Path("unified_hourly_experiment/forecast_cache"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    model, feature_columns, config = load_model(args.checkpoint_dir, args.epoch)
    model = model.to(device)
    seq_len = config.get("sequence_length", 32)

    horizons = sorted({int(c.split("_h")[1]) for c in feature_columns
                       if "_h" in c and c.split("_h")[1].isdigit()}) or [1]

    data_modules = {}
    for symbol in symbols:
        dc = DatasetConfig(
            symbol=symbol, data_root=str(args.data_root),
            forecast_cache_root=str(args.cache_root),
            forecast_horizons=horizons, sequence_length=seq_len,
            min_history_hours=100, validation_days=30, cache_only=True,
        )
        try:
            data_modules[symbol] = BinanceHourlyDataModule(dc)
        except Exception as e:
            logger.warning("Skip {}: {}", symbol, e)

    normalizer = list(data_modules.values())[0].normalizer

    all_bars, all_actions = [], []
    for symbol, dm in data_modules.items():
        frame = dm.frame.copy()
        frame["symbol"] = symbol
        all_bars.append(frame)
        actions_df = generate_actions_from_frame(
            model=model, frame=frame, feature_columns=feature_columns,
            normalizer=normalizer, sequence_length=seq_len, horizon=1, device=device,
        )
        all_actions.append(actions_df)

    bars = pd.concat(all_bars, ignore_index=True)
    actions = pd.concat(all_actions, ignore_index=True)

    cutoff = bars["timestamp"].max() - pd.Timedelta(days=args.days)
    bars = bars[bars["timestamp"] >= cutoff].reset_index(drop=True)
    actions = actions[actions["timestamp"] >= cutoff].reset_index(drop=True)

    long_only = LONG_ONLY & set(symbols)
    short_only = SHORT_ONLY & set(symbols)

    cfg = PortfolioConfig(
        initial_cash=args.initial_cash,
        max_positions=args.max_positions,
        min_edge=0.0,
        max_hold_hours=args.max_hold_hours,
        max_leverage=args.leverage,
        decision_lag_bars=1,
        force_close_slippage=0.003,
        int_qty=True,
        symbols=symbols,
        fee_by_symbol={s: 0.001 for s in symbols},
        long_only_symbols=long_only,
        short_only_symbols=short_only,
    )

    result = run_portfolio_simulation(bars, actions, cfg, horizon=1)

    logger.info("=" * 75)
    logger.info("SIM TRADES (last {} days, {} symbols, pos={}, lev={}, lag=1)",
                args.days, len(symbols), args.max_positions, args.leverage)
    logger.info("=" * 75)
    print(f"{'Timestamp':25s} {'Sym':6s} {'Side':10s} {'Price':>10s} {'Qty':>8s} {'Reason':8s}")
    print("-" * 75)
    for t in result.trades:
        print(f"{str(t.timestamp):25s} {t.symbol:6s} {t.side:10s} {t.price:10.2f} {t.quantity:8.0f} {t.reason or '':8s}")

    m = result.metrics
    logger.info("-" * 75)
    logger.info("Return: {:+.2f}%  Sortino: {:.2f}  Final: ${:.0f}",
                m["total_return"] * 100, m["sortino"], m["final_equity"])
    logger.info("Entries: {}  Target exits: {}  Timeout exits: {}  EOD exits: {}",
                m["num_buys"], m["target_exits"], m["timeout_exits"], m["eod_exits"])

    # Show prod trade log if exists
    log_path = Path("strategy_state/stock_trade_log.jsonl")
    if log_path.exists():
        logger.info("\n=== PROD TRADES ===")
        for line in log_path.read_text().strip().split("\n"):
            if line.strip():
                entry = json.loads(line)
                ts = entry.get("logged_at", "")[:19]
                ev = entry.get("event", "?")
                sym = entry.get("symbol", "?")
                px = entry.get("price", 0)
                qty = entry.get("qty", 0)
                print(f"{ts}  {ev:12s} {sym:6s} qty={qty} px={px}")
    else:
        logger.info("No prod trade log found (will be created on next bot cycle)")


if __name__ == "__main__":
    main()
