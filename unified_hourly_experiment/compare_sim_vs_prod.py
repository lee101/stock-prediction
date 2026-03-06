#!/usr/bin/env python3
"""Compare portfolio simulator output vs actual Alpaca production fills."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import torch
from loguru import logger

from binanceneural.data import BinanceHourlyDataModule, FeatureNormalizer
from binanceneural.config import DatasetConfig
from binanceneural.model import build_policy, policy_config_from_payload
from binanceneural.inference import generate_actions_from_frame
from unified_hourly_experiment.marketsimulator import PortfolioConfig, run_portfolio_simulation
from unified_hourly_experiment.marketsimulator.unified_selector import _edge_score
from src.trade_directions import is_short_only_symbol
from src.torch_load_utils import torch_load_compat

CHECKPOINT_DIR = Path("unified_hourly_experiment/checkpoints/top9_lag1_6L_lr1e5_wd03_rw10_seq48")
EPOCH = 50
SYMBOLS = ["NVDA", "PLTR", "GOOG", "NET", "DBX", "TRIP", "EBAY", "MTCH", "NYT"]
DATA_ROOT = Path("trainingdatahourly/stocks")
CACHE_ROOT = Path("unified_hourly_experiment/forecast_cache")
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(CHECKPOINT_DIR / "config.json") as f:
        config = json.load(f)
    feature_columns = config["feature_columns"]
    seq_len = config.get("sequence_length", 48)

    ckpt = torch_load_compat(CHECKPOINT_DIR / f"epoch_{EPOCH:03d}.pt", map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    policy_cfg = policy_config_from_payload(config, input_dim=len(feature_columns), state_dict=state_dict)
    model = build_policy(policy_cfg)
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)

    normalizer = FeatureNormalizer.from_dict(config["normalizer"])
    horizons = sorted({int(c.split("_h")[1]) for c in feature_columns if "_h" in c and c.split("_h")[1].isdigit()})

    all_bars, all_actions = [], []
    for symbol in SYMBOLS:
        data_config = DatasetConfig(
            symbol=symbol, data_root=str(DATA_ROOT),
            forecast_cache_root=str(CACHE_ROOT),
            forecast_horizons=horizons, sequence_length=seq_len,
            min_history_hours=100, validation_days=30, cache_only=True,
        )
        dm = BinanceHourlyDataModule(data_config)
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

    # 1-day holdout
    cutoff = bars["timestamp"].max() - pd.Timedelta(days=1)
    cut_bars = bars[bars["timestamp"] >= cutoff].reset_index(drop=True)
    cut_acts = actions[actions["timestamp"] >= cutoff].reset_index(drop=True)

    logger.info("=== MODEL SIGNALS (last 1 day, BEFORE lag shift) ===")
    for ts in sorted(cut_acts["timestamp"].unique()):
        group = cut_acts[cut_acts["timestamp"] == ts]
        for _, row in group.iterrows():
            sym = row["symbol"]
            bp = row.get("buy_price", 0)
            sp = row.get("sell_price", 0)
            ba = row.get("buy_amount", 0)
            intensity = min(ba / 100.0, 1.0)
            is_short = is_short_only_symbol(sym)
            fee = 0.001
            if is_short:
                edge = (sp - row.get("predicted_low_p50_h1", 0)) / sp - fee if sp > 0 else 0
            else:
                pred_h = row.get("predicted_high_p50_h1", 0)
                pred_l = row.get("predicted_low_p50_h1", 0)
                pred_c = row.get("predicted_close_p50_h1", 0)
                edge = _edge_score(pred_h, pred_l, pred_c, bp, is_long=True, edge_mode="high_low", fee_rate=fee) if bp > 0 else 0
            if edge and edge >= 0.0 and intensity > 0.001:
                side = "short" if is_short else "long"
                logger.info("  {} {:5s} {:5s} buy={:.2f} sell={:.2f} edge={:.4f} int={:.3f}",
                           str(ts)[:19], sym, side, bp, sp, edge or 0, intensity)

    # Show bar data for key symbols at critical times
    logger.info("\n=== BAR DATA (cached CSV) ===")
    for sym in SYMBOLS:
        sb = cut_bars[cut_bars["symbol"] == sym].sort_values("timestamp")
        logger.info("  {} ({} bars):", sym, len(sb))
        for _, r in sb.iterrows():
            logger.info("    {} O={:.2f} H={:.2f} L={:.2f} C={:.2f} V={}",
                       str(r["timestamp"])[:19], r["open"], r["high"], r["low"], r["close"], int(r["volume"]))

    # Run simulator
    logger.info("\n=== SIMULATOR (lag=1, bar_margin=0.0005, lev=2.0, max_pos=9, int_qty=True) ===")
    cfg = PortfolioConfig(
        initial_cash=56068.0,
        max_positions=9,
        min_edge=0.0,
        max_hold_hours=6,
        enforce_market_hours=True,
        close_at_eod=True,
        symbols=SYMBOLS,
        decision_lag_bars=1,
        bar_margin=0.0005,
        max_leverage=2.0,
        force_close_slippage=0.003,
        int_qty=True,
        fee_by_symbol={s: 0.001 for s in SYMBOLS},
    )
    result = run_portfolio_simulation(cut_bars, cut_acts, cfg, horizon=1)

    print(f"\n{'Timestamp':25s} {'Sym':6s} {'Side':10s} {'Price':>10s} {'Qty':>8s} {'Reason':8s}")
    print("-" * 75)
    for t in result.trades:
        print(f"{str(t.timestamp):25s} {t.symbol:6s} {t.side:10s} {t.price:10.2f} {t.quantity:8.0f} {t.reason or '':8s}")

    m = result.metrics
    logger.info("Return: {:+.4f}%  Sortino: {:.2f}  Final: ${:.0f}",
                m["total_return"] * 100, m["sortino"], m["final_equity"])
    logger.info("Entries: {}  Target exits: {}  Timeout exits: {}  EOD exits: {}",
                m["num_buys"], m["target_exits"], m["timeout_exits"], m["eod_exits"])

    # Actual Alpaca fills
    logger.info("\n=== ACTUAL ALPACA ORDERS (Feb 25 session) ===")
    alpaca = [
        ("19:01 UTC", "TRIP", "SELL 13", "$10.18 lim", "FILLED @$10.1833", "old model"),
        ("19:52 UTC", "TRIP", "BUY 13", "$10.11 lim", "FILLED @$10.11", "GTC exit"),
        ("19:52 UTC", "DBX", "BUY 505", "$24.42 lim", "CANCELED", "DAY, never filled"),
        ("19:52 UTC", "MTCH", "SELL 398", "$31.23 lim", "FILLED @$31.24", "short entry"),
        ("19:52 UTC", "PLTR", "BUY 92", "$134.33 lim", "CANCELED", "DAY, never filled"),
        ("19:52 UTC", "EBAY", "SELL 146", "$84.79 lim", "CANCELED", "DAY, never filled"),
        ("19:52 UTC", "GOOG", "BUY 40", "$310.18 lim", "CANCELED", "DAY, never filled"),
        ("20:01 UTC", "MTCH", "BUY 398", "$31.19 lim", "PENDING GTC", "exit order"),
        ("20:01 UTC", "DBX", "BUY 505", "$24.42 lim", "FILLED @$24.42", "retry"),
        ("20:01 UTC", "PLTR", "BUY 92", "$134.33 lim", "FILLED @$134.33", "retry"),
        ("20:01 UTC", "EBAY", "SELL 146", "$84.79 lim", "FILLED @$84.79", "retry"),
        ("20:01 UTC", "GOOG", "BUY 40", "$310.18 lim", "PENDING DAY", "may not fill"),
    ]
    for row in alpaca:
        logger.info("  {} {:5s} {:10s} {:12s} {:20s} {}", *row)


if __name__ == "__main__":
    main()
