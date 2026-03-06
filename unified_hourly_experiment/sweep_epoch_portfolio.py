#!/usr/bin/env python3
"""Sweep checkpoint epochs using portfolio mode (multi-position).

Supports multi-period evaluation with --holdout-days 1,3,7,30,60,120
Computes smoothness score (harmonic mean of Sortino across periods).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from loguru import logger

from binanceneural.data import BinanceHourlyDataModule
from binanceneural.config import DatasetConfig, PolicyConfig
from binanceneural.model import build_policy
from binanceneural.inference import generate_actions_from_frame
from src.trade_directions import DEFAULT_ALPACA_LIVE8_STOCKS
from unified_hourly_experiment.marketsimulator import (
    PortfolioConfig, run_portfolio_simulation,
)
from src.torch_load_utils import torch_load_compat


def harmonic_mean(values: list[float]) -> float:
    positive = [v for v in values if v > 0]
    if not positive or len(positive) < len(values):
        return 0.0
    return len(positive) / sum(1.0 / v for v in positive)


def parse_holdout_days(s: str) -> list[int]:
    if not s or s == "0":
        return [0]
    return sorted(int(x.strip()) for x in s.split(",") if x.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--symbols", default=",".join(DEFAULT_ALPACA_LIVE8_STOCKS))
    parser.add_argument("--data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--cache-root", type=Path, default=Path("unified_hourly_experiment/forecast_cache"))
    parser.add_argument("--initial-cash", type=float, default=10000)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-hold-hours", type=int, default=6)
    parser.add_argument("--min-edge", type=float, default=0.0)
    parser.add_argument("--decision-lag-bars", type=int, default=1)
    parser.add_argument("--market-order-entry", action="store_true")
    parser.add_argument("--bar-margin", type=float, default=0.0005)
    parser.add_argument("--leverage", type=float, default=2.0)
    parser.add_argument("--force-close-slippage", type=float, default=0.003)
    parser.add_argument("--no-int-qty", action="store_true")
    parser.add_argument("--margin-rate", type=float, default=0.0625)
    parser.add_argument("--no-close-at-eod", action="store_true")
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--holdout-days", type=str, default="0",
                        help="Comma-separated holdout periods, e.g. 1,3,7,30,60,120. 0=all data.")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Run single epoch instead of sweep")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    holdout_periods = parse_holdout_days(args.holdout_days)
    multi_period = len(holdout_periods) > 1 or holdout_periods != [0]

    with open(args.checkpoint_dir / "config.json") as f:
        config = json.load(f)

    feature_columns = config.get("feature_columns", [])
    horizons = sorted({int(c.split("_h")[1]) for c in feature_columns
                       if "_h" in c and c.split("_h")[1].isdigit()}) or [1, 24]

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

    if "normalizer" in config:
        from binanceneural.data import FeatureNormalizer
        normalizer = FeatureNormalizer.from_dict(config["normalizer"])
    else:
        normalizer = list(data_modules.values())[0].normalizer

    checkpoints = sorted(args.checkpoint_dir.glob("epoch_*.pt"),
                         key=lambda p: int(p.stem.split("_")[1]))
    if args.epoch is not None:
        checkpoints = [c for c in checkpoints if int(c.stem.split("_")[1]) == args.epoch]

    periods_str = ",".join(str(d) for d in holdout_periods) if multi_period else "all"
    logger.info("Sweeping {} epochs x {} symbols x {} periods ({})",
                len(checkpoints), len(data_modules), len(holdout_periods), periods_str)

    all_results = []
    epoch_summaries = []

    for ckpt_path in checkpoints:
        epoch = int(ckpt_path.stem.split("_")[1])
        ckpt = torch_load_compat(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)

        pe_key = "pos_encoding.pe"
        max_len = config.get("sequence_length", 32)
        if pe_key in state_dict:
            max_len = max(max_len, state_dict[pe_key].shape[0])
        policy_cfg = PolicyConfig(
            input_dim=len(feature_columns),
            hidden_dim=config.get("transformer_dim", 128),
            num_heads=config.get("transformer_heads", 4),
            num_layers=config.get("transformer_layers", 3),
            num_outputs=config.get("num_outputs", 4),
            model_arch=config.get("model_arch", "gemma"),
            max_len=max_len,
        )
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
            actions_df = generate_actions_from_frame(
                model=model, frame=frame, feature_columns=feature_columns,
                normalizer=normalizer, sequence_length=config.get("sequence_length", 32),
                horizon=1, device=device,
            )
            all_actions.append(actions_df)

        bars_full = pd.concat(all_bars, ignore_index=True)
        actions_full = pd.concat(all_actions, ignore_index=True)

        epoch_period_results = {}
        for holdout in holdout_periods:
            if holdout > 0:
                cutoff = bars_full["timestamp"].max() - pd.Timedelta(days=holdout)
                bars = bars_full[bars_full["timestamp"] >= cutoff].reset_index(drop=True)
                actions = actions_full[actions_full["timestamp"] >= cutoff].reset_index(drop=True)
            else:
                bars = bars_full
                actions = actions_full

            cfg = PortfolioConfig(
                initial_cash=args.initial_cash, max_positions=args.max_positions,
                min_edge=args.min_edge, max_hold_hours=args.max_hold_hours,
                enforce_market_hours=True, close_at_eod=not args.no_close_at_eod,
                symbols=symbols,
                decision_lag_bars=args.decision_lag_bars,
                market_order_entry=args.market_order_entry,
                bar_margin=args.bar_margin,
                max_leverage=args.leverage,
                force_close_slippage=args.force_close_slippage,
                int_qty=not args.no_int_qty,
                fee_by_symbol={s: args.fee_rate for s in symbols},
                margin_annual_rate=args.margin_rate,
            )
            r = run_portfolio_simulation(bars, actions, cfg, horizon=1)
            m = r.metrics
            ret = m["total_return"] * 100
            sort_val = m["sortino"]
            dd = m.get("max_drawdown", 0) * 100
            buys = m.get("num_buys", 0)
            targets = m.get("target_exits", 0)
            wr = (targets / buys * 100) if buys > 0 else 0.0

            period_label = f"{holdout}d" if holdout > 0 else "all"
            row = {
                "epoch": epoch, "period": period_label, "holdout_days": holdout,
                "return": round(ret, 2), "sortino": round(sort_val, 2),
                "max_drawdown": round(dd, 2), "win_rate": round(wr, 1),
                "buys": buys, "target_exits": targets,
            }
            all_results.append(row)
            epoch_period_results[holdout] = row

            if multi_period:
                logger.info("  ep{:3d} {:>4s}: ret={:+7.2f}% sort={:6.2f} dd={:.1f}% wr={:.0f}% buys={}",
                            epoch, period_label, ret, sort_val, dd, wr, buys)
            else:
                logger.info("Epoch {:3d}: ret={:+7.2f}% sort={:6.2f} dd={:.1f}% wr={:.0f}% buys={}",
                            epoch, ret, sort_val, dd, wr, buys)

        if multi_period:
            sortinos = [r["sortino"] for r in epoch_period_results.values()]
            returns = [r["return"] for r in epoch_period_results.values()]
            all_positive = all(r > 0 for r in returns)
            smoothness = harmonic_mean(sortinos)
            avg_sort = float(np.mean(sortinos))
            avg_ret = float(np.mean(returns))
            worst_ret = min(returns)
            worst_period = [r["period"] for r in epoch_period_results.values()
                           if r["return"] == worst_ret][0]
            summary = {
                "epoch": epoch, "smoothness": round(smoothness, 2),
                "avg_sortino": round(avg_sort, 2), "avg_return": round(avg_ret, 2),
                "worst_return": round(worst_ret, 2), "worst_period": worst_period,
                "all_positive": all_positive,
                "periods": {r["period"]: {"ret": r["return"], "sort": r["sortino"]}
                            for r in epoch_period_results.values()},
            }
            epoch_summaries.append(summary)
            status = "PASS" if all_positive else "FAIL"
            logger.info("  ep{:3d} SUMMARY: smooth={:.2f} avg_sort={:.2f} avg_ret={:+.2f}% worst={:+.2f}%({}) [{}]",
                        epoch, smoothness, avg_sort, avg_ret, worst_ret, worst_period, status)

    logger.info("=" * 70)

    if multi_period and epoch_summaries:
        qualified = [s for s in epoch_summaries if s["all_positive"]]
        if qualified:
            best = max(qualified, key=lambda x: x["smoothness"])
            logger.info("BEST QUALIFIED (all periods positive):")
            logger.info("  Epoch {} smooth={:.2f} avg_sort={:.2f} avg_ret={:+.2f}%",
                        best["epoch"], best["smoothness"], best["avg_sortino"], best["avg_return"])
            for p, v in best["periods"].items():
                logger.info("    {}: ret={:+.2f}% sort={:.2f}", p, v["ret"], v["sort"])
        else:
            logger.warning("NO epoch qualified (all periods positive)")
            best = max(epoch_summaries, key=lambda x: x["smoothness"])
            logger.info("Best unqualified: Epoch {} smooth={:.2f} worst={:+.2f}%({})",
                        best["epoch"], best["smoothness"], best["worst_return"], best["worst_period"])
    elif all_results:
        best_sort = max(all_results, key=lambda x: x["sortino"])
        best_ret = max(all_results, key=lambda x: x["return"])
        logger.info("Best Sortino: Epoch {} ({:.2f}%, sort={:.2f})",
                     best_sort["epoch"], best_sort["return"], best_sort["sortino"])
        logger.info("Best Return:  Epoch {} ({:.2f}%, sort={:.2f})",
                     best_ret["epoch"], best_ret["return"], best_ret["sortino"])

    out = {"results": all_results}
    if epoch_summaries:
        out["summaries"] = epoch_summaries
    out_path = args.checkpoint_dir / "epoch_sweep_portfolio.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Saved {} results to {}", len(all_results), out_path)


if __name__ == "__main__":
    main()
