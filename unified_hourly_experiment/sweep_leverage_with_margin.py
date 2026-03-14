#!/usr/bin/env python3
"""Sweep leverage levels (1x-4x) accounting for 6.25% annual margin cost.

Uses the current prod model configs to find optimal leverage.
Simulates EOD forced close at 2x (close_at_eod=True handles this).

Usage:
  python -m unified_hourly_experiment.sweep_leverage_with_margin \
    --checkpoint-dir unified_hourly_experiment/checkpoints/wd_0.06_s42 --epoch 8

  # With specific leverage range
  python -m unified_hourly_experiment.sweep_leverage_with_margin \
    --checkpoint-dir unified_hourly_experiment/checkpoints/wd_0.06_s42 --epoch 8 \
    --leverage-range 1.0,1.5,2.0,2.5,3.0,4.0
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

from binanceneural.config import DatasetConfig
from binanceneural.data import BinanceHourlyDataModule
from binanceneural.inference import generate_actions_from_frame
from unified_hourly_experiment.marketsimulator import (
    PortfolioConfig,
    run_portfolio_simulation,
)
from src.torch_load_utils import torch_load_compat
from unified_hourly_experiment.backtest_portfolio_leverage import load_model_and_actions


MARGIN_RATE = 0.0625  # 6.25% annual


def main():
    parser = argparse.ArgumentParser(description="Leverage sweep with margin cost")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--symbols", default="NVDA,PLTR,GOOG,DBX,TRIP,MTCH,AAPL")
    parser.add_argument("--data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--cache-root", type=Path, default=Path("alpacanewccrosslearning/forecast_cache/mega24_plus_yelp_novol_baseline_20260207_lb2400"))
    parser.add_argument("--initial-cash", type=float, default=46460)  # Current prod equity
    parser.add_argument("--min-edge", type=float, default=0.001)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-hold-hours", type=int, default=5)
    parser.add_argument("--leverage-range", default="1.0,1.5,2.0,2.5,3.0,3.5,4.0")
    parser.add_argument("--margin-rate", type=float, default=MARGIN_RATE)
    parser.add_argument("--holdout-days", type=int, default=30)
    parser.add_argument("--bar-margin", type=float, default=0.0005)
    parser.add_argument("--market-order-entry", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    leverages = [float(x) for x in args.leverage_range.split(",")]

    logger.info(f"Loading model: {args.checkpoint_dir} epoch {args.epoch}")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Leverage range: {leverages}")
    logger.info(f"Margin rate: {args.margin_rate*100:.2f}% annual")

    bars, actions = load_model_and_actions(
        args.checkpoint_dir, args.epoch, symbols,
        args.data_root, args.cache_root, device)

    results = []
    for lev in leverages:
        for margin in [0.0, args.margin_rate]:
            label = f"lev={lev:.1f}x margin={'yes' if margin > 0 else 'no'}"
            cfg = PortfolioConfig(
                initial_cash=args.initial_cash,
                max_positions=args.max_positions,
                min_edge=args.min_edge,
                max_hold_hours=args.max_hold_hours,
                enforce_market_hours=True,
                close_at_eod=True,
                symbols=symbols,
                max_leverage=lev,
                margin_annual_rate=margin,
                market_order_entry=args.market_order_entry,
                bar_margin=args.bar_margin,
                entry_selection_mode="edge_rank",
                fee_by_symbol={s: 0.001 for s in symbols},
                int_qty=True,
                force_close_slippage=0.003,
            )
            r = run_portfolio_simulation(bars, actions, cfg, horizon=1)
            m = r.metrics
            ret = m["total_return"] * 100
            sortino = m["sortino"]
            max_dd = m.get("max_drawdown", 0) * 100
            buys = m["num_buys"]
            sharpe = m.get("sharpe", 0)

            # Compute PnL smoothness: daily return std
            eq = np.array(r.equity_curve)
            if len(eq) > 1:
                daily_rets = np.diff(eq) / eq[:-1]
                vol = np.std(daily_rets) * np.sqrt(252) * 100  # Annualized vol %
            else:
                vol = 0

            logger.info(f"  {label}: ret={ret:+7.2f}% sort={sortino:6.2f} "
                        f"sharpe={sharpe:.2f} maxDD={max_dd:.1f}% vol={vol:.1f}% buys={buys}")
            results.append({
                "leverage": lev, "margin": margin > 0,
                "return_pct": round(ret, 3),
                "sortino": round(sortino, 3),
                "sharpe": round(sharpe, 3),
                "max_drawdown_pct": round(max_dd, 3),
                "annual_vol_pct": round(vol, 1),
                "num_buys": buys,
            })

    # Summary: best configs
    logger.info(f"\n{'='*70}")
    logger.info("=== Results with margin cost (6.25% annual) ===")
    margin_results = [r for r in results if r["margin"]]
    margin_results.sort(key=lambda x: x["sortino"], reverse=True)
    for r in margin_results:
        logger.info(f"  lev={r['leverage']:.1f}x: ret={r['return_pct']:+.2f}% "
                    f"sort={r['sortino']:.2f} sharpe={r['sharpe']:.2f} "
                    f"maxDD={r['max_drawdown_pct']:.1f}% vol={r['annual_vol_pct']:.1f}%")

    logger.info("\n=== Impact of margin cost at each leverage ===")
    for lev in leverages:
        no_margin = next((r for r in results if r["leverage"] == lev and not r["margin"]), None)
        with_margin = next((r for r in results if r["leverage"] == lev and r["margin"]), None)
        if no_margin and with_margin:
            drag = no_margin["return_pct"] - with_margin["return_pct"]
            logger.info(f"  lev={lev:.1f}x: no-margin={no_margin['return_pct']:+.2f}% "
                        f"with-margin={with_margin['return_pct']:+.2f}% "
                        f"margin-drag={drag:.2f}pp")

    out_path = Path("unified_hourly_experiment/leverage_sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
