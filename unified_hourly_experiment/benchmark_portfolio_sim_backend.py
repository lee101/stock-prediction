#!/usr/bin/env python3
"""Benchmark Python vs native backend for portfolio simulator on holdout data."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from unified_hourly_experiment.marketsimulator import PortfolioConfig, run_portfolio_simulation
from unified_hourly_experiment.run_stock_sortino_lag_robust import (
    build_holdout_bars_actions,
    infer_horizons,
    load_model_for_epoch,
    prepare_holdout_data,
)


def parse_symbols(raw: str) -> list[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def time_backend(
    *,
    bars,
    actions,
    cfg: PortfolioConfig,
    horizon: int,
    runs: int,
) -> tuple[float, dict]:
    durations = []
    metrics = {}
    for _ in range(runs):
        t0 = time.perf_counter()
        result = run_portfolio_simulation(bars, actions, cfg, horizon=horizon)
        durations.append(time.perf_counter() - t0)
        metrics = dict(result.metrics)
    return float(np.mean(durations)), metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark portfolio simulator backends.")
    parser.add_argument("--symbols", default="NVDA,PLTR,GOOG,DBX,TRIP,MTCH,NYT")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=9)
    parser.add_argument("--validation-days", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    symbols = parse_symbols(args.symbols)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading model {} epoch {}", args.checkpoint_dir, args.epoch)
    model, feature_columns, sequence_length = load_model_for_epoch(
        args.checkpoint_dir, args.epoch, device
    )
    horizons = infer_horizons(feature_columns, args.horizon)
    symbol_data = prepare_holdout_data(
        symbols,
        data_root=Path("trainingdatahourly/stocks"),
        cache_root=Path("unified_hourly_experiment/forecast_cache"),
        sequence_length=sequence_length,
        validation_days=args.validation_days,
        horizons=horizons,
    )
    bars, actions, used_symbols = build_holdout_bars_actions(
        model,
        feature_columns,
        sequence_length=sequence_length,
        horizon=args.horizon,
        symbol_data=symbol_data,
        device=device,
    )
    logger.info(
        "Prepared benchmark data: {} bars/actions rows, symbols={}",
        len(bars),
        used_symbols,
    )

    base_kwargs = dict(
        initial_cash=10_000.0,
        max_positions=7,
        min_edge=0.0,
        max_hold_hours=6,
        enforce_market_hours=True,
        close_at_eod=True,
        symbols=used_symbols,
        trade_amount_scale=100.0,
        decision_lag_bars=0,
        market_order_entry=False,
        bar_margin=0.002,
        entry_selection_mode="first_trigger",
    )

    python_cfg = PortfolioConfig(**base_kwargs, sim_backend="python")
    native_cfg = PortfolioConfig(**base_kwargs, sim_backend="native")

    logger.info("Warming up python backend")
    run_portfolio_simulation(bars, actions, python_cfg, horizon=args.horizon)
    logger.info("Warming up native backend (includes one-time compile when needed)")
    run_portfolio_simulation(bars, actions, native_cfg, horizon=args.horizon)

    py_mean, py_metrics = time_backend(
        bars=bars, actions=actions, cfg=python_cfg, horizon=args.horizon, runs=args.runs
    )
    native_mean, native_metrics = time_backend(
        bars=bars, actions=actions, cfg=native_cfg, horizon=args.horizon, runs=args.runs
    )

    speedup = py_mean / native_mean if native_mean > 0 else float("inf")
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbols": used_symbols,
        "checkpoint_dir": str(args.checkpoint_dir),
        "epoch": int(args.epoch),
        "validation_days": int(args.validation_days),
        "runs": int(args.runs),
        "python_mean_seconds": py_mean,
        "native_mean_seconds": native_mean,
        "speedup_native_vs_python": speedup,
        "python_metrics": py_metrics,
        "native_metrics": native_metrics,
    }

    logger.info(
        "Benchmark: python={:.4f}s native={:.4f}s speedup={:.2f}x",
        py_mean,
        native_mean,
        speedup,
    )
    output_path = args.output or Path("experiments") / "portfolio_sim_backend_benchmark_20260304.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved benchmark -> {}", output_path)


if __name__ == "__main__":
    main()
