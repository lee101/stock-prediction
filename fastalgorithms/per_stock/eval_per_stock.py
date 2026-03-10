#!/usr/bin/env python3
"""Evaluate per-stock models and output per-bar equity curves.

Usage:
    # Evaluate a single checkpoint dir
    python -u -m fastalgorithms.per_stock.eval_per_stock \
        --checkpoint-dir fastalgorithms/per_stock/checkpoints/NVDA_rw015_wd006_s42 \
        --holdout-days 30

    # Evaluate all per-stock checkpoints and select best per symbol
    python -u -m fastalgorithms.per_stock.eval_per_stock --select-best --holdout-days 30

    # Evaluate best and output equity curves for meta-selector
    python -u -m fastalgorithms.per_stock.eval_per_stock --select-best --output-equity --holdout-days 90
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
from loguru import logger

from binanceneural.config import DatasetConfig, PolicyConfig
from binanceneural.data import BinanceHourlyDataModule, FeatureNormalizer
from binanceneural.inference import generate_actions_from_frame
from binanceneural.model import build_policy
from unified_hourly_experiment.marketsimulator import PortfolioConfig, run_portfolio_simulation
from src.torch_load_utils import torch_load_compat


def evaluate_checkpoint(
    checkpoint_path: Path,
    config: dict,
    symbol: str,
    holdout_days: int = 30,
    device: str = "cuda",
    data_root: Path = Path("trainingdatahourly/stocks"),
    cache_root: Path = Path("unified_hourly_experiment/forecast_cache"),
    fee_rate: float = 0.001,
    margin_rate: float = 0.0625,
    max_leverage: float = 2.0,
    max_hold_hours: int = 6,
    min_edge: float = 0.0,
) -> Tuple[dict, pd.DataFrame]:
    """Evaluate a single checkpoint for a single symbol.

    Returns:
        (metrics_dict, equity_df) where equity_df has columns [timestamp, equity, in_position]
    """
    feature_columns = config.get("feature_columns", [])
    horizons = sorted({int(c.split("_h")[1]) for c in feature_columns
                       if "_h" in c and c.split("_h")[1].isdigit()}) or [1]

    # Load data for single symbol
    data_config = DatasetConfig(
        symbol=symbol,
        data_root=str(data_root),
        forecast_cache_root=str(cache_root),
        forecast_horizons=horizons,
        sequence_length=config.get("sequence_length", 48),
        min_history_hours=100,
        validation_days=30,
        cache_only=True,
    )
    dm = BinanceHourlyDataModule(data_config)

    # Load normalizer
    if "normalizer" in config:
        normalizer = FeatureNormalizer.from_dict(config["normalizer"])
    else:
        normalizer = dm.normalizer

    # Load model
    ckpt = torch_load_compat(checkpoint_path, map_location="cpu", weights_only=False)
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
        model_arch=config.get("model_arch", "classic"),
        max_len=max_len,
    )
    model = build_policy(policy_cfg)
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)

    # Generate actions
    frame = dm.frame.copy()
    frame["symbol"] = symbol
    actions_df = generate_actions_from_frame(
        model=model, frame=frame, feature_columns=feature_columns,
        normalizer=normalizer, sequence_length=config.get("sequence_length", 48),
        horizon=1, device=device,
    )

    bars = frame.copy()

    # Apply holdout filter
    if holdout_days > 0:
        cutoff = bars["timestamp"].max() - pd.Timedelta(days=holdout_days)
        bars = bars[bars["timestamp"] >= cutoff].reset_index(drop=True)
        actions_df = actions_df[actions_df["timestamp"] >= cutoff].reset_index(drop=True)

    # Run portfolio sim with max_positions=1 (single stock)
    sim_config = PortfolioConfig(
        initial_cash=10_000.0,
        max_positions=1,
        min_edge=min_edge,
        max_hold_hours=max_hold_hours,
        enforce_market_hours=True,
        close_at_eod=True,
        symbols=[symbol],
        decision_lag_bars=1,
        bar_margin=0.0005,
        max_leverage=max_leverage,
        force_close_slippage=0.003,
        int_qty=True,
        fee_by_symbol={symbol: fee_rate},
        margin_annual_rate=margin_rate,
    )
    result = run_portfolio_simulation(bars, actions_df, sim_config, horizon=1)

    # Build per-bar equity DataFrame with in_position
    eq_curve = result.equity_curve
    trades = result.trades

    # Determine in_position at each timestamp
    # Track open positions by processing trades chronologically
    position_periods = []
    for t in trades:
        if t.side in ("buy", "short_sell"):
            position_periods.append({"open": t.timestamp, "close": None})
        elif t.side in ("sell", "buy_cover") and position_periods:
            for pp in reversed(position_periods):
                if pp["close"] is None:
                    pp["close"] = t.timestamp
                    break

    # Build DataFrame
    timestamps = eq_curve.index
    equity_vals = eq_curve.values
    in_position = np.zeros(len(timestamps), dtype=bool)

    for pp in position_periods:
        open_ts = pp["open"]
        close_ts = pp["close"] or timestamps[-1]
        mask = (timestamps >= open_ts) & (timestamps <= close_ts)
        in_position[mask] = True

    equity_df = pd.DataFrame({
        "timestamp": timestamps,
        "equity": equity_vals,
        "in_position": in_position,
    })

    metrics = result.metrics.copy()
    metrics["symbol"] = symbol
    metrics["checkpoint"] = str(checkpoint_path)

    # Clean up GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics, equity_df


def evaluate_all_epochs(
    checkpoint_dir: Path,
    holdout_days: int = 30,
    **kwargs,
) -> list[dict]:
    """Evaluate all epoch checkpoints in a directory."""
    with open(checkpoint_dir / "config.json") as f:
        config = json.load(f)

    symbol = config.get("symbol", config.get("stock_symbols", ["UNKNOWN"])[0])
    checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"),
                         key=lambda p: int(p.stem.split("_")[1]))

    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for ckpt_path in checkpoints:
        epoch = int(ckpt_path.stem.split("_")[1])
        try:
            metrics, _ = evaluate_checkpoint(
                ckpt_path, config, symbol,
                holdout_days=holdout_days, device=device, **kwargs,
            )
            metrics["epoch"] = epoch
            results.append(metrics)
            logger.info("  Epoch {:3d}: ret={:+7.2f}% sort={:6.2f} dd={:.1f}% buys={}",
                        epoch, metrics["total_return"] * 100, metrics["sortino"],
                        metrics.get("max_drawdown", 0) * 100, metrics["num_buys"])
        except Exception as e:
            logger.warning("  Epoch {:3d}: FAILED - {}", epoch, e)

    return results


def select_best_per_stock(
    checkpoint_root: Path = Path("fastalgorithms/per_stock/checkpoints"),
    symbols: list[str] = None,
    holdout_days: int = 30,
    selection_metric: str = "sortino",
    **eval_kwargs,
) -> Dict[str, dict]:
    """Scan all per-stock checkpoints and select the best per symbol.

    Returns:
        {symbol: {"checkpoint_dir": Path, "best_epoch": int, "metrics": dict, "equity_df": DataFrame}}
    """
    if symbols is None:
        symbols = ["NVDA", "PLTR", "GOOG", "DBX", "TRIP", "MTCH"]

    best_per_symbol: Dict[str, dict] = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for ckpt_dir in sorted(checkpoint_root.iterdir()):
        if not ckpt_dir.is_dir() or not (ckpt_dir / "config.json").exists():
            continue

        with open(ckpt_dir / "config.json") as f:
            config = json.load(f)

        symbol = config.get("symbol", config.get("stock_symbols", [None])[0])
        if symbol not in symbols:
            continue

        logger.info("Evaluating {} ({})", ckpt_dir.name, symbol)
        results = evaluate_all_epochs(ckpt_dir, holdout_days=holdout_days, **eval_kwargs)

        if not results:
            continue

        best = max(results, key=lambda r: r.get(selection_metric, float("-inf")))

        current_best = best_per_symbol.get(symbol)
        if current_best is None or best.get(selection_metric, float("-inf")) > current_best["metrics"].get(selection_metric, float("-inf")):
            # Re-evaluate best epoch to get equity curve
            best_epoch = best["epoch"]
            ckpt_path = ckpt_dir / f"epoch_{best_epoch:03d}.pt"
            metrics, equity_df = evaluate_checkpoint(
                ckpt_path, config, symbol,
                holdout_days=holdout_days, device=device, **eval_kwargs,
            )
            best_per_symbol[symbol] = {
                "checkpoint_dir": ckpt_dir,
                "best_epoch": best_epoch,
                "metrics": metrics,
                "equity_df": equity_df,
                "config": config,
            }
            logger.info("  -> NEW BEST for {}: epoch {} sort={:.2f} ret={:+.2f}%",
                        symbol, best_epoch, metrics["sortino"], metrics["total_return"] * 100)

    return best_per_symbol


def main():
    parser = argparse.ArgumentParser(description="Per-stock model evaluation")
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="Evaluate a single checkpoint directory")
    parser.add_argument("--checkpoint-root", type=Path,
                        default=Path("fastalgorithms/per_stock/checkpoints"))
    parser.add_argument("--symbols", type=str, default="NVDA,PLTR,GOOG,DBX,TRIP,MTCH")
    parser.add_argument("--holdout-days", type=int, default=30)
    parser.add_argument("--select-best", action="store_true",
                        help="Scan all checkpoints and select best per symbol")
    parser.add_argument("--output-equity", action="store_true",
                        help="Save equity curves for meta-selector")
    parser.add_argument("--min-edge", type=float, default=0.0)
    parser.add_argument("--max-hold-hours", type=int, default=6)
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    if args.checkpoint_dir:
        # Evaluate single checkpoint dir
        logger.info("Evaluating {}", args.checkpoint_dir)
        results = evaluate_all_epochs(
            args.checkpoint_dir, holdout_days=args.holdout_days,
            min_edge=args.min_edge, max_hold_hours=args.max_hold_hours,
        )
        if results:
            best = max(results, key=lambda r: r["sortino"])
            logger.info("\nBest: epoch {} sort={:.2f} ret={:+.2f}%",
                        best["epoch"], best["sortino"], best["total_return"] * 100)

    elif args.select_best:
        # Scan all and select best per symbol
        best = select_best_per_stock(
            checkpoint_root=args.checkpoint_root,
            symbols=symbols,
            holdout_days=args.holdout_days,
            min_edge=args.min_edge,
            max_hold_hours=args.max_hold_hours,
        )

        logger.info("\n=== Best Per-Stock Models ===")
        for sym, info in sorted(best.items()):
            m = info["metrics"]
            logger.info("{:6s}: epoch {:3d} sort={:6.2f} ret={:+7.2f}% dd={:.1f}% buys={} ({})",
                        sym, info["best_epoch"], m["sortino"], m["total_return"] * 100,
                        m.get("max_drawdown", 0) * 100, m["num_buys"],
                        info["checkpoint_dir"].name)

        if args.output_equity:
            # Save equity curves as parquet for meta-selector
            output_dir = args.checkpoint_root / "equity_curves"
            output_dir.mkdir(exist_ok=True)
            for sym, info in best.items():
                out_path = output_dir / f"{sym}_{args.holdout_days}d.parquet"
                info["equity_df"].to_parquet(out_path)
                logger.info("Saved {} equity curve to {}", sym, out_path)

            # Save summary
            summary = {sym: {
                "checkpoint_dir": str(info["checkpoint_dir"]),
                "best_epoch": info["best_epoch"],
                "sortino": info["metrics"]["sortino"],
                "total_return": info["metrics"]["total_return"],
                "max_drawdown": info["metrics"].get("max_drawdown", 0),
                "num_buys": info["metrics"]["num_buys"],
            } for sym, info in best.items()}
            with open(output_dir / f"best_models_{args.holdout_days}d.json", "w") as f:
                json.dump(summary, f, indent=2)
            logger.info("Summary saved to {}", output_dir / f"best_models_{args.holdout_days}d.json")


if __name__ == "__main__":
    main()
