#!/usr/bin/env python3
"""SUI margin/leverage trading experiment.

Trains and evaluates policies with leverage (2x-5x) and optional shorting.
Includes margin interest cost in simulation.
Uses randomized holdout windows for robustness testing.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from binanceneural.config import TrainingConfig
from binanceneural.trainer import BinanceHourlyTrainer
from binanceneural.inference import generate_actions_from_frame
from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import load_policy_checkpoint
from differentiable_loss_utils import (
    simulate_hourly_trades_binary,
    compute_hourly_objective,
    HOURLY_PERIODS_PER_YEAR,
)

SUI_HOURLY_MARGIN_RATE = 0.0000025457  # ~2.23% annual
MAKER_FEE_10BP = 0.001
DEFAULT_MODEL_ID = "chronos2_finetuned/SOLUSDT_lora_20260202_030749/finetuned-ckpt"


@dataclass
class LeverageConfig:
    symbol: str = "SUIUSDT"
    max_leverage: float = 2.0
    can_short: bool = False
    maker_fee: float = MAKER_FEE_10BP
    margin_hourly_rate: float = SUI_HOURLY_MARGIN_RATE
    return_weight: float = 0.012
    learning_rate: float = 1e-4
    epochs: int = 25
    sequence_length: int = 72
    seed: int = 1337
    initial_cash: float = 10_000.0
    val_days: int = 20
    test_days: int = 10
    horizons: str = "1,4,24"
    batch_size: int = 16
    transformer_dim: int = 256
    transformer_layers: int = 4
    transformer_heads: int = 8


def simulate_with_margin_cost(
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    config: LeverageConfig,
) -> dict:
    """Backtest with leverage + margin interest deductions."""
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    actions = actions.sort_values("timestamp").reset_index(drop=True)

    merged = bars.merge(actions, on=["timestamp", "symbol"], how="inner", suffixes=("", "_act"))

    cash = config.initial_cash
    inventory = 0.0
    equity_curve = [cash]
    trades = []
    margin_cost_total = 0.0

    for _, row in merged.iterrows():
        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        buy_price = float(row.get("buy_price", 0) or 0)
        sell_price = float(row.get("sell_price", 0) or 0)
        buy_amount = float(row.get("buy_amount", 0) or 0) / 100.0
        sell_amount = float(row.get("sell_amount", 0) or 0) / 100.0

        equity = cash + inventory * close

        # Margin interest: charged on borrowed amount each hour
        # If cash < 0, we borrowed cash to buy on margin
        if cash < 0:
            interest = abs(cash) * config.margin_hourly_rate
            cash -= interest
            margin_cost_total += interest
        # Short position: borrowed the asset
        if inventory < 0:
            borrowed_value = abs(inventory) * close
            interest = borrowed_value * config.margin_hourly_rate
            cash -= interest
            margin_cost_total += interest

        # Buy execution (can go beyond cash with leverage - cash goes negative = borrowed)
        if buy_amount > 0 and buy_price > 0 and low <= buy_price:
            max_buy_value = config.max_leverage * max(equity, 0) - inventory * buy_price
            if max_buy_value > 0:
                buy_qty = buy_amount * max_buy_value / (buy_price * (1 + config.maker_fee))
                if buy_qty > 0:
                    cost = buy_qty * buy_price * (1 + config.maker_fee)
                    cash -= cost
                    inventory += buy_qty
                    trades.append(("buy", float(row["timestamp"].timestamp()) if hasattr(row["timestamp"], "timestamp") else 0, buy_price, buy_qty))

        # Sell execution
        if sell_amount > 0 and sell_price > 0 and high >= sell_price:
            if inventory > 0:
                sell_qty = min(sell_amount * inventory, inventory)
            elif config.can_short:
                max_short_value = config.max_leverage * max(equity, 0)
                sell_qty = min(sell_amount * max_short_value / (sell_price * (1 + config.maker_fee)),
                               max_short_value / (sell_price * (1 + config.maker_fee)))
            else:
                sell_qty = 0
            sell_qty = max(sell_qty, 0)
            if sell_qty > 0:
                proceeds = sell_qty * sell_price * (1 - config.maker_fee)
                cash += proceeds
                inventory -= sell_qty
                trades.append(("sell", 0, sell_price, sell_qty))

        equity_curve.append(cash + inventory * close)

    # Close any remaining position at last close
    if len(merged) > 0 and inventory != 0:
        last_close = float(merged.iloc[-1]["close"])
        if inventory > 0:
            cash += inventory * last_close * (1 - config.maker_fee)
        else:
            cash -= abs(inventory) * last_close * (1 + config.maker_fee)
        inventory = 0
        equity_curve[-1] = cash

    eq = np.array(equity_curve)
    ret = np.diff(eq) / (np.abs(eq[:-1]) + 1e-10)
    neg = ret[ret < 0]
    sortino = (np.mean(ret) / (np.std(neg) + 1e-10)) * np.sqrt(8760) if len(neg) > 0 else 0
    running_max = np.maximum.accumulate(eq)
    max_dd = float(np.min((eq - running_max) / (running_max + 1e-10)))

    return {
        "total_return": (eq[-1] / eq[0]) - 1 if eq[0] > 0 else 0,
        "sortino": float(sortino),
        "max_drawdown": max_dd,
        "final_equity": float(eq[-1]),
        "num_trades": len(trades),
        "margin_cost_total": margin_cost_total,
        "margin_cost_pct": margin_cost_total / config.initial_cash * 100,
    }


def train_and_evaluate(cfg: LeverageConfig, run_name: str) -> dict:
    horizons = tuple(int(h) for h in cfg.horizons.split(","))
    forecast_cache = Path("binancechronossolexperiment/forecast_cache_sui_10bp")

    dm = ChronosSolDataModule(
        symbol=cfg.symbol,
        data_root=Path("trainingdatahourlybinance"),
        forecast_cache_root=forecast_cache,
        forecast_horizons=horizons,
        context_hours=512,
        quantile_levels=(0.1, 0.5, 0.9),
        batch_size=32,
        model_id="amazon/chronos-t5-small",
        sequence_length=cfg.sequence_length,
        split_config=SplitConfig(val_days=cfg.val_days, test_days=cfg.test_days),
        cache_only=True,
    )

    checkpoint_root = Path("binanceleveragesui/checkpoints") / run_name
    tc = TrainingConfig(
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        sequence_length=cfg.sequence_length,
        learning_rate=cfg.learning_rate,
        weight_decay=1e-4,
        return_weight=cfg.return_weight,
        seed=cfg.seed,
        transformer_dim=cfg.transformer_dim,
        transformer_layers=cfg.transformer_layers,
        transformer_heads=cfg.transformer_heads,
        maker_fee=cfg.maker_fee,
        checkpoint_root=checkpoint_root,
        log_dir=Path("tensorboard_logs/binanceleveragesui"),
        use_compile=False,
    )

    trainer = BinanceHourlyTrainer(tc, dm)
    artifacts = trainer.train()

    # Package checkpoint
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    sd = artifacts.state_dict
    if artifacts.best_checkpoint and artifacts.best_checkpoint.exists():
        ckpt = torch.load(artifacts.best_checkpoint, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
    ckpt_path = checkpoint_root / "policy_checkpoint.pt"
    torch.save({
        "state_dict": sd,
        "config": asdict(tc),
        "feature_columns": list(artifacts.feature_columns),
        "normalizer": artifacts.normalizer.to_dict(),
    }, ckpt_path)

    # Generate actions on test set
    model, normalizer, feature_columns, _ = load_policy_checkpoint(str(ckpt_path))
    test_frame = dm.test_frame.copy()
    actions = generate_actions_from_frame(
        model=model, frame=test_frame, feature_columns=feature_columns,
        normalizer=normalizer, sequence_length=cfg.sequence_length, horizon=horizons[0],
    )

    test_start = dm.test_window_start
    bars = test_frame[test_frame["timestamp"] >= test_start].copy()
    actions = actions[actions["timestamp"] >= test_start].copy()

    # Evaluate at different leverage levels
    results = {"config": asdict(cfg), "run_name": run_name}
    for lev in [1.0, 2.0, 3.0, 4.0, 5.0]:
        eval_cfg = LeverageConfig(**{**asdict(cfg), "max_leverage": lev})
        metrics = simulate_with_margin_cost(bars, actions, eval_cfg)
        results[f"lev_{lev:.1f}x"] = metrics
        logger.info("  {:.1f}x: sortino={:.1f} return={:.4f} dd={:.4f} margin_cost={:.2f}%",
                     lev, metrics["sortino"], metrics["total_return"], metrics["max_drawdown"],
                     metrics["margin_cost_pct"])

    if not cfg.can_short:
        short_cfg = LeverageConfig(**{**asdict(cfg), "max_leverage": 2.0, "can_short": True})
        short_metrics = simulate_with_margin_cost(bars, actions, short_cfg)
        results["lev_2.0x_short"] = short_metrics
        logger.info("  2.0x+short: sortino={:.1f} return={:.4f}", short_metrics["sortino"], short_metrics["total_return"])

    return results


def run_randomized_holdout(cfg: LeverageConfig, n_windows: int = 5) -> list[dict]:
    """Train on random holdout windows to test robustness."""
    horizons = tuple(int(h) for h in cfg.horizons.split(","))
    forecast_cache = Path("binancechronossolexperiment/forecast_cache_sui_10bp")

    # Load full data to get date range
    dm_full = ChronosSolDataModule(
        symbol=cfg.symbol,
        data_root=Path("trainingdatahourlybinance"),
        forecast_cache_root=forecast_cache,
        forecast_horizons=horizons,
        context_hours=512,
        quantile_levels=(0.1, 0.5, 0.9),
        batch_size=32,
        model_id="amazon/chronos-t5-small",
        sequence_length=cfg.sequence_length,
        split_config=SplitConfig(val_days=cfg.val_days, test_days=cfg.test_days),
        cache_only=True,
    )

    total_rows = len(dm_full.full_frame)
    min_train = cfg.sequence_length + 24 * 30  # 30 days min training
    test_hours = cfg.test_days * 24
    val_hours = cfg.val_days * 24

    rng = random.Random(cfg.seed)
    results = []

    for i in range(n_windows):
        # Random split point
        max_test_start = total_rows - test_hours
        min_test_start = min_train + val_hours
        if min_test_start >= max_test_start:
            logger.warning("Not enough data for random window {}", i)
            continue

        test_start_idx = rng.randint(min_test_start, max_test_start)
        val_start_idx = test_start_idx - val_hours

        # Compute equivalent val/test days from end
        rows_from_end = total_rows - test_start_idx
        test_days = rows_from_end // 24
        val_days = val_hours // 24

        window_name = f"rh{i}_vd{val_days}_td{test_days}"
        logger.info("Random window {}: test_start_idx={} ({} days from end)", i, test_start_idx, test_days)

        try:
            dm = ChronosSolDataModule(
                symbol=cfg.symbol,
                data_root=Path("trainingdatahourlybinance"),
                forecast_cache_root=forecast_cache,
                forecast_horizons=horizons,
                context_hours=512,
                quantile_levels=(0.1, 0.5, 0.9),
                batch_size=32,
                model_id="amazon/chronos-t5-small",
                sequence_length=cfg.sequence_length,
                split_config=SplitConfig(val_days=val_days, test_days=test_days),
                cache_only=True,
            )

            checkpoint_root = Path("binanceleveragesui/checkpoints") / f"random_{window_name}"
            tc = TrainingConfig(
                epochs=cfg.epochs,
                batch_size=cfg.batch_size,
                sequence_length=cfg.sequence_length,
                learning_rate=cfg.learning_rate,
                weight_decay=1e-4,
                return_weight=cfg.return_weight,
                seed=cfg.seed,
                transformer_dim=cfg.transformer_dim,
                transformer_layers=cfg.transformer_layers,
                transformer_heads=cfg.transformer_heads,
                maker_fee=cfg.maker_fee,
                checkpoint_root=checkpoint_root,
                log_dir=Path("tensorboard_logs/binanceleveragesui"),
                use_compile=False,
            )

            trainer = BinanceHourlyTrainer(tc, dm)
            artifacts = trainer.train()

            checkpoint_root.mkdir(parents=True, exist_ok=True)
            sd = artifacts.state_dict
            if artifacts.best_checkpoint and artifacts.best_checkpoint.exists():
                ckpt = torch.load(artifacts.best_checkpoint, map_location="cpu", weights_only=False)
                sd = ckpt.get("state_dict", ckpt)
            ckpt_path = checkpoint_root / "policy_checkpoint.pt"
            torch.save({
                "state_dict": sd,
                "config": asdict(tc),
                "feature_columns": list(artifacts.feature_columns),
                "normalizer": artifacts.normalizer.to_dict(),
            }, ckpt_path)

            model, normalizer, feature_columns, _ = load_policy_checkpoint(str(ckpt_path))
            test_frame = dm.test_frame.copy()
            actions = generate_actions_from_frame(
                model=model, frame=test_frame, feature_columns=feature_columns,
                normalizer=normalizer, sequence_length=cfg.sequence_length, horizon=horizons[0],
            )
            test_start = dm.test_window_start
            bars = test_frame[test_frame["timestamp"] >= test_start].copy()
            actions_test = actions[actions["timestamp"] >= test_start].copy()

            window_results = {"window": window_name, "test_start": str(test_start), "test_days": test_days}
            for lev in [1.0, 2.0, 3.0, 4.0, 5.0]:
                eval_cfg = LeverageConfig(**{**asdict(cfg), "max_leverage": lev})
                metrics = simulate_with_margin_cost(bars, actions_test, eval_cfg)
                window_results[f"lev_{lev:.1f}x"] = metrics

            results.append(window_results)
            logger.info("Window {}: 1x={:.1f} 2x={:.1f} 3x={:.1f}",
                         window_name,
                         window_results["lev_1.0x"]["sortino"],
                         window_results["lev_2.0x"]["sortino"],
                         window_results["lev_3.0x"]["sortino"])
        except Exception as e:
            logger.error("Window {} failed: {}", window_name, e)
            results.append({"window": window_name, "error": str(e)})

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="sweep", choices=["sweep", "random", "single"])
    parser.add_argument("--max-leverage", type=float, default=2.0)
    parser.add_argument("--can-short", action="store_true")
    parser.add_argument("--return-weight", type=float, default=0.012)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--n-windows", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) if args.output else Path(f"binanceleveragesui/results_{args.mode}_{timestamp}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_cfg = LeverageConfig(
        max_leverage=args.max_leverage,
        can_short=args.can_short,
        return_weight=args.return_weight,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        seed=args.seed,
    )

    if args.mode == "single":
        run_name = f"lev{args.max_leverage:.0f}x_rw{args.return_weight}_s{args.seed}"
        result = train_and_evaluate(base_cfg, run_name)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Saved: {}", output_path)

    elif args.mode == "random":
        results = run_randomized_holdout(base_cfg, n_windows=args.n_windows)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        # Summary
        for lev in ["1.0", "2.0", "3.0"]:
            key = f"lev_{lev}x"
            sortinos = [r[key]["sortino"] for r in results if key in r and "error" not in r]
            returns = [r[key]["total_return"] for r in results if key in r and "error" not in r]
            if sortinos:
                logger.info("{}: mean_sortino={:.1f} std={:.1f} mean_return={:.4f}",
                             key, np.mean(sortinos), np.std(sortinos), np.mean(returns))
        logger.info("Saved: {}", output_path)

    elif args.mode == "sweep":
        all_results = []
        leverage_levels = [1.0, 2.0, 3.0, 4.0, 5.0]
        rw_values = [0.008, 0.012, 0.016]

        for rw in rw_values:
            cfg = LeverageConfig(**{**asdict(base_cfg), "return_weight": rw})
            run_name = f"sweep_rw{str(rw).replace('.', '')}_lev{args.max_leverage:.0f}x"
            logger.info("Training: rw={} lev={}x", rw, args.max_leverage)
            result = train_and_evaluate(cfg, run_name)
            all_results.append(result)
            with open(output_path, "w") as f:
                json.dump(all_results, f, indent=2)

        # Summary table
        logger.info("\n=== LEVERAGE SWEEP RESULTS ===")
        for r in all_results:
            rw = r["config"]["return_weight"]
            for lev_key in sorted(k for k in r if k.startswith("lev_")):
                m = r[lev_key]
                logger.info("rw={} {}: sortino={:.1f} return={:.4f} dd={:.4f} margin={:.2f}%",
                             rw, lev_key, m["sortino"], m["total_return"], m["max_drawdown"], m["margin_cost_pct"])
        logger.info("Saved: {}", output_path)


if __name__ == "__main__":
    main()
