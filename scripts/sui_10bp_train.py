#!/usr/bin/env python3
"""Train SUI neural policy with 10bp fee and backtest on 7d holdout."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger

import sys
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from binanceneural.config import TrainingConfig
from binanceneural.trainer import BinanceHourlyTrainer
from binanceneural.inference import generate_actions_from_frame
from binanceneural.marketsimulator import (
    BinanceMarketSimulator,
    SimulationConfig,
    run_shared_cash_simulation,
)

SYMBOL = "SUIUSDT"
MAKER_FEE = 0.001  # 10bp


def build_forecast_cache(
    symbol: str,
    data_root: Path,
    cache_root: Path,
    model_id: str = "amazon/chronos-t5-small",
    horizons: Tuple[int, ...] = (1, 4, 24),
    context_hours: int = 512,
) -> None:
    """Build Chronos2 forecast cache for SUI."""
    from binancechronossolexperiment.forecasts import build_hourly_forecast_cache

    csv_path = data_root / f"{symbol}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["timestamp"]).set_index("timestamp")

    for h in horizons:
        logger.info(f"Building forecast cache for horizon={h}...")
        cache_path = cache_root / f"h{h}" / f"{symbol}.parquet"
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            logger.info(f"Cache exists: {cache_path}, skipping")
            continue

        build_hourly_forecast_cache(
            symbol=symbol,
            data_root=data_root,
            output_path=cache_path,
            model_id=model_id,
            horizon=h,
            context_hours=context_hours,
            batch_size=32,
        )
        logger.info(f"Saved cache: {cache_path}")


def load_data_with_forecasts(
    symbol: str,
    data_root: Path,
    forecast_cache_root: Path,
    horizons: Tuple[int, ...] = (1,),
    sequence_length: int = 72,
    val_days: int = 7,
    test_days: int = 7,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load price data merged with Chronos forecasts."""
    from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig

    dm = ChronosSolDataModule(
        symbol=symbol,
        data_root=data_root,
        forecast_cache_root=forecast_cache_root,
        forecast_horizons=horizons,
        context_hours=512,
        quantile_levels=(0.1, 0.5, 0.9),
        batch_size=32,
        model_id="amazon/chronos-t5-small",
        sequence_length=sequence_length,
        split_config=SplitConfig(val_days=val_days, test_days=test_days),
        cache_only=True,
    )

    return dm.train_frame, dm.val_frame, dm.test_frame


def simple_momentum_backtest(
    df: pd.DataFrame,
    fee: float = MAKER_FEE,
    lookback: int = 24,
    threshold: float = 0.003,
) -> dict:
    """Momentum strategy baseline."""
    df = df.copy().sort_index()
    df["returns"] = df["close"].pct_change()
    df["momentum"] = df["close"].pct_change(lookback)

    position = 0.0
    cash = 10000.0
    equity_curve = [cash]
    trades = []

    for i in range(lookback + 1, len(df)):
        price = df["close"].iloc[i]
        mom = df["momentum"].iloc[i]

        if position == 0 and mom > threshold:
            qty = (cash * 0.95) / price
            cost = qty * price * (1 + fee)
            if cost <= cash:
                cash -= cost
                position = qty
                trades.append(("buy", price, i))
        elif position > 0 and mom < -threshold:
            proceeds = position * price * (1 - fee)
            cash += proceeds
            trades.append(("sell", price, i, cash))
            position = 0

        equity = cash + position * price
        equity_curve.append(equity)

    # Close any open position
    if position > 0:
        final_price = df["close"].iloc[-1]
        cash += position * final_price * (1 - fee)
        equity_curve[-1] = cash

    equity_curve = np.array(equity_curve)
    total_return = (equity_curve[-1] / equity_curve[0]) - 1

    returns = np.diff(equity_curve) / (equity_curve[:-1] + 1e-10)
    returns = returns[~np.isnan(returns)]
    neg_returns = returns[returns < 0]
    downside_std = np.std(neg_returns) if len(neg_returns) > 0 else 1e-6
    sortino = (np.mean(returns) / (downside_std + 1e-10)) * np.sqrt(8760)

    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / (running_max + 1e-10)
    max_dd = float(np.min(drawdowns))

    sell_trades = [t for t in trades if t[0] == "sell"]
    win_trades = sum(1 for t in sell_trades if len(t) > 3 and t[3] > 10000)
    win_rate = win_trades / len(sell_trades) if sell_trades else 0

    return {
        "strategy": "momentum",
        "total_return": total_return,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "num_trades": len(sell_trades),
        "win_rate": win_rate,
        "final_equity": equity_curve[-1],
    }


def train_neural_policy(
    symbol: str,
    data_root: Path,
    forecast_cache_root: Path,
    run_name: str,
    maker_fee: float = MAKER_FEE,
    horizons: Tuple[int, ...] = (1, 4, 24),
    sequence_length: int = 72,
    epochs: int = 10,
    val_days: int = 7,
    test_days: int = 7,
    max_history_days: int = 180,
) -> Tuple[Path, dict]:
    """Train neural policy with specified maker fee."""
    from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig

    logger.info(f"Training neural policy for {symbol} with {maker_fee*10000:.0f}bp fee")

    dm = ChronosSolDataModule(
        symbol=symbol,
        data_root=data_root,
        forecast_cache_root=forecast_cache_root,
        forecast_horizons=horizons,
        context_hours=512,
        quantile_levels=(0.1, 0.5, 0.9),
        batch_size=32,
        model_id="amazon/chronos-t5-small",
        sequence_length=sequence_length,
        split_config=SplitConfig(val_days=val_days, test_days=test_days),
        max_history_days=max_history_days,
        cache_only=True,
    )

    checkpoint_root = Path("binancechronossolexperiment/checkpoints") / run_name

    config = TrainingConfig(
        epochs=epochs,
        batch_size=64,
        sequence_length=sequence_length,
        learning_rate=3e-4,
        weight_decay=1e-4,
        optimizer_name="muon_mix",
        model_arch="nano",
        maker_fee=maker_fee,  # 10bp
        checkpoint_root=checkpoint_root.parent,
        run_name=run_name,
        use_compile=False,
    )

    trainer = BinanceHourlyTrainer(config, dm)
    artifacts = trainer.train()

    # Save checkpoint
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    if artifacts.best_checkpoint and artifacts.best_checkpoint.exists():
        best_ckpt = artifacts.best_checkpoint
    else:
        ckpts = sorted(checkpoint_root.glob("*.pt"))
        best_ckpt = ckpts[-1] if ckpts else None

    train_metrics = {
        "final_train_sortino": artifacts.history[-1].train_sortino if artifacts.history else 0,
        "final_val_sortino": artifacts.history[-1].val_sortino if artifacts.history else 0,
        "final_train_return": artifacts.history[-1].train_return if artifacts.history else 0,
        "final_val_return": artifacts.history[-1].val_return if artifacts.history else 0,
    }

    return best_ckpt, train_metrics


def run_neural_backtest(
    symbol: str,
    checkpoint_path: Path,
    test_frame: pd.DataFrame,
    sequence_length: int = 72,
    maker_fee: float = MAKER_FEE,
    initial_cash: float = 10000.0,
) -> dict:
    """Run neural policy backtest."""
    from binancechronossolexperiment.inference import load_policy_checkpoint

    logger.info(f"Running neural backtest with {maker_fee*10000:.0f}bp fee")

    model, normalizer, feature_columns, config = load_policy_checkpoint(str(checkpoint_path))

    actions = generate_actions_from_frame(
        model=model,
        frame=test_frame,
        feature_columns=feature_columns,
        normalizer=normalizer,
        sequence_length=sequence_length,
        horizon=1,
    )

    # Run simulation
    sim_config = SimulationConfig(
        maker_fee=maker_fee,
        initial_cash=initial_cash,
    )
    sim = BinanceMarketSimulator(sim_config)

    bars = test_frame.copy()
    if "timestamp" not in bars.columns and bars.index.name == "timestamp":
        bars = bars.reset_index()

    result = sim.run(bars, actions)
    metrics = dict(result.metrics)

    # Calculate additional metrics
    equity = result.combined_equity
    returns = equity.pct_change().dropna()
    neg_returns = returns[returns < 0]
    downside_std = neg_returns.std() if len(neg_returns) > 0 else 1e-6
    sortino = (returns.mean() / (downside_std + 1e-10)) * np.sqrt(8760)

    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1

    running_max = equity.cummax()
    drawdowns = (equity - running_max) / running_max
    max_dd = float(drawdowns.min())

    # Count trades
    num_trades = 0
    for sym_result in result.per_symbol.values():
        num_trades += len([t for t in sym_result.trades if t.side == "sell"])

    return {
        "strategy": "neural",
        "total_return": total_return,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "num_trades": num_trades,
        "win_rate": 0,  # Would need trade-level analysis
        "final_equity": equity.iloc[-1],
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train SUI with 10bp fee")
    parser.add_argument("--data-root", default="trainingdatahourlybinance")
    parser.add_argument("--forecast-cache-root", default=None)
    parser.add_argument("--lora-model", default=None, help="LoRA checkpoint path")
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--checkpoint", default=None, help="Use existing checkpoint")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eval-days", type=int, default=7)
    parser.add_argument("--maker-fee", type=float, default=MAKER_FEE)
    args = parser.parse_args(list(argv) if argv else None)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    data_root = Path(args.data_root)

    forecast_cache_root = Path(args.forecast_cache_root) if args.forecast_cache_root else \
        Path("binancechronossolexperiment") / f"forecast_cache_sui_{run_id}"

    model_id = args.lora_model or "amazon/chronos-t5-small"

    # 1. Build forecast cache
    if not args.skip_cache:
        logger.info("Building forecast cache...")
        try:
            build_forecast_cache(
                symbol=SYMBOL,
                data_root=data_root,
                cache_root=forecast_cache_root,
                model_id=model_id,
                horizons=(1, 4, 24),
            )
        except Exception as e:
            logger.error(f"Forecast cache build failed: {e}")
            logger.info("Trying with base model...")
            build_forecast_cache(
                symbol=SYMBOL,
                data_root=data_root,
                cache_root=forecast_cache_root,
                model_id="amazon/chronos-t5-small",
                horizons=(1, 4, 24),
            )

    # 2. Train neural policy
    run_name = f"sui_10bp_{run_id}"
    checkpoint = None
    train_metrics = {}

    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
    elif not args.skip_train:
        try:
            checkpoint, train_metrics = train_neural_policy(
                symbol=SYMBOL,
                data_root=data_root,
                forecast_cache_root=forecast_cache_root,
                run_name=run_name,
                maker_fee=args.maker_fee,
                epochs=args.epochs,
                val_days=args.eval_days,
                test_days=args.eval_days,
            )
            logger.info(f"Training complete: {train_metrics}")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()

    # 3. Load test data
    logger.info("Loading test data...")
    csv_path = data_root / f"{SYMBOL}.csv"
    df = pd.read_csv(csv_path, parse_dates=["timestamp"]).set_index("timestamp")

    test_hours = args.eval_days * 24
    test_df = df.iloc[-test_hours:].copy()
    logger.info(f"Test period: {test_df.index.min()} to {test_df.index.max()} ({len(test_df)} bars)")

    # 4. Run backtests
    results = []

    # Momentum baseline
    logger.info("Running momentum backtest...")
    mom_result = simple_momentum_backtest(test_df, fee=args.maker_fee)
    results.append(mom_result)
    logger.info(f"Momentum: return={mom_result['total_return']:.4f}, sortino={mom_result['sortino']:.2f}")

    # Neural policy
    if checkpoint and checkpoint.exists():
        try:
            # Load test frame with forecasts
            from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
            dm = ChronosSolDataModule(
                symbol=SYMBOL,
                data_root=data_root,
                forecast_cache_root=forecast_cache_root,
                forecast_horizons=(1, 4, 24),
                context_hours=512,
                quantile_levels=(0.1, 0.5, 0.9),
                batch_size=32,
                model_id="amazon/chronos-t5-small",
                sequence_length=72,
                split_config=SplitConfig(val_days=args.eval_days, test_days=args.eval_days),
                cache_only=True,
            )

            neural_result = run_neural_backtest(
                symbol=SYMBOL,
                checkpoint_path=checkpoint,
                test_frame=dm.test_frame,
                maker_fee=args.maker_fee,
            )
            results.append(neural_result)
            logger.info(f"Neural: return={neural_result['total_return']:.4f}, sortino={neural_result['sortino']:.2f}")
        except Exception as e:
            logger.error(f"Neural backtest failed: {e}")
            import traceback
            traceback.print_exc()

    # 5. Summary
    print("\n" + "="*70)
    print(f"SUI/USDT Strategy Comparison ({args.eval_days}d holdout, {args.maker_fee*10000:.0f}bp fee)")
    print("="*70)
    for r in results:
        print(f"{r['strategy']:12s}: return={r['total_return']:+.4f}, sortino={r['sortino']:8.2f}, "
              f"maxdd={r['max_drawdown']:.4f}, trades={r['num_trades']}, equity=${r['final_equity']:.2f}")

    # Save results
    output_path = Path(f"reports/sui_10bp_{run_id}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "run_id": run_id,
        "maker_fee": args.maker_fee,
        "eval_days": args.eval_days,
        "train_metrics": train_metrics,
        "backtest_results": results,
    }, indent=2))
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
