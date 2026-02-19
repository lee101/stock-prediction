#!/usr/bin/env python3
"""Backtest SUI strategies with 10bp fee on 7d holdout."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

import sys
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SYMBOL = "SUIUSDT"
MAKER_FEE = 0.001  # 10bp
EVAL_DAYS = 7


def simple_momentum_backtest(df, fee=MAKER_FEE, lookback=24, threshold=0.003):
    """Simple momentum strategy."""
    df = df.copy().sort_index()
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
                trades.append(("buy", price))
        elif position > 0 and mom < -threshold:
            proceeds = position * price * (1 - fee)
            cash += proceeds
            trades.append(("sell", price, cash))
            position = 0

        equity_curve.append(cash + position * price)

    if position > 0:
        cash += position * df["close"].iloc[-1] * (1 - fee)
        equity_curve[-1] = cash

    eq = np.array(equity_curve)
    ret = np.diff(eq) / (eq[:-1] + 1e-10)
    neg = ret[ret < 0]
    sortino = (np.mean(ret) / (np.std(neg) + 1e-10)) * np.sqrt(8760) if len(neg) > 0 else 0

    running_max = np.maximum.accumulate(eq)
    max_dd = float(np.min((eq - running_max) / (running_max + 1e-10)))

    return {
        "strategy": "momentum",
        "total_return": (eq[-1] / eq[0]) - 1,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "num_trades": len([t for t in trades if t[0] == "sell"]),
        "final_equity": eq[-1],
    }


def neural_backtest_with_fee(checkpoint_path, test_frame, fee=MAKER_FEE, seq_len=72):
    """Run neural policy backtest with explicit fee."""
    from binancechronossolexperiment.inference import load_policy_checkpoint
    from binanceneural.inference import generate_actions_from_frame
    from binanceneural.marketsimulator import BinanceMarketSimulator, SimulationConfig

    model, normalizer, feature_columns, _ = load_policy_checkpoint(str(checkpoint_path))

    actions = generate_actions_from_frame(
        model=model,
        frame=test_frame,
        feature_columns=feature_columns,
        normalizer=normalizer,
        sequence_length=seq_len,
        horizon=1,
    )

    config = SimulationConfig(maker_fee=fee, initial_cash=10000.0)
    sim = BinanceMarketSimulator(config)

    bars = test_frame.copy()
    if "timestamp" not in bars.columns and bars.index.name == "timestamp":
        bars = bars.reset_index()

    result = sim.run(bars, actions)
    eq = result.combined_equity

    ret = eq.pct_change().dropna()
    neg = ret[ret < 0]
    sortino = (ret.mean() / (neg.std() + 1e-10)) * np.sqrt(8760) if len(neg) > 0 else 0

    running_max = eq.cummax()
    max_dd = float(((eq - running_max) / running_max).min())

    num_trades = sum(len([t for t in sr.trades if t.side == "sell"]) for sr in result.per_symbol.values())

    return {
        "strategy": "neural (10bp fee)",
        "total_return": (eq.iloc[-1] / eq.iloc[0]) - 1,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "num_trades": num_trades,
        "final_equity": eq.iloc[-1],
    }


def main():
    data_root = Path("trainingdatahourlybinance")
    forecast_cache = Path("binancechronossolexperiment/forecast_cache_sui_10bp")
    checkpoint = Path("binancechronossolexperiment/checkpoints/sui_sortino_rw0012_lr1e4_ep25/policy_checkpoint.pt")

    # Load test data
    csv_path = data_root / f"{SYMBOL}.csv"
    df = pd.read_csv(csv_path, parse_dates=["timestamp"]).set_index("timestamp")

    test_hours = EVAL_DAYS * 24
    test_df = df.iloc[-test_hours:].copy()
    logger.info(f"Test: {test_df.index.min()} to {test_df.index.max()} ({len(test_df)} bars)")

    results = []

    # Momentum
    logger.info("Running momentum backtest...")
    mom = simple_momentum_backtest(test_df)
    results.append(mom)
    logger.info(f"Momentum: return={mom['total_return']:.4f}, sortino={mom['sortino']:.2f}")

    # Neural with 10bp fee
    if checkpoint.exists():
        logger.info("Running neural backtest with 10bp fee...")
        try:
            from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
            dm = ChronosSolDataModule(
                symbol=SYMBOL,
                data_root=data_root,
                forecast_cache_root=forecast_cache,
                forecast_horizons=(1, 4, 24),
                context_hours=512,
                quantile_levels=(0.1, 0.5, 0.9),
                batch_size=32,
                model_id="amazon/chronos-t5-small",
                sequence_length=72,
                split_config=SplitConfig(val_days=EVAL_DAYS, test_days=EVAL_DAYS),
                cache_only=True,
            )

            neural = neural_backtest_with_fee(checkpoint, dm.test_frame)
            results.append(neural)
            logger.info(f"Neural: return={neural['total_return']:.4f}, sortino={neural['sortino']:.2f}")
        except Exception as e:
            logger.error(f"Neural backtest failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*70)
    print(f"SUI/USDT Strategy Comparison ({EVAL_DAYS}d holdout, {MAKER_FEE*10000:.0f}bp fee)")
    print("="*70)
    for r in results:
        print(f"{r['strategy']:20s}: return={r['total_return']:+.4f}, sortino={r['sortino']:8.2f}, "
              f"maxdd={r['max_drawdown']:.4f}, trades={r['num_trades']}, equity=${r['final_equity']:.2f}")

    # Save
    output = Path("reports/sui_10bp_comparison.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    logger.info(f"Saved to {output}")


if __name__ == "__main__":
    main()
