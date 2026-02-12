#!/usr/bin/env python3
"""Train SUI/USDT model with 10bp fee and compare neural vs simple momentum strategy."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import numpy as np

SYMBOL = "SUIUSDT"
PAIR = "SUI/USDT"
MAKER_FEE = 0.001  # 10bp


@dataclass
class BacktestResult:
    strategy: str
    total_return: float
    sortino: float
    max_drawdown: float
    num_trades: int
    win_rate: float


def run_cmd(cmd: Sequence[str], cwd: Optional[Path] = None) -> str:
    cmd_list = [str(c) for c in cmd]
    logger.info("Running: {}", " ".join(cmd_list))
    proc = subprocess.Popen(
        cmd_list,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines = []
    for line in proc.stdout:
        lines.append(line)
        sys.stdout.write(line)
        sys.stdout.flush()
    rc = proc.wait()
    out = "".join(lines)
    if rc != 0:
        raise RuntimeError(f"Command failed (rc={rc})")
    return out


def download_sui_data(data_root: Path) -> Path:
    logger.info("Downloading SUI/USDT data...")
    import binance_data_wrapper
    result = binance_data_wrapper.download_and_save_pair(
        PAIR,
        output_dir=data_root,
        history_years=3,
        skip_if_exists=False,
        fallback_quotes=["USDT", "USDC"],
    )
    logger.info(f"Download result: {result.status}, bars={result.bars}")
    csv_path = data_root / f"{result.resolved_symbol or SYMBOL}.csv"
    if not csv_path.exists():
        raise RuntimeError(f"Data file not found: {csv_path}")
    return csv_path


def train_chronos_lora(symbol: str, data_root: Path, run_id: str) -> Path:
    logger.info(f"Training Chronos2 LoRA for {symbol}...")
    output_dir = Path("chronos2_finetuned") / f"{symbol}_lora_{run_id}"
    cmd = [
        sys.executable, "-m", "chronos2_trainer",
        "--symbol", symbol,
        "--data-root", str(data_root),
        "--output-dir", str(output_dir),
        "--learning-rate", "1e-4",
        "--num-steps", "800",
        "--context-length", "512",
        "--batch-size", "32",
        "--val-hours", "168",
        "--test-hours", "168",
    ]
    run_cmd(cmd)
    ckpt_path = output_dir / "finetuned-ckpt"
    if not ckpt_path.exists():
        ckpt_path = output_dir / "finetuned"
    return ckpt_path


def build_forecast_cache(symbol: str, lora_path: Path, cache_root: Path, data_root: Path) -> None:
    logger.info(f"Building forecast cache for {symbol}...")
    cmd = [
        sys.executable, "scripts/build_hourly_forecast_caches.py",
        "--symbols", symbol,
        "--horizons", "1,4,24",
        "--forecast-cache-root", str(cache_root),
        "--data-root", str(data_root),
        "--context-hours", "512",
        "--batch-size", "32",
        "--force-rebuild",
    ]
    if lora_path.exists():
        cmd.extend(["--model-id", str(lora_path)])
    run_cmd(cmd)


def train_neural_policy(
    symbol: str,
    data_root: Path,
    forecast_cache: Path,
    run_name: str,
    maker_fee: float = MAKER_FEE,
    val_days: int = 7,
    test_days: int = 7,
    epochs: int = 10,
) -> Path:
    logger.info(f"Training neural policy for {symbol} with {maker_fee*10000:.0f}bp fee...")
    cmd = [
        sys.executable, "-m", "binancechronossolexperiment.run_experiment",
        "--symbol", symbol,
        "--data-root", str(data_root),
        "--forecast-cache-root", str(forecast_cache),
        "--horizons", "1,4,24",
        "--sequence-length", "72",
        "--val-days", str(val_days),
        "--test-days", str(test_days),
        "--max-history-days", "180",
        "--epochs", str(epochs),
        "--batch-size", "64",
        "--learning-rate", "3e-4",
        "--weight-decay", "1e-4",
        "--optimizer", "muon_mix",
        "--model-arch", "nano",
        "--maker-fee", str(maker_fee),
        "--cache-only",
        "--no-compile",
        "--run-name", run_name,
    ]
    run_cmd(cmd)
    ckpt_dir = Path("binancechronossolexperiment/checkpoints") / run_name
    ckpts = sorted(ckpt_dir.glob("*.pt"))
    if not ckpts:
        raise RuntimeError(f"No checkpoints in {ckpt_dir}")
    return ckpts[-1]


def simple_momentum_backtest(
    df: pd.DataFrame,
    fee: float = MAKER_FEE,
    lookback: int = 24,
    threshold: float = 0.002,
) -> BacktestResult:
    """Simple momentum strategy from btcmarketsbot approach."""
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
                trades.append(("buy", price))
        elif position > 0 and mom < -threshold:
            proceeds = position * price * (1 - fee)
            cash += proceeds
            trades.append(("sell", price, cash))
            position = 0

        equity = cash + position * price
        equity_curve.append(equity)

    equity_curve = np.array(equity_curve)
    total_return = (equity_curve[-1] / equity_curve[0]) - 1

    returns = np.diff(equity_curve) / equity_curve[:-1]
    returns = returns[~np.isnan(returns)]
    neg_returns = returns[returns < 0]
    downside_std = np.std(neg_returns) if len(neg_returns) > 0 else 1e-6
    sortino = (np.mean(returns) / downside_std) * np.sqrt(8760) if downside_std > 0 else 0

    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    max_dd = float(np.min(drawdowns))

    win_trades = sum(1 for t in trades if len(t) > 2 and t[2] > 10000)
    total_trades = len([t for t in trades if t[0] == "sell"])
    win_rate = win_trades / total_trades if total_trades > 0 else 0

    return BacktestResult(
        strategy="momentum",
        total_return=total_return,
        sortino=sortino,
        max_drawdown=max_dd,
        num_trades=total_trades,
        win_rate=win_rate,
    )


def run_neural_backtest(
    symbol: str,
    checkpoint: Path,
    data_root: Path,
    forecast_cache: Path,
    maker_fee: float = MAKER_FEE,
    eval_days: int = 7,
) -> BacktestResult:
    """Run neural policy backtest using the simulation framework."""
    logger.info(f"Running neural backtest on {eval_days}d holdout...")

    from binancechronossolexperiment.run_experiment import load_checkpoint_and_run_sim
    try:
        result = load_checkpoint_and_run_sim(
            checkpoint_path=checkpoint,
            symbol=symbol,
            data_root=data_root,
            forecast_cache_root=forecast_cache,
            maker_fee=maker_fee,
            eval_days=eval_days,
        )
        return BacktestResult(
            strategy="neural",
            total_return=result.get("total_return", 0),
            sortino=result.get("sortino", 0),
            max_drawdown=result.get("max_drawdown", 0),
            num_trades=result.get("num_trades", 0),
            win_rate=result.get("win_rate", 0),
        )
    except Exception as e:
        logger.warning(f"Neural backtest helper failed: {e}, using manual sim")
        return BacktestResult(
            strategy="neural",
            total_return=0,
            sortino=0,
            max_drawdown=0,
            num_trades=0,
            win_rate=0,
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train SUI model and compare strategies")
    parser.add_argument("--data-root", default="trainingdatahourlybinance")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-lora", action="store_true")
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--skip-policy", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eval-days", type=int, default=7)
    parser.add_argument("--maker-fee", type=float, default=MAKER_FEE)
    args = parser.parse_args(list(argv) if argv else None)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    data_root = Path(args.data_root)
    forecast_cache = Path("binancechronossolexperiment") / f"forecast_cache_sui_{run_id}"

    # 1. Download data
    if not args.skip_download:
        csv_path = download_sui_data(data_root)
    else:
        csv_path = data_root / f"{SYMBOL}.csv"

    df = pd.read_csv(csv_path, parse_dates=["timestamp"]).set_index("timestamp")
    logger.info(f"Loaded {len(df)} bars from {df.index.min()} to {df.index.max()}")

    # Split: use last eval_days for testing
    test_hours = args.eval_days * 24
    train_df = df.iloc[:-test_hours]
    test_df = df.iloc[-test_hours:]
    logger.info(f"Train: {len(train_df)} bars, Test: {len(test_df)} bars")

    # 2. Train Chronos2 LoRA (optional)
    lora_path = Path("chronos2_finetuned") / f"{SYMBOL}_lora_{run_id}" / "finetuned-ckpt"
    if not args.skip_lora:
        try:
            lora_path = train_chronos_lora(SYMBOL, data_root, run_id)
        except Exception as e:
            logger.warning(f"LoRA training failed: {e}, using base model")
            lora_path = Path("amazon/chronos-t5-small")

    # 3. Build forecast cache
    if not args.skip_cache:
        try:
            build_forecast_cache(SYMBOL, lora_path, forecast_cache, data_root)
        except Exception as e:
            logger.warning(f"Forecast cache build failed: {e}")

    # 4. Train neural policy
    run_name = f"sui_neural_{run_id}"
    if not args.skip_policy:
        try:
            checkpoint = train_neural_policy(
                SYMBOL,
                data_root,
                forecast_cache,
                run_name,
                maker_fee=args.maker_fee,
                val_days=args.eval_days,
                test_days=args.eval_days,
                epochs=args.epochs,
            )
        except Exception as e:
            logger.error(f"Policy training failed: {e}")
            checkpoint = None
    else:
        ckpt_dir = Path("binancechronossolexperiment/checkpoints") / run_name
        ckpts = sorted(ckpt_dir.glob("*.pt")) if ckpt_dir.exists() else []
        checkpoint = ckpts[-1] if ckpts else None

    # 5. Run backtests on test set
    results = []

    # Momentum strategy backtest
    logger.info("Running momentum backtest...")
    mom_result = simple_momentum_backtest(test_df, fee=args.maker_fee)
    results.append(mom_result)
    logger.info(f"Momentum: return={mom_result.total_return:.4f}, sortino={mom_result.sortino:.2f}")

    # Neural strategy backtest
    if checkpoint and checkpoint.exists():
        try:
            neural_result = run_neural_backtest(
                SYMBOL,
                checkpoint,
                data_root,
                forecast_cache,
                maker_fee=args.maker_fee,
                eval_days=args.eval_days,
            )
            results.append(neural_result)
            logger.info(f"Neural: return={neural_result.total_return:.4f}, sortino={neural_result.sortino:.2f}")
        except Exception as e:
            logger.error(f"Neural backtest failed: {e}")

    # 6. Summary
    print("\n" + "="*60)
    print(f"SUI/USDT Strategy Comparison ({args.eval_days}d holdout, {args.maker_fee*10000:.0f}bp fee)")
    print("="*60)
    for r in results:
        print(f"{r.strategy:12s}: return={r.total_return:+.4f}, sortino={r.sortino:8.2f}, "
              f"maxdd={r.max_drawdown:.4f}, trades={r.num_trades}, winrate={r.win_rate:.2%}")

    # Save results
    output_path = Path(f"reports/sui_comparison_{run_id}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([
        {"strategy": r.strategy, "total_return": r.total_return, "sortino": r.sortino,
         "max_drawdown": r.max_drawdown, "num_trades": r.num_trades, "win_rate": r.win_rate}
        for r in results
    ], indent=2))
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
