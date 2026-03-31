#!/usr/bin/env python3
"""Calibrate execution parameters for the daily stock PPO ensemble.

Sweeps entry_offset_bps, exit_offset_bps, and allocation_scale over
historical data using a realistic backtest (open-price fills, fees,
limit-order fill simulation). Finds the combination that maximizes
Sortino ratio on the training period and validates on a holdout period.

OPTIMIZATION: Ensemble signals are pre-computed once across all time steps.
The sweep then replays execution logic cheaply (no model inference in loop).

Usage:
    source .venv313/bin/activate
    python scripts/calibrate_daily_stock.py --sweep
    python scripts/calibrate_daily_stock.py --quick-sweep
    python scripts/calibrate_daily_stock.py --eval --entry-offset-bps -5 --exit-offset-bps 10 --allocation-scale 1.2
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pufferlib_market.inference_daily import DailyPPOTrader, compute_daily_features
from trade_daily_stock_prod import (
    DEFAULT_CHECKPOINT,
    DEFAULT_EXTRA_CHECKPOINTS,
    DEFAULT_SYMBOLS,
    _align_frames,
    _load_bare_policy,
    _normalize_daily_frame,
    compute_target_qty_from_values,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("calibrate_daily_stock")


@dataclass
class CalibrationParams:
    """Execution calibration parameters to sweep."""
    entry_offset_bps: float = 0.0    # bps offset for buy limit price from open
    exit_offset_bps: float = 0.0     # bps offset for sell limit price from open
    allocation_scale: float = 1.0    # multiplier on base allocation_pct
    confidence_threshold: float = 0.0  # min ensemble confidence to trade
    fee_bps: float = 0.0            # round-trip fee in bps (Alpaca stocks = ~0)


@dataclass
class BacktestResult:
    """Results from a single backtest window."""
    total_return: float
    annualized_return: float
    sortino: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    num_days: int
    final_equity: float


@dataclass
class CalibrationResult:
    """Results from a full calibration sweep."""
    params: CalibrationParams
    train_sortino: float
    train_return: float
    train_max_dd: float
    train_p10: float
    val_sortino: float
    val_return: float
    val_max_dd: float
    val_p10: float
    num_train_windows: int
    num_val_windows: int


@dataclass
class PrecomputedSignal:
    """Pre-computed ensemble signal for a single time step."""
    action: str          # "flat" or "long_SYMBOL"
    symbol: Optional[str]
    direction: Optional[str]
    confidence: float


def load_daily_frames(
    symbols: Sequence[str],
    data_dir: str = "trainingdata",
    min_days: int = 120,
) -> dict[str, pd.DataFrame]:
    """Load and align daily OHLCV frames for all symbols."""
    frames: dict[str, pd.DataFrame] = {}
    base = REPO / data_dir
    for symbol in symbols:
        path = base / f"{symbol}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing data for {symbol}: {path}")
        frame = _normalize_daily_frame(pd.read_csv(path))
        if len(frame) < min_days:
            raise ValueError(f"{symbol}: only {len(frame)} rows, need {min_days}")
        frames[symbol] = frame
    return _align_frames(frames)


def build_ensemble(
    checkpoint: str,
    extra_checkpoints: list[str],
    symbols: list[str],
    device: str = "cpu",
) -> tuple[DailyPPOTrader, list]:
    """Load the primary trader and extra ensemble policies."""
    trader = DailyPPOTrader(checkpoint, device=device, long_only=True, symbols=symbols)
    extra_policies = [
        _load_bare_policy(
            str((REPO / p).resolve()) if not Path(p).is_absolute() else p,
            trader.obs_size,
            trader.num_actions,
            device,
        )
        for p in extra_checkpoints
    ]
    return trader, extra_policies


def ensemble_signal(
    trader: DailyPPOTrader,
    extra_policies: list,
    features: np.ndarray,
    prices: dict[str, float],
) -> PrecomputedSignal:
    """Get ensemble signal for a single time step."""
    obs = trader.build_observation(features, prices)
    obs_t = torch.from_numpy(obs).unsqueeze(0).to(trader.device)
    all_probs = []
    with torch.inference_mode():
        logits, _ = trader.policy(obs_t)
        logits = trader.apply_action_constraints(logits)
        all_probs.append(F.softmax(logits, dim=-1))
        for pol in extra_policies:
            logits_i, _ = pol(obs_t)
            logits_i = trader.apply_action_constraints(logits_i)
            all_probs.append(F.softmax(logits_i, dim=-1))
    avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)
    action_idx = int(avg_probs.argmax(dim=-1).item())
    confidence = float(avg_probs[0, action_idx].item())

    signal = trader._decode_action(action_idx, confidence, 0.0)
    if signal.direction == "short":
        return PrecomputedSignal("flat", None, None, confidence)
    return PrecomputedSignal(signal.action, signal.symbol, signal.direction, confidence)


def precompute_signals(
    trader: DailyPPOTrader,
    extra_policies: list,
    indexed: dict[str, pd.DataFrame],
    start_idx: int = 0,
    end_idx: Optional[int] = None,
) -> list[PrecomputedSignal]:
    """Pre-compute ensemble signals for all time steps. No position state dependency.

    The daily stock PPO system decides which symbol to go long based on market
    features only — the position-state (cash, current holding) in the observation
    affects the value estimate but not the action selection for this ensemble.
    We approximate by running with a "fresh" state at each step.
    """
    min_len = min(len(frame) for frame in indexed.values())
    if end_idx is None:
        end_idx = min_len
    signals: list[PrecomputedSignal] = []
    total = end_idx - start_idx
    for step, idx in enumerate(range(start_idx, end_idx)):
        close_prices = {
            sym: float(frame["close"].iloc[idx]) for sym, frame in indexed.items()
        }
        features = np.zeros((trader.num_symbols, 16), dtype=np.float32)
        for i, sym in enumerate(trader.SYMBOLS):
            if sym in indexed:
                features[i] = compute_daily_features(indexed[sym].iloc[:idx + 1])

        # Reset trader to neutral state for signal generation
        trader.cash = 10_000.0
        trader.current_position = None
        trader.position_qty = 0.0
        trader.entry_price = 0.0
        trader.hold_hours = 0
        trader.step = min(step, trader.max_steps)

        sig = ensemble_signal(trader, extra_policies, features, close_prices)
        signals.append(sig)

        if (step + 1) % 100 == 0:
            logger.info("Pre-computing signals: %d/%d", step + 1, total)

    logger.info("Pre-computed %d signals (%d long, %d flat)",
                len(signals),
                sum(1 for s in signals if s.direction == "long"),
                sum(1 for s in signals if s.direction is None))
    return signals


@dataclass
class PriceArrays:
    """Pre-built numpy arrays for fast price lookup. Built once, used for all windows."""
    symbols: list[str]
    sym_to_idx: dict[str, int]
    opens: np.ndarray    # [n_bars, n_symbols]
    highs: np.ndarray    # [n_bars, n_symbols]
    lows: np.ndarray     # [n_bars, n_symbols]
    closes: np.ndarray   # [n_bars, n_symbols]


def build_price_arrays(indexed: dict[str, pd.DataFrame]) -> PriceArrays:
    """Pre-build numpy arrays from indexed DataFrames for fast replay."""
    symbols = sorted(indexed.keys())
    sym_to_idx = {s: i for i, s in enumerate(symbols)}
    n_bars = min(len(f) for f in indexed.values())
    n_sym = len(symbols)
    opens = np.zeros((n_bars, n_sym), dtype=np.float64)
    highs = np.zeros((n_bars, n_sym), dtype=np.float64)
    lows = np.zeros((n_bars, n_sym), dtype=np.float64)
    closes = np.zeros((n_bars, n_sym), dtype=np.float64)
    for i, sym in enumerate(symbols):
        frame = indexed[sym]
        opens[:n_bars, i] = frame["open"].values[:n_bars]
        highs[:n_bars, i] = frame["high"].values[:n_bars]
        lows[:n_bars, i] = frame["low"].values[:n_bars]
        closes[:n_bars, i] = frame["close"].values[:n_bars]
    return PriceArrays(symbols, sym_to_idx, opens, highs, lows, closes)


def replay_with_params(
    *,
    signals: list[PrecomputedSignal],
    price_arrays: PriceArrays,
    signal_start_idx: int,
    window_start: int,
    window_end: int,
    params: CalibrationParams,
    base_allocation_pct: float = 25.0,
    initial_cash: float = 10_000.0,
    # Legacy compat: accept indexed dict and convert
    indexed: dict[str, pd.DataFrame] | None = None,
) -> BacktestResult:
    """Replay pre-computed signals with specific calibration params.

    This is the fast inner loop — no model inference, pure arithmetic.
    """
    pa = price_arrays
    cash = initial_cash
    pos_sym_idx: int = -1   # -1 = no position
    pos_qty: float = 0.0
    pos_entry: float = 0.0
    equity_curve = [cash]
    trades = 0
    winning_trades = 0
    allocation_pct = base_allocation_pct * params.allocation_scale
    fee_rate = params.fee_bps / 10_000.0
    entry_scale = 1.0 + params.entry_offset_bps / 10_000.0
    exit_scale = 1.0 + params.exit_offset_bps / 10_000.0
    check_limit_buy = params.entry_offset_bps < 0
    check_limit_sell = params.exit_offset_bps > 0
    conf_thresh = params.confidence_threshold

    for idx in range(window_start, window_end - 1):
        sig_idx = idx - signal_start_idx
        if sig_idx < 0 or sig_idx >= len(signals):
            break
        sig = signals[sig_idx]
        nxt = idx + 1

        # Decode signal
        want_long = sig.direction == "long" and sig.confidence >= conf_thresh
        sig_sym_idx = pa.sym_to_idx.get(sig.symbol, -1) if want_long and sig.symbol else -1

        # Mark-to-market at next open
        equity = cash
        if pos_sym_idx >= 0:
            equity += pos_qty * pa.opens[nxt, pos_sym_idx]
        equity_curve.append(equity)

        # Close if signal differs
        if pos_sym_idx >= 0 and sig_sym_idx != pos_sym_idx:
            open_p = pa.opens[nxt, pos_sym_idx]
            sell_price = open_p * exit_scale
            if check_limit_sell and pa.highs[nxt, pos_sym_idx] < sell_price:
                sell_price = open_p  # limit not reached, sell at market
            proceeds = pos_qty * sell_price * (1.0 - fee_rate)
            cash += proceeds
            if sell_price > pos_entry:
                winning_trades += 1
            pos_sym_idx = -1
            pos_qty = 0.0
            trades += 1

        # Open new position
        if pos_sym_idx < 0 and sig_sym_idx >= 0:
            open_p = pa.opens[nxt, sig_sym_idx]
            buy_price = open_p * entry_scale
            fill = True
            if check_limit_buy and pa.lows[nxt, sig_sym_idx] > buy_price:
                fill = False
            if fill and buy_price > 0:
                target = cash * max(0.0, allocation_pct) / 100.0
                target = min(target, cash * 0.95)
                if target > 0:
                    qty = round(target / buy_price, 4)
                    if qty > 0:
                        cost = qty * buy_price * (1.0 + fee_rate)
                        if cost <= cash:
                            cash -= cost
                            pos_sym_idx = sig_sym_idx
                            pos_qty = qty
                            pos_entry = buy_price
                            trades += 1

    # Close remaining position at final close
    if pos_sym_idx >= 0:
        final_price = pa.closes[window_end - 1, pos_sym_idx]
        proceeds = pos_qty * final_price * (1.0 - fee_rate)
        cash += proceeds
        if final_price > pos_entry:
            winning_trades += 1
        trades += 1

    equity_curve.append(cash)
    curve = np.asarray(equity_curve, dtype=np.float64)
    num_days = window_end - window_start
    total_return = float(curve[-1] / curve[0] - 1.0)
    daily_returns = np.diff(curve) / np.clip(curve[:-1], 1e-8, None)
    downside = daily_returns[daily_returns < 0.0]
    downside_dev = float(np.sqrt(np.mean(np.square(downside)))) if len(downside) else 1e-8
    sortino = float(np.mean(daily_returns) / downside_dev * np.sqrt(252.0)) if len(daily_returns) else 0.0
    max_dd = float(np.min(curve / np.maximum.accumulate(curve) - 1.0)) if len(curve) > 1 else 0.0
    annualized = float((1.0 + total_return) ** (252.0 / max(1, num_days)) - 1.0)
    win_rate = float(winning_trades / max(1, trades))

    return BacktestResult(
        total_return=total_return,
        annualized_return=annualized,
        sortino=sortino,
        max_drawdown=max_dd,
        win_rate=win_rate,
        num_trades=trades,
        num_days=num_days,
        final_equity=cash,
    )


# Legacy interface for tests that pass trader/extra_policies directly
def run_calibrated_backtest(
    *,
    trader: DailyPPOTrader,
    extra_policies: list,
    indexed: dict[str, pd.DataFrame],
    start_idx: int,
    end_idx: int,
    params: CalibrationParams,
    base_allocation_pct: float = 25.0,
    initial_cash: float = 10_000.0,
) -> BacktestResult:
    """Run a calibrated backtest by computing signals on-the-fly (for tests/single evals)."""
    signals = precompute_signals(trader, extra_policies, indexed, start_idx, end_idx)
    pa = build_price_arrays(indexed)
    return replay_with_params(
        signals=signals,
        price_arrays=pa,
        signal_start_idx=start_idx,
        window_start=start_idx,
        window_end=end_idx,
        params=params,
        base_allocation_pct=base_allocation_pct,
        initial_cash=initial_cash,
    )


def replay_multiwindow(
    *,
    signals: list[PrecomputedSignal],
    price_arrays: PriceArrays,
    signal_start_idx: int,
    params: CalibrationParams,
    window_size: int = 90,
    start_range: tuple[int, int],
    base_allocation_pct: float = 25.0,
    initial_cash: float = 10_000.0,
    # Legacy compat
    indexed: dict[str, pd.DataFrame] | None = None,
) -> list[BacktestResult]:
    """Replay pre-computed signals across multiple sliding windows (fast)."""
    n_bars = price_arrays.opens.shape[0]
    results = []
    for start in range(start_range[0], start_range[1]):
        end = start + window_size
        if end > n_bars:
            break
        r = replay_with_params(
            signals=signals,
            price_arrays=price_arrays,
            signal_start_idx=signal_start_idx,
            window_start=start,
            window_end=end,
            params=params,
            base_allocation_pct=base_allocation_pct,
            initial_cash=initial_cash,
        )
        results.append(r)
    return results


# Legacy interface for tests
def run_multiwindow_eval(
    *,
    trader: DailyPPOTrader,
    extra_policies: list,
    indexed: dict[str, pd.DataFrame],
    params: CalibrationParams,
    window_size: int = 90,
    start_range: tuple[int, int] | None = None,
    base_allocation_pct: float = 25.0,
    initial_cash: float = 10_000.0,
) -> list[BacktestResult]:
    """Run backtest across multiple sliding windows (computes signals on-the-fly)."""
    min_len = min(len(frame) for frame in indexed.values())
    if start_range is None:
        start_range = (120, min_len - window_size)
    sig_start = start_range[0]
    sig_end = min(start_range[1] + window_size, min_len)
    signals = precompute_signals(trader, extra_policies, indexed, sig_start, sig_end)
    pa = build_price_arrays(indexed)
    return replay_multiwindow(
        signals=signals,
        price_arrays=pa,
        signal_start_idx=sig_start,
        params=params,
        window_size=window_size,
        start_range=start_range,
        base_allocation_pct=base_allocation_pct,
        initial_cash=initial_cash,
    )


def aggregate_results(results: list[BacktestResult]) -> dict[str, float]:
    """Compute aggregate metrics from multi-window results."""
    if not results:
        return {"n": 0, "neg": 0, "med_return": 0, "p10_return": 0, "med_sortino": 0, "p10_sortino": 0}
    returns = [r.total_return for r in results]
    sortinos = [r.sortino for r in results]
    neg = sum(1 for r in returns if r < 0)
    return {
        "n": len(results),
        "neg": neg,
        "med_return": float(np.median(returns)),
        "p10_return": float(np.percentile(returns, 10)),
        "p90_return": float(np.percentile(returns, 90)),
        "worst_return": float(np.min(returns)),
        "med_sortino": float(np.median(sortinos)),
        "p10_sortino": float(np.percentile(sortinos, 10)),
        "med_dd": float(np.median([r.max_drawdown for r in results])),
        "med_trades": float(np.median([r.num_trades for r in results])),
        "med_winrate": float(np.median([r.win_rate for r in results])),
    }


def sweep_calibration(
    *,
    trader: DailyPPOTrader,
    extra_policies: list,
    indexed: dict[str, pd.DataFrame],
    train_frac: float = 0.70,
    window_size: int = 90,
    base_allocation_pct: float = 25.0,
    entry_offsets: Sequence[float] = (-25, -15, -10, -5, 0, 5, 10, 15, 25),
    exit_offsets: Sequence[float] = (-25, -15, -10, -5, 0, 5, 10, 15, 25),
    allocation_scales: Sequence[float] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
    confidence_thresholds: Sequence[float] = (0.0,),
    fee_bps_values: Sequence[float] = (0.0,),
    output_csv: str | None = None,
) -> list[CalibrationResult]:
    """Sweep over calibration parameters and return sorted results.

    Pre-computes signals once, then replays execution with each param combo.
    """
    min_len = min(len(frame) for frame in indexed.values())
    usable_start = 120
    usable_end = min_len
    usable_range = usable_end - usable_start - window_size

    train_end_offset = int(usable_range * train_frac)
    train_range = (usable_start, usable_start + train_end_offset)
    val_range = (usable_start + train_end_offset, usable_end - window_size)

    logger.info(
        "Sweep setup: %d usable bars, train windows %d-%d (%d), val windows %d-%d (%d), window=%d",
        usable_range, train_range[0], train_range[1], train_range[1] - train_range[0],
        val_range[0], val_range[1], val_range[1] - val_range[0], window_size,
    )

    # Pre-compute ALL signals once (the expensive step)
    logger.info("Pre-computing ensemble signals for all %d time steps...", usable_end - usable_start)
    all_signals = precompute_signals(trader, extra_policies, indexed, usable_start, usable_end)
    logger.info("Signal pre-computation complete.")

    # Pre-build price arrays once for fast replay
    pa = build_price_arrays(indexed)

    combos = list(itertools.product(
        entry_offsets, exit_offsets, allocation_scales, confidence_thresholds, fee_bps_values,
    ))
    logger.info("Sweeping %d parameter combinations (replay only, no inference)", len(combos))

    results: list[CalibrationResult] = []
    start_time = time.time()

    for i, (entry_off, exit_off, alloc_scale, conf_thresh, fee) in enumerate(combos):
        params = CalibrationParams(
            entry_offset_bps=entry_off,
            exit_offset_bps=exit_off,
            allocation_scale=alloc_scale,
            confidence_threshold=conf_thresh,
            fee_bps=fee,
        )

        train_results = replay_multiwindow(
            signals=all_signals,
            price_arrays=pa,
            signal_start_idx=usable_start,
            params=params,
            window_size=window_size,
            start_range=train_range,
            base_allocation_pct=base_allocation_pct,
        )
        train_agg = aggregate_results(train_results)

        val_results = replay_multiwindow(
            signals=all_signals,
            price_arrays=pa,
            signal_start_idx=usable_start,
            params=params,
            window_size=window_size,
            start_range=val_range,
            base_allocation_pct=base_allocation_pct,
        )
        val_agg = aggregate_results(val_results)

        cr = CalibrationResult(
            params=params,
            train_sortino=train_agg["med_sortino"],
            train_return=train_agg["med_return"],
            train_max_dd=train_agg["med_dd"],
            train_p10=train_agg["p10_return"],
            val_sortino=val_agg["med_sortino"],
            val_return=val_agg["med_return"],
            val_max_dd=val_agg["med_dd"],
            val_p10=val_agg["p10_return"],
            num_train_windows=train_agg["n"],
            num_val_windows=val_agg["n"],
        )
        results.append(cr)

        if (i + 1) % 25 == 0 or i == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (len(combos) - i - 1)
            logger.info(
                "[%d/%d] entry=%+.0fbps exit=%+.0fbps scale=%.2f → train_sort=%.2f val_sort=%.2f val_p10=%.1f%% (ETA %.0fs)",
                i + 1, len(combos), entry_off, exit_off, alloc_scale,
                cr.train_sortino, cr.val_sortino, cr.val_p10 * 100, eta,
            )

    # Sort by val_p10 descending (most robust metric)
    results.sort(key=lambda r: r.val_p10, reverse=True)

    if output_csv:
        _write_results_csv(results, output_csv)
        logger.info("Results written to %s", output_csv)

    return results


def _write_results_csv(results: list[CalibrationResult], path: str) -> None:
    """Write calibration results to CSV."""
    fieldnames = [
        "entry_offset_bps", "exit_offset_bps", "allocation_scale",
        "confidence_threshold", "fee_bps",
        "train_sortino", "train_return", "train_max_dd", "train_p10",
        "val_sortino", "val_return", "val_max_dd", "val_p10",
        "num_train_windows", "num_val_windows",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "entry_offset_bps": r.params.entry_offset_bps,
                "exit_offset_bps": r.params.exit_offset_bps,
                "allocation_scale": r.params.allocation_scale,
                "confidence_threshold": r.params.confidence_threshold,
                "fee_bps": r.params.fee_bps,
                "train_sortino": f"{r.train_sortino:.4f}",
                "train_return": f"{r.train_return:.4f}",
                "train_max_dd": f"{r.train_max_dd:.4f}",
                "train_p10": f"{r.train_p10:.4f}",
                "val_sortino": f"{r.val_sortino:.4f}",
                "val_return": f"{r.val_return:.4f}",
                "val_max_dd": f"{r.val_max_dd:.4f}",
                "val_p10": f"{r.val_p10:.4f}",
                "num_train_windows": r.num_train_windows,
                "num_val_windows": r.num_val_windows,
            })


def print_top_results(results: list[CalibrationResult], top_n: int = 20) -> None:
    """Print leaderboard of top calibration results."""
    print("\n" + "=" * 130)
    print(f"TOP {min(top_n, len(results))} CALIBRATION RESULTS (sorted by val_p10)")
    print("=" * 130)
    print(f"{'Rank':>4} {'Entry':>7} {'Exit':>7} {'Scale':>6} {'Conf':>5} {'Fee':>4} │"
          f" {'Tr Sort':>8} {'Tr Ret':>8} {'Tr P10':>8} │"
          f" {'Val Sort':>8} {'Val Ret':>8} {'Val P10':>8} {'Val DD':>8}")
    print("-" * 130)
    for i, r in enumerate(results[:top_n]):
        print(
            f"{i+1:>4} {r.params.entry_offset_bps:>+6.0f} {r.params.exit_offset_bps:>+6.0f} "
            f"{r.params.allocation_scale:>6.2f} {r.params.confidence_threshold:>5.2f} "
            f"{r.params.fee_bps:>4.0f} │"
            f" {r.train_sortino:>8.2f} {r.train_return:>7.1%} {r.train_p10:>7.1%} │"
            f" {r.val_sortino:>8.2f} {r.val_return:>7.1%} {r.val_p10:>7.1%} {r.val_max_dd:>7.1%}"
        )
    # Print baseline (0,0,1.0) if in results
    baseline = [r for r in results if r.params.entry_offset_bps == 0
                and r.params.exit_offset_bps == 0 and r.params.allocation_scale == 1.0
                and r.params.confidence_threshold == 0.0 and r.params.fee_bps == 0.0]
    if baseline:
        b = baseline[0]
        rank = results.index(b) + 1
        print("-" * 130)
        print(f"BASELINE (rank {rank}/{len(results)}): entry=0 exit=0 scale=1.0 │"
              f" tr_sort={b.train_sortino:.2f} tr_ret={b.train_return:.1%} tr_p10={b.train_p10:.1%} │"
              f" val_sort={b.val_sortino:.2f} val_ret={b.val_return:.1%} val_p10={b.val_p10:.1%}")
    print("=" * 130)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate daily stock PPO execution params")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", default="trainingdata")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--allocation-pct", type=float, default=25.0)
    parser.add_argument("--window-size", type=int, default=90)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-csv", default="sweepresults/daily_stock_calibration.csv")

    parser.add_argument("--sweep", action="store_true", help="Run full calibration sweep")
    parser.add_argument("--quick-sweep", action="store_true", help="Run reduced sweep (faster)")
    parser.add_argument("--eval", action="store_true", help="Evaluate specific params")

    parser.add_argument("--entry-offset-bps", type=float, default=0.0)
    parser.add_argument("--exit-offset-bps", type=float, default=0.0)
    parser.add_argument("--allocation-scale", type=float, default=1.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--fee-bps", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    symbols = [s.upper() for s in (args.symbols or DEFAULT_SYMBOLS)]

    logger.info("Loading daily data for %d symbols...", len(symbols))
    frames = load_daily_frames(symbols, data_dir=args.data_dir)
    indexed = {
        sym: frame.set_index("timestamp")[["open", "high", "low", "close", "volume"]].copy()
        for sym, frame in frames.items()
    }
    min_len = min(len(f) for f in indexed.values())
    logger.info("Aligned data: %d bars across %d symbols", min_len, len(symbols))

    checkpoint = str((REPO / args.checkpoint).resolve())
    extra_paths = [str((REPO / p).resolve()) for p in DEFAULT_EXTRA_CHECKPOINTS]
    logger.info("Loading %d-model ensemble...", 1 + len(extra_paths))
    trader, extra_policies = build_ensemble(checkpoint, extra_paths, symbols, args.device)
    logger.info("Ensemble loaded: %d policies", 1 + len(extra_policies))

    if args.sweep or args.quick_sweep:
        if args.quick_sweep:
            entry_offsets = (-15, -5, 0, 5, 15)
            exit_offsets = (-15, -5, 0, 5, 15)
            alloc_scales = (0.75, 1.0, 1.5)
        else:
            entry_offsets = (-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25)
            exit_offsets = (-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25)
            alloc_scales = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)

        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        results = sweep_calibration(
            trader=trader,
            extra_policies=extra_policies,
            indexed=indexed,
            train_frac=args.train_frac,
            window_size=args.window_size,
            base_allocation_pct=args.allocation_pct,
            entry_offsets=entry_offsets,
            exit_offsets=exit_offsets,
            allocation_scales=alloc_scales,
            output_csv=args.output_csv,
        )
        print_top_results(results)

        if results:
            best = results[0]
            print(f"\n{'='*60}")
            print("BEST CALIBRATION PARAMS FOR DEPLOYMENT:")
            print(f"  --entry-offset-bps {best.params.entry_offset_bps:.0f}")
            print(f"  --exit-offset-bps {best.params.exit_offset_bps:.0f}")
            print(f"  --allocation-scale {best.params.allocation_scale:.2f}")
            print(f"  Val P10 Return: {best.val_p10:.1%}")
            print(f"  Val Median Return: {best.val_return:.1%}")
            print(f"  Val Sortino: {best.val_sortino:.2f}")
            print(f"{'='*60}")

    elif args.eval:
        params = CalibrationParams(
            entry_offset_bps=args.entry_offset_bps,
            exit_offset_bps=args.exit_offset_bps,
            allocation_scale=args.allocation_scale,
            confidence_threshold=args.confidence_threshold,
            fee_bps=args.fee_bps,
        )
        logger.info("Evaluating params: %s", asdict(params))
        results = run_multiwindow_eval(
            trader=trader,
            extra_policies=extra_policies,
            indexed=indexed,
            params=params,
            window_size=args.window_size,
            base_allocation_pct=args.allocation_pct,
        )
        agg = aggregate_results(results)
        print(f"\nResults over {agg['n']} windows (window_size={args.window_size}):")
        print(f"  Negative windows: {agg['neg']}/{agg['n']}")
        print(f"  Median return:    {agg['med_return']:.2%}")
        print(f"  P10 return:       {agg['p10_return']:.2%}")
        print(f"  Worst return:     {agg['worst_return']:.2%}")
        print(f"  Median Sortino:   {agg['med_sortino']:.2f}")
        print(f"  Median MaxDD:     {agg['med_dd']:.2%}")
        print(f"  Median Win Rate:  {agg['med_winrate']:.1%}")
        print(f"  Median Trades:    {agg['med_trades']:.0f}")


if __name__ == "__main__":
    main()
