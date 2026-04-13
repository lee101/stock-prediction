#!/usr/bin/env python3
"""Meta-strategy backtest: forecast per-model PnL, pick top-K, follow their signals.

Instead of running a single softmax ensemble, we:
1. Simulate each model individually over daily bars
2. Track per-model daily equity curves
3. At each day, select top-K models via trailing momentum or Chronos2 forecast
4. The meta-portfolio follows the selected models' signals (which symbols to be long)
5. Compare vs the ensemble baseline

Usage:
    python scripts/meta_strategy_backtest.py --data-dir trainingdata --top-k 2
    python scripts/meta_strategy_backtest.py --data-dir trainingdata --top-k 2 --use-chronos2
"""
from __future__ import annotations

import argparse
import json
import logging
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pufferlib_market.evaluate_holdout import load_policy, _mask_all_shorts
from pufferlib_market.hourly_replay import (
    DailySimResult,
    MktdData,
    Position,
    read_mktd,
    simulate_daily_policy,
    _build_obs,
    _compute_equity,
    _close_position,
    _open_long_limit,
    INITIAL_CASH,
    P_OPEN,
    P_HIGH,
    P_LOW,
    P_CLOSE,
    P_VOL,
)
from pufferlib_market.inference_daily import compute_daily_features as compute_16_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("meta_strategy")

DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "GOOG", "META", "TSLA",
    "SPY", "QQQ", "PLTR", "JPM", "V", "AMZN",
]

ENSEMBLE_DIR = REPO / "pufferlib_market" / "prod_ensemble"


@dataclass
class ModelTrace:
    name: str
    actions: np.ndarray       # [T] int - action at each day
    equity_curve: np.ndarray  # [T+1] float - equity at each day
    held_symbols: list        # [T] str|None - which symbol is held after action
    daily_returns: np.ndarray # [T] float


@dataclass
class MetaResult:
    total_return: float
    sortino: float
    max_drawdown: float
    num_trades: int
    daily_returns: np.ndarray
    equity_curve: np.ndarray
    selected_models: list     # per-day list of selected model names
    held_symbols: list        # per-day list of held symbols


def build_mktd_from_csvs(
    data_dir: Path,
    symbols: list[str],
    min_days: int = 120,
) -> tuple[MktdData, pd.DatetimeIndex]:
    """Build MKTD-like arrays directly from daily CSV files."""
    dfs = {}
    for sym in symbols:
        csv_path = data_dir / f"{sym}.csv"
        if not csv_path.exists():
            log.warning("missing %s", csv_path)
            continue
        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        if len(df) >= min_days:
            dfs[sym] = df.set_index("timestamp")

    available = [s for s in symbols if s in dfs]
    if not available:
        raise ValueError(f"No symbol data found in {data_dir}")
    log.info("loaded %d/%d symbols", len(available), len(symbols))

    all_dates = None
    for sym in available:
        dates = dfs[sym].index.normalize().unique()
        all_dates = dates if all_dates is None else all_dates.intersection(dates)
    all_dates = all_dates.sort_values()
    log.info("common dates: %d (from %s to %s)", len(all_dates), all_dates[0].date(), all_dates[-1].date())

    S = len(available)
    T = len(all_dates)
    features = np.zeros((T, S, 16), dtype=np.float32)
    prices = np.zeros((T, S, 5), dtype=np.float32)

    for si, sym in enumerate(available):
        df = dfs[sym]
        # For each date, get the row and compute features
        for ti, date in enumerate(all_dates):
            mask = df.index.normalize() == date
            if mask.any():
                row = df[mask].iloc[-1]
                prices[ti, si, P_OPEN] = float(row.get("open", 0))
                prices[ti, si, P_HIGH] = float(row.get("high", 0))
                prices[ti, si, P_LOW] = float(row.get("low", 0))
                prices[ti, si, P_CLOSE] = float(row.get("close", 0))
                prices[ti, si, P_VOL] = float(row.get("volume", 0))

        # Compute rolling features using the full history up to each date
        df_aligned = df[df.index.normalize().isin(all_dates)].copy()
        df_aligned = df_aligned[~df_aligned.index.duplicated(keep="last")]
        df_aligned = df_aligned.reindex(all_dates, method="ffill")
        if len(df_aligned) >= 60:
            feat_df = pd.DataFrame({
                "open": df_aligned["open"].values,
                "high": df_aligned["high"].values,
                "low": df_aligned["low"].values,
                "close": df_aligned["close"].values,
                "volume": df_aligned["volume"].values,
            })
            # Compute features for all days at once using rolling windows
            close = feat_df["close"].astype(float)
            high = feat_df["high"].astype(float)
            low = feat_df["low"].astype(float)
            volume = feat_df["volume"].astype(float)

            ret_1d = close.pct_change(1).fillna(0.0).clip(-0.5, 0.5)
            ret_5d = close.pct_change(5).fillna(0.0).clip(-1.0, 1.0)
            ret_20d = close.pct_change(20).fillna(0.0).clip(-2.0, 2.0)
            vol_5d = ret_1d.rolling(5, min_periods=1).std(ddof=0).fillna(0.01).clip(0.0, 1.0)
            vol_20d = ret_1d.rolling(20, min_periods=1).std(ddof=0).fillna(0.01).clip(0.0, 1.0)
            ma5 = close.rolling(5, min_periods=1).mean()
            ma20 = close.rolling(20, min_periods=1).mean()
            ma60 = close.rolling(60, min_periods=1).mean()
            ma_delta_5d = ((close - ma5) / ma5.clip(lower=1e-8)).fillna(0.0).clip(-0.5, 0.5)
            ma_delta_20d = ((close - ma20) / ma20.clip(lower=1e-8)).fillna(0.0).clip(-0.5, 0.5)
            ma_delta_60d = ((close - ma60) / ma60.clip(lower=1e-8)).fillna(0.0).clip(-0.5, 0.5)
            tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
            atr14 = tr.rolling(14, min_periods=1).mean()
            atr_pct_14d = (atr14 / close.clip(lower=1e-8)).fillna(0.0).clip(0.0, 0.5)
            range_pct_1d = ((high - low) / close.clip(lower=1e-8)).fillna(0.0).clip(0.0, 0.5)
            delta = close.diff()
            gain = delta.clip(lower=0.0).rolling(14, min_periods=1).mean()
            loss = (-delta.clip(upper=0.0)).rolling(14, min_periods=1).mean()
            rs = gain / loss.clip(lower=1e-8)
            rsi_14 = (2.0 * (100.0 - 100.0 / (1.0 + rs)) / 100.0 - 1.0).fillna(0.0).clip(-1.0, 1.0)
            trend_60d = close.pct_change(60).fillna(0.0).clip(-3.0, 3.0)
            roll_max_20 = close.rolling(20, min_periods=1).max()
            roll_max_60 = close.rolling(60, min_periods=1).max()
            drawdown_20d = ((close - roll_max_20) / roll_max_20.clip(lower=1e-8)).fillna(0.0).clip(-1.0, 0.0)
            drawdown_60d = ((close - roll_max_60) / roll_max_60.clip(lower=1e-8)).fillna(0.0).clip(-1.0, 0.0)
            log_vol = np.log1p(volume.clip(lower=0.0))
            log_vol_mean20 = log_vol.rolling(20, min_periods=1).mean()
            log_vol_std20 = log_vol.rolling(20, min_periods=1).std(ddof=0).replace(0.0, 1.0)
            log_volume_z20d = ((log_vol - log_vol_mean20) / log_vol_std20).fillna(0.0).clip(-5.0, 5.0)
            log_volume_delta_5d = (log_vol - log_vol.rolling(5, min_periods=1).mean()).fillna(0.0).clip(-10.0, 10.0)

            all_feats = np.stack([
                ret_1d.values, ret_5d.values, ret_20d.values,
                vol_5d.values, vol_20d.values,
                ma_delta_5d.values, ma_delta_20d.values, ma_delta_60d.values,
                atr_pct_14d.values, range_pct_1d.values,
                rsi_14.values, trend_60d.values,
                drawdown_20d.values, drawdown_60d.values,
                log_volume_z20d.values, log_volume_delta_5d.values,
            ], axis=1).astype(np.float32)
            features[:, si, :] = all_feats[:T]

    tradable = np.ones((T, S), dtype=np.uint8)
    data = MktdData(version=2, symbols=available, features=features, prices=prices, tradable=tradable)
    return data, all_dates


def simulate_single_model(
    data: MktdData,
    checkpoint_path: str | Path,
    *,
    fee_rate: float = 0.001,
    slippage_bps: float = 0.0,
    fill_buffer_bps: float = 5.0,
    decision_lag: int = 2,
    device: str = "cpu",
) -> ModelTrace:
    """Run a single model through the full MKTD data, recording per-day state."""
    S = data.num_symbols
    T = data.num_timesteps
    max_steps = T - 1

    loaded = load_policy(checkpoint_path, S, features_per_sym=16, device=torch.device(device))
    policy = loaded.policy

    import collections
    pending = collections.deque(maxlen=max(1, decision_lag + 1))

    def policy_fn(obs):
        obs_t = torch.from_numpy(obs.astype(np.float32)).to(device).view(1, -1)
        with torch.no_grad():
            logits, _ = policy(obs_t)
        logits = _mask_all_shorts(logits, num_symbols=S, per_symbol_actions=1)
        action_now = int(torch.argmax(logits, dim=-1).item())
        if decision_lag <= 0:
            return action_now
        pending.append(action_now)
        if len(pending) <= decision_lag:
            return 0
        return pending.popleft()

    result = simulate_daily_policy(
        data, policy_fn,
        max_steps=max_steps,
        fee_rate=fee_rate,
        slippage_bps=slippage_bps,
        fill_buffer_bps=fill_buffer_bps,
        max_leverage=1.0,
        periods_per_year=252.0,
        enable_drawdown_profit_early_exit=False,
    )

    eq = np.array(result.equity_curve) if result.equity_curve is not None else np.ones(max_steps + 1) * INITIAL_CASH
    daily_returns = np.diff(eq) / np.maximum(eq[:-1], 1e-8)

    # Decode actions to held symbols
    held = []
    for a in result.actions[:max_steps]:
        a = int(a)
        if a == 0:
            held.append(None)
        elif 1 <= a <= S:
            held.append(data.symbols[a - 1])
        else:
            held.append(None)

    name = Path(checkpoint_path).stem
    return ModelTrace(
        name=name,
        actions=result.actions[:max_steps],
        equity_curve=eq,
        held_symbols=held,
        daily_returns=daily_returns,
    )


def simulate_ensemble(
    data: MktdData,
    checkpoint_paths: list[str | Path],
    *,
    fee_rate: float = 0.001,
    slippage_bps: float = 0.0,
    fill_buffer_bps: float = 5.0,
    decision_lag: int = 2,
    device: str = "cpu",
) -> ModelTrace:
    """Run the softmax ensemble (baseline)."""
    S = data.num_symbols
    T = data.num_timesteps
    max_steps = T - 1

    policies = []
    for cp in checkpoint_paths:
        loaded = load_policy(cp, S, features_per_sym=16, device=torch.device(device))
        policies.append(loaded.policy)

    import collections
    pending = collections.deque(maxlen=max(1, decision_lag + 1))

    def policy_fn(obs):
        obs_t = torch.from_numpy(obs.astype(np.float32)).to(device).view(1, -1)
        with torch.no_grad():
            probs_sum = None
            for p in policies:
                lg, _ = p(obs_t)
                pr = torch.softmax(lg, dim=-1)
                probs_sum = pr if probs_sum is None else probs_sum + pr
            logits = torch.log(probs_sum / len(policies) + 1e-8)
        logits = _mask_all_shorts(logits, num_symbols=S, per_symbol_actions=1)
        action_now = int(torch.argmax(logits, dim=-1).item())
        if decision_lag <= 0:
            return action_now
        pending.append(action_now)
        if len(pending) <= decision_lag:
            return 0
        return pending.popleft()

    result = simulate_daily_policy(
        data, policy_fn,
        max_steps=max_steps,
        fee_rate=fee_rate,
        slippage_bps=slippage_bps,
        fill_buffer_bps=fill_buffer_bps,
        max_leverage=1.0,
        periods_per_year=252.0,
        enable_drawdown_profit_early_exit=False,
    )

    eq = np.array(result.equity_curve) if result.equity_curve is not None else np.ones(max_steps + 1) * INITIAL_CASH
    daily_returns = np.diff(eq) / np.maximum(eq[:-1], 1e-8)
    held = []
    for a in result.actions[:max_steps]:
        a = int(a)
        if a == 0:
            held.append(None)
        elif 1 <= a <= S:
            held.append(data.symbols[a - 1])
        else:
            held.append(None)

    return ModelTrace(
        name="ensemble",
        actions=result.actions[:max_steps],
        equity_curve=eq,
        held_symbols=held,
        daily_returns=daily_returns,
    )


def select_top_k_momentum(
    traces: list[ModelTrace],
    day_idx: int,
    lookback: int = 20,
    top_k: int = 2,
) -> list[int]:
    """Select top-K models by trailing momentum (cumulative return over lookback).

    equity_curve[t] = equity valued at day t prices (available at end of day t).
    We use end=day_idx so we only see equity through TODAY, not tomorrow.
    """
    scores = []
    for i, tr in enumerate(traces):
        start = max(0, day_idx - lookback)
        end = day_idx  # no lookahead: equity through today's close
        if end >= len(tr.equity_curve) or start >= end:
            scores.append((i, 0.0))
            continue
        ret = (tr.equity_curve[end] - tr.equity_curve[start]) / max(tr.equity_curve[start], 1e-8)
        scores.append((i, ret))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scores[:top_k]]


def select_top_k_ema_momentum(
    traces: list[ModelTrace],
    day_idx: int,
    lookback: int = 5,
    top_k: int = 1,
    halflife: int = 2,
) -> list[int]:
    """Select top-K models by exponentially-weighted momentum (recent days weighted more)."""
    scores = []
    for i, tr in enumerate(traces):
        start = max(0, day_idx - lookback)
        end = day_idx  # no lookahead
        if end >= len(tr.equity_curve) or start >= end - 1:
            scores.append((i, 0.0))
            continue
        daily_rets = []
        for d in range(start + 1, end):
            if tr.equity_curve[d - 1] > 0:
                daily_rets.append((tr.equity_curve[d] - tr.equity_curve[d - 1]) / tr.equity_curve[d - 1])
            else:
                daily_rets.append(0.0)
        if not daily_rets:
            scores.append((i, 0.0))
            continue
        decay = np.log(2) / max(halflife, 1)
        weights = np.array([np.exp(decay * j) for j in range(len(daily_rets))])
        weights /= weights.sum()
        score = float(np.dot(weights, daily_rets))
        scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scores[:top_k]]


def select_top_k_sortino(
    traces: list[ModelTrace],
    day_idx: int,
    lookback: int = 10,
    top_k: int = 1,
) -> list[int]:
    """Select top-K models by trailing Sortino ratio."""
    scores = []
    for i, tr in enumerate(traces):
        start = max(0, day_idx - lookback)
        end = day_idx  # no lookahead
        if end >= len(tr.equity_curve) or start >= end - 1:
            scores.append((i, 0.0))
            continue
        daily_rets = []
        for d in range(start + 1, end):
            if tr.equity_curve[d - 1] > 0:
                daily_rets.append((tr.equity_curve[d] - tr.equity_curve[d - 1]) / tr.equity_curve[d - 1])
            else:
                daily_rets.append(0.0)
        if len(daily_rets) < 2:
            scores.append((i, 0.0))
            continue
        arr = np.array(daily_rets)
        mean_ret = arr.mean()
        downside = arr[arr < 0]
        if len(downside) == 0:
            sortino = mean_ret * 100  # all positive = very high score
        else:
            dd = float(np.sqrt(np.mean(downside**2)))
            sortino = mean_ret / max(dd, 1e-8)
        scores.append((i, sortino))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scores[:top_k]]


def select_top_k_chronos2(
    traces: list[ModelTrace],
    day_idx: int,
    lookback: int = 60,
    top_k: int = 2,
    wrapper=None,
) -> list[int]:
    """Select top-K models by Chronos2 forecast of their PnL curves."""
    if wrapper is None:
        return select_top_k_momentum(traces, day_idx, lookback=lookback, top_k=top_k)

    scores = []
    for i, tr in enumerate(traces):
        start = max(0, day_idx - lookback + 1)
        end = day_idx  # no lookahead
        if end >= len(tr.equity_curve):
            scores.append((i, 0.0))
            continue
        eq_slice = tr.equity_curve[start:end + 1]
        if len(eq_slice) < 10:
            # Not enough history for Chronos2, fall back to momentum
            ret = (eq_slice[-1] - eq_slice[0]) / max(eq_slice[0], 1e-8)
            scores.append((i, ret))
            continue

        # Build a simple dataframe for Chronos2
        df = pd.DataFrame({
            "close": eq_slice,
            "open": eq_slice,
            "high": eq_slice,
            "low": eq_slice,
        })
        try:
            pred = wrapper.predict_ohlc(
                df,
                prediction_length=1,
                quantile_levels=(0.5,),
            )
            predicted_close = float(pred.median["close"].iloc[0])
            current = float(eq_slice[-1])
            forecast_ret = (predicted_close - current) / max(current, 1e-8)
            scores.append((i, forecast_ret))
        except Exception as e:
            log.warning("chronos2 failed for model %s: %s", tr.name, e)
            ret = (eq_slice[-1] - eq_slice[0]) / max(eq_slice[0], 1e-8)
            scores.append((i, ret))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scores[:top_k]]


def run_meta_portfolio(
    data: MktdData,
    traces: list[ModelTrace],
    *,
    top_k: int = 2,
    lookback: int = 20,
    warmup: int = 20,
    fee_rate: float = 0.001,
    slippage_bps: float = 0.0,
    selector: str = "momentum",
    chronos2_wrapper=None,
) -> MetaResult:
    """Simulate the meta-portfolio that selects top-K models daily."""
    S = data.num_symbols
    T = len(traces[0].daily_returns)  # number of trading days
    slip = max(0.0, slippage_bps) / 10_000.0
    eff_fee = fee_rate + slip

    cash = INITIAL_CASH
    positions: dict[str, float] = {}  # symbol -> qty
    entry_prices: dict[str, float] = {}

    equity_history = [float(cash)]
    daily_rets = []
    selected_models_history = []
    held_symbols_history = []
    num_trades = 0

    for t in range(T):
        # Current equity before any action
        eq_before = cash
        for sym, qty in positions.items():
            si = data.symbols.index(sym) if sym in data.symbols else -1
            if si >= 0 and t + 1 < data.num_timesteps:
                price = float(data.prices[t, si, P_CLOSE])
                eq_before += qty * price

        if t < warmup:
            # During warmup, stay flat
            selected_models_history.append([])
            held_symbols_history.append(list(positions.keys()))
            equity_history.append(eq_before)
            daily_rets.append(0.0)
            continue

        # Select top-K models
        if selector == "chronos2":
            selected = select_top_k_chronos2(traces, t, lookback=lookback, top_k=top_k, wrapper=chronos2_wrapper)
        elif selector == "ema":
            selected = select_top_k_ema_momentum(traces, t, lookback=lookback, top_k=top_k)
        elif selector == "sortino":
            selected = select_top_k_sortino(traces, t, lookback=lookback, top_k=top_k)
        else:
            selected = select_top_k_momentum(traces, t, lookback=lookback, top_k=top_k)

        selected_names = [traces[i].name for i in selected]
        selected_models_history.append(selected_names)

        # Get target symbols from selected models
        target_symbols = set()
        for i in selected:
            sym = traces[i].held_symbols[t] if t < len(traces[i].held_symbols) else None
            if sym is not None:
                target_symbols.add(sym)

        # Rebalance: close positions not in target, open positions that are new
        alloc_per_sym = 1.0 / max(len(target_symbols), 1) if target_symbols else 0.0

        # Close unwanted positions
        for sym in list(positions.keys()):
            if sym not in target_symbols:
                si = data.symbols.index(sym) if sym in data.symbols else -1
                if si >= 0:
                    price = float(data.prices[t, si, P_CLOSE])
                    proceeds = positions[sym] * price * (1.0 - eff_fee)
                    cash += proceeds
                    num_trades += 1
                del positions[sym]
                entry_prices.pop(sym, None)

        # Open new positions
        for sym in target_symbols:
            if sym not in positions:
                si = data.symbols.index(sym) if sym in data.symbols else -1
                if si >= 0:
                    price = float(data.prices[t, si, P_CLOSE])
                    budget = cash * alloc_per_sym
                    if budget > 10.0 and price > 0:
                        qty = budget / (price * (1.0 + eff_fee))
                        cost = qty * price * (1.0 + eff_fee)
                        cash -= cost
                        positions[sym] = qty
                        entry_prices[sym] = price
                        num_trades += 1

        # End of day equity (using next day's close for mark-to-market)
        t_mark = min(t + 1, data.num_timesteps - 1)
        eq_after = cash
        for sym, qty in positions.items():
            si = data.symbols.index(sym) if sym in data.symbols else -1
            if si >= 0:
                price = float(data.prices[t_mark, si, P_CLOSE])
                eq_after += qty * price

        daily_ret = (eq_after - eq_before) / max(eq_before, 1e-8) if eq_before > 1e-8 else 0.0
        daily_rets.append(daily_ret)
        equity_history.append(eq_after)
        held_symbols_history.append(list(positions.keys()))

    # Final close
    for sym in list(positions.keys()):
        si = data.symbols.index(sym) if sym in data.symbols else -1
        if si >= 0:
            price = float(data.prices[-1, si, P_CLOSE])
            cash += positions[sym] * price * (1.0 - eff_fee)
            num_trades += 1
        del positions[sym]

    eq_arr = np.array(equity_history)
    ret_arr = np.array(daily_rets)
    total_return = (eq_arr[-1] - eq_arr[0]) / max(eq_arr[0], 1e-8)

    # Sortino
    sortino = 0.0
    if len(ret_arr) > 1:
        neg_rets = ret_arr[ret_arr < 0]
        if len(neg_rets) > 0:
            downside_dev = float(np.sqrt(np.mean(neg_rets ** 2)))
            if downside_dev > 1e-12:
                sortino = float(np.mean(ret_arr) / downside_dev * np.sqrt(252.0))

    # Max drawdown
    peak = np.maximum.accumulate(eq_arr)
    dd = (peak - eq_arr) / np.maximum(peak, 1e-8)
    max_dd = float(np.max(dd))

    return MetaResult(
        total_return=total_return,
        sortino=sortino,
        max_drawdown=max_dd,
        num_trades=num_trades,
        daily_returns=ret_arr,
        equity_curve=eq_arr,
        selected_models=selected_models_history,
        held_symbols=held_symbols_history,
    )


def main():
    parser = argparse.ArgumentParser(description="Meta-strategy backtest")
    parser.add_argument("--data-dir", type=str, default="trainingdata")
    parser.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--ensemble-dir", type=str, default=str(ENSEMBLE_DIR))
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument("--fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--decision-lag", type=int, default=2)
    parser.add_argument("--use-chronos2", action="store_true")
    parser.add_argument("--selector", type=str, default="momentum",
                        choices=["momentum", "ema", "sortino", "chronos2"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--max-models", type=int, default=0, help="Limit number of models to test (0=all)")
    parser.add_argument("--eval-days", type=int, default=0, help="Limit eval to last N days (0=all)")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    data_dir = Path(args.data_dir)
    ensemble_dir = Path(args.ensemble_dir)

    # Load data
    log.info("building MKTD from CSVs...")
    data, dates = build_mktd_from_csvs(data_dir, symbols)
    log.info("data: %d days x %d symbols", data.num_timesteps, data.num_symbols)

    if args.eval_days > 0 and args.eval_days < data.num_timesteps:
        start = data.num_timesteps - args.eval_days
        data = MktdData(
            version=data.version,
            symbols=list(data.symbols),
            features=data.features[start:].copy(),
            prices=data.prices[start:].copy(),
            tradable=data.tradable[start:].copy() if data.tradable is not None else None,
        )
        dates = dates[start:]
        log.info("trimmed to last %d days", args.eval_days)

    # Discover checkpoints
    checkpoints = sorted(ensemble_dir.glob("*.pt"))
    if args.max_models > 0:
        checkpoints = checkpoints[:args.max_models]
    if not checkpoints:
        log.error("no checkpoints found in %s", ensemble_dir)
        return
    log.info("found %d checkpoints", len(checkpoints))

    # Phase 1: simulate each model individually
    traces = []
    for i, cp in enumerate(checkpoints):
        log.info("[%d/%d] simulating %s...", i + 1, len(checkpoints), cp.stem)
        try:
            tr = simulate_single_model(
                data, cp,
                fee_rate=args.fee_rate,
                slippage_bps=args.slippage_bps,
                fill_buffer_bps=args.fill_buffer_bps,
                decision_lag=args.decision_lag,
                device=args.device,
            )
            traces.append(tr)
            log.info("  %s: return=%.2f%% sortino=n/a trades=%d",
                     tr.name, tr.equity_curve[-1] / tr.equity_curve[0] * 100 - 100, int(np.sum(tr.actions > 0)))
        except Exception as e:
            log.warning("  %s failed: %s", cp.stem, e)

    if len(traces) < 2:
        log.error("need at least 2 models for meta-strategy, got %d", len(traces))
        return

    # Phase 2: simulate ensemble baseline
    log.info("simulating %d-model ensemble baseline...", len(checkpoints))
    ensemble_trace = simulate_ensemble(
        data, checkpoints,
        fee_rate=args.fee_rate,
        slippage_bps=args.slippage_bps,
        fill_buffer_bps=args.fill_buffer_bps,
        decision_lag=args.decision_lag,
        device=args.device,
    )
    ens_ret = ensemble_trace.equity_curve[-1] / ensemble_trace.equity_curve[0] * 100 - 100
    log.info("ensemble: return=%.2f%%", ens_ret)

    # Phase 3: meta-strategy with selected method
    sel = args.selector if not args.use_chronos2 else "momentum"
    log.info("running meta-strategy (%s, top-%d, lookback=%d)...", sel, args.top_k, args.lookback)
    meta_mom = run_meta_portfolio(
        data, traces,
        top_k=args.top_k,
        lookback=args.lookback,
        warmup=args.warmup,
        fee_rate=args.fee_rate,
        slippage_bps=args.slippage_bps,
        selector=sel,
    )

    # Phase 4: optionally run with Chronos2
    meta_c2 = None
    if args.use_chronos2:
        log.info("running meta-strategy with Chronos2 selection...")
        try:
            from src.models.chronos2_wrapper import Chronos2OHLCWrapper
            wrapper = Chronos2OHLCWrapper.from_pretrained("amazon/chronos-2", device_map=args.device)
            meta_c2 = run_meta_portfolio(
                data, traces,
                top_k=args.top_k,
                lookback=60,
                warmup=args.warmup,
                fee_rate=args.fee_rate,
                slippage_bps=args.slippage_bps,
                selector="chronos2",
                chronos2_wrapper=wrapper,
            )
        except Exception as e:
            log.warning("chronos2 meta failed: %s", e)

    # Report
    print("\n" + "=" * 70)
    print("META-STRATEGY BACKTEST RESULTS")
    print("=" * 70)
    print(f"Data: {data.num_timesteps} days, {data.num_symbols} symbols")
    print(f"Models: {len(traces)}")
    print(f"Fee: {args.fee_rate*100:.1f}bps, Slippage: {args.slippage_bps:.0f}bps, Fill buffer: {args.fill_buffer_bps:.0f}bps")
    print(f"Decision lag: {args.decision_lag}")
    print()

    # Per-model results
    print("PER-MODEL INDIVIDUAL RESULTS:")
    print(f"{'Model':<20} {'Return%':>10} {'Trades':>8}")
    print("-" * 40)
    for tr in sorted(traces, key=lambda t: t.equity_curve[-1], reverse=True):
        ret = tr.equity_curve[-1] / tr.equity_curve[0] * 100 - 100
        trades = int(np.sum(np.diff(tr.actions.astype(int)) != 0))
        print(f"{tr.name:<20} {ret:>+10.2f}% {trades:>8}")

    print()
    monthly_days = 21.0

    def _monthly(total_ret, n_days):
        if n_days <= 0:
            return 0.0
        months = n_days / monthly_days
        if months <= 0:
            return 0.0
        return ((1 + total_ret) ** (1.0 / months) - 1) * 100

    n_days = data.num_timesteps
    print("COMPARISON:")
    print(f"{'Strategy':<30} {'Total%':>10} {'Monthly%':>10} {'Sortino':>10} {'MaxDD%':>10} {'Trades':>8}")
    print("-" * 80)

    # Best individual model
    best_tr = max(traces, key=lambda t: t.equity_curve[-1])
    best_ret = best_tr.equity_curve[-1] / best_tr.equity_curve[0] - 1
    best_dr = np.diff(best_tr.equity_curve) / np.maximum(best_tr.equity_curve[:-1], 1e-8)
    neg = best_dr[best_dr < 0]
    best_sort = float(np.mean(best_dr) / np.sqrt(np.mean(neg**2)) * np.sqrt(252)) if len(neg) > 0 else 0
    best_peak = np.maximum.accumulate(best_tr.equity_curve)
    best_dd = float(np.max((best_peak - best_tr.equity_curve) / np.maximum(best_peak, 1e-8)))
    print(f"{'Best model ('+best_tr.name+')':<30} {best_ret*100:>+10.2f}% {_monthly(best_ret, n_days):>+10.2f}% {best_sort:>10.2f} {best_dd*100:>10.2f}% {'-':>8}")

    # Ensemble
    ens_ret_f = ensemble_trace.equity_curve[-1] / ensemble_trace.equity_curve[0] - 1
    ens_dr = np.diff(ensemble_trace.equity_curve) / np.maximum(ensemble_trace.equity_curve[:-1], 1e-8)
    neg = ens_dr[ens_dr < 0]
    ens_sort = float(np.mean(ens_dr) / np.sqrt(np.mean(neg**2)) * np.sqrt(252)) if len(neg) > 0 else 0
    ens_peak = np.maximum.accumulate(ensemble_trace.equity_curve)
    ens_dd = float(np.max((ens_peak - ensemble_trace.equity_curve) / np.maximum(ens_peak, 1e-8)))
    print(f"{'Ensemble ('+str(len(checkpoints))+'m)':<30} {ens_ret_f*100:>+10.2f}% {_monthly(ens_ret_f, n_days):>+10.2f}% {ens_sort:>10.2f} {ens_dd*100:>10.2f}% {'-':>8}")

    # Meta momentum
    print(f"{'Meta momentum (top-'+str(args.top_k)+')':<30} {meta_mom.total_return*100:>+10.2f}% {_monthly(meta_mom.total_return, n_days):>+10.2f}% {meta_mom.sortino:>10.2f} {meta_mom.max_drawdown*100:>10.2f}% {meta_mom.num_trades:>8}")

    # Meta Chronos2
    if meta_c2 is not None:
        print(f"{'Meta Chronos2 (top-'+str(args.top_k)+')':<30} {meta_c2.total_return*100:>+10.2f}% {_monthly(meta_c2.total_return, n_days):>+10.2f}% {meta_c2.sortino:>10.2f} {meta_c2.max_drawdown*100:>10.2f}% {meta_c2.num_trades:>8}")

    print()

    # Save results
    if args.out:
        out_path = Path(args.out)
        results = {
            "config": {
                "symbols": symbols,
                "n_models": len(traces),
                "top_k": args.top_k,
                "lookback": args.lookback,
                "warmup": args.warmup,
                "fee_rate": args.fee_rate,
                "slippage_bps": args.slippage_bps,
                "fill_buffer_bps": args.fill_buffer_bps,
                "decision_lag": args.decision_lag,
                "n_days": n_days,
            },
            "per_model": {
                tr.name: {
                    "total_return": float(tr.equity_curve[-1] / tr.equity_curve[0] - 1),
                }
                for tr in traces
            },
            "ensemble": {
                "total_return": float(ens_ret_f),
                "sortino": float(ens_sort),
                "max_drawdown": float(ens_dd),
            },
            "meta_momentum": {
                "total_return": float(meta_mom.total_return),
                "sortino": float(meta_mom.sortino),
                "max_drawdown": float(meta_mom.max_drawdown),
                "num_trades": int(meta_mom.num_trades),
            },
        }
        if meta_c2 is not None:
            results["meta_chronos2"] = {
                "total_return": float(meta_c2.total_return),
                "sortino": float(meta_c2.sortino),
                "max_drawdown": float(meta_c2.max_drawdown),
                "num_trades": int(meta_c2.num_trades),
            }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        log.info("saved to %s", out_path)


if __name__ == "__main__":
    main()
