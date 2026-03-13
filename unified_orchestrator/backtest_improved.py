"""Backtest multiple improvement strategies independently and combined.

Tests momentum baseline, RL-only, RL + vol scaling, RL + correlation
derisking, RL + wider targets, and a combined mode. All modes use the
HourlyTraderMarketSimulator for realistic fill modelling.

No LLM API calls -- pure offline simulation.

Usage:
  python -m unified_orchestrator.backtest_improved --days 30
  python -m unified_orchestrator.backtest_improved --days 7 --symbols BTCUSD ETHUSD
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from newnanoalpacahourlyexp.marketsimulator.hourly_trader import (
    HourlyTraderMarketSimulator,
    HourlyTraderSimulationConfig,
)
from src.rolling_risk_metrics import RollingRiskMetrics

LOG_DIR = REPO / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = LOG_DIR / "backtest_improved_results.log"

logger = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────────

DEFAULT_SYMBOLS = ["BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD", "AVAXUSD"]
DEFAULT_CHECKPOINT = str(
    REPO / "pufferlib_market" / "checkpoints" / "autoresearch" / "slip_5bps" / "best.pt"
)
INITIAL_CASH = 46_000.0


# ── Data loading ────────────────────────────────────────────────────

DATA_DIR = REPO / "trainingdatahourly" / "crypto"
BINANCE_DIR = REPO / "binance_spot_hourly"


def load_bars(symbol: str) -> pd.DataFrame:
    """Load hourly OHLCV bars for *symbol*, trying multiple paths."""
    csv = DATA_DIR / f"{symbol}.csv"
    if not csv.exists():
        csv = BINANCE_DIR / f"{symbol.replace('USD', 'USDT')}.csv"
    if not csv.exists():
        raise FileNotFoundError(f"No data found for {symbol}")
    df = pd.read_csv(csv)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    elif "date" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"], utc=True)
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    else:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    return df.sort_values("timestamp").dropna(subset=["close"]).reset_index(drop=True)


# ── RL observation builder (matches C env / inference.py) ───────────

def _compute_features_for_symbol(
    bars_df: pd.DataFrame, idx: int
) -> np.ndarray:
    """Compute 16 features at bar *idx* matching ``inference.py:compute_hourly_features``."""
    features = np.zeros(16, dtype=np.float32)

    if idx < 24:
        return features

    window = bars_df.iloc[max(0, idx - 72) : idx + 1]
    close = window["close"].values
    high = window["high"].values
    low = window["low"].values
    vol = window["volume"].values if "volume" in window.columns else np.zeros(len(window))

    if len(close) < 2 or close[-1] == 0:
        return features

    # Returns
    ret_1h = close[-1] / close[-2] - 1.0 if len(close) >= 2 else 0.0
    ret_4h = close[-1] / close[-4] - 1.0 if len(close) >= 5 else 0.0
    ret_24h = close[-1] / close[-24] - 1.0 if len(close) >= 25 else 0.0

    ma_24 = close[-24:].mean() if len(close) >= 24 else close.mean()
    ma_72 = close.mean()
    ema_alpha = 2.0 / 25.0
    ema_24 = close[-1]  # simplified
    if len(close) >= 24:
        ema_24 = close[-24:].copy().astype(np.float64)[-1]
        ema_val = float(ema_24)
        for p in close[-24:]:
            ema_val = ema_alpha * p + (1 - ema_alpha) * ema_val
        ema_24 = ema_val

    atr_vals = high[-24:] - low[-24:] if len(close) >= 24 else high - low
    atr_24 = atr_vals.mean()

    vol_mean = vol[-24:].mean() if len(vol) >= 24 else max(vol.mean(), 1e-8)
    vol_ratio = vol[-1] / max(vol_mean, 1e-8) if vol_mean > 0 else 0.0

    range_pos = (close[-1] - low[-1]) / max(high[-1] - low[-1], 1e-8)
    ma_ratio = close[-1] / max(ma_24, 1e-8)

    macd_proxy = ema_24 - ma_72

    def _safe_zscore(val: float, series: np.ndarray) -> float:
        if len(series) < 2:
            return 0.0
        m = series.mean()
        s = series.std()
        if s < 1e-12:
            return 0.0
        return float((val - m) / s)

    # Build rolling series for z-scoring
    features[0] = _safe_zscore(ret_1h, np.diff(close) / np.maximum(close[:-1], 1e-8))
    features[1] = ret_4h  # skip full z-score for speed
    features[2] = ret_24h
    features[3] = _safe_zscore(ma_ratio - 1.0, close / np.maximum(
        pd.Series(close).rolling(24, min_periods=1).mean().values, 1e-8) - 1.0)
    features[4] = range_pos - 0.5
    features[5] = _safe_zscore(vol_ratio - 1.0, np.zeros(1))  # simplified
    features[6] = atr_24 / max(close[-1], 1e-8)
    features[7] = macd_proxy / max(close[-1], 1e-8)
    # features 8-15 remain zero (placeholders, matching inference.py)

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def build_multi_symbol_obs(
    all_bars: Dict[str, pd.DataFrame],
    bar_indices: Dict[str, int],
    symbols: List[str],
    ppo_trader: object,
) -> np.ndarray:
    """Build full observation vector for PPOTrader.get_signal()."""
    num_symbols = len(symbols)
    features = np.zeros((num_symbols, 16), dtype=np.float32)
    prices = {}

    for i, sym in enumerate(symbols):
        if sym in all_bars and sym in bar_indices:
            idx = bar_indices[sym]
            df = all_bars[sym]
            features[i] = _compute_features_for_symbol(df, idx)
            prices[sym] = float(df.iloc[idx]["close"])

    return features, prices


# ── RL policy wrapper ───────────────────────────────────────────────

def load_ppo_trader(
    checkpoint_path: str,
    symbols: List[str],
    device: str = "cpu",
) -> object:
    """Load PPOTrader with proper symbol list."""
    from pufferlib_market.inference import PPOTrader
    return PPOTrader(
        checkpoint_path=checkpoint_path,
        device=device,
        symbols=symbols,
    )


# ── Rolling volatility / correlation helpers ────────────────────────

def compute_rolling_vol(
    closes: np.ndarray, window: int = 24
) -> float:
    """Compute rolling annualised hourly volatility over last *window* bars."""
    if len(closes) < max(2, window):
        return 0.01
    recent = closes[-window:]
    log_rets = np.diff(np.log(np.maximum(recent, 1e-12)))
    if len(log_rets) < 2:
        return 0.01
    return float(np.std(log_rets, ddof=1))


def compute_pairwise_corr(
    all_bars: Dict[str, pd.DataFrame],
    bar_indices: Dict[str, int],
    symbols: List[str],
    window: int = 72,
) -> pd.DataFrame:
    """Compute rolling return correlation matrix over last *window* bars."""
    returns_map: Dict[str, np.ndarray] = {}
    for sym in symbols:
        if sym not in all_bars or sym not in bar_indices:
            continue
        idx = bar_indices[sym]
        df = all_bars[sym]
        start = max(0, idx - window)
        closes = df.iloc[start : idx + 1]["close"].values
        if len(closes) < 3:
            continue
        rets = np.diff(np.log(np.maximum(closes, 1e-12)))
        returns_map[sym] = rets

    if len(returns_map) < 2:
        return pd.DataFrame(np.eye(len(symbols)), index=symbols, columns=symbols)

    # Align lengths
    min_len = min(len(r) for r in returns_map.values())
    aligned = {sym: rets[-min_len:] for sym, rets in returns_map.items()}
    df_rets = pd.DataFrame(aligned)
    corr = df_rets.corr().fillna(0.0)
    # Fill missing symbols with zeros
    for sym in symbols:
        if sym not in corr.columns:
            corr[sym] = 0.0
            corr.loc[sym] = 0.0
            corr.at[sym, sym] = 1.0
    return corr.reindex(index=symbols, columns=symbols).fillna(0.0)


def avg_portfolio_corr_if_added(
    corr_matrix: pd.DataFrame,
    current_positions: set,
    new_symbol: str,
) -> float:
    """Return avg off-diagonal correlation if *new_symbol* were added."""
    combined = current_positions | {new_symbol}
    if len(combined) < 2:
        return 0.0
    syms = sorted(combined)
    total = 0.0
    count = 0
    for i, s1 in enumerate(syms):
        for j, s2 in enumerate(syms):
            if i >= j:
                continue
            if s1 in corr_matrix.columns and s2 in corr_matrix.columns:
                total += abs(corr_matrix.at[s1, s2])
            count += 1
    return total / max(count, 1)


# ── Mode result container ──────────────────────────────────────────

@dataclass
class ModeResult:
    name: str
    return_pct: float = 0.0
    sortino: float = 0.0
    max_drawdown_pct: float = 0.0
    fills: int = 0
    trades: int = 0
    win_rate: float = 0.0
    elapsed_s: float = 0.0
    error: str = ""
    extra: Dict = field(default_factory=dict)


# ── Action generators per mode ─────────────────────────────────────

def _momentum_actions(
    all_bars: Dict[str, pd.DataFrame],
    symbols: List[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Mode 1: SMA crossover baseline."""
    all_bar_rows = []
    all_action_rows = []

    for sym in symbols:
        df = all_bars[sym]
        window = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)].copy()
        if len(window) < 48:
            continue

        closes = df["close"].values
        for _, bar in window.iterrows():
            ts = bar["timestamp"]
            close = float(bar["close"])
            bar_idx = df.index.get_loc(bar.name)

            all_bar_rows.append({
                "timestamp": ts, "symbol": sym,
                "open": float(bar["open"]), "high": float(bar["high"]),
                "low": float(bar["low"]), "close": close,
            })

            if bar_idx >= 48:
                sma_12 = closes[max(0, bar_idx - 12) : bar_idx].mean()
                sma_48 = closes[max(0, bar_idx - 48) : bar_idx].mean()
                buy_signal = sma_12 > sma_48
            else:
                buy_signal = False

            if buy_signal:
                all_action_rows.append({
                    "timestamp": ts, "symbol": sym,
                    "buy_price": close * 0.998,
                    "sell_price": close * 1.010,
                    "buy_amount": 100.0,
                    "sell_amount": 0.0,
                })
            else:
                all_action_rows.append({
                    "timestamp": ts, "symbol": sym,
                    "buy_price": 0.0,
                    "sell_price": close * 0.998,
                    "buy_amount": 0.0,
                    "sell_amount": 100.0,
                })

    return pd.DataFrame(all_bar_rows), pd.DataFrame(all_action_rows)


def _rl_actions(
    all_bars: Dict[str, pd.DataFrame],
    symbols: List[str],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    ppo_trader: object,
    buy_spread: float = 0.002,
    sell_target: float = 0.012,
    vol_scaling: bool = False,
    vol_target: float = 0.01,
    corr_filter: bool = False,
    corr_threshold: float = 0.70,
    base_alloc_pct: float = 0.20,
    conf_threshold: float = 0.40,
    trend_filter: bool = False,
    trend_window: int = 48,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate actions using PPO signals with optional vol/corr/trend adjustments."""
    all_bar_rows = []
    all_action_rows = []

    # Precompute timestamps that fall within the window for each symbol
    sym_windows: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = all_bars[sym]
        win = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)].copy()
        if len(win) < 24:
            continue
        sym_windows[sym] = win

    if not sym_windows:
        return pd.DataFrame(), pd.DataFrame()

    # Get all unique timestamps across symbols
    all_timestamps = sorted(
        set().union(*(set(w["timestamp"]) for w in sym_windows.values()))
    )

    # Track which symbols have open positions for correlation filter
    active_positions: set = set()

    for ts in all_timestamps:
        # Build bar indices at this timestamp
        bar_indices = {}
        ts_bars = {}
        for sym in sym_windows:
            df = all_bars[sym]
            win = sym_windows[sym]
            mask = win["timestamp"] == ts
            if not mask.any():
                continue
            bar = win.loc[mask].iloc[-1]
            bar_idx = df.index.get_loc(bar.name)
            bar_indices[sym] = bar_idx
            ts_bars[sym] = bar

        if not bar_indices:
            continue

        # Build observation and get signal
        features, prices = build_multi_symbol_obs(
            all_bars, bar_indices, symbols, ppo_trader
        )
        signal = ppo_trader.get_signal(features, prices)

        # Compute correlation matrix if needed
        corr_matrix = None
        if corr_filter:
            corr_matrix = compute_pairwise_corr(
                all_bars, bar_indices, list(bar_indices.keys()), window=72
            )

        for sym in ts_bars:
            bar = ts_bars[sym]
            close = float(bar["close"])
            bar_idx = bar_indices[sym]

            all_bar_rows.append({
                "timestamp": ts, "symbol": sym,
                "open": float(bar["open"]), "high": float(bar["high"]),
                "low": float(bar["low"]), "close": close,
            })

            # Determine if the signal targets this symbol
            sym_direction = None
            sym_confidence = 0.0

            if signal.symbol == sym and signal.direction in ("long", "short"):
                sym_direction = signal.direction
                sym_confidence = signal.confidence
            elif signal.symbol is None:
                # Flat signal
                sym_direction = None
                sym_confidence = 0.0

            # Volatility scaling
            alloc_mult = 1.0
            if vol_scaling and bar_idx >= 24:
                closes = all_bars[sym].iloc[max(0, bar_idx - 24) : bar_idx + 1]["close"].values
                realized_vol = compute_rolling_vol(closes, window=24)
                alloc_mult = vol_target / max(realized_vol, 0.001)
                alloc_mult = max(0.2, min(2.0, alloc_mult))

            # Correlation filter
            skip_entry = False
            if corr_filter and corr_matrix is not None and sym_direction == "long":
                if sym not in active_positions and len(active_positions) > 0:
                    avg_corr = avg_portfolio_corr_if_added(
                        corr_matrix, active_positions, sym
                    )
                    if avg_corr > corr_threshold:
                        skip_entry = True

            # Trend filter: don't buy if price is below its SMA (downtrend)
            if trend_filter and sym_direction == "long" and bar_idx >= trend_window:
                closes_tw = all_bars[sym].iloc[max(0, bar_idx - trend_window) : bar_idx + 1]["close"].values
                sma = np.mean(closes_tw)
                if close < sma:
                    skip_entry = True  # don't buy in downtrend

            # Build action
            buy_amount = 100.0 * alloc_mult
            if sym_direction == "long" and sym_confidence > conf_threshold and not skip_entry:
                all_action_rows.append({
                    "timestamp": ts, "symbol": sym,
                    "buy_price": close * (1.0 - buy_spread),
                    "sell_price": close * (1.0 + sell_target),
                    "buy_amount": buy_amount,
                    "sell_amount": 0.0,
                })
                active_positions.add(sym)
            elif sym_direction == "short" and sym_confidence > conf_threshold and not skip_entry:
                all_action_rows.append({
                    "timestamp": ts, "symbol": sym,
                    "buy_price": 0.0,
                    "sell_price": close * (1.0 + buy_spread),
                    "buy_amount": 0.0,
                    "sell_amount": buy_amount,
                })
                active_positions.discard(sym)
            else:
                # Flat / no action -- still submit row so simulator can fill exits
                all_action_rows.append({
                    "timestamp": ts, "symbol": sym,
                    "buy_price": 0.0,
                    "sell_price": 0.0,
                    "buy_amount": 0.0,
                    "sell_amount": 0.0,
                })
                active_positions.discard(sym)

    return pd.DataFrame(all_bar_rows), pd.DataFrame(all_action_rows)


# ── Simulation runner ──────────────────────────────────────────────

def _run_sim(
    bars_df: pd.DataFrame,
    actions_df: pd.DataFrame,
    symbols: List[str],
    initial_cash: float,
    alloc_pct: float,
) -> ModeResult:
    """Run the hourly trader simulator and compute metrics."""
    if bars_df.empty or actions_df.empty:
        return ModeResult(name="", error="empty data")

    cfg = HourlyTraderSimulationConfig(
        initial_cash=initial_cash,
        allocation_pct=alloc_pct,
        max_leverage=1.0,
        enforce_market_hours=False,
        allow_short=False,
        decision_lag_bars=1,
        fill_buffer_bps=5.0,
        partial_fill_on_touch=True,
    )

    sim = HourlyTraderMarketSimulator(cfg)
    result = sim.run(bars_df, actions_df)

    total_return = (result.equity_curve.iloc[-1] / initial_cash - 1) * 100

    # Max drawdown
    eq = result.equity_curve.values
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.maximum(peak, 1e-8) * 100
    max_dd = float(abs(dd.min())) if len(dd) > 0 else 0.0

    # Win rate from entry/exit pairs
    wins = 0
    total_trades = 0
    entry_prices: Dict[str, float] = {}
    for fill_rec in result.fills:
        sym = fill_rec.symbol
        if fill_rec.kind == "entry":
            entry_prices[sym] = fill_rec.price
        elif fill_rec.kind == "exit" and sym in entry_prices:
            total_trades += 1
            if fill_rec.price > entry_prices[sym]:
                wins += 1
            del entry_prices[sym]

    win_rate = wins / total_trades if total_trades > 0 else 0.0

    return ModeResult(
        name="",
        return_pct=total_return,
        sortino=result.metrics.get("sortino", 0.0),
        max_drawdown_pct=max_dd,
        fills=len(result.fills),
        trades=total_trades,
        win_rate=win_rate,
    )


# ── Main backtest orchestrator ─────────────────────────────────────

def run_backtest(
    symbols: List[str],
    checkpoint_path: str,
    days: int = 30,
    initial_cash: float = INITIAL_CASH,
    target_spread_pcts: Optional[List[float]] = None,
) -> List[ModeResult]:
    """Run all backtest modes and return results."""

    if target_spread_pcts is None:
        target_spread_pcts = [1.5, 2.0, 2.5]

    # Load data
    print("Loading data...")
    all_bars: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            all_bars[sym] = load_bars(sym)
            print(f"  {sym}: {len(all_bars[sym])} bars")
        except Exception as e:
            print(f"  {sym}: SKIP ({e})")

    usable = [s for s in symbols if s in all_bars]
    if not usable:
        print("No usable symbols, exiting.")
        return []

    # Determine time window
    end_ts = min(all_bars[s]["timestamp"].max() for s in usable)
    start_ts = end_ts - pd.Timedelta(days=days)
    print(f"\nBacktest window: {start_ts} to {end_ts} ({days}d)")
    print(f"Symbols: {', '.join(usable)}")

    alloc_pct = 1.0 / len(usable)

    # Load RL policy
    ppo_trader = None
    ckpt = Path(checkpoint_path)
    if ckpt.exists():
        try:
            ppo_trader = load_ppo_trader(str(ckpt), usable)
            print(f"RL checkpoint loaded: {ckpt.name}")
        except Exception as e:
            print(f"RL checkpoint load failed: {e}")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")

    results: List[ModeResult] = []

    # ── Mode 1: Momentum baseline ──────────────────────────────────
    print(f"\n{'=' * 65}")
    print("MODE 1: Momentum baseline (SMA12 > SMA48)")
    print(f"{'=' * 65}")
    t0 = time.time()
    try:
        bars_df, actions_df = _momentum_actions(all_bars, usable, start_ts, end_ts)
        res = _run_sim(bars_df, actions_df, usable, initial_cash, alloc_pct)
        res.name = "momentum_baseline"
        res.elapsed_s = time.time() - t0
        results.append(res)
        _print_result(res)
    except Exception as e:
        logger.exception("Mode 1 failed")
        results.append(ModeResult(name="momentum_baseline", error=str(e)))

    if ppo_trader is None:
        print("\nSkipping RL modes (no checkpoint)")
        return results

    # ── Mode 2: RL-only ────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("MODE 2: RL-only (PPO signals, 1.2% target)")
    print(f"{'=' * 65}")
    t0 = time.time()
    try:
        bars_df, actions_df = _rl_actions(
            all_bars, usable, start_ts, end_ts, ppo_trader,
            buy_spread=0.002, sell_target=0.012,
        )
        res = _run_sim(bars_df, actions_df, usable, initial_cash, alloc_pct)
        res.name = "rl_only"
        res.elapsed_s = time.time() - t0
        results.append(res)
        _print_result(res)
    except Exception as e:
        logger.exception("Mode 2 failed")
        results.append(ModeResult(name="rl_only", error=str(e)))

    # ── Mode 3: RL + volatility scaling ────────────────────────────
    print(f"\n{'=' * 65}")
    print("MODE 3: RL + volatility scaling (target_vol=0.01, cap 0.2x-2x)")
    print(f"{'=' * 65}")
    t0 = time.time()
    try:
        bars_df, actions_df = _rl_actions(
            all_bars, usable, start_ts, end_ts, ppo_trader,
            buy_spread=0.002, sell_target=0.012,
            vol_scaling=True, vol_target=0.01,
        )
        res = _run_sim(bars_df, actions_df, usable, initial_cash, alloc_pct)
        res.name = "rl_vol_scaling"
        res.elapsed_s = time.time() - t0
        results.append(res)
        _print_result(res)
    except Exception as e:
        logger.exception("Mode 3 failed")
        results.append(ModeResult(name="rl_vol_scaling", error=str(e)))

    # ── Mode 4: RL + correlation derisking ─────────────────────────
    print(f"\n{'=' * 65}")
    print("MODE 4: RL + correlation derisking (threshold=0.70)")
    print(f"{'=' * 65}")
    t0 = time.time()
    try:
        bars_df, actions_df = _rl_actions(
            all_bars, usable, start_ts, end_ts, ppo_trader,
            buy_spread=0.002, sell_target=0.012,
            corr_filter=True, corr_threshold=0.70,
        )
        res = _run_sim(bars_df, actions_df, usable, initial_cash, alloc_pct)
        res.name = "rl_corr_filter"
        res.elapsed_s = time.time() - t0
        results.append(res)
        _print_result(res)
    except Exception as e:
        logger.exception("Mode 4 failed")
        results.append(ModeResult(name="rl_corr_filter", error=str(e)))

    # ── Mode 5: RL + wider targets ─────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"MODE 5: RL + wider targets ({target_spread_pcts}%)")
    print(f"{'=' * 65}")
    for tgt_pct in target_spread_pcts:
        sell_target = tgt_pct / 100.0
        label = f"rl_target_{tgt_pct:.1f}pct"
        t0 = time.time()
        try:
            bars_df, actions_df = _rl_actions(
                all_bars, usable, start_ts, end_ts, ppo_trader,
                buy_spread=0.002, sell_target=sell_target,
            )
            res = _run_sim(bars_df, actions_df, usable, initial_cash, alloc_pct)
            res.name = label
            res.elapsed_s = time.time() - t0
            results.append(res)
            _print_result(res)
        except Exception as e:
            logger.exception(f"Mode 5 ({tgt_pct}%) failed")
            results.append(ModeResult(name=label, error=str(e)))

    # ── Mode 6: Combined best ─────────────────────────────────────
    # Pick the best target from Mode 5 by sortino, then combine with vol + corr
    mode5_results = [r for r in results if r.name.startswith("rl_target_") and not r.error]
    if mode5_results:
        best_target_res = max(mode5_results, key=lambda r: r.sortino)
        # Extract target pct from name
        best_target_pct = float(best_target_res.name.split("_")[-1].replace("pct", ""))
    else:
        best_target_pct = 2.0  # fallback

    best_sell_target = best_target_pct / 100.0

    print(f"\n{'=' * 65}")
    print(f"MODE 6: Combined (RL + vol_scaling + corr_filter + {best_target_pct:.1f}% target)")
    print(f"{'=' * 65}")
    t0 = time.time()
    try:
        bars_df, actions_df = _rl_actions(
            all_bars, usable, start_ts, end_ts, ppo_trader,
            buy_spread=0.002, sell_target=best_sell_target,
            vol_scaling=True, vol_target=0.01,
            corr_filter=True, corr_threshold=0.70,
        )
        res = _run_sim(bars_df, actions_df, usable, initial_cash, alloc_pct)
        res.name = "combined_best"
        res.elapsed_s = time.time() - t0
        results.append(res)
        _print_result(res)
    except Exception as e:
        logger.exception("Mode 6 failed")
        results.append(ModeResult(name="combined_best", error=str(e)))

    # ── Mode 7: RL + high confidence filter ────────────────────────
    for conf_thresh in [0.55, 0.65, 0.75]:
        label = f"rl_conf_{int(conf_thresh*100)}"
        print(f"\n{'=' * 65}")
        print(f"MODE 7: RL + confidence threshold ({conf_thresh:.0%})")
        print(f"{'=' * 65}")
        t0 = time.time()
        try:
            bars_df, actions_df = _rl_actions(
                all_bars, usable, start_ts, end_ts, ppo_trader,
                buy_spread=0.002, sell_target=best_sell_target,
                conf_threshold=conf_thresh,
            )
            res = _run_sim(bars_df, actions_df, usable, initial_cash, alloc_pct)
            res.name = label
            res.elapsed_s = time.time() - t0
            results.append(res)
            _print_result(res)
        except Exception as e:
            logger.exception(f"Mode 7 ({label}) failed")
            results.append(ModeResult(name=label, error=str(e)))

    # ── Mode 8: RL + trend filter (only buy above SMA) ──────────
    for tw in [24, 48, 72]:
        label = f"rl_trend_{tw}h"
        print(f"\n{'=' * 65}")
        print(f"MODE 8: RL + trend filter (SMA{tw}h, {best_target_pct:.1f}% target)")
        print(f"{'=' * 65}")
        t0 = time.time()
        try:
            bars_df, actions_df = _rl_actions(
                all_bars, usable, start_ts, end_ts, ppo_trader,
                buy_spread=0.002, sell_target=best_sell_target,
                trend_filter=True, trend_window=tw,
            )
            res = _run_sim(bars_df, actions_df, usable, initial_cash, alloc_pct)
            res.name = label
            res.elapsed_s = time.time() - t0
            results.append(res)
            _print_result(res)
        except Exception as e:
            logger.exception(f"Mode 8 ({label}) failed")
            results.append(ModeResult(name=label, error=str(e)))

    # ── Mode 9: Ultimate combined ────────────────────────────────
    # Best target + vol scaling + corr filter + trend filter + higher confidence
    best_conf_results = [r for r in results if r.name.startswith("rl_conf_") and not r.error]
    best_conf = 0.55  # default
    if best_conf_results:
        best_conf_res = max(best_conf_results, key=lambda r: r.sortino)
        best_conf = float(best_conf_res.name.split("_")[-1]) / 100.0

    best_trend_results = [r for r in results if r.name.startswith("rl_trend_") and not r.error]
    best_tw = 48  # default
    if best_trend_results:
        best_trend_res = max(best_trend_results, key=lambda r: r.sortino)
        best_tw = int(best_trend_res.name.split("_")[-1].replace("h", ""))

    print(f"\n{'=' * 65}")
    print(f"MODE 9: Ultimate (RL + vol + corr + trend{best_tw}h + conf>{best_conf:.0%} + {best_target_pct:.1f}%)")
    print(f"{'=' * 65}")
    t0 = time.time()
    try:
        bars_df, actions_df = _rl_actions(
            all_bars, usable, start_ts, end_ts, ppo_trader,
            buy_spread=0.002, sell_target=best_sell_target,
            vol_scaling=True, vol_target=0.01,
            corr_filter=True, corr_threshold=0.70,
            conf_threshold=best_conf,
            trend_filter=True, trend_window=best_tw,
        )
        res = _run_sim(bars_df, actions_df, usable, initial_cash, alloc_pct)
        res.name = "ultimate_combined"
        res.elapsed_s = time.time() - t0
        results.append(res)
        _print_result(res)
    except Exception as e:
        logger.exception("Mode 9 failed")
        results.append(ModeResult(name="ultimate_combined", error=str(e)))

    return results


def _print_result(res: ModeResult) -> None:
    """Print a single mode result."""
    if res.error:
        print(f"  ERROR: {res.error}")
        return
    print(f"  Return:   {res.return_pct:+.2f}%")
    print(f"  Sortino:  {res.sortino:.2f}")
    print(f"  Max DD:   {res.max_drawdown_pct:.2f}%")
    print(f"  Fills:    {res.fills}")
    print(f"  Trades:   {res.trades}")
    print(f"  Win rate: {res.win_rate:.1%}")
    print(f"  Time:     {res.elapsed_s:.1f}s")


def print_comparison_table(results: List[ModeResult], days: int, symbols: List[str]) -> str:
    """Print and return a formatted comparison table."""
    lines = []
    header = (
        f"\n{'=' * 90}\n"
        f"BACKTEST COMPARISON  ({days}d, symbols: {', '.join(symbols)})\n"
        f"{'=' * 90}"
    )
    lines.append(header)

    col_fmt = "{:<25s} {:>8s} {:>8s} {:>8s} {:>6s} {:>6s} {:>8s}"
    lines.append(col_fmt.format(
        "Mode", "Return%", "Sortino", "MaxDD%", "Fills", "Trades", "WinRate"
    ))
    lines.append("-" * 90)

    for r in results:
        if r.error:
            lines.append(f"{r.name:<25s}  ERROR: {r.error}")
        else:
            lines.append(col_fmt.format(
                r.name,
                f"{r.return_pct:+.2f}",
                f"{r.sortino:.2f}",
                f"{r.max_drawdown_pct:.2f}",
                str(r.fills),
                str(r.trades),
                f"{r.win_rate:.1%}",
            ))

    lines.append("-" * 90)

    # Determine winner
    valid = [r for r in results if not r.error]
    if len(valid) >= 2:
        best_sortino = max(valid, key=lambda r: r.sortino)
        best_return = max(valid, key=lambda r: r.return_pct)
        lines.append(f"  Best Sortino: {best_sortino.name} ({best_sortino.sortino:.2f})")
        lines.append(f"  Best Return:  {best_return.name} ({best_return.return_pct:+.2f}%)")

        # Compare RL-only vs combined
        rl_only = next((r for r in valid if r.name == "rl_only"), None)
        combined = next((r for r in valid if r.name == "combined_best"), None)
        if rl_only and combined:
            delta_ret = combined.return_pct - rl_only.return_pct
            delta_sort = combined.sortino - rl_only.sortino
            lines.append(f"\n  Combined vs RL-only:")
            lines.append(f"    Return delta: {delta_ret:+.2f}%")
            lines.append(f"    Sortino delta: {delta_sort:+.2f}")
            if delta_sort > 0 and delta_ret > 0:
                lines.append("    VERDICT: Combined WINS")
            elif delta_sort < 0 and delta_ret < 0:
                lines.append("    VERDICT: RL-only WINS")
            else:
                lines.append("    VERDICT: MIXED")

    lines.append(f"{'=' * 90}")
    text = "\n".join(lines)
    print(text)
    return text


def save_results(results: List[ModeResult], days: int, symbols: List[str]) -> None:
    """Save results to log file and JSON."""
    table_text = print_comparison_table(results, days, symbols)

    # Log file
    with open(LOG_PATH, "w") as f:
        f.write(table_text)
        f.write("\n")
    print(f"\nLog saved to {LOG_PATH}")

    # JSON
    json_path = LOG_PATH.with_suffix(".json")
    data = []
    for r in results:
        d = {
            "mode": r.name,
            "return_pct": r.return_pct,
            "sortino": r.sortino,
            "max_drawdown_pct": r.max_drawdown_pct,
            "fills": r.fills,
            "trades": r.trades,
            "win_rate": r.win_rate,
            "elapsed_s": r.elapsed_s,
        }
        if r.error:
            d["error"] = r.error
        data.append(d)
    with open(json_path, "w") as f:
        json.dump({"days": days, "symbols": symbols, "results": data}, f, indent=2)
    print(f"JSON saved to {json_path}")


# ── CLI ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Backtest improved strategies independently and combined"
    )
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cash", type=float, default=INITIAL_CASH)
    parser.add_argument(
        "--target-spreads", nargs="+", type=float, default=[1.5, 2.0, 2.5],
        help="Profit target percentages to test in Mode 5",
    )
    args = parser.parse_args()

    results = run_backtest(
        symbols=args.symbols,
        checkpoint_path=args.checkpoint,
        days=args.days,
        initial_cash=args.cash,
        target_spread_pcts=args.target_spreads,
    )

    if results:
        save_results(results, args.days, args.symbols)


if __name__ == "__main__":
    main()
