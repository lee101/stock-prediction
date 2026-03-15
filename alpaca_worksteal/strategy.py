"""Work-stealing strategy for Alpaca (stocks + crypto), adapted from binance_worksteal.

Key differences from Binance version:
- Alpaca is commission-free for stocks (like FDUSD 0% fee)
- Stocks have market hours (9:30-16:00 ET, Mon-Fri)
- 4x intraday leverage for stocks (margin account)
- Crypto 24/7 but limited symbols on Alpaca
- Work-stealing: only create orders when price is near target, freeing buying power
- Supports RL signal overlay: RL model suggests direction, worksteal times entry

Usage with RL overlay:
  1. RL daily model produces per-symbol signals (long/short/flat + confidence)
  2. For each "long" signal, compute dip buy target from recent high
  3. For each "short" signal, compute pump sell target from recent low
  4. Work-stealer monitors prices and only commits buying power when proximity < threshold
  5. Multiple fills possible per day if price oscillates around targets
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class AlpacaWorkStealConfig:
    # Entry parameters
    dip_pct: float = 0.05            # Buy when X% below recent high
    proximity_pct: float = 0.005     # Only place order within this % of target
    profit_target_pct: float = 0.03  # Take profit at this % gain
    stop_loss_pct: float = 0.05      # Stop loss at this % loss
    # Position management
    max_positions: int = 8           # Max concurrent positions
    max_hold_hours: int = 48         # Force exit after N hours (0=no limit)
    max_position_pct: float = 0.15   # Max % of equity per position
    # Market structure
    lookback_bars: int = 48          # Bars to compute reference price (hourly)
    ref_price_method: str = "high"   # "high", "sma", "close"
    # Fees (Alpaca is commission-free)
    stock_fee: float = 0.0           # Commission-free
    crypto_fee: float = 0.0015       # 15bps for Alpaca crypto
    # Risk
    trailing_stop_pct: float = 0.003  # 0.3% trailing stop (proven in prod)
    reentry_cooldown_bars: int = 4   # Wait N bars before re-entering same symbol
    max_leverage: float = 2.0        # 2x overnight, 4x intraday
    max_drawdown_exit: float = 0.15  # Emergency exit at 15% portfolio drawdown
    # Trend filters
    sma_filter_period: int = 24      # Only long above 24h SMA
    # Cash management
    initial_cash: float = 50_000.0
    min_cash_reserve_pct: float = 0.10  # Keep 10% cash reserve always
    # RL integration
    use_rl_signals: bool = False     # If True, only enter in RL signal direction
    min_rl_confidence: float = 0.4   # Minimum RL confidence to trade
    # Oscillation trading
    enable_oscillation: bool = True  # Allow multiple round-trips per day
    oscillation_min_spread: float = 0.005  # Min % between buy/sell for oscillation


CRYPTO_SYMBOLS = {"BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD", "AVAXUSD",
                  "DOGEUSD", "LINKUSD", "AAVEUSD", "UNIUSD"}


@dataclass
class Position:
    symbol: str
    direction: str  # "long" or "short"
    entry_price: float
    entry_bar: int  # bar index at entry
    quantity: float
    cost_basis: float
    peak_price: float
    target_exit_price: float
    stop_price: float
    is_crypto: bool = False


@dataclass
class TradeLog:
    bar_idx: int
    timestamp: pd.Timestamp
    symbol: str
    side: str
    price: float
    quantity: float
    notional: float
    fee: float
    pnl: float = 0.0
    reason: str = ""


def get_fee(symbol: str, config: AlpacaWorkStealConfig) -> float:
    if symbol in CRYPTO_SYMBOLS:
        return config.crypto_fee
    return config.stock_fee


def is_tradable_hour(timestamp: pd.Timestamp, symbol: str) -> bool:
    """Check if this symbol can trade at this time."""
    if symbol in CRYPTO_SYMBOLS:
        return True  # 24/7
    # Stock market hours: 9:30-16:00 ET (14:30-21:00 UTC)
    hour_utc = timestamp.hour
    minute = timestamp.minute
    weekday = timestamp.weekday()
    if weekday >= 5:  # Saturday/Sunday
        return False
    # Approximate: 14:30-21:00 UTC = 9:30-16:00 ET
    if hour_utc < 14 or hour_utc >= 21:
        return False
    if hour_utc == 14 and minute < 30:
        return False
    return True


def compute_ref_price(bars: pd.DataFrame, method: str, lookback: int) -> float:
    if len(bars) < 2:
        return float(bars["close"].iloc[-1])
    window = bars.tail(lookback)
    if method == "high":
        return float(window["high"].max())
    elif method == "sma":
        return float(window["close"].mean())
    return float(window["close"].max())


def run_alpaca_worksteal_backtest(
    all_bars: Dict[str, pd.DataFrame],
    config: AlpacaWorkStealConfig,
    rl_signals: Optional[Dict[str, Dict[int, dict]]] = None,  # symbol -> {bar_idx: {direction, confidence}}
) -> Tuple[pd.DataFrame, List[TradeLog], Dict[str, float]]:
    """Run hourly work-stealing backtest for Alpaca stocks+crypto.

    Args:
        all_bars: {symbol: DataFrame with timestamp, open, high, low, close, volume}
        config: Strategy configuration
        rl_signals: Optional RL signal overlay {symbol: {bar_idx: {direction, confidence}}}
    """
    # Align all symbols to common timestamps
    for sym in list(all_bars.keys()):
        df = all_bars[sym].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        all_bars[sym] = df

    # Build common timeline
    all_timestamps = sorted(set().union(*[set(df["timestamp"].tolist()) for df in all_bars.values()]))

    cash = config.initial_cash
    positions: Dict[str, Position] = {}
    trades: List[TradeLog] = []
    equity_rows: List[Dict] = []
    last_exit_bar: Dict[str, int] = {}

    for bar_idx, ts in enumerate(all_timestamps):
        # Get current bars for all symbols
        current_bars: Dict[str, pd.Series] = {}
        history: Dict[str, pd.DataFrame] = {}

        for sym, df in all_bars.items():
            mask = df["timestamp"] <= ts
            hist = df[mask]
            if hist.empty:
                continue
            last_bar = hist.iloc[-1]
            if last_bar["timestamp"] != ts:
                continue
            if not is_tradable_hour(ts, sym):
                continue
            current_bars[sym] = last_bar
            history[sym] = hist

        if not current_bars:
            continue

        # Compute current equity
        inv_value = 0.0
        for sym, pos in positions.items():
            if sym in current_bars:
                price = float(current_bars[sym]["close"])
                if pos.direction == "long":
                    inv_value += pos.quantity * price
                else:
                    inv_value += pos.quantity * (2 * pos.entry_price - price)
        current_equity = cash + inv_value

        # === 1. Check exits ===
        exits = []
        for sym, pos in list(positions.items()):
            if sym not in current_bars:
                continue
            bar = current_bars[sym]
            close = float(bar["close"])
            high = float(bar["high"])
            low = float(bar["low"])

            exit_price = None
            exit_reason = ""

            if pos.direction == "long":
                pos.peak_price = max(pos.peak_price, high)
                if high >= pos.target_exit_price:
                    exit_price = pos.target_exit_price
                    exit_reason = "profit_target"
                elif low <= pos.stop_price:
                    exit_price = pos.stop_price
                    exit_reason = "stop_loss"
                elif config.trailing_stop_pct > 0:
                    trail = pos.peak_price * (1 - config.trailing_stop_pct)
                    if low <= trail:
                        exit_price = trail
                        exit_reason = "trailing_stop"

            # Max hold
            if exit_price is None and config.max_hold_hours > 0:
                held = bar_idx - pos.entry_bar
                if held >= config.max_hold_hours:
                    exit_price = close
                    exit_reason = "max_hold"

            if exit_price is not None:
                exits.append((sym, exit_price, exit_reason))

        for sym, exit_price, reason in exits:
            pos = positions[sym]
            fee_rate = get_fee(sym, config)
            if pos.direction == "long":
                proceeds = pos.quantity * exit_price * (1 - fee_rate)
                pnl = proceeds - pos.cost_basis
                cash += proceeds
            else:
                pnl = pos.quantity * (pos.entry_price - exit_price)
                cash += pos.cost_basis + pnl
            trades.append(TradeLog(
                bar_idx=bar_idx, timestamp=ts, symbol=sym,
                side="sell" if pos.direction == "long" else "cover",
                price=exit_price, quantity=pos.quantity,
                notional=pos.quantity * exit_price,
                fee=pos.quantity * exit_price * fee_rate,
                pnl=pnl, reason=reason,
            ))
            last_exit_bar[sym] = bar_idx
            del positions[sym]

        # === 2. Check entries (work-stealing) ===
        if len(positions) < config.max_positions:
            candidates = []
            cash_reserve = config.initial_cash * config.min_cash_reserve_pct
            available_cash = max(0, cash - cash_reserve)

            for sym, bar in current_bars.items():
                if sym in positions:
                    continue
                if sym in last_exit_bar and (bar_idx - last_exit_bar[sym]) < config.reentry_cooldown_bars:
                    continue
                if sym not in history or len(history[sym]) < config.lookback_bars:
                    continue

                close = float(bar["close"])
                low_bar = float(bar["low"])
                high_bar = float(bar["high"])

                # SMA filter
                if config.sma_filter_period > 0:
                    sma = float(history[sym]["close"].iloc[-config.sma_filter_period:].mean())
                    if close < sma:
                        continue

                # RL signal filter
                if config.use_rl_signals and rl_signals:
                    sig = rl_signals.get(sym, {}).get(bar_idx)
                    if sig is None or sig.get("confidence", 0) < config.min_rl_confidence:
                        continue
                    if sig.get("direction") != "long":
                        continue

                # Dip buy target
                ref = compute_ref_price(history[sym], config.ref_price_method, config.lookback_bars)
                buy_target = ref * (1 - config.dip_pct)
                proximity = (close - buy_target) / ref

                if proximity <= config.proximity_pct:
                    dip_score = -proximity  # more dipped = higher priority
                    fill_price = max(buy_target, low_bar)
                    candidates.append((sym, dip_score, fill_price, bar))

            # Sort by best dip score (work-stealing: best opportunities first)
            candidates.sort(key=lambda x: x[1], reverse=True)

            slots = config.max_positions - len(positions)
            for sym, score, fill_price, bar in candidates[:slots]:
                if sym in positions or available_cash <= 0:
                    continue

                fee_rate = get_fee(sym, config)
                max_alloc = current_equity * config.max_position_pct * config.max_leverage
                alloc = min(max_alloc, available_cash)
                quantity = alloc / (fill_price * (1 + fee_rate))
                if quantity <= 0:
                    continue

                actual_cost = quantity * fill_price * (1 + fee_rate)
                cash -= actual_cost
                available_cash -= actual_cost

                positions[sym] = Position(
                    symbol=sym, direction="long",
                    entry_price=fill_price, entry_bar=bar_idx,
                    quantity=quantity, cost_basis=actual_cost,
                    peak_price=float(bar["high"]),
                    target_exit_price=fill_price * (1 + config.profit_target_pct),
                    stop_price=fill_price * (1 - config.stop_loss_pct),
                    is_crypto=sym in CRYPTO_SYMBOLS,
                )
                trades.append(TradeLog(
                    bar_idx=bar_idx, timestamp=ts, symbol=sym, side="buy",
                    price=fill_price, quantity=quantity,
                    notional=quantity * fill_price,
                    fee=quantity * fill_price * fee_rate,
                    reason=f"dip_buy(score={score:.4f})",
                ))

        # === 3. Equity snapshot ===
        inv_value = 0.0
        for sym, pos in positions.items():
            if sym in current_bars:
                price = float(current_bars[sym]["close"])
                if pos.direction == "long":
                    inv_value += pos.quantity * price
        equity = cash + inv_value
        equity_rows.append({
            "timestamp": ts,
            "equity": equity,
            "cash": cash,
            "n_positions": len(positions),
            "positions": ",".join(sorted(positions.keys())),
        })

        # Max drawdown exit
        if config.max_drawdown_exit > 0 and len(equity_rows) > 1:
            peak_eq = max(r["equity"] for r in equity_rows)
            dd = (equity - peak_eq) / peak_eq if peak_eq > 0 else 0
            if dd < -config.max_drawdown_exit:
                for sym, pos in list(positions.items()):
                    price = float(current_bars.get(sym, pd.Series({"close": pos.entry_price}))["close"])
                    fee_rate = get_fee(sym, config)
                    proceeds = pos.quantity * price * (1 - fee_rate)
                    pnl = proceeds - pos.cost_basis
                    cash += proceeds
                    trades.append(TradeLog(
                        bar_idx=bar_idx, timestamp=ts, symbol=sym,
                        side="sell", price=price, quantity=pos.quantity,
                        notional=pos.quantity * price,
                        fee=pos.quantity * price * fee_rate,
                        pnl=pnl, reason="max_dd_exit",
                    ))
                positions.clear()
                equity_rows[-1]["equity"] = cash
                print(f"  MAX DD EXIT: dd={dd:.1%} at bar {bar_idx}, equity=${cash:.0f}")
                break

    equity_df = pd.DataFrame(equity_rows)
    metrics = _compute_metrics(equity_df, trades)
    return equity_df, trades, metrics


def _compute_metrics(equity_df: pd.DataFrame, trades: List[TradeLog]) -> Dict[str, float]:
    if equity_df.empty or len(equity_df) < 2:
        return {}
    values = equity_df["equity"].values.astype(float)
    returns = np.diff(values) / np.clip(values[:-1], 1e-8, None)
    total_return = (values[-1] - values[0]) / values[0]
    mean_ret = returns.mean()
    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) > 1 else 1e-8
    sortino = mean_ret / max(downside_std, 1e-8) * np.sqrt(8760)  # hourly annualization
    peak = np.maximum.accumulate(values)
    max_dd = float(((values - peak) / peak).min())
    exits = [t for t in trades if t.side in ("sell", "cover")]
    win_rate = len([t for t in exits if t.pnl > 0]) / len(exits) * 100 if exits else 0

    return {
        "total_return_pct": float(total_return * 100),
        "sortino": float(sortino),
        "max_drawdown_pct": float(max_dd * 100),
        "win_rate": win_rate,
        "n_trades": len(trades),
        "final_equity": float(values[-1]),
        "n_bars": len(equity_df),
    }


def load_hourly_bars(data_dir: str, symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Load hourly OHLCV for stocks and crypto."""
    from pathlib import Path
    result = {}
    data_path = Path(data_dir)

    for sym in symbols:
        # Try multiple paths
        candidates = [
            data_path / "stocks" / f"{sym}.csv",
            data_path / "crypto" / f"{sym}.csv",
            data_path / f"{sym}.csv",
        ]
        for fpath in candidates:
            if fpath.exists():
                df = pd.read_csv(fpath)
                if "timestamp" in df.columns and len(df) > 48:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
                    result[sym] = df[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
                    break
    return result
