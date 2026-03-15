"""Work-stealing dip-buying strategy for large crypto universe.

Supports:
- Long dip-buying (buy on X% dip from recent high)
- Short pump-selling (short on X% pump from recent low)
- Leverage (margin mode, configurable max)
- FDUSD 0% fee for BTC/ETH, USDT 10bps for rest
- Margin interest costs
- Work-stealing: limited positions, best candidates fill first
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

FDUSD_SYMBOLS = {"BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD"}
MARGIN_ANNUAL_RATE = 0.0625  # 6.25% per year


@dataclass
class WorkStealConfig:
    dip_pct: float = 0.10
    proximity_pct: float = 0.005
    profit_target_pct: float = 0.05
    stop_loss_pct: float = 0.08
    max_positions: int = 5
    max_hold_days: int = 14
    lookback_days: int = 20
    ref_price_method: str = "high"
    maker_fee: float = 0.001  # default for USDT pairs
    fdusd_fee: float = 0.0    # 0% for FDUSD pairs
    initial_cash: float = 10_000.0
    equal_weight: bool = True
    trailing_stop_pct: float = 0.0
    reentry_cooldown_days: int = 1
    max_leverage: float = 1.0  # 1.0 = no leverage
    enable_shorts: bool = False
    short_pump_pct: float = 0.10
    margin_annual_rate: float = MARGIN_ANNUAL_RATE
    max_position_pct: float = 0.25
    # Trend filters
    sma_filter_period: int = 0
    market_breadth_filter: float = 0.0
    rsi_filter: int = 0
    volume_spike_filter: float = 0.0
    # Risk management
    max_drawdown_exit: float = 0.25  # bail out if DD exceeds this (0=disabled)


@dataclass
class Position:
    symbol: str
    direction: str  # "long" or "short"
    entry_price: float
    entry_date: pd.Timestamp
    quantity: float  # positive for both long/short (abs size)
    cost_basis: float
    peak_price: float  # for trailing stop (long: highest, short: lowest)
    target_exit_price: float
    stop_price: float
    margin_borrowed: float = 0.0  # USDT borrowed for leverage


@dataclass
class TradeLog:
    timestamp: pd.Timestamp
    symbol: str
    side: str  # "buy" or "sell" or "short" or "cover"
    price: float
    quantity: float
    notional: float
    fee: float
    pnl: float = 0.0
    reason: str = ""
    direction: str = "long"


def get_fee(symbol: str, config: WorkStealConfig) -> float:
    if symbol in FDUSD_SYMBOLS:
        return config.fdusd_fee
    return config.maker_fee


def compute_ref_price(bars: pd.DataFrame, method: str, lookback: int) -> float:
    if len(bars) < 2:
        return bars["close"].iloc[-1]
    window = bars.tail(lookback)
    if method == "high":
        return float(window["high"].max())
    elif method == "sma":
        return float(window["close"].mean())
    elif method == "close":
        return float(window["close"].max())
    return float(window["high"].max())


def compute_ref_low(bars: pd.DataFrame, lookback: int) -> float:
    if len(bars) < 2:
        return bars["close"].iloc[-1]
    window = bars.tail(lookback)
    return float(window["low"].min())


def compute_sma(bars: pd.DataFrame, period: int) -> float:
    if len(bars) < period:
        return float(bars["close"].mean())
    return float(bars["close"].iloc[-period:].mean())


def compute_rsi(bars: pd.DataFrame, period: int = 14) -> float:
    if len(bars) < period + 1:
        return 50.0
    closes = bars["close"].values[-(period+1):]
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def compute_volume_ratio(bars: pd.DataFrame, period: int = 20) -> float:
    if len(bars) < period + 1 or "volume" not in bars.columns:
        return 1.0
    avg_vol = bars["volume"].iloc[-(period+1):-1].mean()
    current_vol = bars["volume"].iloc[-1]
    if avg_vol <= 0:
        return 1.0
    return float(current_vol / avg_vol)


def compute_atr(bars: pd.DataFrame, period: int = 14) -> float:
    if len(bars) < period + 1:
        return float(bars["high"].iloc[-1] - bars["low"].iloc[-1])
    high = bars["high"].values[-period:]
    low = bars["low"].values[-period:]
    close = bars["close"].values[-(period+1):-1]
    tr = np.maximum(high - low, np.maximum(np.abs(high - close), np.abs(low - close)))
    return float(np.mean(tr))


def _compute_margin_interest(pos: Position, current_date: pd.Timestamp, rate: float) -> float:
    if pos.margin_borrowed <= 0:
        return 0.0
    days_held = max(1, (current_date - pos.entry_date).days)
    daily_rate = rate / 365.0
    return pos.margin_borrowed * daily_rate * days_held


def run_worksteal_backtest(
    all_bars: Dict[str, pd.DataFrame],
    config: WorkStealConfig,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[TradeLog], Dict[str, float]]:
    for sym in list(all_bars.keys()):
        df = all_bars[sym].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        all_bars[sym] = df

    all_dates = sorted(set().union(*[set(df["timestamp"].tolist()) for df in all_bars.values()]))
    if start_date:
        start_ts = pd.Timestamp(start_date, tz="UTC")
        all_dates = [d for d in all_dates if d >= start_ts]
    if end_date:
        end_ts = pd.Timestamp(end_date, tz="UTC")
        all_dates = [d for d in all_dates if d <= end_ts]

    cash = config.initial_cash
    positions: Dict[str, Position] = {}
    trades: List[TradeLog] = []
    equity_rows: List[Dict] = []
    last_exit: Dict[str, pd.Timestamp] = {}

    for i, date in enumerate(all_dates):
        current_bars: Dict[str, pd.Series] = {}
        history: Dict[str, pd.DataFrame] = {}
        for sym, df in all_bars.items():
            mask = df["timestamp"] <= date
            hist = df[mask]
            if hist.empty:
                continue
            bar = hist.iloc[-1]
            if bar["timestamp"] != date:
                continue
            current_bars[sym] = bar
            history[sym] = hist

        if not current_bars:
            continue

        # Compute current equity for position sizing
        inv_value = 0.0
        for sym, pos in positions.items():
            if sym in current_bars:
                close = float(current_bars[sym]["close"])
                if pos.direction == "long":
                    inv_value += pos.quantity * close
                else:  # short
                    inv_value += pos.quantity * (2 * pos.entry_price - close)
        current_equity = cash + inv_value

        # 1. Check exits
        symbols_to_exit = []
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
            else:  # short
                pos.peak_price = min(pos.peak_price, low)

                if low <= pos.target_exit_price:
                    exit_price = pos.target_exit_price
                    exit_reason = "profit_target"
                elif high >= pos.stop_price:
                    exit_price = pos.stop_price
                    exit_reason = "stop_loss"
                elif config.trailing_stop_pct > 0:
                    trail = pos.peak_price * (1 + config.trailing_stop_pct)
                    if high >= trail:
                        exit_price = trail
                        exit_reason = "trailing_stop"

            # Margin call: force close short if loss exceeds collateral
            if exit_price is None and pos.direction == "short":
                unrealized_loss = pos.quantity * (close - pos.entry_price)
                if unrealized_loss > pos.cost_basis * 0.8:  # 80% of collateral
                    exit_price = close
                    exit_reason = "margin_call"

            if exit_price is None and config.max_hold_days > 0:
                held = (date - pos.entry_date).days
                if held >= config.max_hold_days:
                    exit_price = close
                    exit_reason = "max_hold"

            if exit_price is not None:
                symbols_to_exit.append((sym, exit_price, exit_reason))

        for sym, exit_price, reason in symbols_to_exit:
            pos = positions[sym]
            fee_rate = get_fee(sym, config)
            margin_interest = _compute_margin_interest(pos, date, config.margin_annual_rate)

            if pos.direction == "long":
                proceeds = pos.quantity * exit_price * (1 - fee_rate)
                pnl = proceeds - pos.cost_basis - margin_interest
                cash += proceeds
                side = "sell"
            else:  # short cover
                cover_cost = pos.quantity * exit_price * (1 + fee_rate)
                # PnL = (entry_price - exit_price) * quantity - fees - interest
                pnl = pos.quantity * (pos.entry_price - exit_price) - \
                      pos.quantity * exit_price * fee_rate - \
                      pos.quantity * pos.entry_price * fee_rate - margin_interest
                # Return margin collateral + P&L
                cash += pos.cost_basis + pnl  # cost_basis = margin locked
                side = "cover"

            trades.append(TradeLog(
                timestamp=date, symbol=sym, side=side,
                price=exit_price, quantity=pos.quantity,
                notional=pos.quantity * exit_price,
                fee=pos.quantity * exit_price * fee_rate + margin_interest,
                pnl=pnl, reason=reason, direction=pos.direction,
            ))
            last_exit[sym] = date
            del positions[sym]

        # 2. Market breadth filter
        if config.market_breadth_filter > 0:
            n_dipping = 0
            n_total = 0
            for sym2 in current_bars:
                if sym2 not in history or len(history[sym2]) < 5:
                    continue
                n_total += 1
                prev_close = float(history[sym2].iloc[-2]["close"])
                if float(current_bars[sym2]["close"]) < prev_close:
                    n_dipping += 1
            breadth_ratio = n_dipping / max(n_total, 1)
            skip_entries = breadth_ratio > config.market_breadth_filter
        else:
            skip_entries = False

        # 3. Check entries (work-stealing)
        if len(positions) < config.max_positions and not skip_entries:
            candidates = []

            for sym, bar in current_bars.items():
                if sym in positions:
                    continue
                if sym in last_exit:
                    if (date - last_exit[sym]).days < config.reentry_cooldown_days:
                        continue
                if sym not in history or len(history[sym]) < config.lookback_days:
                    continue

                close = float(bar["close"])
                low_bar = float(bar["low"])
                high_bar = float(bar["high"])

                # SMA trend filter: only enter longs if above SMA
                if config.sma_filter_period > 0:
                    sma = compute_sma(history[sym], config.sma_filter_period)
                    if close < sma:
                        continue  # skip: below trend

                # RSI filter: only buy oversold
                if config.rsi_filter > 0:
                    rsi = compute_rsi(history[sym], 14)
                    if rsi > config.rsi_filter:
                        continue  # skip: not oversold enough

                # Volume spike filter
                if config.volume_spike_filter > 0:
                    vol_ratio = compute_volume_ratio(history[sym], 20)
                    if vol_ratio < config.volume_spike_filter:
                        continue  # skip: no volume confirmation

                # Long candidate: dip from recent high
                ref_high = compute_ref_price(history[sym], config.ref_price_method, config.lookback_days)
                buy_target = ref_high * (1 - config.dip_pct)
                dist_long = (close - buy_target) / ref_high

                if dist_long <= config.proximity_pct:
                    dip_score = -dist_long
                    fill_price = max(buy_target, low_bar)
                    candidates.append((sym, "long", dip_score, fill_price, bar))

                # Short candidate: pump from recent low
                if config.enable_shorts:
                    ref_low = compute_ref_low(history[sym], config.lookback_days)
                    short_target = ref_low * (1 + config.short_pump_pct)
                    dist_short = (short_target - close) / ref_low

                    if dist_short <= config.proximity_pct:
                        pump_score = -dist_short
                        fill_price = min(short_target, high_bar)
                        candidates.append((sym, "short", pump_score, fill_price, bar))

            candidates.sort(key=lambda x: x[2], reverse=True)

            slots = config.max_positions - len(positions)
            # Use initial_cash for sizing shorts to prevent compounding spiral
            base_equity = config.initial_cash
            for sym, direction, score, fill_price, bar in candidates[:slots]:
                if sym in positions:
                    continue
                if cash <= 0 and direction == "long":
                    continue

                fee_rate = get_fee(sym, config)
                # Size based on initial equity, not current (prevents leverage spiral)
                max_alloc = base_equity * config.max_position_pct * config.max_leverage

                if direction == "long":
                    alloc = min(max_alloc, cash)
                    quantity = alloc / (fill_price * (1 + fee_rate))
                    if quantity <= 0:
                        continue

                    actual_cost = quantity * fill_price * (1 + fee_rate)
                    borrowed = max(0, actual_cost - cash)
                    cash -= min(actual_cost, cash)

                    positions[sym] = Position(
                        symbol=sym, direction="long",
                        entry_price=fill_price, entry_date=date,
                        quantity=quantity, cost_basis=actual_cost,
                        peak_price=float(bar["high"]),
                        target_exit_price=fill_price * (1 + config.profit_target_pct),
                        stop_price=fill_price * (1 - config.stop_loss_pct),
                        margin_borrowed=borrowed,
                    )
                    trades.append(TradeLog(
                        timestamp=date, symbol=sym, side="buy",
                        price=fill_price, quantity=quantity,
                        notional=quantity * fill_price,
                        fee=quantity * fill_price * fee_rate,
                        reason=f"dip_buy(score={score:.4f})",
                        direction="long",
                    ))
                else:  # short
                    alloc = min(max_alloc, base_equity * config.max_position_pct)
                    quantity = alloc / fill_price
                    if quantity <= 0:
                        continue
                    # Borrow asset, sell it - but margin collateral comes from cash
                    margin_required = alloc * 0.5  # 50% margin requirement
                    if cash < margin_required:
                        continue
                    proceeds = quantity * fill_price * (1 - fee_rate)
                    cash -= margin_required  # lock up margin
                    borrowed = quantity * fill_price

                    positions[sym] = Position(
                        symbol=sym, direction="short",
                        entry_price=fill_price, entry_date=date,
                        quantity=quantity, cost_basis=margin_required,
                        peak_price=float(bar["low"]),
                        target_exit_price=fill_price * (1 - config.profit_target_pct),
                        stop_price=fill_price * (1 + config.stop_loss_pct),
                        margin_borrowed=borrowed,
                    )
                    trades.append(TradeLog(
                        timestamp=date, symbol=sym, side="short",
                        price=fill_price, quantity=quantity,
                        notional=quantity * fill_price,
                        fee=quantity * fill_price * fee_rate,
                        reason=f"pump_short(score={score:.4f})",
                        direction="short",
                    ))

        # 3. Compute equity
        inventory_value = 0.0
        for sym, pos in positions.items():
            if sym in current_bars:
                close = float(current_bars[sym]["close"])
                interest = _compute_margin_interest(pos, date, config.margin_annual_rate)
                if pos.direction == "long":
                    inventory_value += pos.quantity * close - interest
                else:
                    # Short unrealized P&L (margin collateral already in cash)
                    unrealized_pnl = pos.quantity * (pos.entry_price - close) - interest
                    inventory_value += unrealized_pnl
            else:
                if pos.direction == "long":
                    inventory_value += pos.quantity * pos.entry_price

        equity = cash + inventory_value
        equity_rows.append({
            "timestamp": date,
            "equity": equity,
            "cash": cash,
            "inventory_value": inventory_value,
            "n_positions": len(positions),
            "n_long": sum(1 for p in positions.values() if p.direction == "long"),
            "n_short": sum(1 for p in positions.values() if p.direction == "short"),
            "positions": ",".join(f"{p.direction[0]}:{s}" for s, p in positions.items()),
            "leverage": (abs(inventory_value) + cash) / max(equity, 1) if equity > 0 else 0,
        })

        # Early exit on max drawdown
        if config.max_drawdown_exit > 0 and len(equity_rows) > 1:
            peak_eq = max(r["equity"] for r in equity_rows)
            dd = (equity - peak_eq) / peak_eq if peak_eq > 0 else 0
            if dd < -config.max_drawdown_exit:
                # Force close all positions
                for sym, pos in list(positions.items()):
                    if sym in current_bars:
                        close_p = float(current_bars[sym]["close"])
                    else:
                        close_p = pos.entry_price
                    fee_rate = get_fee(sym, config)
                    if pos.direction == "long":
                        proceeds = pos.quantity * close_p * (1 - fee_rate)
                        pnl = proceeds - pos.cost_basis
                        cash += proceeds
                    else:
                        pnl = pos.quantity * (pos.entry_price - close_p)
                        cash += pos.cost_basis + pnl
                    trades.append(TradeLog(
                        timestamp=date, symbol=sym,
                        side="sell" if pos.direction == "long" else "cover",
                        price=close_p, quantity=pos.quantity,
                        notional=pos.quantity * close_p,
                        fee=pos.quantity * close_p * fee_rate,
                        pnl=pnl, reason="max_dd_exit",
                        direction=pos.direction,
                    ))
                positions.clear()
                n_days_active = len(equity_rows)
                print(f"  EARLY EXIT: DD={dd:.1%} after {n_days_active}d, "
                      f"equity=${equity:.0f} -> ${cash:.0f}")
                equity_rows[-1]["equity"] = cash
                equity_rows[-1]["n_positions"] = 0
                break

    equity_df = pd.DataFrame(equity_rows)
    metrics = compute_metrics(equity_df, config, trades)
    return equity_df, trades, metrics


def compute_metrics(equity_df: pd.DataFrame, config: WorkStealConfig,
                    trades: Optional[List[TradeLog]] = None) -> Dict[str, float]:
    if equity_df.empty or len(equity_df) < 2:
        return {}
    values = equity_df["equity"].values.astype(float)
    returns = np.diff(values) / np.clip(values[:-1], 1e-8, None)

    total_return = (values[-1] - values[0]) / values[0]
    mean_ret = returns.mean()
    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) > 1 else 1e-8

    sortino = mean_ret / max(downside_std, 1e-8) * np.sqrt(365)
    sharpe = mean_ret / max(returns.std(), 1e-8) * np.sqrt(365)

    peak = np.maximum.accumulate(values)
    drawdown = (values - peak) / peak
    max_dd = float(drawdown.min())

    n_days = len(equity_df)
    win_rate = 0.0
    if trades:
        exits = [t for t in trades if t.side in ("sell", "cover")]
        wins = [t for t in exits if t.pnl > 0]
        win_rate = len(wins) / len(exits) * 100 if exits else 0

    return {
        "total_return": float(total_return),
        "total_return_pct": float(total_return * 100),
        "sortino": float(sortino),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "max_drawdown_pct": float(max_dd * 100),
        "n_days": n_days,
        "final_equity": float(values[-1]),
        "mean_daily_return": float(mean_ret),
        "win_rate": win_rate,
    }


def load_daily_bars(data_dir: str, symbols: List[str]) -> Dict[str, pd.DataFrame]:
    from pathlib import Path
    data_path = Path(data_dir)
    result = {}
    for sym in symbols:
        base = sym.replace("USD", "").replace("USDT", "")
        # Prefer USDT (Binance Vision, more up-to-date) over USD (Alpaca)
        candidates = [f"{base}USDT.csv", f"{sym}.csv", f"{base}USD.csv"]
        for fname in candidates:
            fpath = data_path / fname
            if fpath.exists():
                df = pd.read_csv(fpath)
                if "timestamp" in df.columns and len(df) > 30:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                    df["symbol"] = f"{base}USD"
                    result[f"{base}USD"] = df
                    break
    return result


def print_results(equity_df: pd.DataFrame, trades: List[TradeLog], metrics: Dict[str, float]):
    print(f"\n{'='*60}")
    print(f"WORK-STEALING BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"Total Return: {metrics.get('total_return_pct', 0):.2f}%")
    print(f"Sortino:      {metrics.get('sortino', 0):.2f}")
    print(f"Sharpe:       {metrics.get('sharpe', 0):.2f}")
    print(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"Win Rate:     {metrics.get('win_rate', 0):.1f}%")
    print(f"Final Equity: ${metrics.get('final_equity', 0):.2f}")
    print(f"Days:         {metrics.get('n_days', 0)}")

    buys = [t for t in trades if t.side in ("buy", "short")]
    exits = [t for t in trades if t.side in ("sell", "cover")]
    winning = [t for t in exits if t.pnl > 0]
    losing = [t for t in exits if t.pnl <= 0]

    print(f"\nEntries: {len(buys)} ({sum(1 for t in buys if t.direction=='long')} long, "
          f"{sum(1 for t in buys if t.direction=='short')} short)")
    print(f"Exits:   {len(exits)}")
    if exits:
        if winning:
            print(f"Avg Win:  ${np.mean([t.pnl for t in winning]):.2f}")
        if losing:
            print(f"Avg Loss: ${np.mean([t.pnl for t in losing]):.2f}")
        total_pnl = sum(t.pnl for t in exits)
        total_fees = sum(t.fee for t in trades)
        print(f"Total PnL: ${total_pnl:.2f}")
        print(f"Total Fees: ${total_fees:.2f}")

    sym_pnl: Dict[str, float] = {}
    sym_trades: Dict[str, int] = {}
    for t in exits:
        sym_pnl[t.symbol] = sym_pnl.get(t.symbol, 0) + t.pnl
        sym_trades[t.symbol] = sym_trades.get(t.symbol, 0) + 1

    if sym_pnl:
        print(f"\nPer-Symbol PnL:")
        for sym in sorted(sym_pnl, key=lambda s: sym_pnl[s], reverse=True):
            print(f"  {sym:12s} ${sym_pnl[sym]:>8.2f} ({sym_trades[sym]} trades)")

    reasons: Dict[str, int] = {}
    for t in exits:
        reasons[t.reason] = reasons.get(t.reason, 0) + 1
    if reasons:
        print(f"\nExit Reasons:")
        for r, c in sorted(reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"  {r:20s} {c}")
