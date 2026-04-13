import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from .config import GStockConfig
from .prompt import load_daily_bars, build_prompt
from .llm_client import call_llm, parse_allocation


@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    direction: str  # "long" or "short"
    exit_price: float = 0.0
    stop_price: float = 0.0
    entry_date: str = ""


@dataclass
class SimState:
    cash: float
    positions: dict = field(default_factory=dict)  # sym -> Position
    equity_curve: list = field(default_factory=list)
    trade_log: list = field(default_factory=list)
    daily_returns: list = field(default_factory=list)


def get_current_prices(bars_dict: dict, date: pd.Timestamp) -> dict:
    prices = {}
    for sym, df in bars_dict.items():
        row = df[df["timestamp"] <= date]
        if len(row) > 0:
            prices[sym] = float(row.iloc[-1]["close"])
    return prices


def get_bar_for_date(bars_dict: dict, symbol: str, date: pd.Timestamp) -> dict:
    df = bars_dict.get(symbol)
    if df is None:
        return {}
    mask = (df["timestamp"].dt.date == date.date())
    if mask.any():
        row = df[mask].iloc[0]
        return {"open": float(row["open"]), "high": float(row["high"]),
                "low": float(row["low"]), "close": float(row["close"])}
    return {}


def portfolio_value(state: SimState, prices: dict) -> float:
    val = state.cash
    for sym, pos in state.positions.items():
        p = prices.get(sym, pos.entry_price)
        if pos.direction == "long":
            val += pos.qty * p
        else:  # short
            val += pos.qty * (2 * pos.entry_price - p)
    return val


def close_position(state: SimState, sym: str, price: float, reason: str, date: str):
    pos = state.positions[sym]
    if pos.direction == "long":
        pnl = pos.qty * (price - pos.entry_price)
    else:
        pnl = pos.qty * (pos.entry_price - price)
    state.cash += pos.qty * price if pos.direction == "long" else pos.qty * (2 * pos.entry_price - price)
    state.trade_log.append({
        "symbol": sym, "direction": pos.direction, "entry": pos.entry_price,
        "exit": price, "pnl": pnl, "qty": pos.qty, "reason": reason,
        "entry_date": pos.entry_date, "exit_date": date,
    })
    del state.positions[sym]


def apply_fees(notional: float, fee_bps: float) -> float:
    return notional * fee_bps / 10000


def run_simulation(config: GStockConfig, start_date: str, end_date: str,
                   use_cache: bool = True, verbose: bool = False) -> dict:
    bars_dict = {}
    for sym in config.symbols:
        df = load_daily_bars(sym, config.data_dir)
        if not df.empty:
            bars_dict[sym] = df

    available_syms = list(bars_dict.keys())
    if verbose:
        print(f"loaded {len(available_syms)} symbols")

    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC")

    all_dates = set()
    for df in bars_dict.values():
        dates = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]["timestamp"]
        all_dates.update(dates.tolist())
    all_dates = sorted(all_dates)

    if not all_dates:
        return {"error": "no dates in range"}

    state = SimState(cash=config.initial_capital)
    prev_equity = config.initial_capital

    for i, date in enumerate(all_dates):
        prices = get_current_prices(bars_dict, date)
        eq = portfolio_value(state, prices)

        # check exits/stops from previous day's orders
        for sym in list(state.positions.keys()):
            bar = get_bar_for_date(bars_dict, sym, date)
            if not bar:
                continue
            pos = state.positions[sym]
            if pos.direction == "long":
                if pos.stop_price > 0 and bar["low"] <= pos.stop_price:
                    close_position(state, sym, pos.stop_price, "stop", str(date.date()))
                    state.cash -= apply_fees(pos.qty * pos.stop_price, config.fee_bps)
                elif pos.exit_price > 0 and bar["high"] >= pos.exit_price:
                    close_position(state, sym, pos.exit_price, "tp", str(date.date()))
                    state.cash -= apply_fees(pos.qty * pos.exit_price, config.fee_bps)
            else:  # short
                if pos.stop_price > 0 and bar["high"] >= pos.stop_price:
                    close_position(state, sym, pos.stop_price, "stop", str(date.date()))
                    state.cash -= apply_fees(pos.qty * pos.stop_price, config.fee_bps)
                elif pos.exit_price > 0 and bar["low"] <= pos.exit_price:
                    close_position(state, sym, pos.exit_price, "tp", str(date.date()))
                    state.cash -= apply_fees(pos.qty * pos.exit_price, config.fee_bps)

        # margin interest on borrowed
        if config.leverage > 1.0:
            borrowed = max(0, sum(
                pos.qty * prices.get(pos.symbol, pos.entry_price)
                for pos in state.positions.values()
            ) - config.initial_capital)
            daily_interest = borrowed * config.margin_annual_rate / 365
            state.cash -= daily_interest

        eq = portfolio_value(state, prices)

        # build positions dict for prompt
        pos_dict = {}
        for sym, pos in state.positions.items():
            pos_dict[sym] = {"qty": pos.qty, "entry_price": pos.entry_price}

        # call LLM for new allocation
        prompt = build_prompt(
            available_syms, config.data_dir, config.forecast_cache_dir,
            date, pos_dict, prices, eq, config.leverage, config.max_positions
        )

        date_str = str(date.date())
        try:
            resp = call_llm(prompt, config.model, config.temperature,
                           date_str=date_str, use_cache=use_cache)
            alloc = parse_allocation(resp)
        except Exception as e:
            if verbose:
                print(f"{date_str}: LLM error: {e}")
            alloc = {}

        # rebalance to target allocation
        total_capital = eq * config.leverage
        target_syms = set()

        for sym, spec in alloc.items():
            sym = sym.upper()
            if sym not in prices:
                continue
            if not isinstance(spec, dict):
                continue
            pct = float(spec.get("allocation_pct", 0))
            direction = spec.get("direction", "long")
            exit_price = float(spec.get("exit_price", 0))
            stop_price = float(spec.get("stop_price", 0))

            if pct <= 0:
                continue
            target_syms.add(sym)
            target_notional = total_capital * pct / 100
            target_qty = target_notional / prices[sym]

            if sym in state.positions:
                cur_pos = state.positions[sym]
                if cur_pos.direction != direction:
                    # close and reopen
                    close_position(state, sym, prices[sym], "flip", date_str)
                    state.cash -= apply_fees(cur_pos.qty * prices[sym], config.fee_bps)
                else:
                    # adjust size
                    diff_qty = target_qty - cur_pos.qty
                    if abs(diff_qty) > target_qty * 0.05:
                        diff_notional = abs(diff_qty) * prices[sym]
                        state.cash -= apply_fees(diff_notional, config.fee_bps)
                        if direction == "long":
                            state.cash -= diff_qty * prices[sym]
                        cur_pos.qty = target_qty
                        cur_pos.exit_price = exit_price
                        cur_pos.stop_price = stop_price
                    continue

            # open new position
            if sym not in state.positions:
                cost = target_qty * prices[sym]
                fee = apply_fees(cost, config.fee_bps)
                if direction == "long":
                    state.cash -= cost + fee
                else:
                    state.cash -= fee  # short: margin posted
                state.positions[sym] = Position(
                    symbol=sym, qty=target_qty, entry_price=prices[sym],
                    direction=direction, exit_price=exit_price,
                    stop_price=stop_price, entry_date=date_str
                )

        # close positions not in target
        for sym in list(state.positions.keys()):
            if sym not in target_syms:
                pos = state.positions[sym]
                close_position(state, sym, prices[sym], "rebalance", date_str)
                state.cash -= apply_fees(pos.qty * prices[sym], config.fee_bps)

        eq = portfolio_value(state, prices)
        daily_ret = (eq / prev_equity - 1) if prev_equity > 0 else 0
        state.equity_curve.append({"date": date_str, "equity": eq})
        state.daily_returns.append(daily_ret)
        prev_equity = eq

        if verbose and i % 10 == 0:
            print(f"{date_str}: eq=${eq:.2f} pos={len(state.positions)} cash=${state.cash:.2f}")

    return compute_metrics(state, config)


def compute_metrics(state: SimState, config: GStockConfig) -> dict:
    if not state.equity_curve:
        return {"error": "no data"}

    equities = [e["equity"] for e in state.equity_curve]
    returns = np.array(state.daily_returns)

    total_return = (equities[-1] / config.initial_capital - 1) * 100
    peak = np.maximum.accumulate(equities)
    drawdowns = (np.array(equities) - peak) / peak
    max_dd = float(drawdowns.min()) * 100

    # sortino
    neg_rets = returns[returns < 0]
    downside_std = np.sqrt(np.mean(neg_rets ** 2)) if len(neg_rets) > 0 else 1e-8
    mean_ret = np.mean(returns)
    sortino = float(mean_ret / downside_std * np.sqrt(365)) if downside_std > 0 else 0

    # sharpe
    std = np.std(returns) if len(returns) > 1 else 1e-8
    sharpe = float(mean_ret / std * np.sqrt(365)) if std > 0 else 0

    n_days = len(equities)
    n_trades = len(state.trade_log)
    win_trades = [t for t in state.trade_log if t["pnl"] > 0]
    win_rate = len(win_trades) / n_trades * 100 if n_trades > 0 else 0

    monthly_ret = total_return / max(1, n_days / 30)

    return {
        "total_return_pct": round(total_return, 2),
        "monthly_return_pct": round(monthly_ret, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sortino": round(sortino, 2),
        "sharpe": round(sharpe, 2),
        "n_trades": n_trades,
        "win_rate_pct": round(win_rate, 1),
        "n_days": n_days,
        "final_equity": round(equities[-1], 2),
        "equity_curve": state.equity_curve,
        "trade_log": state.trade_log,
    }
