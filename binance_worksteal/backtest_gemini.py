#!/usr/bin/env python3
"""Backtest work-stealing strategy with Gemini LLM overlay.

Compares: rule-only vs Gemini-enhanced on same data.
Gemini recalibrates buy/sell/stop prices for each candidate.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from binance_worksteal.strategy import (
    WorkStealConfig, Position, TradeLog,
    compute_ref_price, compute_sma, get_fee, load_daily_bars,
    FDUSD_SYMBOLS,
)
from binance_worksteal.gemini_overlay import (
    DailyTradePlan, build_daily_prompt, call_gemini_daily,
    load_forecast_daily,
)

CACHE_DIR = Path("binance_worksteal/gemini_cache")


def _cache_key(symbol: str, date: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{symbol}_{date}.json"


def _get_cached_plan(symbol: str, date: str) -> Optional[DailyTradePlan]:
    path = _cache_key(symbol, date)
    if path.exists():
        data = json.loads(path.read_text())
        return DailyTradePlan(**data)
    return None


def _set_cached_plan(symbol: str, date: str, plan: DailyTradePlan):
    path = _cache_key(symbol, date)
    path.write_text(json.dumps({
        "action": plan.action, "buy_price": plan.buy_price,
        "sell_price": plan.sell_price, "stop_price": plan.stop_price,
        "confidence": plan.confidence, "reasoning": plan.reasoning,
    }))


def run_gemini_backtest(
    all_bars: Dict[str, pd.DataFrame],
    config: WorkStealConfig,
    start_date: str,
    end_date: str,
    model: str = "gemini-2.5-flash",
    use_cache: bool = True,
    api_key: Optional[str] = None,
    forecast_cache_root: Optional[Path] = None,
) -> tuple:
    """Run backtest with Gemini overlay on candidates identified by rule-based system."""
    for sym in list(all_bars.keys()):
        df = all_bars[sym].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        all_bars[sym] = df

    all_dates = sorted(set().union(*[set(df["timestamp"].tolist()) for df in all_bars.values()]))
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC")
    all_dates = [d for d in all_dates if start_ts <= d <= end_ts]

    cash = config.initial_cash
    positions: Dict[str, Position] = {}
    trades: List[TradeLog] = []
    equity_rows = []
    last_exit: Dict[str, pd.Timestamp] = {}
    recent_trades: List[dict] = []
    llm_calls = 0
    llm_overrides = 0

    for date in all_dates:
        current_bars = {}
        history = {}
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

        # Compute equity
        inv_value = 0.0
        for sym, pos in positions.items():
            if sym in current_bars:
                inv_value += pos.quantity * float(current_bars[sym]["close"])
        current_equity = cash + inv_value

        # 1. Exits -- check all positions
        exits = []
        for sym, pos in list(positions.items()):
            if sym not in current_bars:
                continue
            bar = current_bars[sym]
            close = float(bar["close"])
            high = float(bar["high"])
            low = float(bar["low"])
            pos.peak_price = max(pos.peak_price, high)

            exit_price = None
            reason = ""

            if high >= pos.target_exit_price:
                exit_price = pos.target_exit_price
                reason = "profit_target"
            elif low <= pos.stop_price:
                exit_price = pos.stop_price
                reason = "stop_loss"
            elif config.trailing_stop_pct > 0:
                trail = pos.peak_price * (1 - config.trailing_stop_pct)
                if low <= trail:
                    exit_price = trail
                    reason = "trailing_stop"

            if exit_price is None and config.max_hold_days > 0:
                held = (date - pos.entry_date).days
                if held >= config.max_hold_days:
                    exit_price = close
                    reason = "max_hold"

            # Ask Gemini if we should adjust exit for open positions
            if exit_price is None and sym in history:
                date_str = str(date.date())
                cached = _get_cached_plan(f"exit_{sym}", date_str) if use_cache else None
                if cached:
                    plan = cached
                else:
                    pos_info = {
                        "quantity": pos.quantity, "entry_price": pos.entry_price,
                        "held_days": (date - pos.entry_date).days,
                        "peak_price": pos.peak_price,
                        "target_sell": pos.target_exit_price,
                        "stop_price": pos.stop_price,
                    }
                    fc = load_forecast_daily(sym, cache_root=forecast_cache_root, as_of=date)
                    prompt = build_daily_prompt(
                        symbol=sym, bars=history[sym], current_price=close,
                        rule_signal={"action": "hold_or_exit"},
                        position_info=pos_info, recent_trades=recent_trades[-5:],
                        forecast_24h=fc, fee_bps=0 if sym in FDUSD_SYMBOLS else 10,
                    )
                    plan = call_gemini_daily(prompt, model=model, api_key=api_key)
                    llm_calls += 1
                    if plan and use_cache:
                        _set_cached_plan(f"exit_{sym}", date_str, plan)

                if plan and plan.action == "sell" and plan.confidence > 0.5:
                    exit_price = close  # market exit
                    reason = f"gemini_exit(conf={plan.confidence:.2f})"
                    llm_overrides += 1
                elif plan and plan.action == "adjust":
                    if plan.sell_price > 0:
                        pos.target_exit_price = plan.sell_price
                    if plan.stop_price > 0:
                        pos.stop_price = plan.stop_price
                    llm_overrides += 1

            if exit_price is not None:
                exits.append((sym, exit_price, reason))

        for sym, exit_price, reason in exits:
            pos = positions[sym]
            fee_rate = get_fee(sym, config)
            proceeds = pos.quantity * exit_price * (1 - fee_rate)
            pnl = proceeds - pos.cost_basis
            cash += proceeds
            trade = {
                "timestamp": str(date), "symbol": sym, "side": "sell",
                "price": exit_price, "pnl": pnl, "reason": reason,
            }
            trades.append(TradeLog(
                timestamp=date, symbol=sym, side="sell",
                price=exit_price, quantity=pos.quantity,
                notional=pos.quantity * exit_price,
                fee=pos.quantity * exit_price * fee_rate,
                pnl=pnl, reason=reason,
            ))
            recent_trades.append(trade)
            last_exit[sym] = date
            del positions[sym]

        # 2. Entries -- find rule-based candidates, enhance with Gemini
        if len(positions) < config.max_positions:
            candidates = []
            for sym, bar in current_bars.items():
                if sym in positions:
                    continue
                if sym in last_exit and (date - last_exit[sym]).days < config.reentry_cooldown_days:
                    continue
                if sym not in history or len(history[sym]) < config.lookback_days:
                    continue

                close = float(bar["close"])
                low_bar = float(bar["low"])

                if config.sma_filter_period > 0:
                    sma = compute_sma(history[sym], config.sma_filter_period)
                    if close < sma:
                        continue
                    sma_ok = True
                else:
                    sma_ok = None

                ref_high = compute_ref_price(history[sym], config.ref_price_method, config.lookback_days)
                buy_target = ref_high * (1 - config.dip_pct)
                dist = (close - buy_target) / ref_high

                if dist <= config.proximity_pct:
                    dip_score = -dist
                    fill_price = max(buy_target, low_bar)
                    candidates.append((sym, dip_score, fill_price, bar, {
                        "buy_target": buy_target,
                        "dip_score": dip_score,
                        "ref_price": ref_high,
                        "sma_ok": sma_ok,
                    }))

            candidates.sort(key=lambda x: x[1], reverse=True)
            slots = config.max_positions - len(positions)

            for sym, score, rule_fill_price, bar, rule_signal in candidates[:slots]:
                if sym in positions or cash <= 0:
                    continue

                close = float(bar["close"])
                date_str = str(date.date())

                # Ask Gemini to recalibrate
                cached = _get_cached_plan(f"entry_{sym}", date_str) if use_cache else None
                if cached:
                    plan = cached
                else:
                    fc = load_forecast_daily(sym, cache_root=forecast_cache_root, as_of=date)
                    # Build universe snapshot
                    uni_lines = []
                    for s2, b2 in sorted(current_bars.items()):
                        if s2 in history and len(history[s2]) >= 5:
                            c2 = float(b2["close"])
                            prev = float(history[s2].iloc[-2]["close"])
                            chg = (c2 - prev) / prev * 100
                            in_pos = "HELD" if s2 in positions else ""
                            uni_lines.append(f"  {s2:10s} ${c2:>10.2f} {chg:+5.1f}% {in_pos}")
                    universe_summary = "\n".join(uni_lines[:15])  # top 15

                    prompt = build_daily_prompt(
                        symbol=sym, bars=history[sym], current_price=close,
                        rule_signal=rule_signal, recent_trades=recent_trades[-5:],
                        forecast_24h=fc, universe_summary=universe_summary,
                        fee_bps=0 if sym in FDUSD_SYMBOLS else 10,
                    )
                    plan = call_gemini_daily(prompt, model=model, api_key=api_key)
                    llm_calls += 1
                    if plan and use_cache:
                        _set_cached_plan(f"entry_{sym}", date_str, plan)

                # Use Gemini prices if available, else fall back to rule
                if plan and plan.action in ("buy", "adjust") and plan.confidence > 0.3:
                    buy_price = plan.buy_price if plan.buy_price > 0 else rule_fill_price
                    sell_target = plan.sell_price if plan.sell_price > 0 else rule_fill_price * (1 + config.profit_target_pct)
                    stop = plan.stop_price if plan.stop_price > 0 else rule_fill_price * (1 - config.stop_loss_pct)
                    confidence = plan.confidence
                    llm_overrides += 1
                    source = f"gemini(conf={confidence:.2f})"
                elif plan and plan.action == "hold" and plan.confidence > 0.5:
                    continue  # Gemini says skip
                else:
                    buy_price = rule_fill_price
                    sell_target = rule_fill_price * (1 + config.profit_target_pct)
                    stop = rule_fill_price * (1 - config.stop_loss_pct)
                    confidence = 0.5
                    source = "rule_only"

                # Check fill: did low reach buy_price?
                low_bar = float(bar["low"])
                if low_bar > buy_price:
                    continue  # no fill

                fee_rate = get_fee(sym, config)
                alloc = min(cash, current_equity * config.max_position_pct)
                quantity = alloc / (buy_price * (1 + fee_rate))
                if quantity <= 0:
                    continue

                # Scale by confidence
                quantity *= min(confidence, 1.0)
                cost = quantity * buy_price * (1 + fee_rate)
                cash -= cost

                positions[sym] = Position(
                    symbol=sym, direction="long",
                    entry_price=buy_price, entry_date=date,
                    quantity=quantity, cost_basis=cost,
                    peak_price=float(bar["high"]),
                    target_exit_price=sell_target,
                    stop_price=stop,
                )
                trades.append(TradeLog(
                    timestamp=date, symbol=sym, side="buy",
                    price=buy_price, quantity=quantity,
                    notional=quantity * buy_price,
                    fee=quantity * buy_price * fee_rate,
                    reason=f"dip_buy({source})",
                ))
                recent_trades.append({
                    "timestamp": str(date), "symbol": sym, "side": "buy",
                    "price": buy_price, "pnl": 0, "reason": source,
                })

        # Equity
        inv_value = sum(
            pos.quantity * float(current_bars[sym]["close"])
            for sym, pos in positions.items()
            if sym in current_bars
        )
        equity = cash + inv_value
        equity_rows.append({
            "timestamp": date, "equity": equity, "cash": cash,
            "n_positions": len(positions),
        })

        # Max DD exit
        if config.max_drawdown_exit > 0 and len(equity_rows) > 1:
            peak_eq = max(r["equity"] for r in equity_rows)
            dd = (equity - peak_eq) / peak_eq if peak_eq > 0 else 0
            if dd < -config.max_drawdown_exit:
                for sym, pos in list(positions.items()):
                    cp = float(current_bars.get(sym, {}).get("close", pos.entry_price)) if sym in current_bars else pos.entry_price
                    fee_rate = get_fee(sym, config)
                    proceeds = pos.quantity * cp * (1 - fee_rate)
                    pnl = proceeds - pos.cost_basis
                    cash += proceeds
                    trades.append(TradeLog(
                        timestamp=date, symbol=sym, side="sell",
                        price=cp, quantity=pos.quantity,
                        notional=pos.quantity * cp,
                        fee=pos.quantity * cp * fee_rate,
                        pnl=pnl, reason="max_dd_exit",
                    ))
                positions.clear()
                print(f"  EARLY EXIT: DD={dd:.1%} after {len(equity_rows)}d, eq=${equity:.0f}->${cash:.0f}")
                equity_rows[-1]["equity"] = cash
                break

    equity_df = pd.DataFrame(equity_rows)
    metrics = _compute_metrics(equity_df, trades)
    metrics["llm_calls"] = llm_calls
    metrics["llm_overrides"] = llm_overrides
    return equity_df, trades, metrics


def _compute_metrics(equity_df, trades):
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
    dd = (values - peak) / peak
    exits = [t for t in trades if t.side == "sell"]
    wins = [t for t in exits if t.pnl > 0]
    return {
        "total_return_pct": float(total_return * 100),
        "sortino": float(sortino),
        "sharpe": float(sharpe),
        "max_drawdown_pct": float(dd.min() * 100),
        "win_rate": len(wins) / len(exits) * 100 if exits else 0,
        "n_trades": len(trades),
        "n_days": len(equity_df),
        "final_equity": float(values[-1]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="trainingdata/train")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--symbols", nargs="+", default=None, help="Optional symbol subset (default: full work-steal universe).")
    parser.add_argument("--forecast-cache-root", type=Path, default=None, help="Optional forecast cache root override.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON path for summary metrics.")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--rule-only", action="store_true", help="Run without Gemini for comparison")
    args = parser.parse_args()

    from binance_worksteal.backtest import FULL_UNIVERSE
    symbols = list(args.symbols) if args.symbols else list(FULL_UNIVERSE)
    all_bars = load_daily_bars(args.data_dir, symbols)
    if not all_bars:
        raise SystemExit(f"No daily bars loaded from {args.data_dir} for symbols={symbols}")
    print(f"Loaded {len(all_bars)} symbols")

    if not args.start_date:
        latest = max(df["timestamp"].max() for df in all_bars.values())
        args.end_date = str(latest.date())
        args.start_date = str((latest - pd.Timedelta(days=args.days)).date())
    print(f"Period: {args.start_date} to {args.end_date}")

    config = WorkStealConfig(
        dip_pct=0.20, proximity_pct=0.02, profit_target_pct=0.15,
        stop_loss_pct=0.10, max_positions=5, max_hold_days=14,
        lookback_days=20, ref_price_method="high",
        sma_filter_period=20, trailing_stop_pct=0.03,
        max_drawdown_exit=0.25,
    )

    if args.rule_only:
        from binance_worksteal.strategy import run_worksteal_backtest, print_results
        eq, trades, m = run_worksteal_backtest(
            {k: v.copy() for k, v in all_bars.items()}, config,
            start_date=args.start_date, end_date=args.end_date,
        )
        print("\n=== RULE-ONLY ===")
        print_results(eq, trades, m)
        payload = {
            "mode": "rule_only",
            "symbols": sorted(all_bars),
            "data_dir": str(args.data_dir),
            "forecast_cache_root": str(args.forecast_cache_root) if args.forecast_cache_root else None,
            "start_date": str(args.start_date),
            "end_date": str(args.end_date),
            "metrics": m,
            "trade_count": len(trades),
        }
    else:
        eq, trades, m = run_gemini_backtest(
            {k: v.copy() for k, v in all_bars.items()}, config,
            start_date=args.start_date, end_date=args.end_date,
            model=args.model, use_cache=not args.no_cache,
            forecast_cache_root=args.forecast_cache_root,
        )
        print(f"\n=== GEMINI ENHANCED ({args.model}) ===")
        print(f"Return: {m.get('total_return_pct',0):.2f}%")
        print(f"Sortino: {m.get('sortino',0):.2f}")
        print(f"MaxDD: {m.get('max_drawdown_pct',0):.2f}%")
        print(f"WinRate: {m.get('win_rate',0):.1f}%")
        print(f"Trades: {m.get('n_trades',0)}")
        print(f"LLM calls: {m.get('llm_calls',0)}, overrides: {m.get('llm_overrides',0)}")
        payload = {
            "mode": "gemini_overlay",
            "symbols": sorted(all_bars),
            "model": str(args.model),
            "data_dir": str(args.data_dir),
            "forecast_cache_root": str(args.forecast_cache_root) if args.forecast_cache_root else None,
            "start_date": str(args.start_date),
            "end_date": str(args.end_date),
            "metrics": m,
            "trade_count": len(trades),
        }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    sys.exit(main() or 0)
