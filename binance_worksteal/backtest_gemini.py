#!/usr/bin/env python3
"""Backtest work-stealing strategy with Gemini LLM overlay.

Compares: rule-only vs Gemini-enhanced on same data.
Gemini recalibrates buy/sell/stop prices for each candidate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from binance_worksteal.strategy import (
    WorkStealConfig, Position, TradeLog, compute_metrics,
    compute_ref_price, compute_sma, get_fee, load_daily_bars,
    FDUSD_SYMBOLS,
)
from binance_worksteal.gemini_overlay import (
    DailyTradePlan, build_daily_prompt, call_gemini_daily,
    load_forecast_daily, list_forecast_coverage,
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
    llm_skips = 0

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

        inv_value = 0.0
        for sym, pos in positions.items():
            if sym in current_bars:
                inv_value += pos.quantity * float(current_bars[sym]["close"])
        current_equity = cash + inv_value

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
                    exit_price = close
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

                cached = _get_cached_plan(f"entry_{sym}", date_str) if use_cache else None
                if cached:
                    plan = cached
                else:
                    fc = load_forecast_daily(sym, cache_root=forecast_cache_root, as_of=date)
                    uni_lines = []
                    for s2, b2 in sorted(current_bars.items()):
                        if s2 in history and len(history[s2]) >= 5:
                            c2 = float(b2["close"])
                            prev = float(history[s2].iloc[-2]["close"])
                            chg = (c2 - prev) / prev * 100
                            in_pos = "HELD" if s2 in positions else ""
                            uni_lines.append(f"  {s2:10s} ${c2:>10.2f} {chg:+5.1f}% {in_pos}")
                    universe_summary = "\n".join(uni_lines[:15])

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

                if plan and plan.action in ("buy", "adjust") and plan.confidence > 0.3:
                    buy_price = plan.buy_price if plan.buy_price > 0 else rule_fill_price
                    sell_target = plan.sell_price if plan.sell_price > 0 else rule_fill_price * (1 + config.profit_target_pct)
                    stop = plan.stop_price if plan.stop_price > 0 else rule_fill_price * (1 - config.stop_loss_pct)
                    confidence = plan.confidence
                    llm_overrides += 1
                    source = f"gemini(conf={confidence:.2f})"
                elif plan and plan.action == "hold" and plan.confidence > 0.5:
                    llm_skips += 1
                    continue
                else:
                    buy_price = rule_fill_price
                    sell_target = rule_fill_price * (1 + config.profit_target_pct)
                    stop = rule_fill_price * (1 - config.stop_loss_pct)
                    confidence = 0.5
                    source = "rule_only"

                low_bar = float(bar["low"])
                if low_bar > buy_price:
                    continue

                fee_rate = get_fee(sym, config)
                alloc = min(cash, current_equity * config.max_position_pct)
                quantity = alloc / (buy_price * (1 + fee_rate))
                if quantity <= 0:
                    continue

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
    metrics = compute_metrics(equity_df, config, trades)
    metrics["n_trades"] = len(trades)
    metrics["llm_calls"] = llm_calls
    metrics["llm_overrides"] = llm_overrides
    metrics["llm_skips"] = llm_skips
    return equity_df, trades, metrics


def _print_comparison(rule_m: dict, gemini_m: dict, model_name: str):
    print(f"\n{'='*70}")
    print(f"COMPARISON: Rule-Only vs Gemini ({model_name})")
    print(f"{'='*70}")
    print(f"{'Metric':<20s} {'Rule-Only':>12s} {'Gemini':>12s} {'Delta':>12s}")
    print(f"{'-'*56}")
    for key, label, fmt in [
        ("total_return_pct", "Return %", ".2f"),
        ("sortino", "Sortino", ".2f"),
        ("sharpe", "Sharpe", ".2f"),
        ("max_drawdown_pct", "Max DD %", ".2f"),
        ("win_rate", "Win Rate %", ".1f"),
        ("n_trades", "Trades", ".0f"),
    ]:
        rv = rule_m.get(key, 0)
        gv = gemini_m.get(key, 0)
        delta = gv - rv
        print(f"{label:<20s} {rv:>12{fmt}} {gv:>12{fmt}} {delta:>+12{fmt}}")
    print(f"{'='*70}")
    if "llm_calls" in gemini_m:
        print(f"LLM calls: {gemini_m['llm_calls']}, overrides: {gemini_m.get('llm_overrides',0)}, skips: {gemini_m.get('llm_skips',0)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="trainingdata/train")
    parser.add_argument("--start-date", "--start", default=None)
    parser.add_argument("--end-date", "--end", default=None)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--gemini-model", "--model", default="gemini-2.5-flash")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--forecast-cache-root", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--use-cache", action="store_true", default=True)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--rule-only", action="store_true")
    parser.add_argument("--compare", action="store_true", help="Run both rule-only and Gemini, print comparison table")
    parser.add_argument("--forecast-coverage", action="store_true", help="Print Chronos2 forecast coverage report")
    args = parser.parse_args()

    if args.no_cache:
        args.use_cache = False

    from binance_worksteal.backtest import FULL_UNIVERSE
    symbols = list(args.symbols) if args.symbols else list(FULL_UNIVERSE)

    if args.forecast_coverage:
        cov = list_forecast_coverage(symbols, cache_root=args.forecast_cache_root)
        print(f"Chronos2 h24 forecast coverage: {len(cov['covered'])}/{len(symbols)}")
        print(f"  Cache dir: {cov['cache_dir']}")
        if cov["covered"]:
            print(f"  Covered: {', '.join(sorted(cov['covered']))}")
        if cov["missing"]:
            print(f"  Missing:  {', '.join(sorted(cov['missing']))}")
        if not args.compare and not args.rule_only and not args.start_date:
            return 0

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

    if args.compare:
        from binance_worksteal.strategy import run_worksteal_backtest, print_results
        print("\n--- Running rule-only baseline ---")
        req, rtrades, rm = run_worksteal_backtest(
            {k: v.copy() for k, v in all_bars.items()}, config,
            start_date=args.start_date, end_date=args.end_date,
        )
        print("\n--- Running Gemini overlay ---")
        geq, gtrades, gm = run_gemini_backtest(
            {k: v.copy() for k, v in all_bars.items()}, config,
            start_date=args.start_date, end_date=args.end_date,
            model=args.gemini_model, use_cache=args.use_cache,
            forecast_cache_root=args.forecast_cache_root,
        )
        _print_comparison(rm, gm, args.gemini_model)
        payload = {
            "mode": "compare",
            "symbols": sorted(all_bars),
            "model": args.gemini_model,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "rule_metrics": rm,
            "gemini_metrics": gm,
        }
    elif args.rule_only:
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
            model=args.gemini_model, use_cache=args.use_cache,
            forecast_cache_root=args.forecast_cache_root,
        )
        print(f"\n=== GEMINI ENHANCED ({args.gemini_model}) ===")
        print(f"Return: {m.get('total_return_pct',0):.2f}%")
        print(f"Sortino: {m.get('sortino',0):.2f}")
        print(f"MaxDD: {m.get('max_drawdown_pct',0):.2f}%")
        print(f"WinRate: {m.get('win_rate',0):.1f}%")
        print(f"Trades: {m.get('n_trades',0)}")
        print(f"LLM calls: {m.get('llm_calls',0)}, overrides: {m.get('llm_overrides',0)}, skips: {m.get('llm_skips',0)}")
        payload = {
            "mode": "gemini_overlay",
            "symbols": sorted(all_bars),
            "model": str(args.gemini_model),
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
