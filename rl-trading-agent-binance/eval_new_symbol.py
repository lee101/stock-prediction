#!/usr/bin/env python3
"""Evaluate adding a new symbol to the hybrid spot portfolio.

Generates Gemini signals for a symbol over a backtest window,
simulates margin trading, and compares vs baseline (without the symbol).
Caches all LLM responses so re-runs are free.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rl_trading_agent_binance_prompt import build_live_prompt, load_latest_forecast
from llm_hourly_trader.providers import call_llm
from llm_hourly_trader.gemini_wrapper import TradePlan
from src.dynamic_position_sizing import compute_position_scale, compute_atr_pct, NO_FORECAST_FALLBACK_SCALE

CACHE_DIR = Path("rl-trading-agent-binance/signal_cache")

SYMBOL_FEE = {
    "BTCUSD": 0.0, "ETHUSD": 0.0,
    "SOLUSD": 0.001, "DOGEUSD": 0.001,
    "AAVEUSD": 0.001, "LINKUSD": 0.001,
    "XRPUSD": 0.001, "AVAXUSD": 0.001,
}

SYMBOL_MAX_POS = {
    "BTCUSD": 0.25, "ETHUSD": 0.20,
    "SOLUSD": 0.15, "DOGEUSD": 0.10,
    "AAVEUSD": 0.15, "LINKUSD": 0.10,
    "XRPUSD": 0.10, "AVAXUSD": 0.10,
}

BASE_MAE_PCT = {
    "BTCUSD": 0.0055, "ETHUSD": 0.008,
    "SOLUSD": 0.012, "DOGEUSD": 0.018,
    "AAVEUSD": 0.017, "LINKUSD": 0.008,
    "XRPUSD": 0.008, "AVAXUSD": 0.013,
}



def _cache_key(symbol: str, ts_str: str, model: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h = hashlib.md5(f"{symbol}_{ts_str}_{model}".encode()).hexdigest()[:12]
    return CACHE_DIR / f"{symbol}_{ts_str}_{h}.json"


def _get_cached(symbol: str, ts_str: str, model: str) -> Optional[dict]:
    path = _cache_key(symbol, ts_str, model)
    if path.exists():
        return json.loads(path.read_text())
    return None


def _set_cached(symbol: str, ts_str: str, model: str, plan: TradePlan):
    path = _cache_key(symbol, ts_str, model)
    path.write_text(json.dumps({
        "direction": plan.direction,
        "buy_price": plan.buy_price,
        "sell_price": plan.sell_price,
        "confidence": plan.confidence,
        "reasoning": plan.reasoning,
    }))


def load_hourly_bars(symbol: str, data_root: str = "trainingdatahourly/crypto") -> pd.DataFrame:
    path = Path(data_root) / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No hourly data: {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return df


def load_forecast_at(symbol: str, timestamp: pd.Timestamp, horizon: int,
                     cache_root: Optional[Path] = None) -> Optional[dict]:
    root = cache_root or (REPO / "binanceneural" / "forecast_cache")
    path = root / f"h{horizon}" / f"{symbol}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return None
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df[df["timestamp"] <= timestamp].sort_values("timestamp")
            if df.empty:
                return None
            row = df.iloc[-1]
        else:
            row = df.iloc[-1]
        result = {}
        for col in df.columns:
            try:
                result[col] = float(row[col])
            except (ValueError, TypeError):
                pass
        return result
    except Exception:
        return None


def generate_signals(
    symbol: str,
    bars: pd.DataFrame,
    start_date: str,
    end_date: str,
    model: str = "gemini-3.1-flash-lite-preview",
    thinking_level: str = "HIGH",
    rate_limit: float = 1.5,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC")
    bars = bars.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    mask = (bars["timestamp"] >= start_ts) & (bars["timestamp"] <= end_ts)
    window_bars = bars[mask].reset_index(drop=True)

    all_bars_sorted = bars.sort_values("timestamp").reset_index(drop=True)

    results = []
    api_calls = 0
    for idx in range(len(window_bars)):
        ts = window_bars.iloc[idx]["timestamp"]
        ts_str = ts.strftime("%Y%m%d_%H")

        fc_1h_pre = load_forecast_at(symbol, ts, 1)
        fc_quantiles = {}
        if fc_1h_pre:
            fc_quantiles = {
                "fc_p10": fc_1h_pre.get("predicted_close_p10", 0.0),
                "fc_p50": fc_1h_pre.get("predicted_close_p50", 0.0),
                "fc_p90": fc_1h_pre.get("predicted_close_p90", 0.0),
            }

        cached = _get_cached(symbol, ts_str, model)
        if cached:
            results.append({
                "timestamp": ts, "symbol": symbol,
                **cached, **fc_quantiles,
            })
            continue

        ts_idx = all_bars_sorted[all_bars_sorted["timestamp"] <= ts].index
        if len(ts_idx) < 24:
            continue
        context_start = max(0, ts_idx[-1] - 72)
        context_end = ts_idx[-1] + 1
        context_rows = all_bars_sorted.iloc[context_start:context_end].to_dict("records")

        current_price = float(context_rows[-1]["close"])
        fc_1h = fc_1h_pre
        fc_24h = load_forecast_at(symbol, ts, 24)

        fee_bps = int(SYMBOL_FEE.get(symbol, 0.001) * 10000)
        prompt = build_live_prompt(
            symbol, context_rows, current_price,
            fc_1h=fc_1h, fc_24h=fc_24h,
            fee_bps=fee_bps,
        )

        try:
            plan = call_llm(prompt, model=model, thinking_level=thinking_level)
            _set_cached(symbol, ts_str, model, plan)
            results.append({
                "timestamp": ts, "symbol": symbol,
                "direction": plan.direction,
                "buy_price": plan.buy_price,
                "sell_price": plan.sell_price,
                "confidence": plan.confidence,
                "reasoning": plan.reasoning,
                **fc_quantiles,
            })
            api_calls += 1
            if api_calls % 50 == 0:
                print(f"  {symbol}: {api_calls} API calls, {len(results)} total signals")
            time.sleep(rate_limit)
        except Exception as e:
            print(f"  {symbol} @ {ts_str}: API error: {e}")
            time.sleep(5)

    if api_calls > 0:
        print(f"  {symbol}: {api_calls} new API calls, {len(results)} total signals")
    return pd.DataFrame(results)


def simulate_portfolio(
    bars_map: dict[str, pd.DataFrame],
    signals_map: dict[str, pd.DataFrame],
    initial_cash: float = 10000.0,
    leverage: float = 5.0,
    margin_rate_annual: float = 0.10,
    max_hold_hours: float = 6.0,
    margin_fee: float = 0.001,
) -> dict:
    all_signals = pd.concat(list(signals_map.values()), ignore_index=True)
    all_signals["timestamp"] = pd.to_datetime(all_signals["timestamp"], utc=True)

    all_bars_list = []
    for sym, df in bars_map.items():
        b = df.copy()
        b["symbol"] = sym
        b["timestamp"] = pd.to_datetime(b["timestamp"], utc=True)
        all_bars_list.append(b)
    all_bars = pd.concat(all_bars_list, ignore_index=True)
    bars_by_symbol = {sym: grp for sym, grp in all_bars.groupby("symbol")}

    merged = all_bars.merge(all_signals, on=["timestamp", "symbol"], how="inner")
    merged = merged.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    cash = initial_cash
    positions = {}
    equity_history = []
    trades = []
    hourly_margin_rate = margin_rate_annual / 8760

    for ts, chunk in merged.groupby("timestamp", sort=True):
        for sym, pos in list(positions.items()):
            if pos["qty"] <= 0:
                continue
            borrowed = pos["qty"] * pos["cost_basis"] - pos["equity_used"]
            if borrowed > 0:
                interest = borrowed * hourly_margin_rate
                cash -= interest
                pos["total_interest"] = pos.get("total_interest", 0) + interest

        for _, row in chunk.iterrows():
            sym = row["symbol"]
            pos = positions.get(sym)
            if pos is None:
                continue
            held_hours = (ts - pos["open_time"]).total_seconds() / 3600.0
            if held_hours >= max_hold_hours:
                close_price = float(row["close"]) * 0.995
                pnl = (close_price - pos["cost_basis"]) * pos["qty"]
                fee = pos["qty"] * close_price * margin_fee
                interest = pos.get("total_interest", 0)
                cash += pos["equity_used"] + pnl - fee
                trades.append({
                    "timestamp": str(ts), "symbol": sym, "side": "close",
                    "price": close_price, "pnl": pnl - fee - interest,
                    "reason": "max_hold",
                })
                del positions[sym]

        for _, row in chunk.iterrows():
            sym = row["symbol"]
            pos = positions.get(sym)
            if pos is None:
                continue
            sell_price = float(row.get("sell_price", 0) or 0)
            if sell_price <= 0:
                continue
            if float(row["high"]) >= sell_price:
                pnl = (sell_price - pos["cost_basis"]) * pos["qty"]
                fee = pos["qty"] * sell_price * margin_fee
                interest = pos.get("total_interest", 0)
                cash += pos["equity_used"] + pnl - fee
                trades.append({
                    "timestamp": str(ts), "symbol": sym, "side": "sell",
                    "price": sell_price, "pnl": pnl - fee - interest,
                    "reason": "take_profit",
                })
                del positions[sym]

        for _, row in chunk.iterrows():
            sym = row["symbol"]
            if sym in positions:
                continue
            direction = str(row.get("direction", "hold")).lower().strip()
            if direction != "long":
                continue
            buy_price = float(row.get("buy_price", 0) or 0)
            confidence = float(row.get("confidence", 0) or 0)
            if buy_price <= 0 or confidence <= 0:
                continue
            if float(row["low"]) > buy_price:
                continue

            max_pct = SYMBOL_MAX_POS.get(sym, 0.10)
            fc_p10 = float(row.get("fc_p10", 0) or 0)
            fc_p50 = float(row.get("fc_p50", 0) or 0)
            fc_p90 = float(row.get("fc_p90", 0) or 0)
            current_price = float(row["close"])
            sym_bars = bars_by_symbol.get(sym, pd.DataFrame())
            atr_pct = compute_atr_pct(sym_bars, ts)
            base_mae = BASE_MAE_PCT.get(sym, 0.01)
            scale = compute_position_scale(fc_p10, fc_p50, fc_p90, current_price, atr_pct, base_mae)
            if scale <= 0:
                scale = NO_FORECAST_FALLBACK_SCALE
            equity_alloc = cash * max_pct * scale
            if equity_alloc < 12:
                continue
            notional = equity_alloc * leverage
            qty = notional / buy_price
            fee = qty * buy_price * margin_fee
            cash -= equity_alloc + fee
            positions[sym] = {
                "qty": qty, "cost_basis": buy_price,
                "equity_used": equity_alloc,
                "open_time": ts, "total_interest": 0,
            }
            trades.append({
                "timestamp": str(ts), "symbol": sym, "side": "buy",
                "price": buy_price, "pnl": 0, "reason": "entry",
            })

        total_equity = cash
        for sym, pos in positions.items():
            bar = chunk[chunk["symbol"] == sym]
            if len(bar) > 0:
                current_price = float(bar.iloc[0]["close"])
            else:
                current_price = pos["cost_basis"]
            pos_value = pos["qty"] * current_price
            borrowed = pos["qty"] * pos["cost_basis"] - pos["equity_used"]
            unrealized = pos_value - pos["qty"] * pos["cost_basis"]
            total_equity += pos["equity_used"] + unrealized
        equity_history.append({"timestamp": ts, "equity": total_equity})

    for sym, pos in list(positions.items()):
        last_bars = all_bars[all_bars["symbol"] == sym].sort_values("timestamp")
        if len(last_bars) > 0:
            close_price = float(last_bars.iloc[-1]["close"])
            pnl = (close_price - pos["cost_basis"]) * pos["qty"]
            fee = pos["qty"] * close_price * margin_fee
            interest = pos.get("total_interest", 0)
            cash += pos["equity_used"] + pnl - fee
            trades.append({
                "timestamp": "final", "symbol": sym, "side": "close",
                "price": close_price, "pnl": pnl - fee - interest,
                "reason": "end_of_backtest",
            })

    equity_df = pd.DataFrame(equity_history)
    final_equity = equity_df["equity"].iloc[-1] if len(equity_df) > 0 else initial_cash
    ret_pct = (final_equity - initial_cash) / initial_cash * 100

    if len(equity_df) > 1:
        returns = equity_df["equity"].pct_change().dropna()
        neg_returns = returns[returns < 0]
        sortino = (returns.mean() / neg_returns.std() * np.sqrt(8760)) if len(neg_returns) > 1 else 0
        max_dd = 0
        peak = equity_df["equity"].iloc[0]
        for eq in equity_df["equity"]:
            peak = max(peak, eq)
            dd = (peak - eq) / peak * 100
            max_dd = max(max_dd, dd)
    else:
        sortino = 0
        max_dd = 0

    trades_df = pd.DataFrame(trades)
    per_symbol_pnl = {}
    if len(trades_df) > 0:
        for sym in trades_df["symbol"].unique():
            sym_trades = trades_df[trades_df["symbol"] == sym]
            per_symbol_pnl[sym] = float(sym_trades["pnl"].sum())

    return {
        "initial_cash": initial_cash,
        "final_equity": final_equity,
        "return_pct": ret_pct,
        "sortino": sortino,
        "max_dd_pct": max_dd,
        "n_trades": len([t for t in trades if t["side"] == "buy"]),
        "per_symbol_pnl": per_symbol_pnl,
        "equity_df": equity_df,
        "trades_df": trades_df,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["BTCUSD", "ETHUSD", "SOLUSD"])
    parser.add_argument("--add-symbol", type=str, help="New symbol to evaluate adding")
    parser.add_argument("--start", default="2026-02-16")
    parser.add_argument("--end", default="2026-03-18")
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--thinking", default="HIGH")
    parser.add_argument("--leverage", type=float, default=5.0)
    parser.add_argument("--cash", type=float, default=10000.0)
    parser.add_argument("--rate-limit", type=float, default=1.0)
    parser.add_argument("--data-root", default="trainingdatahourly/crypto")
    args = parser.parse_args()

    all_symbols = list(args.symbols)
    if args.add_symbol and args.add_symbol not in all_symbols:
        all_symbols.append(args.add_symbol)

    print(f"Loading hourly bars for {all_symbols}...")
    bars_map = {}
    for sym in all_symbols:
        bars_map[sym] = load_hourly_bars(sym, args.data_root)
        print(f"  {sym}: {len(bars_map[sym])} rows")

    print(f"\nGenerating Gemini signals ({args.start} to {args.end})...")
    signals_map = {}
    for sym in all_symbols:
        print(f"  Processing {sym}...")
        signals_map[sym] = generate_signals(
            sym, bars_map[sym], args.start, args.end,
            model=args.model, thinking_level=args.thinking,
            rate_limit=args.rate_limit,
        )
        print(f"  {sym}: {len(signals_map[sym])} signals")

    # Baseline: original symbols only
    baseline_symbols = list(args.symbols)
    print(f"\n{'='*60}")
    print(f"BASELINE: {baseline_symbols}")
    baseline = simulate_portfolio(
        {s: bars_map[s] for s in baseline_symbols},
        {s: signals_map[s] for s in baseline_symbols},
        initial_cash=args.cash, leverage=args.leverage,
    )
    print(f"  Return: {baseline['return_pct']:+.2f}%")
    print(f"  Sortino: {baseline['sortino']:.2f}")
    print(f"  Max DD: {baseline['max_dd_pct']:.2f}%")
    print(f"  Trades: {baseline['n_trades']}")
    for sym, pnl in baseline['per_symbol_pnl'].items():
        print(f"    {sym}: ${pnl:+.2f}")

    if args.add_symbol:
        print(f"\n{'='*60}")
        print(f"WITH {args.add_symbol}: {all_symbols}")
        extended = simulate_portfolio(
            bars_map, signals_map,
            initial_cash=args.cash, leverage=args.leverage,
        )
        print(f"  Return: {extended['return_pct']:+.2f}%")
        print(f"  Sortino: {extended['sortino']:.2f}")
        print(f"  Max DD: {extended['max_dd_pct']:.2f}%")
        print(f"  Trades: {extended['n_trades']}")
        for sym, pnl in extended['per_symbol_pnl'].items():
            print(f"    {sym}: ${pnl:+.2f}")

        print(f"\n{'='*60}")
        print(f"COMPARISON:")
        delta_ret = extended['return_pct'] - baseline['return_pct']
        delta_sort = extended['sortino'] - baseline['sortino']
        delta_dd = extended['max_dd_pct'] - baseline['max_dd_pct']
        print(f"  Return delta: {delta_ret:+.2f}%")
        print(f"  Sortino delta: {delta_sort:+.2f}")
        print(f"  Max DD delta: {delta_dd:+.2f}%")
        new_sym_pnl = extended['per_symbol_pnl'].get(args.add_symbol, 0)
        verdict = "ACCEPT" if new_sym_pnl > 0 and delta_sort >= -0.5 else "REJECT"
        print(f"  {args.add_symbol} standalone P&L: ${new_sym_pnl:+.2f}")
        print(f"  Verdict: {verdict}")

    # Save results
    out_dir = Path(f"analysis/hybrid_symbol_eval_{time.strftime('%Y%m%d')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "baseline_symbols": baseline_symbols,
        "add_symbol": args.add_symbol,
        "start": args.start, "end": args.end,
        "model": args.model, "leverage": args.leverage,
        "baseline": {k: v for k, v in baseline.items() if k not in ("equity_df", "trades_df")},
    }
    if args.add_symbol:
        result["extended"] = {k: v for k, v in extended.items() if k not in ("equity_df", "trades_df")}

    tag = args.add_symbol or "baseline"
    result_path = out_dir / f"eval_{tag}.json"
    result_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nResults saved to {result_path}")


if __name__ == "__main__":
    main()
