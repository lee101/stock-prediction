#!/usr/bin/env python3
"""Sweep reprompt configurations on cached backtest data.

Evaluates different reprompt_passes / review_model / reprompt_policy combos
using the existing experiment_runner backtest loop and LLM cache.

Usage:
    source .venv313/bin/activate
    python scripts/sweep_reprompt_configs.py --symbols BTCUSD ETHUSD SOLUSD --days 30
    python scripts/sweep_reprompt_configs.py --days 30 --cache-only  # replay from cache only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_hourly_trader.experiment_runner import (
    _compute_metrics,
    load_bars,
    load_forecasts,
    get_forecast_at,
    _portfolio_equity_at_ts,
)
from llm_hourly_trader.windowing import resolve_window_end
from llm_hourly_trader.config import SYMBOL_UNIVERSE, SymbolConfig
from llm_hourly_trader.gemini_wrapper import build_prompt
from llm_hourly_trader.providers import CacheMissError, call_llm

DEFAULT_CONFIGS = [
    {"reprompt_passes": 1, "review_model": None, "reprompt_policy": "entry_only"},
    {"reprompt_passes": 2, "review_model": None, "reprompt_policy": "entry_only"},
    {"reprompt_passes": 2, "review_model": None, "reprompt_policy": "always"},
    {"reprompt_passes": 3, "review_model": None, "reprompt_policy": "entry_only"},
    {"reprompt_passes": 2, "review_model": "gemini-2.5-pro", "reprompt_policy": "entry_only"},
    {"reprompt_passes": 2, "review_model": "gemini-2.5-pro", "reprompt_policy": "always"},
    {"reprompt_passes": 3, "review_model": "gemini-2.5-pro", "reprompt_policy": "entry_only"},
]


def _config_label(cfg: dict) -> str:
    rm = cfg["review_model"] or "same"
    return f"passes={cfg['reprompt_passes']}_review={rm}_policy={cfg['reprompt_policy']}"


def _load_data(symbols: list[str]) -> tuple[list[str], dict, dict, dict, dict]:
    all_bars = {}
    all_fc_h1 = {}
    all_fc_h24 = {}
    usable = []
    sym_configs = {}
    for sym in symbols:
        bars = load_bars(sym)
        fc1 = load_forecasts(sym, "h1")
        fc24 = load_forecasts(sym, "h24")
        if bars.empty or fc1.empty:
            continue
        all_bars[sym] = bars
        all_fc_h1[sym] = fc1
        all_fc_h24[sym] = fc24
        sym_configs[sym] = SYMBOL_UNIVERSE.get(sym, SymbolConfig(sym, "crypto"))
        usable.append(sym)
    return usable, all_bars, all_fc_h1, all_fc_h24, sym_configs


def run_backtest_with_reprompt(
    usable: list[str],
    all_bars: dict,
    all_fc_h1: dict,
    all_fc_h24: dict,
    sym_configs: dict,
    days: int,
    model: str,
    variant: str,
    reprompt_passes: int,
    review_model: str | None,
    reprompt_policy: str,
    initial_cash: float = 2000.0,
    max_hold_hours: int = 6,
    leverage: float = 1.0,
    cache_only: bool = False,
    rate_limit: float = 0.0,
    end_timestamp: str | None = None,
) -> dict:
    if not usable:
        return {"error": "no usable symbols"}

    fc_ends = [all_fc_h1[s]["timestamp"].max() for s in usable if not all_fc_h1[s].empty]
    end_ts = resolve_window_end(fc_ends, end_timestamp)
    start_ts = end_ts - timedelta(days=days)

    sym_windows = {}
    for sym in usable:
        w = all_bars[sym]
        sym_windows[sym] = w[(w["timestamp"] >= start_ts) & (w["timestamp"] <= end_ts)]

    all_ts = sorted(set(ts for sym in usable for ts in sym_windows[sym]["timestamp"]))

    cash = initial_cash
    positions: dict[str, dict] = {}
    equity_history = []
    trades = []
    confidences = []
    num_entries = 0
    review_cache_ns = f"review_{review_model}" if review_model else None

    for ts in all_ts:
        prices_by_symbol = {}
        for sym in usable:
            bars_at = sym_windows[sym][sym_windows[sym]["timestamp"] == ts]
            if not bars_at.empty:
                prices_by_symbol[sym] = float(bars_at.iloc[0]["close"])

        for sym in list(positions.keys()):
            pos = positions[sym]
            held = (ts - pos["open_time"]).total_seconds() / 3600.0
            if held >= max_hold_hours:
                bars_at = sym_windows[sym][sym_windows[sym]["timestamp"] == ts]
                if bars_at.empty:
                    continue
                close_price = float(bars_at.iloc[0]["close"])
                fee_rate = sym_configs[sym].maker_fee
                sell_fee = pos["qty"] * close_price * fee_rate
                realized = (close_price - pos["cost_basis"]) * pos["qty"]
                margin_back = pos["qty"] * pos["cost_basis"] * (1 + fee_rate) / leverage
                cash += margin_back + realized - sell_fee
                trades.append({"ts": str(ts), "sym": sym, "side": "close",
                               "price": close_price, "pnl": realized, "reason": "max_hold"})
                del positions[sym]

        for sym in usable:
            bars_at = sym_windows[sym][sym_windows[sym]["timestamp"] == ts]
            if bars_at.empty:
                continue
            bar = bars_at.iloc[0]
            close_price = float(bar["close"])
            high = float(bar["high"])
            low = float(bar["low"])
            fee_rate = sym_configs[sym].maker_fee

            pos = positions.get(sym)
            position_info = {}
            if pos:
                held = (ts - pos["open_time"]).total_seconds() / 3600.0
                position_info = {
                    "qty": pos["qty"],
                    "entry_price": pos["cost_basis"],
                    "held_hours": held,
                }

            hist_slice = all_bars[sym][all_bars[sym]["timestamp"] <= ts].tail(25)
            if len(hist_slice) < 5:
                continue
            history = hist_slice.to_dict("records")
            fc_1h = get_forecast_at(all_fc_h1[sym], ts)
            fc_24h = get_forecast_at(all_fc_h24[sym], ts)

            current_pos_str = "flat"
            if pos:
                pnl_pct = (close_price - pos["cost_basis"]) / pos["cost_basis"] * 100
                current_pos_str = f"long {pos['qty']:.6f} @ ${pos['cost_basis']:.2f} ({pnl_pct:+.1f}%)"

            prompt = build_prompt(
                symbol=sym, history_rows=history,
                forecast_1h=fc_1h, forecast_24h=fc_24h,
                current_position=current_pos_str,
                cash=cash,
                equity=_portfolio_equity_at_ts(cash, positions, prices_by_symbol),
                allowed_directions=sym_configs[sym].allowed_directions,
                asset_class=sym_configs[sym].asset_class,
                maker_fee=fee_rate,
                variant=variant,
                position_info=position_info,
            )

            try:
                plan = call_llm(
                    prompt,
                    model=model,
                    cache_only=cache_only,
                    reprompt_passes=reprompt_passes,
                    review_model=review_model,
                    reprompt_policy=reprompt_policy,
                    review_cache_namespace=review_cache_ns,
                )
            except CacheMissError:
                continue

            if rate_limit > 0:
                time.sleep(rate_limit)

            if "exhausted" in plan.reasoning:
                continue

            confidences.append(float(plan.confidence))

            if pos and plan.sell_price > 0 and high >= plan.sell_price:
                sell_fee = pos["qty"] * plan.sell_price * fee_rate
                realized = (plan.sell_price - pos["cost_basis"]) * pos["qty"]
                margin_back = pos["qty"] * pos["cost_basis"] * (1 + fee_rate) / leverage
                cash += margin_back + realized - sell_fee
                trades.append({"ts": str(ts), "sym": sym, "side": "sell",
                               "price": plan.sell_price, "pnl": realized})
                del positions[sym]
                pos = None

            if not pos and plan.direction == "long" and plan.buy_price > 0 and low <= plan.buy_price:
                equity = _portfolio_equity_at_ts(cash, positions, prices_by_symbol)
                alloc = min(0.25, max(0.05, plan.confidence * 0.3)) * equity * leverage
                alloc = min(alloc, cash * 0.95)
                if alloc > 10:
                    qty = alloc / plan.buy_price
                    buy_fee = qty * plan.buy_price * fee_rate
                    cost = qty * plan.buy_price * (1 + fee_rate) / leverage
                    if cost <= cash:
                        cash -= cost
                        positions[sym] = {"qty": qty, "cost_basis": plan.buy_price, "open_time": ts}
                        num_entries += 1
                        trades.append({"ts": str(ts), "sym": sym, "side": "buy",
                                       "price": plan.buy_price, "qty": qty})

        equity = _portfolio_equity_at_ts(cash, positions, prices_by_symbol)
        equity_history.append(equity)

    equity_arr = np.array(equity_history) if equity_history else np.array([initial_cash])
    metrics = _compute_metrics(equity_arr)
    metrics["num_entries"] = num_entries
    metrics["num_trades"] = len(trades)
    metrics["avg_confidence"] = float(np.mean(confidences)) if confidences else 0.0
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Sweep reprompt configurations")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSD", "ETHUSD", "SOLUSD"])
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--model", type=str, default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--variant", type=str, default="position_context")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--rate-limit", type=float, default=0.0)
    parser.add_argument("--end-timestamp", type=str, default=None)
    args = parser.parse_args()

    print(f"Sweep: {len(DEFAULT_CONFIGS)} configs, {args.days}d, symbols={args.symbols}")
    print(f"Model: {args.model}, variant: {args.variant}, cache_only: {args.cache_only}")
    print()

    print("Loading data...", end=" ", flush=True)
    usable, all_bars, all_fc_h1, all_fc_h24, sym_configs = _load_data(args.symbols)
    print(f"{len(usable)} symbols loaded")

    results = []
    for cfg in DEFAULT_CONFIGS:
        label = _config_label(cfg)
        print(f"Running: {label} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            metrics = run_backtest_with_reprompt(
                usable=usable,
                all_bars=all_bars,
                all_fc_h1=all_fc_h1,
                all_fc_h24=all_fc_h24,
                sym_configs=sym_configs,
                days=args.days,
                model=args.model,
                variant=args.variant,
                reprompt_passes=cfg["reprompt_passes"],
                review_model=cfg["review_model"],
                reprompt_policy=cfg["reprompt_policy"],
                cache_only=args.cache_only,
                rate_limit=args.rate_limit,
                end_timestamp=args.end_timestamp,
            )
        except Exception as e:
            metrics = {"error": str(e)}
        elapsed = time.time() - t0
        metrics["config"] = label
        metrics["elapsed_s"] = round(elapsed, 1)
        results.append(metrics)
        if "error" in metrics:
            print(f"ERROR: {metrics['error']}")
        else:
            print(f"Sort={metrics.get('sortino', 0):.2f} Ret={metrics.get('return_pct', 0):+.2f}% "
                  f"Entries={metrics.get('num_entries', 0)} AvgConf={metrics.get('avg_confidence', 0):.3f} "
                  f"({elapsed:.1f}s)")

    print(f"\n{'='*90}")
    print(f"{'Config':<55} {'Sortino':>8} {'Return%':>8} {'Entries':>8} {'AvgConf':>8}")
    print(f"{'-'*55} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in sorted(results, key=lambda x: x.get("sortino", -999), reverse=True):
        if "error" in r:
            print(f"{r['config']:<55} {'ERROR':>8}")
            continue
        print(f"{r['config']:<55} {r.get('sortino', 0):>8.2f} {r.get('return_pct', 0):>+8.2f} "
              f"{r.get('num_entries', 0):>8} {r.get('avg_confidence', 0):>8.3f}")

    out_path = Path("scripts/reprompt_sweep_results.json")
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
