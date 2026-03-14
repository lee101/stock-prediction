"""
Run independent A/B experiments for prompt variants.

Leverage values above 1x in this runner model hypothetical borrowed notional.
They are not parity with the live spot executor.

Experiments:
  1. position_context - tell LLM about existing positions, P&L, hold time
  2. fractional - allow partial position exits (exit_pct field)
  3. anonymized - replace symbol names with generic labels

Usage:
  python -m llm_hourly_trader.experiment_runner --symbols BTCUSD ETHUSD SOLUSD --days 3
  python -m llm_hourly_trader.experiment_runner --variants baseline anonymized --days 1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_hourly_trader.config import SYMBOL_UNIVERSE, SymbolConfig
from llm_hourly_trader.gemini_wrapper import TradePlan, build_prompt
from llm_hourly_trader.historical_error_bands import HistoricalForecastErrorEstimator
from llm_hourly_trader.providers import CacheMissError, call_llm
from llm_hourly_trader.windowing import parse_utc_timestamp, resolve_window_end

DATA_DIRS = [
    Path(__file__).resolve().parent.parent / "trainingdatahourly" / "crypto",
    Path(__file__).resolve().parent.parent / "trainingdatahourly" / "stocks",
]
FORECAST_DIR = Path(__file__).resolve().parent.parent / "binanceneural" / "forecast_cache"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "llm_hourly_trader" / "results"


def load_bars(symbol: str) -> pd.DataFrame:
    for d in DATA_DIRS:
        path = d / f"{symbol}.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            return df.sort_values("timestamp").reset_index(drop=True)
    return pd.DataFrame()


def load_forecasts(symbol: str, horizon: str) -> pd.DataFrame:
    path = FORECAST_DIR / horizon / f"{symbol}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def get_forecast_at(fc_df: pd.DataFrame, ts: pd.Timestamp) -> dict | None:
    if fc_df.empty:
        return None
    mask = fc_df["timestamp"] <= ts
    if not mask.any():
        return None
    row = fc_df[mask].iloc[-1]
    result = {}
    for c in fc_df.columns:
        if c in ("timestamp", "symbol"):
            continue
        try:
            val = float(row[c])
            if pd.notna(val):
                result[c] = val
        except (ValueError, TypeError):
            pass
    return result if result else None


def _compute_metrics(equity: np.ndarray) -> dict:
    if len(equity) < 2:
        return {"return_pct": 0, "sortino": 0, "max_dd_pct": 0}
    total_return = (equity[-1] - equity[0]) / equity[0]
    returns = np.diff(equity) / np.clip(equity[:-1], 1e-8, None)
    mean_ret = returns.mean()
    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) else 0
    sortino = mean_ret / downside_std * np.sqrt(8760) if downside_std > 0 else 0
    peaks = np.maximum.accumulate(equity)
    drawdowns = np.where(peaks > 0, (peaks - equity) / peaks, 0)
    max_dd = float(np.max(drawdowns)) if len(drawdowns) else 0
    return {
        "return_pct": float(total_return * 100),
        "sortino": float(sortino),
        "max_dd_pct": float(max_dd * 100),
        "final_equity": float(equity[-1]),
        "num_hours": len(equity),
    }


def _portfolio_equity_at_ts(
    cash: float,
    positions: dict[str, dict],
    prices_by_symbol: dict[str, float],
) -> float:
    equity = cash
    for sym, pos in positions.items():
        mark_price = prices_by_symbol.get(sym, pos["cost_basis"])
        equity += pos["qty"] * mark_price
    return equity


def _parse_exit_pct(plan: TradePlan) -> float:
    """Extract exit_pct from reasoning if not in standard fields."""
    try:
        data = json.loads(plan.reasoning) if plan.reasoning.startswith("{") else {}
        return float(data.get("exit_pct", 1.0))
    except Exception:
        pass
    match = re.search(r'"exit_pct"\s*:\s*([\d.]+)', plan.reasoning)
    if match:
        return float(match.group(1))
    return 1.0


def run_sequential_backtest(
    symbols: list[str],
    days: int,
    variant: str,
    model: str,
    initial_cash: float = 2000.0,
    max_hold_hours: int = 6,
    max_position_pct: float = 0.25,
    rate_limit: float = 4.2,
    leverage: float = 1.0,
    end_timestamp: str | None = None,
    cache_only: bool = False,
) -> dict:
    """Run a sequential backtest where each bar sees current position state."""
    all_bars = {}
    all_fc_h1 = {}
    all_fc_h24 = {}
    usable = []

    for sym in symbols:
        bars = load_bars(sym)
        fc1 = load_forecasts(sym, "h1")
        fc24 = load_forecasts(sym, "h24")
        if bars.empty or fc1.empty:
            print(f"  {sym}: skipping (no data)")
            continue
        all_bars[sym] = bars
        all_fc_h1[sym] = fc1
        all_fc_h24[sym] = fc24
        usable.append(sym)
        print(f"  {sym}: {len(bars)} bars, fc_h1={len(fc1)} rows")

    if not usable:
        return {"error": "no usable symbols"}

    use_mae_bands = variant in {"mae_bands", "historical_mae_bands"}
    error_estimators: dict[str, dict[int, HistoricalForecastErrorEstimator]] = {}
    if use_mae_bands:
        for sym in usable:
            error_estimators[sym] = {
                1: HistoricalForecastErrorEstimator.from_frames(
                    bars=all_bars[sym],
                    forecasts=all_fc_h1[sym],
                    horizon_hours=1,
                ),
                24: HistoricalForecastErrorEstimator.from_frames(
                    bars=all_bars[sym],
                    forecasts=all_fc_h24[sym],
                    horizon_hours=24,
                ),
            }

    fc_ends = [all_fc_h1[s]["timestamp"].max() for s in usable if not all_fc_h1[s].empty]
    end_ts = resolve_window_end(fc_ends, end_timestamp)
    start_ts = end_ts - timedelta(days=days)
    print(f"  Window: {start_ts} -> {end_ts}")

    sym_configs = {s: SYMBOL_UNIVERSE.get(s, SymbolConfig(s, "crypto")) for s in usable}

    # Build per-symbol bar windows
    sym_windows = {}
    for sym in usable:
        w = all_bars[sym]
        w = w[(w["timestamp"] >= start_ts) & (w["timestamp"] <= end_ts)]
        sym_windows[sym] = w

    # Collect all unique timestamps
    all_ts = sorted(set(ts for sym in usable for ts in sym_windows[sym]["timestamp"]))
    print(f"  {len(all_ts)} timestamps, {len(usable)} symbols")

    # State
    cash = initial_cash
    positions: dict[str, dict] = {}  # sym -> {qty, cost_basis, open_time}
    equity_history = []
    trades = []
    api_calls = 0

    for ts_idx, ts in enumerate(all_ts):
        prices_by_symbol = {}
        for sym in usable:
            bars_at = sym_windows[sym][sym_windows[sym]["timestamp"] == ts]
            if not bars_at.empty:
                prices_by_symbol[sym] = float(bars_at.iloc[0]["close"])

        # Force-close positions at max hold
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
                               "price": close_price, "qty": pos["qty"],
                               "pnl": realized, "reason": "max_hold"})
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

            # Build position info for context-aware variants
            pos = positions.get(sym)
            position_info = {}
            if pos:
                held = (ts - pos["open_time"]).total_seconds() / 3600.0
                position_info = {
                    "qty": pos["qty"],
                    "entry_price": pos["cost_basis"],
                    "held_hours": held,
                }

            # Build prompt
            hist_slice = all_bars[sym][all_bars[sym]["timestamp"] <= ts].tail(25)
            if len(hist_slice) < 5:
                continue
            history = hist_slice.to_dict("records")
            fc_1h = get_forecast_at(all_fc_h1[sym], ts)
            fc_24h = get_forecast_at(all_fc_h24[sym], ts)
            forecast_error_1h = None
            forecast_error_24h = None
            if use_mae_bands:
                band_1h = error_estimators[sym][1].band_at(ts)
                band_24h = error_estimators[sym][24].band_at(ts)
                forecast_error_1h = band_1h.as_prompt_context() if band_1h else None
                forecast_error_24h = band_24h.as_prompt_context() if band_24h else None

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
                forecast_error_1h=forecast_error_1h,
                forecast_error_24h=forecast_error_24h,
            )

            # Call LLM
            try:
                plan = call_llm(prompt, model=model, cache_only=cache_only)
            except CacheMissError as exc:
                raise RuntimeError(
                    f"Cache-only replay missing response for {sym} at {parse_utc_timestamp(ts).isoformat()}"
                ) from exc
            api_calls += 1
            if rate_limit > 0:
                time.sleep(rate_limit)

            if "exhausted" in plan.reasoning:
                continue

            # Process exit (take-profit)
            if pos and plan.sell_price > 0 and high >= plan.sell_price:
                exit_pct = 1.0
                if variant == "fractional":
                    exit_pct = _parse_exit_pct(plan)
                    exit_pct = max(0.0, min(1.0, exit_pct))
                sell_qty = pos["qty"] * exit_pct
                if sell_qty > 0:
                    sell_fee = sell_qty * plan.sell_price * fee_rate
                    realized = (plan.sell_price - pos["cost_basis"]) * sell_qty
                    margin_back = sell_qty * pos["cost_basis"] * (1 + fee_rate) / leverage
                    cash += margin_back + realized - sell_fee
                    trades.append({"ts": str(ts), "sym": sym, "side": "sell",
                                   "price": plan.sell_price, "qty": sell_qty,
                                   "pnl": realized, "reason": f"tp_{exit_pct:.0%}"})
                    remaining = pos["qty"] - sell_qty
                    if remaining < 1e-10:
                        del positions[sym]
                    else:
                        positions[sym]["qty"] = remaining

            # Process entry
            if sym not in positions and plan.direction == "long" and plan.buy_price > 0:
                if plan.confidence <= 0:
                    continue
                if low <= plan.buy_price:
                    max_notional = cash * max_position_pct * leverage
                    qty = max_notional / (plan.buy_price * (1 + fee_rate))
                    qty *= plan.confidence
                    if qty <= 0:
                        continue
                    notional = qty * plan.buy_price * (1 + fee_rate)
                    margin_req = notional / leverage
                    cash -= margin_req
                    positions[sym] = {
                        "qty": qty, "cost_basis": plan.buy_price, "open_time": ts,
                    }
                    trades.append({"ts": str(ts), "sym": sym, "side": "buy",
                                   "price": plan.buy_price, "qty": qty,
                                   "pnl": 0, "reason": "entry"})

        # Equity snapshot: cash + margin_locked + unrealized_pnl per position
        pos_value = 0.0
        for sym, pos in positions.items():
            margin_locked = pos["qty"] * pos["cost_basis"] * (1 + sym_configs[sym].maker_fee) / leverage
            mark = pos["cost_basis"]
            bars_at = sym_windows.get(sym, pd.DataFrame())
            if not bars_at.empty:
                bars_now = bars_at[bars_at["timestamp"] == ts]
                if not bars_now.empty:
                    mark = float(bars_now.iloc[0]["close"])
            unrealized = (mark - pos["cost_basis"]) * pos["qty"]
            pos_value += margin_locked + unrealized
        equity_history.append(cash + pos_value)

        if ts_idx % max(1, len(all_ts) // 10) == 0:
            eq = equity_history[-1]
            print(f"  [{ts}] eq=${eq:.2f} pos={list(positions.keys())} trades={len(trades)}")

    equity_arr = np.array(equity_history, dtype=float)
    metrics = _compute_metrics(equity_arr)
    entries = sum(1 for t in trades if t["side"] == "buy")
    exits = sum(1 for t in trades if t["side"] in ("sell", "close"))
    realized = sum(t["pnl"] for t in trades)

    return {
        "variant": variant,
        "model": model,
        "leverage": leverage,
        "symbols": usable,
        "days": days,
        "window_start": start_ts.isoformat(),
        "window_end": end_ts.isoformat(),
        "cache_only": cache_only,
        "api_calls": api_calls,
        "entries": entries,
        "exits": exits,
        "total_trades": entries + exits,
        "realized_pnl": realized,
        **metrics,
        "trades": trades,
    }


def main():
    parser = argparse.ArgumentParser(description="Prompt experiment runner")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSD", "ETHUSD", "SOLUSD"])
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--model", type=str, default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--variants", nargs="+",
                        default=["default", "position_context", "fractional", "anonymized"])
    parser.add_argument("--initial-cash", type=float, default=2000.0)
    parser.add_argument("--rate-limit", type=float, default=4.2)
    parser.add_argument("--leverage", type=float, nargs="+", default=[1.0])
    parser.add_argument("--end-ts", type=str, default=None,
                        help="Fixed UTC end timestamp for reproducible reruns (e.g. 2026-03-14T00:00:00Z)")
    parser.add_argument("--cache-only", action="store_true",
                        help="Replay from llm_hourly_trader/api_cache only; fail on any cache miss")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"PROMPT EXPERIMENT RUNNER")
    print(f"Model: {args.model}")
    print(f"Symbols: {args.symbols}")
    print(f"Days: {args.days}")
    print(f"Variants: {args.variants}")
    print(f"Leverage: {args.leverage}")
    if args.end_ts:
        print(f"End timestamp: {args.end_ts}")
    if args.cache_only:
        print("Mode: cache-only replay")
    if any(float(lev) > 1.0 for lev in args.leverage):
        print("NOTE: leverage >1x here is hypothetical backtest leverage, not spot live-parity.")
    print(f"{'='*70}\n")

    results = {}
    for variant in args.variants:
        for lev in args.leverage:
            label = f"{variant}_{lev:.0f}x"
            print(f"\n{'='*50}")
            print(f"Running: {label}")
            print(f"{'='*50}")
            result = run_sequential_backtest(
                symbols=args.symbols,
                days=args.days,
                variant=variant,
                model=args.model,
                initial_cash=args.initial_cash,
                rate_limit=args.rate_limit,
                leverage=lev,
                end_timestamp=args.end_ts,
                cache_only=args.cache_only,
            )
            results[label] = result

    # Comparison table
    print(f"\n{'='*70}")
    print(f"EXPERIMENT RESULTS COMPARISON")
    print(f"{'='*70}")
    print(f"{'Config':<25} {'Return%':>8} {'Sortino':>8} {'MaxDD%':>8} {'Trades':>7} {'PnL':>10}")
    print("-" * 75)
    for v, r in results.items():
        if "error" in r:
            print(f"{v:<25} {'ERROR':>8}")
            continue
        print(f"{v:<25} {r['return_pct']:>+8.3f} {r['sortino']:>8.2f} "
              f"{r['max_dd_pct']:>8.3f} {r['total_trades']:>7} ${r['realized_pnl']:>+9.2f}")
    print(f"{'='*75}\n")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"experiment_{args.model.replace('/', '_')}_{args.days}d.json"
    save_data = {}
    for v, r in results.items():
        save_r = {k: val for k, val in r.items() if k != "trades"}
        save_data[v] = save_r
    with open(out, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
