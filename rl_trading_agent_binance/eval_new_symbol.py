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
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_hourly_trader.gemini_wrapper import TradePlan
from llm_hourly_trader.providers import call_llm
from rl_trading_agent_binance_prompt import build_live_prompt

from src.binance_symbol_utils import forecast_cache_symbol_candidates
from src.forecast_cache_lookup import load_latest_forecast_from_cache
from src.hourly_data_utils import resolve_hourly_symbol_path


CACHE_DIR = Path("rl_trading_agent_binance/signal_cache")
SIGNAL_COLUMNS = [
    "timestamp",
    "symbol",
    "direction",
    "buy_price",
    "sell_price",
    "confidence",
    "reasoning",
]

SYMBOL_FEE = {
    "BTCUSD": 0.0,
    "ETHUSD": 0.0,
    "SOLUSD": 0.001,
    "DOGEUSD": 0.001,
    "AAVEUSD": 0.001,
    "LINKUSD": 0.001,
    "XRPUSD": 0.001,
    "AVAXUSD": 0.001,
}

SYMBOL_MAX_POS = {
    "BTCUSD": 0.25,
    "ETHUSD": 0.20,
    "SOLUSD": 0.15,
    "DOGEUSD": 0.10,
    "AAVEUSD": 0.15,
    "LINKUSD": 0.10,
    "XRPUSD": 0.10,
    "AVAXUSD": 0.10,
}


def _cache_namespace(
    *,
    signal_mode: str,
    model: str,
    thinking_level: str,
    forecast_cache_root: Path | None,
) -> str:
    root = ""
    if forecast_cache_root is not None:
        root = str(Path(forecast_cache_root).resolve())
    return f"{signal_mode}|{model}|{thinking_level}|{root}"


def _cache_key(symbol: str, ts_str: str, cache_namespace: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h = hashlib.md5(f"{symbol}_{ts_str}_{cache_namespace}".encode()).hexdigest()[:12]
    return CACHE_DIR / f"{symbol}_{ts_str}_{h}.json"


def _get_cached(symbol: str, ts_str: str, cache_namespace: str) -> dict | None:
    path = _cache_key(symbol, ts_str, cache_namespace)
    if path.exists():
        return json.loads(path.read_text())
    return None


def _set_cached(symbol: str, ts_str: str, cache_namespace: str, plan: TradePlan):
    path = _cache_key(symbol, ts_str, cache_namespace)
    path.write_text(
        json.dumps(
            {
                "direction": plan.direction,
                "buy_price": plan.buy_price,
                "sell_price": plan.sell_price,
                "confidence": plan.confidence,
                "reasoning": plan.reasoning,
            }
        )
    )


def load_hourly_bars(symbol: str, data_root: str = "trainingdatahourly/crypto") -> pd.DataFrame:
    path = None
    for candidate in forecast_cache_symbol_candidates(symbol):
        path = resolve_hourly_symbol_path(candidate, Path(data_root))
        if path is not None:
            break
    if path is None:
        raise FileNotFoundError(f"No hourly data for {symbol} under {data_root}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    df["symbol"] = symbol
    return df


def load_forecast_at(symbol: str, timestamp: pd.Timestamp, horizon: int, cache_root: Path | None = None) -> dict | None:
    default_root = REPO / "binanceneural" / "forecast_cache"
    candidate_roots: list[Path] = []
    if cache_root is not None:
        candidate_roots.append(Path(cache_root))
    if not candidate_roots or candidate_roots[-1] != default_root:
        candidate_roots.append(default_root)
    for root in candidate_roots:
        forecast = load_latest_forecast_from_cache(symbol, int(horizon), root, as_of=timestamp)
        if forecast is not None:
            return forecast
    return None


def build_forecast_rule_signal(
    *,
    symbol: str,
    current_price: float,
    fc_1h: dict | None,
    fc_24h: dict | None,
    total_cost: float = 0.0020,
    min_reward_risk: float = 1.10,
) -> dict:
    del symbol
    if current_price <= 0.0:
        return {
            "direction": "hold",
            "buy_price": 0.0,
            "sell_price": 0.0,
            "confidence": 0.0,
            "reasoning": "invalid_price",
        }

    forecasts = [fc for fc in (fc_1h, fc_24h) if fc]
    if not forecasts:
        return {
            "direction": "hold",
            "buy_price": 0.0,
            "sell_price": 0.0,
            "confidence": 0.0,
            "reasoning": "missing_forecast",
        }

    close_targets = [float(fc["predicted_close_p50"]) for fc in forecasts if fc.get("predicted_close_p50") is not None]
    high_targets = [
        float(fc.get("predicted_high_p50", fc.get("predicted_close_p50", current_price))) for fc in forecasts
    ]
    low_targets = [float(fc.get("predicted_low_p50", current_price)) for fc in forecasts]
    if not close_targets:
        return {
            "direction": "hold",
            "buy_price": 0.0,
            "sell_price": 0.0,
            "confidence": 0.0,
            "reasoning": "missing_close_target",
        }

    expected_close = float(np.mean(close_targets))
    expected_high = float(max(high_targets)) if high_targets else expected_close
    expected_low = float(min(low_targets)) if low_targets else current_price
    close_edge = (expected_close - current_price) / current_price
    upside_edge = (expected_high - current_price) / current_price
    downside_risk = max(0.0, (current_price - expected_low) / current_price)
    usable_edge = max(close_edge, upside_edge)
    reward_risk = usable_edge / max(downside_risk, 0.0025)

    if usable_edge <= total_cost or reward_risk < min_reward_risk:
        return {
            "direction": "hold",
            "buy_price": 0.0,
            "sell_price": 0.0,
            "confidence": 0.0,
            "reasoning": "edge_below_threshold",
        }

    entry_discount = min(max(usable_edge * 0.35, 0.0005), 0.0030)
    buy_price = current_price * (1.0 - entry_discount)
    sell_floor = buy_price * 1.0025
    sell_price = max(sell_floor, min(expected_high, current_price * (1.0 + max(total_cost, usable_edge * 0.85))))
    confidence = min(0.95, max(0.05, usable_edge / 0.02 * min(reward_risk / 2.0, 1.0)))
    return {
        "direction": "long",
        "buy_price": float(buy_price),
        "sell_price": float(sell_price),
        "confidence": float(confidence),
        "reasoning": f"edge={usable_edge:.4f},rr={reward_risk:.2f}",
    }


def generate_signals(
    symbol: str,
    bars: pd.DataFrame,
    start_date: str,
    end_date: str,
    model: str = "gemini-3.1-flash-lite-preview",
    thinking_level: str = "HIGH",
    rate_limit: float = 1.5,
    signal_mode: str = "gemini",
    forecast_cache_root: Path | None = None,
    forecast_rule_total_cost: float = 0.0020,
    forecast_rule_min_reward_risk: float = 1.10,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC")
    bars = bars.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    mask = (bars["timestamp"] >= start_ts) & (bars["timestamp"] <= end_ts)
    window_bars = bars[mask].reset_index(drop=True)

    all_bars_sorted = bars.sort_values("timestamp").reset_index(drop=True)

    cache_namespace = _cache_namespace(
        signal_mode=signal_mode,
        model=model,
        thinking_level=thinking_level,
        forecast_cache_root=forecast_cache_root,
    )

    results = []
    api_calls = 0
    for idx in range(len(window_bars)):
        ts = window_bars.iloc[idx]["timestamp"]
        ts_str = ts.strftime("%Y%m%d_%H")

        cached = _get_cached(symbol, ts_str, cache_namespace)
        if cached:
            results.append(
                {
                    "timestamp": ts,
                    "symbol": symbol,
                    **cached,
                }
            )
            continue

        ts_idx = all_bars_sorted[all_bars_sorted["timestamp"] <= ts].index
        if len(ts_idx) < 24:
            continue
        context_start = max(0, ts_idx[-1] - 72)
        context_end = ts_idx[-1] + 1
        context_rows = all_bars_sorted.iloc[context_start:context_end].to_dict("records")

        current_price = float(context_rows[-1]["close"])
        fc_1h = load_forecast_at(symbol, ts, 1, cache_root=forecast_cache_root)
        fc_4h = load_forecast_at(symbol, ts, 4, cache_root=forecast_cache_root)
        fc_12h = load_forecast_at(symbol, ts, 12, cache_root=forecast_cache_root)
        fc_24h = load_forecast_at(symbol, ts, 24, cache_root=forecast_cache_root)

        if signal_mode == "forecast_rule":
            signal = build_forecast_rule_signal(
                symbol=symbol,
                current_price=current_price,
                fc_1h=fc_1h,
                fc_24h=fc_24h,
                total_cost=float(forecast_rule_total_cost),
                min_reward_risk=float(forecast_rule_min_reward_risk),
            )
            results.append(
                {
                    "timestamp": ts,
                    "symbol": symbol,
                    **signal,
                }
            )
            continue

        fee_bps = int(SYMBOL_FEE.get(symbol, 0.001) * 10000)
        prompt = build_live_prompt(
            symbol,
            context_rows,
            current_price,
            fc_1h=fc_1h,
            fc_24h=fc_24h,
            fee_bps=fee_bps,
            fc_4h=fc_4h,
            fc_12h=fc_12h,
        )

        try:
            plan = call_llm(prompt, model=model, thinking_level=thinking_level)
            _set_cached(symbol, ts_str, cache_namespace, plan)
            results.append(
                {
                    "timestamp": ts,
                    "symbol": symbol,
                    "direction": plan.direction,
                    "buy_price": plan.buy_price,
                    "sell_price": plan.sell_price,
                    "confidence": plan.confidence,
                    "reasoning": plan.reasoning,
                }
            )
            api_calls += 1
            if api_calls % 50 == 0:
                print(f"  {symbol}: {api_calls} API calls, {len(results)} total signals")
            time.sleep(rate_limit)
        except Exception as e:
            print(f"  {symbol} @ {ts_str}: API error: {e}")
            time.sleep(5)

    if not results:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)
    if api_calls > 0:
        print(f"  {symbol}: {api_calls} new API calls, {len(results)} total signals")
    return pd.DataFrame(results, columns=SIGNAL_COLUMNS)


def _parse_window_days(raw: str | None) -> list[int]:
    values: list[int] = []
    if raw is None:
        return values
    for raw_token in str(raw).split(","):
        token = raw_token.strip()
        if not token:
            continue
        day_count = int(token)
        if day_count <= 0 or day_count in values:
            continue
        values.append(day_count)
    return values


def _slice_frame_map(
    frames: dict[str, pd.DataFrame],
    *,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    sliced: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        local = frame.copy()
        if local.empty or "timestamp" not in local.columns:
            continue
        local["timestamp"] = pd.to_datetime(local["timestamp"], utc=True)
        window = local[(local["timestamp"] >= start_ts) & (local["timestamp"] <= end_ts)].reset_index(drop=True)
        if not window.empty:
            sliced[symbol] = window
    return sliced


def simulate_portfolio(
    bars_map: dict[str, pd.DataFrame],
    signals_map: dict[str, pd.DataFrame],
    initial_cash: float = 10000.0,
    leverage: float = 5.0,
    margin_rate_annual: float = 0.10,
    max_hold_hours: float = 6.0,
    margin_fee: float = 0.001,
    symbol_max_pos_overrides: dict[str, float] | None = None,
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

    merged = all_bars.merge(all_signals, on=["timestamp", "symbol"], how="inner")
    merged = merged.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    cash = initial_cash
    positions = {}
    equity_history = []
    trades = []
    hourly_margin_rate = margin_rate_annual / 8760

    for ts, chunk in merged.groupby("timestamp", sort=True):
        for _sym, pos in list(positions.items()):
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
                trades.append(
                    {
                        "timestamp": str(ts),
                        "symbol": sym,
                        "side": "close",
                        "price": close_price,
                        "pnl": pnl - fee - interest,
                        "reason": "max_hold",
                    }
                )
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
                trades.append(
                    {
                        "timestamp": str(ts),
                        "symbol": sym,
                        "side": "sell",
                        "price": sell_price,
                        "pnl": pnl - fee - interest,
                        "reason": "take_profit",
                    }
                )
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

            override_max_pos = None
            if symbol_max_pos_overrides:
                override_max_pos = symbol_max_pos_overrides.get(sym)
            max_pct = float(override_max_pos if override_max_pos is not None else SYMBOL_MAX_POS.get(sym, 0.10))
            equity_alloc = cash * max_pct
            if equity_alloc < 12:
                continue
            notional = equity_alloc * leverage
            qty = notional / buy_price
            fee = qty * buy_price * margin_fee
            cash -= equity_alloc + fee
            positions[sym] = {
                "qty": qty,
                "cost_basis": buy_price,
                "equity_used": equity_alloc,
                "open_time": ts,
                "total_interest": 0,
            }
            trades.append(
                {
                    "timestamp": str(ts),
                    "symbol": sym,
                    "side": "buy",
                    "price": buy_price,
                    "pnl": 0,
                    "reason": "entry",
                }
            )

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
            trades.append(
                {
                    "timestamp": "final",
                    "symbol": sym,
                    "side": "close",
                    "price": close_price,
                    "pnl": pnl - fee - interest,
                    "reason": "end_of_backtest",
                }
            )

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
    parser.add_argument("--signal-mode", choices=("gemini", "forecast_rule"), default="gemini")
    parser.add_argument("--thinking", default="HIGH")
    parser.add_argument("--leverage", type=float, default=5.0)
    parser.add_argument("--cash", type=float, default=10000.0)
    parser.add_argument("--rate-limit", type=float, default=1.0)
    parser.add_argument("--data-root", default="trainingdatahourly/crypto")
    parser.add_argument("--forecast-cache-root", type=Path, default=None)
    parser.add_argument(
        "--windows", default=None, help="Optional comma-separated window lengths in days ending at --end."
    )
    parser.add_argument("--forecast-rule-total-cost-bps", type=float, default=20.0)
    parser.add_argument("--forecast-rule-min-reward-risk", type=float, default=1.10)
    parser.add_argument("--add-symbol-forecast-rule-total-cost-bps", type=float, default=None)
    parser.add_argument("--add-symbol-forecast-rule-min-reward-risk", type=float, default=None)
    parser.add_argument("--add-symbol-max-pos", type=float, default=None)
    args = parser.parse_args()
    if args.add_symbol_max_pos is not None and float(args.add_symbol_max_pos) <= 0:
        raise SystemExit("--add-symbol-max-pos must be positive")

    window_days = _parse_window_days(args.windows)
    end_ts = pd.Timestamp(args.end, tz="UTC")
    signal_start = args.start
    if window_days:
        signal_start = str((end_ts - pd.Timedelta(days=max(window_days))).date())

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
        forecast_rule_total_cost = float(args.forecast_rule_total_cost_bps) / 10000.0
        forecast_rule_min_reward_risk = float(args.forecast_rule_min_reward_risk)
        if args.add_symbol and sym == args.add_symbol:
            if args.add_symbol_forecast_rule_total_cost_bps is not None:
                forecast_rule_total_cost = float(args.add_symbol_forecast_rule_total_cost_bps) / 10000.0
            if args.add_symbol_forecast_rule_min_reward_risk is not None:
                forecast_rule_min_reward_risk = float(args.add_symbol_forecast_rule_min_reward_risk)
        signals_map[sym] = generate_signals(
            sym,
            bars_map[sym],
            signal_start,
            args.end,
            model=args.model,
            thinking_level=args.thinking,
            rate_limit=args.rate_limit,
            signal_mode=args.signal_mode,
            forecast_cache_root=args.forecast_cache_root,
            forecast_rule_total_cost=forecast_rule_total_cost,
            forecast_rule_min_reward_risk=forecast_rule_min_reward_risk,
        )
        print(f"  {sym}: {len(signals_map[sym])} signals")

    baseline_symbols = list(args.symbols)
    if not window_days:
        window_days = []

    window_specs = (
        [(f"{days}d", end_ts - pd.Timedelta(days=days), end_ts) for days in window_days]
        if window_days
        else [("full", pd.Timestamp(signal_start, tz="UTC"), end_ts)]
    )

    baseline = None
    extended = None
    window_results: list[dict[str, object]] = []
    add_symbol_max_pos_overrides: dict[str, float] = {}
    if args.add_symbol and args.add_symbol_max_pos is not None:
        add_symbol_max_pos_overrides[str(args.add_symbol).strip().upper()] = float(args.add_symbol_max_pos)
    for label, start_ts, window_end_ts in window_specs:
        baseline_bars = _slice_frame_map(
            {s: bars_map[s] for s in baseline_symbols},
            start_ts=start_ts,
            end_ts=window_end_ts,
        )
        baseline_signals = _slice_frame_map(
            {s: signals_map[s] for s in baseline_symbols},
            start_ts=start_ts,
            end_ts=window_end_ts,
        )
        baseline = simulate_portfolio(
            baseline_bars,
            baseline_signals,
            initial_cash=args.cash,
            leverage=args.leverage,
        )
        print(f"\n{'=' * 60}")
        print(f"BASELINE {label}: {baseline_symbols} ({start_ts.date()} -> {window_end_ts.date()})")
        print(f"  Return: {baseline['return_pct']:+.2f}%")
        print(f"  Sortino: {baseline['sortino']:.2f}")
        print(f"  Max DD: {baseline['max_dd_pct']:.2f}%")
        print(f"  Trades: {baseline['n_trades']}")

        row: dict[str, object] = {
            "window": label,
            "start": str(start_ts.date()),
            "end": str(window_end_ts.date()),
            "baseline": {k: v for k, v in baseline.items() if k not in ("equity_df", "trades_df")},
        }

        if args.add_symbol:
            extended_bars = _slice_frame_map(bars_map, start_ts=start_ts, end_ts=window_end_ts)
            extended_signals = _slice_frame_map(signals_map, start_ts=start_ts, end_ts=window_end_ts)
            extended = simulate_portfolio(
                extended_bars,
                extended_signals,
                initial_cash=args.cash,
                leverage=args.leverage,
                symbol_max_pos_overrides=add_symbol_max_pos_overrides,
            )
            print(f"WITH {args.add_symbol} {label}: {all_symbols}")
            print(f"  Return: {extended['return_pct']:+.2f}%")
            print(f"  Sortino: {extended['sortino']:.2f}")
            print(f"  Max DD: {extended['max_dd_pct']:.2f}%")
            print(f"  Trades: {extended['n_trades']}")
            delta_ret = extended["return_pct"] - baseline["return_pct"]
            delta_sort = extended["sortino"] - baseline["sortino"]
            delta_dd = extended["max_dd_pct"] - baseline["max_dd_pct"]
            new_sym_pnl = extended["per_symbol_pnl"].get(args.add_symbol, 0)
            verdict = "ACCEPT" if new_sym_pnl > 0 and delta_sort >= -0.5 else "REJECT"
            print(f"  Return delta: {delta_ret:+.2f}%")
            print(f"  Sortino delta: {delta_sort:+.2f}")
            print(f"  Max DD delta: {delta_dd:+.2f}%")
            print(f"  {args.add_symbol} standalone P&L: ${new_sym_pnl:+.2f}")
            print(f"  Verdict: {verdict}")
            row["extended"] = {k: v for k, v in extended.items() if k not in ("equity_df", "trades_df")}
            row["comparison"] = {
                "return_delta": float(delta_ret),
                "sortino_delta": float(delta_sort),
                "max_dd_delta": float(delta_dd),
                "new_symbol_pnl": float(new_sym_pnl),
                "verdict": verdict,
            }

        window_results.append(row)

    out_dir = Path(f"analysis/hybrid_symbol_eval_{time.strftime('%Y%m%d')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "baseline_symbols": baseline_symbols,
        "add_symbol": args.add_symbol,
        "start": signal_start,
        "end": args.end,
        "model": args.model,
        "signal_mode": args.signal_mode,
        "leverage": args.leverage,
        "forecast_cache_root": str(args.forecast_cache_root) if args.forecast_cache_root else None,
        "forecast_rule_total_cost_bps": float(args.forecast_rule_total_cost_bps),
        "forecast_rule_min_reward_risk": float(args.forecast_rule_min_reward_risk),
        "add_symbol_forecast_rule_total_cost_bps": (
            float(args.add_symbol_forecast_rule_total_cost_bps)
            if args.add_symbol_forecast_rule_total_cost_bps is not None
            else None
        ),
        "add_symbol_forecast_rule_min_reward_risk": (
            float(args.add_symbol_forecast_rule_min_reward_risk)
            if args.add_symbol_forecast_rule_min_reward_risk is not None
            else None
        ),
        "add_symbol_max_pos": (float(args.add_symbol_max_pos) if args.add_symbol_max_pos is not None else None),
        "windows": window_results,
    }

    tag = args.add_symbol or "baseline"
    result_path = out_dir / f"eval_{tag}_{args.signal_mode}.json"
    result_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nResults saved to {result_path}")


if __name__ == "__main__":
    main()
