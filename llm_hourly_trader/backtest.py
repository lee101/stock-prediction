"""
Backtest LLM hourly trader across multiple providers/models.

Supports: Gemini, OpenAI (GPT-4.1/o3/o4), Anthropic (Claude Sonnet 4.6)
Supports crypto + stocks with direction constraints, max drawdown tracking,
and realistic simulation via shared-cash market simulator.

Usage:
  python -m llm_hourly_trader.backtest --symbols BTCUSD ETHUSD --days 7
  python -m llm_hourly_trader.backtest --group crypto --days 30 --prompt conservative
  python -m llm_hourly_trader.backtest --model o3 --group crypto --days 7
  python -m llm_hourly_trader.backtest --model claude-sonnet-4-6 --symbols BTCUSD --days 7
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_hourly_trader.config import (
    FORECAST_CUTOFFS,
    STOCK_TRADING_HOURS_UTC,
    SYMBOL_UNIVERSE,
    BacktestConfig,
    SymbolConfig,
)
from llm_hourly_trader.gemini_wrapper import TradePlan, build_prompt
from llm_hourly_trader.historical_error_bands import HistoricalForecastErrorEstimator
from llm_hourly_trader.providers import call_llm

DATA_DIRS = [
    Path(__file__).resolve().parent.parent / "trainingdatahourly" / "crypto",
    Path(__file__).resolve().parent.parent / "trainingdatahourly" / "stocks",
]
FORECAST_DIR = Path(__file__).resolve().parent.parent / "binanceneural" / "forecast_cache"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "llm_hourly_trader" / "results"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_bars(symbol: str) -> pd.DataFrame:
    for d in DATA_DIRS:
        path = d / f"{symbol}.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["timestamp"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df["symbol"] = symbol
            return df.sort_values("timestamp").reset_index(drop=True)
    raise FileNotFoundError(f"No bar data for {symbol}")


def load_forecasts(symbol: str, horizon: str) -> pd.DataFrame:
    path = FORECAST_DIR / horizon / f"{symbol}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def get_forecast_at(forecasts: pd.DataFrame, ts: pd.Timestamp) -> dict | None:
    if forecasts.empty:
        return None
    match = forecasts[forecasts["timestamp"] == ts]
    if match.empty:
        before = forecasts[forecasts["timestamp"] <= ts]
        if before.empty:
            return None
        match = before.iloc[[-1]]
    row = match.iloc[0]
    return {
        "predicted_close_p50": float(row["predicted_close_p50"]),
        "predicted_close_p10": float(row["predicted_close_p10"]),
        "predicted_close_p90": float(row["predicted_close_p90"]),
        "predicted_high_p50": float(row["predicted_high_p50"]),
        "predicted_low_p50": float(row["predicted_low_p50"]),
    }


def is_stock_trading_hour(ts: pd.Timestamp) -> bool:
    """Check if timestamp falls within US stock trading hours."""
    h = ts.hour
    return STOCK_TRADING_HOURS_UTC[0] <= h < STOCK_TRADING_HOURS_UTC[1]


# ---------------------------------------------------------------------------
# Simulation (shared-cash, long+short, drawdown tracking)
# ---------------------------------------------------------------------------

def simulate(
    bars_df: pd.DataFrame,
    actions_df: pd.DataFrame,
    config: BacktestConfig,
    symbol_configs: dict[str, SymbolConfig],
) -> dict:
    """Simulate trades with shared cash, direction constraints, drawdown tracking."""
    cash = config.initial_cash
    # Positions: symbol -> {qty, cost_basis, open_time, direction}
    positions: dict[str, dict] = {}
    equity_history: list[float] = []
    trades: list[dict] = []
    timestamps: list[pd.Timestamp] = []

    merged = bars_df.merge(actions_df, on=["timestamp", "symbol"], how="inner", suffixes=("", "_act"))
    merged = merged.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    for ts, chunk in merged.groupby("timestamp", sort=True):
        # ---- Close positions at max hold or take-profit/stop ----
        for _, row in chunk.iterrows():
            sym = row["symbol"]
            pos = positions.get(sym)
            if pos is None or pos["qty"] <= 0:
                continue
            held_hours = (ts - pos["open_time"]).total_seconds() / 3600.0
            if held_hours >= config.max_hold_hours:
                close_price = float(row["close"])
                fee_rate = symbol_configs[sym].maker_fee
                if pos["direction"] == "long":
                    proceeds = pos["qty"] * close_price * (1 - fee_rate)
                    realized = (close_price - pos["cost_basis"]) * pos["qty"]
                else:  # short
                    proceeds = pos["qty"] * (2 * pos["cost_basis"] - close_price) * (1 - fee_rate)
                    realized = (pos["cost_basis"] - close_price) * pos["qty"]
                cash += proceeds
                trades.append({
                    "timestamp": str(ts), "symbol": sym, "side": "close",
                    "price": close_price, "quantity": pos["qty"],
                    "direction": pos["direction"], "realized_pnl": realized,
                    "fee": pos["qty"] * close_price * fee_rate, "reason": "max_hold",
                })
                del positions[sym]

        # ---- Process exits (sell signals) ----
        for _, row in chunk.iterrows():
            sym = row["symbol"]
            pos = positions.get(sym)
            if pos is None or pos["qty"] <= 0:
                continue
            sell_price = float(row.get("sell_price", 0) or 0)
            if sell_price <= 0:
                continue
            high = float(row["high"])
            low = float(row["low"])
            fee_rate = symbol_configs[sym].maker_fee

            if pos["direction"] == "long" and high >= sell_price:
                proceeds = pos["qty"] * sell_price * (1 - fee_rate)
                realized = (sell_price - pos["cost_basis"]) * pos["qty"]
                cash += proceeds
                trades.append({
                    "timestamp": str(ts), "symbol": sym, "side": "sell",
                    "price": sell_price, "quantity": pos["qty"],
                    "direction": "long", "realized_pnl": realized,
                    "fee": pos["qty"] * sell_price * fee_rate, "reason": "take_profit",
                })
                del positions[sym]
            elif pos["direction"] == "short" and low <= sell_price:
                # Short cover at sell_price (take profit on short = price went down)
                proceeds = pos["qty"] * (2 * pos["cost_basis"] - sell_price) * (1 - fee_rate)
                realized = (pos["cost_basis"] - sell_price) * pos["qty"]
                cash += proceeds
                trades.append({
                    "timestamp": str(ts), "symbol": sym, "side": "cover",
                    "price": sell_price, "quantity": pos["qty"],
                    "direction": "short", "realized_pnl": realized,
                    "fee": pos["qty"] * sell_price * fee_rate, "reason": "take_profit",
                })
                del positions[sym]

        # ---- Process entries ----
        for _, row in chunk.iterrows():
            sym = row["symbol"]
            if sym in positions:
                continue  # already have a position
            direction = str(row.get("direction", "hold")).lower().strip()
            if direction == "hold":
                continue

            sym_cfg = symbol_configs.get(sym)
            if sym_cfg is None:
                continue
            if direction not in sym_cfg.allowed_directions:
                continue  # direction not allowed for this symbol

            buy_price = float(row.get("buy_price", 0) or 0)
            confidence = float(row.get("confidence", 0) or 0)
            if buy_price <= 0 or confidence <= 0:
                continue

            high = float(row["high"])
            low = float(row["low"])
            fee_rate = sym_cfg.maker_fee

            # Check fill
            if direction == "long" and low <= buy_price:
                max_notional = cash * config.max_position_pct
                qty = max_notional / (buy_price * (1 + fee_rate))
                qty *= confidence  # scale by confidence
                if qty <= 0:
                    continue
                cost = qty * buy_price * (1 + fee_rate)
                cash -= cost
                positions[sym] = {
                    "qty": qty, "cost_basis": buy_price,
                    "open_time": ts, "direction": "long",
                }
                trades.append({
                    "timestamp": str(ts), "symbol": sym, "side": "buy",
                    "price": buy_price, "quantity": qty,
                    "direction": "long", "realized_pnl": 0,
                    "fee": qty * buy_price * fee_rate, "reason": "entry",
                })
            elif direction == "short" and high >= buy_price:
                # Short entry: sell at buy_price (which is above current for shorts)
                max_notional = cash * config.max_position_pct
                qty = max_notional / (buy_price * (1 + fee_rate))
                qty *= confidence
                if qty <= 0:
                    continue
                cost = qty * buy_price * (1 + fee_rate)  # collateral
                cash -= cost
                positions[sym] = {
                    "qty": qty, "cost_basis": buy_price,
                    "open_time": ts, "direction": "short",
                }
                trades.append({
                    "timestamp": str(ts), "symbol": sym, "side": "short",
                    "price": buy_price, "quantity": qty,
                    "direction": "short", "realized_pnl": 0,
                    "fee": qty * buy_price * fee_rate, "reason": "entry",
                })

        # ---- Equity snapshot ----
        close_prices = {row["symbol"]: float(row["close"]) for _, row in chunk.iterrows()}
        inv_value = 0.0
        for sym, pos in positions.items():
            cp = close_prices.get(sym, pos["cost_basis"])
            if pos["direction"] == "long":
                inv_value += pos["qty"] * cp
            else:  # short
                inv_value += pos["qty"] * (2 * pos["cost_basis"] - cp)
        equity = cash + inv_value
        equity_history.append(equity)
        timestamps.append(ts)

    # ---- Compute metrics ----
    equity_arr = np.array(equity_history, dtype=float)
    metrics = _compute_metrics(equity_arr, config.initial_cash)

    return {
        "equity_history": equity_history,
        "timestamps": [str(t) for t in timestamps],
        "trades": trades,
        "metrics": metrics,
    }


def _compute_metrics(equity: np.ndarray, initial_cash: float) -> dict:
    if len(equity) < 2:
        return {"total_return_pct": 0, "sortino": 0, "max_drawdown_pct": 0, "mean_hourly_return": 0}

    total_return = (equity[-1] - equity[0]) / equity[0]
    returns = np.diff(equity) / np.clip(equity[:-1], 1e-8, None)
    mean_ret = returns.mean()
    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) else 0
    sortino = mean_ret / downside_std * np.sqrt(8760) if downside_std > 0 else 0

    # Max drawdown
    peaks = np.maximum.accumulate(equity)
    drawdowns = np.where(peaks > 0, (peaks - equity) / peaks, 0)
    max_dd = float(np.max(drawdowns)) if len(drawdowns) else 0

    return {
        "total_return_pct": float(total_return * 100),
        "sortino": float(sortino),
        "max_drawdown_pct": float(max_dd * 100),
        "mean_hourly_return": float(mean_ret * 100),
        "final_equity": float(equity[-1]),
        "num_hours": len(equity),
    }


# ---------------------------------------------------------------------------
# Main backtest orchestrator
# ---------------------------------------------------------------------------

def run_backtest(
    symbols: list[str],
    days: int,
    config: BacktestConfig,
) -> dict:
    """Run the full backtest."""
    print(f"\n{'='*60}")
    print(f"LLM Hourly Trader Backtest v2")
    print(f"Model: {config.model}")
    print(f"Prompt: {config.prompt_variant}")
    print(f"Symbols: {symbols}")
    print(f"Days: {days} | Max hold: {config.max_hold_hours}h | Max pos: {config.max_position_pct*100:.0f}%")
    print(f"Initial cash: ${config.initial_cash:,.2f}")
    print(f"{'='*60}\n")

    # Load data
    all_bars = {}
    all_fc_h1 = {}
    all_fc_h24 = {}
    usable_symbols = []

    for sym in symbols:
        try:
            bars = load_bars(sym)
            fc_h1 = load_forecasts(sym, "h1")
            fc_h24 = load_forecasts(sym, "h24")
            all_bars[sym] = bars
            all_fc_h1[sym] = fc_h1
            all_fc_h24[sym] = fc_h24
            usable_symbols.append(sym)
            fc_end = fc_h1["timestamp"].max() if not fc_h1.empty else "none"
            print(f"  {sym}: {len(bars)} bars, fc_h1 to {fc_end}")
        except FileNotFoundError:
            print(f"  {sym}: SKIPPED (no data)")

    if not usable_symbols:
        return {"error": "no usable symbols"}

    use_mae_bands = config.prompt_variant in {"mae_bands", "historical_mae_bands"}
    error_estimators: dict[str, dict[int, HistoricalForecastErrorEstimator]] = {}
    if use_mae_bands:
        for sym in usable_symbols:
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

    # Determine test window: end at earliest forecast cutoff, go back N days
    fc_ends = []
    for sym in usable_symbols:
        if not all_fc_h1[sym].empty:
            fc_ends.append(all_fc_h1[sym]["timestamp"].max())
    end_ts = min(fc_ends)
    start_ts = end_ts - timedelta(days=days)
    print(f"\n  Window: {start_ts} -> {end_ts}")

    # Collect LLM actions
    action_rows = []
    bar_rows = []
    api_calls = 0
    errors = 0

    sym_configs = {sym: SYMBOL_UNIVERSE.get(sym, SymbolConfig(sym, "crypto")) for sym in usable_symbols}

    # Prepare all (symbol, bar) pairs grouped by timestamp for parallel dispatch
    sym_windows = {}
    for sym in usable_symbols:
        sym_cfg = sym_configs[sym]
        sym_bars = all_bars[sym]
        window = sym_bars[(sym_bars["timestamp"] >= start_ts) & (sym_bars["timestamp"] <= end_ts)].copy()
        if sym_cfg.asset_class == "stock":
            window = window[window["timestamp"].apply(is_stock_trading_hour)]
        sym_windows[sym] = window
        print(f"\n  {sym} ({sym_cfg.asset_class}): {len(window)} bars, "
              f"directions={sym_cfg.allowed_directions}")

    # Build all tasks: list of (sym, bar, prompt) for each bar
    tasks: list[tuple[str, dict, str | None]] = []
    for sym in usable_symbols:
        sym_bars = all_bars[sym]
        for _, bar in sym_windows[sym].iterrows():
            ts = bar["timestamp"]
            hist_slice = sym_bars[sym_bars["timestamp"] <= ts].tail(25)
            if len(hist_slice) < 5:
                tasks.append((sym, bar.to_dict(), None))
                continue
            sym_cfg = sym_configs[sym]
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
            prompt = build_prompt(
                symbol=sym,
                history_rows=history,
                forecast_1h=fc_1h,
                forecast_24h=fc_24h,
                current_position="flat",
                cash=config.initial_cash,
                equity=config.initial_cash,
                allowed_directions=sym_cfg.allowed_directions,
                asset_class=sym_cfg.asset_class,
                maker_fee=sym_cfg.maker_fee,
                variant=config.prompt_variant,
                forecast_error_1h=forecast_error_1h,
                forecast_error_24h=forecast_error_24h,
            )
            tasks.append((sym, bar.to_dict(), prompt))

    total_tasks = len(tasks)
    total_api = sum(1 for _, _, p in tasks if p is not None)
    parallel = config.parallel_workers
    print(f"\n  Total bars: {total_tasks}, API calls needed: {total_api}, "
          f"parallel workers: {parallel}")

    def _do_call(sym: str, bar: dict, prompt: str | None, idx: int) -> tuple[str, dict, TradePlan | None, int]:
        if prompt is None:
            return sym, bar, None, idx
        plan = call_llm(prompt, model=config.model)
        return sym, bar, plan, idx

    # Dispatch with thread pool for parallel calls (codex/deepseek are I/O bound)
    results_ordered: list[tuple[str, dict, TradePlan | None, int]] = [None] * total_tasks
    t0 = time.time()
    done_count = 0

    if parallel > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {
                pool.submit(_do_call, sym, bar, prompt, idx): idx
                for idx, (sym, bar, prompt) in enumerate(tasks)
            }
            for future in concurrent.futures.as_completed(futures):
                sym, bar, plan, idx = future.result()
                results_ordered[idx] = (sym, bar, plan, idx)
                done_count += 1
                if done_count % max(1, total_tasks // 20) == 0 or done_count == total_tasks:
                    elapsed = time.time() - t0
                    rate = done_count / elapsed if elapsed > 0 else 0
                    eta = (total_tasks - done_count) / rate if rate > 0 else 0
                    print(f"  Progress: {done_count}/{total_tasks} ({rate:.1f}/s, ETA {eta:.0f}s)")
    else:
        for idx, (sym, bar, prompt) in enumerate(tasks):
            results_ordered[idx] = _do_call(sym, bar, prompt, idx)
            done_count += 1
            if config.rate_limit_seconds > 0 and prompt is not None:
                time.sleep(config.rate_limit_seconds)
            if done_count % max(1, total_tasks // 20) == 0:
                elapsed = time.time() - t0
                rate = done_count / elapsed if elapsed > 0 else 0
                eta = (total_tasks - done_count) / rate if rate > 0 else 0
                print(f"  Progress: {done_count}/{total_tasks} ({rate:.1f}/s, ETA {eta:.0f}s)")

    # Process results in order
    sym_bar_counts: dict[str, int] = {}
    for sym, bar, plan, idx in results_ordered:
        ts = bar["timestamp"] if isinstance(bar["timestamp"], pd.Timestamp) else pd.Timestamp(bar["timestamp"])
        sym_bar_counts[sym] = sym_bar_counts.get(sym, 0) + 1
        i = sym_bar_counts[sym] - 1

        if plan is None:
            action_rows.append({
                "timestamp": ts, "symbol": sym,
                "buy_price": 0, "sell_price": 0, "direction": "hold", "confidence": 0,
            })
            bar_rows.append(bar)
            continue

        api_calls += 1
        if plan.reasoning.startswith("API error") or "exhausted" in plan.reasoning:
            errors += 1

        sym_cfg = sym_configs[sym]
        if plan.direction not in sym_cfg.allowed_directions and plan.direction != "hold":
            plan = TradePlan("hold", 0, 0, 0, f"direction {plan.direction} not allowed")

        last_close = float(bar["close"])
        if plan.direction == "short":
            if plan.buy_price > 0 and plan.buy_price < last_close:
                plan.buy_price = last_close * 1.001
            if plan.sell_price > 0 and plan.sell_price > last_close:
                plan.sell_price = last_close * 0.995

        action_rows.append({
            "timestamp": ts, "symbol": sym,
            "buy_price": plan.buy_price, "sell_price": plan.sell_price,
            "direction": plan.direction, "confidence": plan.confidence,
        })
        bar_rows.append(bar)

        if i % 24 == 0:
            print(
                f"  [{ts}] {sym} -> {plan.direction.upper()} "
                f"(conf={plan.confidence:.1f}, buy={plan.buy_price:.2f}, sell={plan.sell_price:.2f})"
            )

    bars_df = pd.DataFrame(bar_rows)
    actions_df = pd.DataFrame(action_rows)

    if bars_df.empty or actions_df.empty:
        return {"error": "no data"}

    bars_df["timestamp"] = pd.to_datetime(bars_df["timestamp"], utc=True)
    actions_df["timestamp"] = pd.to_datetime(actions_df["timestamp"], utc=True)

    # Run simulation
    print(f"\n  Running market simulation ({len(bars_df)} bars, {len(actions_df)} actions)...")
    result = simulate(bars_df, actions_df, config, sym_configs)

    metrics = result["metrics"]
    all_trades = result["trades"]
    buys = sum(1 for t in all_trades if t["side"] in ("buy", "short"))
    exits = sum(1 for t in all_trades if t["side"] in ("sell", "cover", "close"))
    realized_pnl = sum(t["realized_pnl"] for t in all_trades)
    total_fees = sum(t["fee"] for t in all_trades)

    # Per-symbol breakdown
    per_sym = {}
    for t in all_trades:
        sym = t["symbol"]
        if sym not in per_sym:
            per_sym[sym] = {"entries": 0, "exits": 0, "realized_pnl": 0, "fees": 0}
        if t["side"] in ("buy", "short"):
            per_sym[sym]["entries"] += 1
        else:
            per_sym[sym]["exits"] += 1
        per_sym[sym]["realized_pnl"] += t["realized_pnl"]
        per_sym[sym]["fees"] += t["fee"]

    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS: {config.model} ({config.prompt_variant} prompt)")
    print(f"{'='*60}")
    print(f"  Window: {start_ts} -> {end_ts} ({days}d)")
    print(f"  API calls: {api_calls} (errors: {errors})")
    print(f"  Total return: {metrics['total_return_pct']:+.4f}%")
    print(f"  Max drawdown: {metrics['max_drawdown_pct']:.4f}%")
    print(f"  Sortino: {metrics['sortino']:.4f}")
    print(f"  Entries: {buys}, Exits: {exits}")
    print(f"  Realized PnL: ${realized_pnl:+.2f}")
    print(f"  Total fees: ${total_fees:.2f}")
    print(f"  Final equity: ${metrics['final_equity']:,.2f}")
    print()
    for sym, stats in sorted(per_sym.items()):
        cfg = sym_configs.get(sym)
        dirs = cfg.allowed_directions if cfg else ["?"]
        print(f"  {sym:10s} ({'/'.join(dirs):>5s}): "
              f"{stats['entries']} entries, {stats['exits']} exits, "
              f"PnL=${stats['realized_pnl']:+.2f}, fees=${stats['fees']:.2f}")
    print(f"{'='*60}\n")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_tag = config.model.replace("/", "_").replace(".", "")
    tag = f"{model_tag}_{config.prompt_variant}_{days}d_{'_'.join(usable_symbols)}"
    result_data = {
        "model": config.model,
        "prompt_variant": config.prompt_variant,
        "symbols": usable_symbols,
        "symbol_directions": {s: sym_configs[s].allowed_directions for s in usable_symbols},
        "days": days,
        "window": f"{start_ts} -> {end_ts}",
        **{k: v for k, v in config.__dict__.items() if k != "model"},
        "api_calls": api_calls,
        "errors": errors,
        **metrics,
        "total_trades": buys + exits,
        "entries": buys,
        "exits": exits,
        "realized_pnl": realized_pnl,
        "total_fees": total_fees,
        "per_symbol": per_sym,
    }

    out_path = RESULTS_DIR / f"{tag}.json"
    with open(out_path, "w") as f:
        json.dump(result_data, f, indent=2, default=str)
    print(f"  Results: {out_path}")

    trades_path = RESULTS_DIR / f"{tag}_trades.json"
    with open(trades_path, "w") as f:
        json.dump(all_trades, f, indent=2, default=str)
    print(f"  Trades: {trades_path}")

    eq_path = RESULTS_DIR / f"{tag}_equity.csv"
    pd.DataFrame({
        "timestamp": result["timestamps"],
        "equity": result["equity_history"],
    }).to_csv(eq_path, index=False)
    print(f"  Equity: {eq_path}")

    return result_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

GROUPS = {
    "crypto": ["BTCUSD", "ETHUSD", "SOLUSD", "LINKUSD", "UNIUSD"],
    "ai_stocks": ["NVDA", "META", "AAPL", "TSLA", "MSFT", "GOOG", "PLTR", "NFLX", "NET"],
    "short_stocks": ["DBX", "TRIP", "NYT", "YELP"],
    "all": list(SYMBOL_UNIVERSE.keys()),
}


def main():
    parser = argparse.ArgumentParser(description="Backtest Gemini Flash Lite hourly trader v2")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--group", choices=list(GROUPS.keys()), default=None)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument(
        "--prompt",
        choices=[
            "default",
            "conservative",
            "aggressive",
            "position_context",
            "no_forecast",
            "h1_only",
            "h24_only",
            "uncertainty_gated",
            "uncertainty_strict",
            "freeform",
            "mae_bands",
            "fractional",
            "anonymized",
        ],
        default="default",
    )
    parser.add_argument("--max-hold-hours", type=int, default=6)
    parser.add_argument("--max-position-pct", type=float, default=0.25)
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--rate-limit", type=float, default=4.2)
    parser.add_argument("--model", type=str, default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--parallel", type=int, default=1, help="Parallel API workers (for codex/deepseek)")
    args = parser.parse_args()

    symbols = args.symbols or GROUPS.get(args.group, GROUPS["crypto"])

    config = BacktestConfig(
        initial_cash=args.initial_cash,
        max_hold_hours=args.max_hold_hours,
        max_position_pct=args.max_position_pct,
        rate_limit_seconds=args.rate_limit,
        model=args.model,
        prompt_variant=args.prompt,
        parallel_workers=args.parallel,
    )

    run_backtest(symbols=symbols, days=args.days, config=config)


if __name__ == "__main__":
    main()
