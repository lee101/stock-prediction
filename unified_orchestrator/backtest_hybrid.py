"""Backtest RL+Gemini hybrid vs RL-only vs Gemini-only vs Claude.

Uses actual LLM API calls with Chronos2 forecasts on real price data,
then runs through the HourlyTrader simulator for realistic fill modeling.

Prompts include time/day-of-week context and previous position state
to match production behavior.

Usage:
  python -m unified_orchestrator.backtest_hybrid \
      --symbols BTCUSD ETHUSD SOLUSD --days 7

  # Compare Gemini vs Sonnet:
  python -m unified_orchestrator.backtest_hybrid \
      --symbols BTCUSD ETHUSD SOLUSD --days 7 \
      --model claude-sonnet-4-6 --thinking-level HIGH --reasoning-effort low
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from env_real import *  # noqa: F401,F403 — exports API keys to os.environ

from llm_hourly_trader.gemini_wrapper import TradePlan, build_prompt
from llm_hourly_trader.providers import call_llm
from newnanoalpacahourlyexp.marketsimulator.hourly_trader import (
    HourlyTraderMarketSimulator,
    HourlyTraderSimulationConfig,
)
from unified_orchestrator.rl_gemini_bridge import (
    RLSignal,
    build_hybrid_prompt,
)

FORECAST_DIR = REPO / "binanceneural" / "forecast_cache"
DATA_DIR = REPO / "trainingdatahourly" / "crypto"


@dataclass
class PositionState:
    """Track simulated position state per symbol for sequential prompts."""
    direction: str = "flat"  # flat, long
    entry_price: float = 0.0
    qty: float = 0.0
    entry_time: Optional[pd.Timestamp] = None
    hold_hours: int = 0
    unrealized_pnl_pct: float = 0.0


@dataclass
class PrevPlanOutcome:
    """What happened with the previous bar's plan."""
    direction: str = "hold"
    buy_price: float = 0.0
    sell_price: float = 0.0
    confidence: float = 0.0
    was_filled: bool = False
    pnl_pct: float = 0.0


def _build_context_header(
    ts: pd.Timestamp,
    position: PositionState,
    prev_outcome: Optional[PrevPlanOutcome],
    cash: float,
    equity: float,
) -> str:
    """Build time/day/position context header to prepend to prompts."""
    lines = []

    # Time context
    day_name = ts.strftime("%A")
    hour_str = ts.strftime("%H:%M UTC")
    lines.append(f"TIME: {day_name} {hour_str}")

    # Position context
    if position.direction == "long":
        lines.append(
            f"CURRENT POSITION: LONG @ ${position.entry_price:.2f} "
            f"(held {position.hold_hours}h, unrealized {position.unrealized_pnl_pct:+.2f}%)"
        )
        lines.append("  ** You MUST set sell_price as your exit target for this position. **")
    else:
        lines.append("CURRENT POSITION: FLAT (no position)")

    # Previous plan outcome
    if prev_outcome:
        filled_str = "FILLED" if prev_outcome.was_filled else "NOT FILLED"
        lines.append(
            f"PREVIOUS PLAN: {prev_outcome.direction} | "
            f"buy=${prev_outcome.buy_price:.2f} sell=${prev_outcome.sell_price:.2f} | "
            f"conf={prev_outcome.confidence:.2f} | {filled_str}"
        )
        if prev_outcome.was_filled and prev_outcome.pnl_pct != 0:
            lines.append(f"  PnL: {prev_outcome.pnl_pct:+.2f}%")

    # Account
    lines.append(f"ACCOUNT: Cash ${cash:,.0f} | Equity ${equity:,.0f}")

    return "\n".join(lines) + "\n\n"


def load_bars(symbol: str) -> pd.DataFrame:
    """Load hourly bars for a crypto symbol."""
    csv = DATA_DIR / f"{symbol}.csv"
    if csv.exists():
        df = pd.read_csv(csv)
    else:
        csv = REPO / f"binance_spot_hourly/{symbol.replace('USD', 'USDT')}.csv"
        df = pd.read_csv(csv)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    elif "date" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def load_forecasts(symbol: str, horizon: str) -> pd.DataFrame:
    """Load Chronos2 forecasts."""
    path = FORECAST_DIR / horizon / f"{symbol}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def get_forecast_at(fc_df: pd.DataFrame, ts: pd.Timestamp) -> dict | None:
    """Get forecast at or just before timestamp."""
    if fc_df.empty:
        return None
    match = fc_df[fc_df["timestamp"] <= ts]
    if match.empty:
        return None
    row = match.iloc[-1]
    return {
        "predicted_close_p50": float(row.get("predicted_close_p50", 0)),
        "predicted_close_p10": float(row.get("predicted_close_p10", 0)),
        "predicted_close_p90": float(row.get("predicted_close_p90", 0)),
        "predicted_high_p50": float(row.get("predicted_high_p50", 0)),
        "predicted_low_p50": float(row.get("predicted_low_p50", 0)),
    }


def generate_rl_signal(close_prices: np.ndarray, idx: int) -> RLSignal:
    """Generate a mock RL signal from price momentum (proxy for actual RL model)."""
    if idx < 48:
        return RLSignal(0, "", "flat", 0.5, 0.0, 0.0)

    sma_12 = close_prices[max(0, idx-12):idx].mean()
    sma_48 = close_prices[max(0, idx-48):idx].mean()

    if sma_12 > sma_48:
        gap = (sma_12 - sma_48) / sma_48
        confidence = min(0.95, 0.5 + gap * 50)
        return RLSignal(0, "", "long", confidence, gap * 100, 1.0)
    else:
        return RLSignal(0, "", "flat", 0.5, 0.0, 0.0)


def _update_position(
    position: PositionState,
    plan: TradePlan,
    close: float,
    prev_close: float,
) -> PrevPlanOutcome:
    """Simple position tracking: check if limit prices would have filled."""
    outcome = PrevPlanOutcome(
        direction=plan.direction,
        buy_price=plan.buy_price,
        sell_price=plan.sell_price,
        confidence=plan.confidence,
    )

    if position.direction == "long":
        position.hold_hours += 1
        position.unrealized_pnl_pct = (close - position.entry_price) / position.entry_price * 100
        # Check if sell limit was hit (price went above sell_price)
        if plan.sell_price > 0 and close >= plan.sell_price:
            outcome.was_filled = True
            outcome.pnl_pct = (plan.sell_price - position.entry_price) / position.entry_price * 100
            # Exit position
            position.direction = "flat"
            position.entry_price = 0
            position.qty = 0
            position.hold_hours = 0
            position.unrealized_pnl_pct = 0
    elif plan.direction == "long" and plan.buy_price > 0:
        # Check if buy limit was hit (price went below buy_price)
        if close <= plan.buy_price:
            outcome.was_filled = True
            position.direction = "long"
            position.entry_price = plan.buy_price
            position.qty = 1.0
            position.hold_hours = 0
            position.unrealized_pnl_pct = (close - plan.buy_price) / plan.buy_price * 100

    return outcome


def run_backtest(
    symbols: list[str],
    days: int = 7,
    initial_cash: float = 10_000.0,
    modes: list[str] = None,
    model: str = "gemini-3.1-flash-lite-preview",
    thinking_level: str = "HIGH",
    reasoning_effort: str | None = None,
) -> dict:
    """Run backtest across modes with sequential position tracking.

    Modes:
      - gemini_only: LLM prompt with Chronos2 (current production)
      - rl_only: RL signal → fixed spread prices (no LLM)
      - rl_gemini: RL signal → LLM refinement with Chronos2 + RL context
    """
    if modes is None:
        modes = ["gemini_only", "rl_only", "rl_gemini"]

    # Load data
    print("Loading data...")
    all_bars = {}
    all_fc_h1 = {}
    all_fc_h24 = {}
    for sym in symbols:
        try:
            all_bars[sym] = load_bars(sym)
            all_fc_h1[sym] = load_forecasts(sym, "h1")
            all_fc_h24[sym] = load_forecasts(sym, "h24")
            print(f"  {sym}: {len(all_bars[sym])} bars, "
                  f"fc_h1={len(all_fc_h1[sym])}, fc_h24={len(all_fc_h24[sym])}")
        except Exception as e:
            print(f"  {sym}: SKIP ({e})")

    usable = [s for s in symbols if s in all_bars]
    if not usable:
        print("No usable symbols")
        return {}

    # Determine time window
    end_ts = min(all_bars[s]["timestamp"].max() for s in usable)
    start_ts = end_ts - pd.Timedelta(days=days)
    print(f"\nBacktest window: {start_ts} to {end_ts} ({days}d)")
    print(f"Model: {model} | thinking={thinking_level} | effort={reasoning_effort}")

    results = {}

    for mode in modes:
        print(f"\n{'=' * 60}")
        print(f"MODE: {mode.upper()}")
        print(f"{'=' * 60}")

        all_bar_dfs = []
        all_action_dfs = []
        api_calls = 0
        t0 = time.time()

        for sym in usable:
            bars = all_bars[sym]
            window = bars[(bars["timestamp"] >= start_ts) & (bars["timestamp"] <= end_ts)].copy()
            if len(window) < 12:
                continue

            all_bar_dfs.append(window[["timestamp", "open", "high", "low", "close"]].assign(symbol=sym))

            closes_all = bars["close"].values
            actions = []
            position = PositionState()
            prev_outcome: Optional[PrevPlanOutcome] = None
            cash = initial_cash
            equity = initial_cash

            for i, (_, bar) in enumerate(window.iterrows()):
                ts = bar["timestamp"]
                close = bar["close"]
                bar_idx = bars.index.get_loc(bar.name)
                prev_close = closes_all[bar_idx - 1] if bar_idx > 0 else close

                # Update position state from previous bar
                if position.direction == "long":
                    position.unrealized_pnl_pct = (close - position.entry_price) / position.entry_price * 100
                    equity = cash + (close / position.entry_price) * 100  # rough
                else:
                    equity = cash

                if mode == "rl_only":
                    rl_sig = generate_rl_signal(closes_all, bar_idx)
                    if rl_sig.direction == "long" and rl_sig.confidence > 0.55:
                        buy_price = close * 0.998
                        sell_price = close * 1.008
                    else:
                        buy_price = 0
                        sell_price = close * 0.998
                    plan = TradePlan(
                        direction="long" if buy_price > 0 else "hold",
                        buy_price=buy_price, sell_price=sell_price,
                        confidence=rl_sig.confidence, reasoning="rl_momentum",
                    )

                elif mode == "gemini_only":
                    hist_slice = bars[bars["timestamp"] <= ts].tail(25)
                    fc_1h = get_forecast_at(all_fc_h1[sym], ts)
                    fc_24h = get_forecast_at(all_fc_h24[sym], ts)

                    # Build context header with time/day/position
                    ctx_header = _build_context_header(
                        ts, position, prev_outcome, cash, equity,
                    )

                    try:
                        base_prompt = build_prompt(
                            symbol=sym,
                            history_rows=hist_slice.to_dict("records"),
                            forecast_1h=fc_1h, forecast_24h=fc_24h,
                            current_position=position.direction,
                            cash=cash, equity=equity,
                            allowed_directions=["long"],
                            asset_class="crypto", maker_fee=0.0008,
                        )
                        prompt = ctx_header + base_prompt
                        plan = call_llm(prompt, model=model, thinking_level=thinking_level,
                                        reasoning_effort=reasoning_effort)
                        api_calls += 1
                        buy_price = plan.buy_price if plan.direction == "long" else 0
                        sell_price = plan.sell_price if plan.sell_price > 0 else close * 1.01
                    except Exception as e:
                        buy_price = 0
                        sell_price = 0
                        plan = TradePlan("hold", 0, 0, 0, f"error: {e}")

                elif mode == "rl_gemini":
                    rl_sig = generate_rl_signal(closes_all, bar_idx)

                    if rl_sig.direction == "flat" and position.direction != "long":
                        buy_price = 0
                        sell_price = close * 0.998
                        plan = TradePlan("hold", 0, sell_price, 0.3, "rl_flat")
                    else:
                        hist_slice = bars[bars["timestamp"] <= ts].tail(25)
                        fc_1h = get_forecast_at(all_fc_h1[sym], ts)
                        fc_24h = get_forecast_at(all_fc_h24[sym], ts)

                        ctx_header = _build_context_header(
                            ts, position, prev_outcome, cash, equity,
                        )

                        rl_sig_named = RLSignal(
                            symbol_idx=0, symbol_name=sym,
                            direction=rl_sig.direction,
                            confidence=rl_sig.confidence,
                            logit_gap=rl_sig.logit_gap,
                            allocation_pct=rl_sig.allocation_pct,
                        )
                        try:
                            base_prompt = build_hybrid_prompt(
                                symbol=sym,
                                rl_signal=rl_sig_named,
                                history_rows=hist_slice.to_dict("records"),
                                current_price=close,
                                forecast_1h=fc_1h,
                                forecast_24h=fc_24h,
                            )
                            prompt = ctx_header + base_prompt
                            plan = call_llm(prompt, model=model, thinking_level=thinking_level,
                                            reasoning_effort=reasoning_effort)
                            api_calls += 1
                            buy_price = plan.buy_price if plan.direction == "long" else 0
                            sell_price = plan.sell_price if plan.sell_price > 0 else close * 1.01
                        except Exception as e:
                            buy_price = close * 0.998 if rl_sig.confidence > 0.6 else 0
                            sell_price = close * 1.01
                            plan = TradePlan(
                                "long" if buy_price > 0 else "hold",
                                buy_price, sell_price, 0.3, f"fallback: {e}",
                            )

                # Update position tracking
                prev_outcome = _update_position(position, plan, close, prev_close)

                actions.append({
                    "timestamp": ts, "symbol": sym,
                    "buy_price": buy_price, "sell_price": sell_price,
                    "buy_amount": 100 if buy_price > 0 else 0,
                    "sell_amount": 100 if buy_price == 0 else 0,
                })

                # Progress
                if (i + 1) % 50 == 0:
                    elapsed = time.time() - t0
                    rate = api_calls / max(elapsed, 0.1)
                    print(f"  {sym}: {i+1}/{len(window)} bars, "
                          f"{api_calls} API calls, {elapsed:.0f}s ({rate:.1f} calls/s)")

            all_action_dfs.append(pd.DataFrame(actions))

        if not all_bar_dfs:
            results[mode] = {"error": "no data"}
            continue

        bars_df = pd.concat(all_bar_dfs, ignore_index=True)
        actions_df = pd.concat(all_action_dfs, ignore_index=True)

        elapsed = time.time() - t0
        print(f"  Generated {len(actions_df)} actions in {elapsed:.0f}s "
              f"({api_calls} API calls)")

        # Run simulator
        cfg = HourlyTraderSimulationConfig(
            initial_cash=initial_cash,
            allocation_pct=1.0 / len(usable),
            max_leverage=1.0,
            enforce_market_hours=False,
            allow_short=False,
            decision_lag_bars=1,
            fill_buffer_bps=5.0,
            partial_fill_on_touch=True,
        )

        try:
            sim = HourlyTraderMarketSimulator(cfg)
            result = sim.run(bars_df, actions_df)
            total_return = (result.equity_curve.iloc[-1] / initial_cash - 1) * 100
            results[mode] = {
                "return_pct": total_return,
                "final_equity": result.equity_curve.iloc[-1],
                "fills": len(result.fills),
                "sortino": result.metrics.get("sortino", 0),
                "max_drawdown": result.metrics.get("max_drawdown_pct", 0),
                "api_calls": api_calls,
                "elapsed_s": elapsed,
            }
            print(f"  Return: {total_return:+.2f}%")
            print(f"  Sortino: {results[mode]['sortino']:.2f}")
            print(f"  Fills: {len(result.fills)}")
        except Exception as e:
            print(f"  Sim error: {e}")
            import traceback
            traceback.print_exc()
            results[mode] = {"error": str(e)}

    # Summary
    print(f"\n{'=' * 60}")
    print("BACKTEST COMPARISON")
    print(f"{'=' * 60}")
    for mode, data in results.items():
        if "error" in data:
            print(f"  {mode}: ERROR - {data['error']}")
        else:
            print(f"  {mode}: {data['return_pct']:+.2f}% | "
                  f"Sortino={data['sortino']:.2f} | "
                  f"DD={data['max_drawdown']:.2f}% | "
                  f"{data['fills']} fills | "
                  f"{data['api_calls']} API calls | "
                  f"{data['elapsed_s']:.0f}s")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["BTCUSD", "ETHUSD", "SOLUSD"])
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--cash", type=float, default=10000.0)
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--thinking-level", default="HIGH")
    parser.add_argument("--reasoning-effort", default=None,
                        help="Effort level for Anthropic/OpenAI reasoning models (low/medium/high/max)")
    parser.add_argument("--modes", nargs="+", default=["rl_only", "gemini_only", "rl_gemini"])
    args = parser.parse_args()

    run_backtest(
        symbols=args.symbols,
        days=args.days,
        initial_cash=args.cash,
        modes=args.modes,
        model=args.model,
        thinking_level=args.thinking_level,
        reasoning_effort=args.reasoning_effort,
    )


if __name__ == "__main__":
    main()
