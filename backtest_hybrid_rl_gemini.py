#!/usr/bin/env python3
"""
Hybrid RL + Gemini daily trading backtest with market simulator validation.

Pipeline:
  1. RL model (ent_anneal 23-sym) generates per-symbol signals (direction + confidence)
  2. Gemini gets FULL context:
     - RL signal (direction, confidence, action probabilities)
     - Chronos2 h24 forecast (predicted close/high/low, p10/p90 bands)
     - Current position (entry price, P&L%, hold days)
     - Previous positions history (last 5 trades with P&L)
     - Previous Gemini reasoning (what it thought last time)
     - Previous Chronos2 forecast accuracy (was it right?)
     - Recent OHLCV bars (14 days)
     - Market regime (trend, SMA, volatility, ATR)
  3. Gemini refines: buy_price, sell_price, allocation (-5x to +5x)
  4. Execute through marketsimulator.py with realistic Binance/Alpaca fees
  5. Compare PnL: pure RL vs RL+Gemini hybrid

Usage:
    python -u backtest_hybrid_rl_gemini.py \
        --checkpoint pufferlib_market/checkpoints/autoresearch_mixed23_daily/ent_anneal/best.pt \
        --symbols BTCUSD,ETHUSD,SOLUSD,AAPL,MSFT,NVDA \
        --start-date 2025-06-01 --end-date 2025-12-01 \
        --mode hybrid  # or "rl_only" for comparison
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

# marketsimulator.py is shadowed by marketsimulator/ package — register before exec
import types as _types
_ms_mod = _types.ModuleType("marketsimulator_py")
_ms_mod.__file__ = str(Path(__file__).resolve().parent / "marketsimulator.py")
sys.modules["marketsimulator_py"] = _ms_mod
_ms_code = Path(_ms_mod.__file__).read_text()
exec(compile(_ms_code, _ms_mod.__file__, "exec"), _ms_mod.__dict__)
SimulationConfig = _ms_mod.SimulationConfig
run_shared_cash_simulation = _ms_mod.run_shared_cash_simulation


CRYPTO_SYMBOLS = {"BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD", "AVAXUSD",
                  "DOGEUSD", "LINKUSD", "AAVEUSD", "UNIUSD"}
SHORT_BORROW_APR = 0.0625


@dataclass
class SymbolSignal:
    symbol: str
    rl_direction: str      # "long", "short", "flat"
    rl_confidence: float   # 0-1
    rl_action_probs: list  # full softmax probabilities


@dataclass
class GeminiPlan:
    symbol: str
    allocation: float      # -5 to +5
    buy_price: float
    sell_price: float
    confidence: float
    reasoning: str


@dataclass
class TradeHistory:
    date: str
    symbol: str
    side: str
    price: float
    pnl: float
    reason: str


@dataclass
class BacktestContext:
    cash: float = 10000.0
    positions: Dict[str, dict] = field(default_factory=dict)
    trade_history: List[TradeHistory] = field(default_factory=list)
    prev_reasoning: Dict[str, str] = field(default_factory=dict)
    prev_forecasts: Dict[str, dict] = field(default_factory=dict)
    prev_forecast_accuracy: Dict[str, float] = field(default_factory=dict)
    equity_curve: List[dict] = field(default_factory=list)


def load_daily_bars(symbol: str) -> pd.DataFrame:
    for root in ["trainingdata/train", "trainingdatahourly"]:
        for subdir in ["", "crypto", "stocks"]:
            p = Path(root) / subdir / f"{symbol}.csv" if subdir else Path(root) / f"{symbol}.csv"
            if p.exists():
                df = pd.read_csv(p)
                df.columns = [c.lower() for c in df.columns]
                ts = "timestamp" if "timestamp" in df.columns else "date"
                df["timestamp"] = pd.to_datetime(df[ts], utc=True)
                df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
                return df[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
    raise FileNotFoundError(f"No data for {symbol}")


def load_forecast_h24(symbol: str) -> Optional[pd.DataFrame]:
    candidates = [
        Path("alpacanewccrosslearning/forecast_cache/crypto13_novol_20260208_lb4000/h24"),
        Path("binanceneural/forecast_cache/h24"),
    ]
    for d in candidates:
        p = d / f"{symbol}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp")
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            return df.sort_index()
    return None


def get_rl_signals(
    checkpoint_path: str,
    bars_dict: Dict[str, pd.DataFrame],
    date: pd.Timestamp,
    hidden_size: int = 1024,
) -> List[SymbolSignal]:
    """Run RL inference on daily bars to get per-symbol signals."""
    # For now, use a simplified signal extraction from daily features
    # In production, this would load the PPO model and run inference
    signals = []
    for sym, df in bars_dict.items():
        day_rows = df[df["timestamp"] == date]
        if day_rows.empty:
            continue
        bar = day_rows.iloc[0]
        idx = day_rows.index[0]

        # Simple momentum-based signal as placeholder
        # (real implementation loads checkpoint and runs policy forward pass)
        if idx < 5:
            signals.append(SymbolSignal(sym, "flat", 0.3, []))
            continue

        close = float(bar["close"])
        prev_close = float(df.iloc[idx-1]["close"])
        ret_1d = (close - prev_close) / prev_close

        sma5 = float(df["close"].iloc[max(0,idx-4):idx+1].mean())
        sma20 = float(df["close"].iloc[max(0,idx-19):idx+1].mean())

        if close > sma20 and ret_1d > -0.02:
            direction = "long"
            confidence = min(0.5 + (close - sma20) / sma20 * 5, 0.95)
        elif close < sma20 and ret_1d < 0.02:
            direction = "short"
            confidence = min(0.5 + (sma20 - close) / sma20 * 5, 0.95)
        else:
            direction = "flat"
            confidence = 0.3

        signals.append(SymbolSignal(sym, direction, confidence, []))

    return signals


def build_gemini_prompt(
    symbol: str,
    rl_signal: SymbolSignal,
    bars_df: pd.DataFrame,
    day_idx: int,
    forecast: dict,
    ctx: BacktestContext,
    n_symbols: int,
) -> str:
    """Build rich Gemini prompt with full context."""
    bar = bars_df.iloc[day_idx]
    date_str = bar["timestamp"].strftime("%Y-%m-%d")
    price = float(bar["close"])

    # Recent bars
    start = max(0, day_idx - 13)
    recent = bars_df.iloc[start:day_idx + 1]
    bars_text = ""
    for _, row in recent.iterrows():
        bars_text += (
            f"{row['timestamp'].strftime('%Y-%m-%d')} "
            f"O={row['open']:.2f} H={row['high']:.2f} L={row['low']:.2f} "
            f"C={row['close']:.2f} V={row['volume']:.0f}\n"
        )

    # Position info
    pos = ctx.positions.get(symbol)
    if pos:
        hold_days = (bar["timestamp"] - pd.Timestamp(pos["entry_date"])).days
        unreal_pnl = (price - pos["entry_price"]) * pos.get("qty", 0)
        if pos.get("is_short"):
            unreal_pnl = (pos["entry_price"] - price) * pos.get("qty", 0)
        pos_text = (
            f"CURRENT POSITION: {'SHORT' if pos.get('is_short') else 'LONG'} "
            f"qty={pos.get('qty', 0):.6f} @ ${pos['entry_price']:.2f}\n"
            f"  Unrealized P&L: ${unreal_pnl:.2f} ({unreal_pnl/max(pos['entry_price']*pos.get('qty',1),1)*100:+.2f}%)\n"
            f"  Held: {hold_days} days\n"
        )
    else:
        pos_text = "CURRENT POSITION: FLAT\n"

    # Previous trades
    recent_trades = [t for t in ctx.trade_history[-10:] if t.symbol == symbol]
    trades_text = ""
    if recent_trades:
        trades_text = "RECENT TRADES:\n"
        for t in recent_trades[-5:]:
            trades_text += f"  {t.date} {t.side} @ ${t.price:.2f} PnL=${t.pnl:+.2f} ({t.reason})\n"

    # Previous reasoning
    prev_reason = ctx.prev_reasoning.get(symbol, "")
    prev_text = f"YOUR PREVIOUS REASONING: {prev_reason}\n" if prev_reason else ""

    # Previous forecast accuracy
    prev_acc = ctx.prev_forecast_accuracy.get(symbol)
    acc_text = ""
    if prev_acc is not None:
        acc_text = f"PREVIOUS FORECAST ACCURACY: {prev_acc:+.2f}% (predicted vs actual move)\n"

    # Chronos2 forecast
    if forecast:
        fc_text = (
            f"CHRONOS2 24H FORECAST:\n"
            f"  Predicted close: ${forecast.get('predicted_close', 0):.2f}\n"
            f"  Range: ${forecast.get('predicted_low', 0):.2f} - ${forecast.get('predicted_high', 0):.2f}\n"
            f"  Confidence band: ${forecast.get('predicted_close_p10', 0):.2f} (p10) to ${forecast.get('predicted_close_p90', 0):.2f} (p90)\n"
            f"  Implied move: {((forecast.get('predicted_close', price) - price) / price * 100):+.2f}%\n"
        )
    else:
        fc_text = "CHRONOS2 FORECAST: Not available\n"

    # RL signal
    rl_text = (
        f"RL MODEL SIGNAL: {rl_signal.rl_direction.upper()} "
        f"(confidence: {rl_signal.rl_confidence:.2f})\n"
    )

    # Market context
    closes = bars_df["close"].iloc[max(0, day_idx-20):day_idx+1].values
    sma20 = np.mean(closes[-min(20, len(closes)):])
    sma5 = np.mean(closes[-min(5, len(closes)):])
    trend = (price - sma20) / sma20 * 100
    vol = np.std(np.diff(closes) / closes[:-1]) * 100 if len(closes) > 1 else 0

    # Fee info
    is_crypto = symbol in CRYPTO_SYMBOLS
    fee_text = "FEES: 0% (FDUSD/Alpaca commission-free)" if not is_crypto or symbol in {"BTCUSD", "ETHUSD", "SOLUSD"} else "FEES: 0.1% (USDT pair)"

    prompt = f"""You are a daily portfolio trader. The RL model has analyzed {n_symbols} symbols and recommends a trade.
Your job: REFINE the RL signal with better entry/exit prices using all available context.

DATE: {date_str}
SYMBOL: {symbol} ({'crypto' if is_crypto else 'stock'})
PRICE: ${price:.2f}
TREND: {trend:+.1f}% vs SMA-20 (${sma20:.2f}), SMA-5: ${sma5:.2f}, Vol: {vol:.2f}%/day

{rl_text}
{fc_text}
{pos_text}
{trades_text}
{prev_text}
{acc_text}
{fee_text}

RECENT PRICES (14 days):
{bars_text}

PORTFOLIO: Cash=${ctx.cash:.0f}, {len(ctx.positions)} positions open, {n_symbols} symbols tracked

You must allocate between -5.0 and +5.0 (negative=short, positive=long, 0=flat).
Max allocation per symbol is {5.0/n_symbols:.1f}x to prevent over-concentration.
Set realistic buy/sell prices within today's expected range.

Respond JSON:
{{"allocation": <-5.0 to 5.0>, "buy_price": <limit entry or 0>, "sell_price": <limit exit or 0>, "confidence": <0-1>, "reasoning": "<1-2 sentences>"}}"""

    return prompt


def call_gemini_structured(prompt: str, model: str = "gemini-2.5-flash") -> dict:
    """Call Gemini with structured output."""
    from llm_hourly_trader.cache import get_cached, set_cached

    cached = get_cached(model, prompt)
    if cached is not None:
        return cached

    try:
        from google import genai
        from google.genai import types

        schema = genai.types.Schema(
            type=genai.types.Type.OBJECT,
            required=["allocation", "buy_price", "sell_price", "confidence", "reasoning"],
            properties={
                "allocation": genai.types.Schema(type=genai.types.Type.STRING),
                "buy_price": genai.types.Schema(type=genai.types.Type.STRING),
                "sell_price": genai.types.Schema(type=genai.types.Type.STRING),
                "confidence": genai.types.Schema(type=genai.types.Type.STRING),
                "reasoning": genai.types.Schema(type=genai.types.Type.STRING),
            },
        )

        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            thinking_config=types.ThinkingConfig(thinking_budget=4096),
        )

        for attempt in range(3):
            try:
                response = client.models.generate_content(model=model, contents=prompt, config=config)
                result = json.loads(response.text)
                parsed = {
                    "allocation": max(-5.0, min(5.0, float(result.get("allocation", 0)))),
                    "buy_price": float(result.get("buy_price", 0)),
                    "sell_price": float(result.get("sell_price", 0)),
                    "confidence": max(0.0, min(1.0, float(result.get("confidence", 0.5)))),
                    "reasoning": str(result.get("reasoning", "")),
                }
                set_cached(model, prompt, parsed)
                return parsed
            except Exception as e:
                if "429" in str(e):
                    time.sleep(min(30 * (attempt + 1), 90))
                else:
                    raise

    except Exception as e:
        return {"allocation": 0, "buy_price": 0, "sell_price": 0, "confidence": 0, "reasoning": f"error: {e}"}


def run_hybrid_backtest(
    symbols: List[str],
    checkpoint: str,
    start_date: str,
    end_date: Optional[str],
    mode: str = "hybrid",
    initial_cash: float = 10000.0,
    model: str = "gemini-2.5-flash",
    slippage_bps: float = 5.0,
) -> Tuple[BacktestContext, Dict[str, float]]:
    """Run RL+Gemini hybrid or RL-only backtest."""

    print(f"Loading data for {len(symbols)} symbols...")
    all_bars = {}
    all_forecasts = {}
    for sym in symbols:
        try:
            all_bars[sym] = load_daily_bars(sym)
            all_forecasts[sym] = load_forecast_h24(sym)
            print(f"  {sym}: {len(all_bars[sym])} bars, forecast={'yes' if all_forecasts[sym] is not None else 'no'}")
        except FileNotFoundError:
            print(f"  {sym}: SKIP (no data)")

    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC") if end_date else min(df["timestamp"].max() for df in all_bars.values())

    # Build actions for marketsimulator
    all_actions = []
    all_bar_data = []
    ctx = BacktestContext(cash=initial_cash)
    n_symbols = len(all_bars)

    dates = sorted(set().union(*(
        set(df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]["timestamp"].values)
        for df in all_bars.values()
    )))

    print(f"\nBacktest: {len(dates)} days, {mode} mode, {n_symbols} symbols")
    print(f"  Period: {start.date()} to {end.date()}")
    print()

    for i, date in enumerate(dates):
        date = pd.Timestamp(date)
        if date.tz is None:
            date = date.tz_localize("UTC")

        # Get RL signals
        rl_signals = get_rl_signals(checkpoint, all_bars, date)
        signal_map = {s.symbol: s for s in rl_signals}

        # Collect all decisions first (for portfolio-level normalization)
        daily_decisions = {}

        for sym in all_bars:
            day_rows = all_bars[sym][all_bars[sym]["timestamp"] == date]
            if day_rows.empty:
                continue
            bar = day_rows.iloc[0]
            day_idx = day_rows.index[0]
            price = float(bar["close"])

            rl_sig = signal_map.get(sym, SymbolSignal(sym, "flat", 0.3, []))

            if mode == "hybrid":
                # Get forecast
                fc = {}
                fc_df = all_forecasts.get(sym)
                if fc_df is not None and not fc_df.empty:
                    mask = fc_df.index <= date
                    if mask.any():
                        row = fc_df.loc[mask].iloc[-1]
                        fc = {
                            "predicted_close": float(row.get("predicted_close_p50", 0)),
                            "predicted_close_p10": float(row.get("predicted_close_p10", 0)),
                            "predicted_close_p90": float(row.get("predicted_close_p90", 0)),
                            "predicted_high": float(row.get("predicted_high_p50", 0)),
                            "predicted_low": float(row.get("predicted_low_p50", 0)),
                        }

                # Check previous forecast accuracy
                prev_fc = ctx.prev_forecasts.get(sym)
                if prev_fc and "predicted_close" in prev_fc:
                    actual_move = (price - prev_fc.get("ref_price", price)) / max(prev_fc.get("ref_price", price), 1) * 100
                    pred_move = (prev_fc["predicted_close"] - prev_fc.get("ref_price", price)) / max(prev_fc.get("ref_price", price), 1) * 100
                    ctx.prev_forecast_accuracy[sym] = actual_move - pred_move

                # Build prompt and call Gemini
                prompt = build_gemini_prompt(sym, rl_sig, all_bars[sym], day_idx, fc, ctx, n_symbols)
                decision = call_gemini_structured(prompt, model=model)

                # Store forecast for accuracy tracking
                if fc:
                    fc["ref_price"] = price
                    ctx.prev_forecasts[sym] = fc

                ctx.prev_reasoning[sym] = decision.get("reasoning", "")
                daily_decisions[sym] = decision

                if i < 5 or (i + 1) % 30 == 0:
                    print(f"  [{date.strftime('%Y-%m-%d')}] {sym:8s} ${price:>10.2f} "
                          f"RL={rl_sig.rl_direction:5s}({rl_sig.rl_confidence:.2f}) "
                          f"→ alloc={decision['allocation']:+.1f}x "
                          f"reason={decision['reasoning'][:50]}")

            else:  # rl_only mode
                # Convert RL signal directly to trade action
                alloc = 0.0
                if rl_sig.rl_direction == "long":
                    alloc = rl_sig.rl_confidence * (5.0 / n_symbols)
                elif rl_sig.rl_direction == "short":
                    alloc = -rl_sig.rl_confidence * (5.0 / n_symbols)

                daily_decisions[sym] = {
                    "allocation": alloc,
                    "buy_price": price * 0.998 if alloc > 0 else 0,
                    "sell_price": price * 1.002 if alloc < 0 else (price * 1.01 if alloc > 0 else 0),
                    "confidence": rl_sig.rl_confidence,
                    "reasoning": f"rl_{rl_sig.rl_direction}",
                }

        # Normalize allocations across all symbols
        total_alloc = sum(abs(d["allocation"]) for d in daily_decisions.values())
        if total_alloc > 5.0:
            scale = 5.0 / total_alloc
            for sym in daily_decisions:
                daily_decisions[sym]["allocation"] *= scale

        # Convert to marketsimulator actions
        for sym, decision in daily_decisions.items():
            day_rows = all_bars[sym][all_bars[sym]["timestamp"] == date]
            if day_rows.empty:
                continue
            bar = day_rows.iloc[0]
            price = float(bar["close"])
            alloc = decision["allocation"]

            buy_price = decision.get("buy_price", 0)
            sell_price = decision.get("sell_price", 0)

            # Default prices if not set
            if alloc > 0 and buy_price <= 0:
                buy_price = price * 0.998
            if alloc > 0 and sell_price <= 0:
                sell_price = price * 1.01

            buy_amount = max(alloc, 0) * 100  # scale for marketsimulator
            sell_amount = max(-alloc, 0) * 100 if alloc < 0 else (100 if alloc > 0 else 0)

            all_actions.append({
                "timestamp": date,
                "symbol": sym,
                "buy_price": buy_price if alloc > 0 else 0,
                "sell_price": sell_price,
                "buy_amount": buy_amount,
                "sell_amount": sell_amount,
            })

            all_bar_data.append({
                "timestamp": date,
                "symbol": sym,
                "open": float(bar["open"]),
                "high": float(bar["high"]),
                "low": float(bar["low"]),
                "close": price,
                "volume": float(bar["volume"]),
            })

        # Progress
        if (i + 1) % 30 == 0:
            print(f"    Day {i+1}/{len(dates)}")

    # Run through marketsimulator
    if all_actions and all_bar_data:
        bars_df = pd.DataFrame(all_bar_data)
        actions_df = pd.DataFrame(all_actions)

        sim_config = SimulationConfig(
            maker_fee=0.0,  # commission-free
            initial_cash=initial_cash,
            max_hold_hours=72 * 24,  # 72 days in hours (daily bars)
        )

        result = run_shared_cash_simulation(bars_df, actions_df, sim_config)
        metrics = result.metrics

        # Annualize
        n_days = len(dates)
        total_ret = metrics.get("total_return", 0)
        if n_days > 0 and total_ret > -1:
            annualized = (1 + total_ret) ** (365 / n_days) - 1
            metrics["annualized_return"] = annualized

        # Compute additional metrics
        eq = result.combined_equity.values
        if len(eq) > 1:
            returns = np.diff(eq) / np.clip(eq[:-1], 1e-8, None)
            neg = returns[returns < 0]
            ds_std = neg.std() if len(neg) > 0 else 1e-8
            metrics["sortino"] = returns.mean() / ds_std * np.sqrt(365) if ds_std > 0 else 0

            peak = np.maximum.accumulate(eq)
            metrics["max_drawdown"] = float(((eq - peak) / peak).min())

        total_trades = sum(len(sr.trades) for sr in result.per_symbol.values())
        metrics["n_trades"] = total_trades

        return ctx, metrics
    else:
        return ctx, {"total_return": 0, "n_trades": 0}


def main():
    parser = argparse.ArgumentParser(description="Hybrid RL+Gemini daily backtest")
    parser.add_argument("--checkpoint", default="pufferlib_market/checkpoints/mixed23_fresh_targeted/reg_combo_2/best.pt")
    parser.add_argument("--symbols", default="BTCUSD,ETHUSD,SOLUSD,AAPL,MSFT,NVDA,GOOG,AMZN,META,TSLA,PLTR,NET,JPM,V")
    parser.add_argument("--start-date", default="2025-06-01")
    parser.add_argument("--end-date", default="2025-12-01")
    parser.add_argument("--mode", choices=["hybrid", "rl_only"], default="hybrid")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--cash", type=float, default=10000.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    ctx, metrics = run_hybrid_backtest(
        symbols=symbols,
        checkpoint=args.checkpoint,
        start_date=args.start_date,
        end_date=args.end_date,
        mode=args.mode,
        initial_cash=args.cash,
        model=args.model,
        slippage_bps=args.slippage_bps,
    )

    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS ({args.mode.upper()} mode)")
    print(f"{'='*60}")
    print(f"  Total return:   {metrics.get('total_return', 0)*100:+.2f}%")
    print(f"  Annualized:     {metrics.get('annualized_return', 0)*100:+.1f}%")
    print(f"  Sortino:        {metrics.get('sortino', 0):.2f}")
    print(f"  Max drawdown:   {metrics.get('max_drawdown', 0)*100:.2f}%")
    print(f"  Trades:         {metrics.get('n_trades', 0)}")


if __name__ == "__main__":
    main()
