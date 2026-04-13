#!/usr/bin/env python3
"""
End-to-end daily LLM trading backtest for Binance crypto.

The LLM (Gemini) gets full state context each day:
  - Current position (symbol, qty, entry price, P&L, hold days)
  - Previous positions history
  - Chronos2 h24 forecasts (predicted close/high/low)
  - Recent OHLCV bars (last 30 days)
  - Previous LLM reasoning/thinking
  - Market regime (trend, volatility)

The LLM decides:
  - allocation: float from -5.0 to +5.0 (negative = short, positive = long, 0 = flat)
    e.g., 3.0 means 3x leverage long, -2.0 means 2x leverage short
  - buy_price: limit entry price for the day
  - sell_price: limit exit price for the day
  - reasoning: brief explanation

Usage:
    python backtest_daily_llm.py --symbols BTCUSD,ETHUSD,SOLUSD \
        --start-date 2025-06-01 --end-date 2026-03-01 \
        --model gemini-3.1-flash-lite-preview --fee-tier fdusd
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


# Ensure project imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))


@dataclass
class DailyDecision:
    """LLM decision for one symbol on one day."""

    symbol: str
    allocation: float  # -5.0 to +5.0 (leverage × direction)
    buy_price: float  # limit entry price (0 = no entry)
    sell_price: float  # limit exit price (0 = no exit)
    confidence: float  # 0-1
    reasoning: str = ""


@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    entry_date: pd.Timestamp
    is_short: bool = False

    @property
    def direction_str(self):
        return "SHORT" if self.is_short else "LONG"


@dataclass
class TradeRecord:
    date: pd.Timestamp
    symbol: str
    side: str  # "buy", "sell", "short", "cover"
    price: float
    qty: float
    notional: float
    fee: float
    realized_pnl: float
    reason: str


@dataclass
class BacktestState:
    cash: float = 10000.0
    positions: dict = field(default_factory=dict)  # symbol -> Position
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)
    daily_decisions: list = field(default_factory=list)
    prev_reasoning: dict = field(default_factory=dict)  # symbol -> last reasoning
    prev_decisions: list = field(default_factory=list)  # last N decisions for context


FEE_TIERS = {
    "fdusd": 0.0,
    "usdt": 0.001,
    "conservative": 0.0015,
}

SHORT_BORROW_APR = 0.0625  # 6.25% annual


def load_daily_bars(symbol: str, data_root: str = "trainingdata/train") -> pd.DataFrame:
    root = Path(data_root)
    for subdir in ["crypto", "stocks", ""]:
        p = root / subdir / f"{symbol}.csv" if subdir else root / f"{symbol}.csv"
        if p.exists():
            df = pd.read_csv(p)
            df.columns = [c.lower() for c in df.columns]
            ts_col = "timestamp" if "timestamp" in df.columns else "date"
            df["timestamp"] = pd.to_datetime(df[ts_col], utc=True)
            df = df.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last")
            return df[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
    raise FileNotFoundError(f"No daily data for {symbol}")


def load_forecast_h24(symbol: str) -> pd.DataFrame | None:
    """Load h24 Chronos2 forecasts if available."""
    candidates = [
        Path("alpacanewccrosslearning/forecast_cache/crypto13_novol_20260208_lb4000/h24"),
        Path("binanceneural/forecast_cache/h24"),
        Path("alpacanewccrosslearning/forecast_cache/h24"),
    ]
    for cache_dir in candidates:
        path = cache_dir / f"{symbol}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp")
            df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
            return df.sort_index()
    return None


def get_forecast_for_date(fc_df: pd.DataFrame | None, date: pd.Timestamp) -> dict:
    """Get the most recent h24 forecast on or before this date."""
    if fc_df is None or fc_df.empty:
        return {}
    # Find forecasts issued before this date
    mask = fc_df.index <= date
    if not mask.any():
        return {}
    row = fc_df.loc[mask].iloc[-1]
    return {
        "predicted_close": float(row.get("predicted_close_p50", 0)),
        "predicted_close_p10": float(row.get("predicted_close_p10", 0)),
        "predicted_close_p90": float(row.get("predicted_close_p90", 0)),
        "predicted_high": float(row.get("predicted_high_p50", 0)),
        "predicted_low": float(row.get("predicted_low_p50", 0)),
    }


def compute_market_context(bars: pd.DataFrame, idx: int) -> dict:
    """Compute market regime context from recent bars."""
    if idx < 5:
        return {"regime": "unknown", "trend": 0, "volatility": 0}

    closes = bars["close"].iloc[max(0, idx - 20) : idx + 1].values
    returns = np.diff(closes) / closes[:-1]

    sma20 = np.mean(closes[-min(20, len(closes)) :])
    sma5 = np.mean(closes[-min(5, len(closes)) :])
    current = closes[-1]

    trend = (current - sma20) / sma20 if sma20 > 0 else 0
    vol = np.std(returns) if len(returns) > 1 else 0

    if trend > 0.05:
        regime = "strong_uptrend"
    elif trend > 0.02:
        regime = "mild_uptrend"
    elif trend < -0.05:
        regime = "strong_downtrend"
    elif trend < -0.02:
        regime = "mild_downtrend"
    else:
        regime = "sideways"

    return {
        "regime": regime,
        "trend_pct": round(trend * 100, 2),
        "volatility_daily": round(vol * 100, 2),
        "sma5": round(sma5, 2),
        "sma20": round(sma20, 2),
        "price": round(current, 2),
    }


def build_daily_prompt(
    symbol: str,
    bars_df: pd.DataFrame,
    day_idx: int,
    forecast: dict,
    position: Position | None,
    state: BacktestState,
    fee_rate: float,
) -> str:
    """Build a rich daily prompt for the LLM."""
    current_bar = bars_df.iloc[day_idx]
    date_str = current_bar["timestamp"].strftime("%Y-%m-%d")
    price = current_bar["close"]

    # Recent bars (last 14 days)
    start = max(0, day_idx - 13)
    recent = bars_df.iloc[start : day_idx + 1]
    bars_text = "Date        | Open      | High      | Low       | Close     | Volume\n"
    for _, row in recent.iterrows():
        bars_text += (
            f"{row['timestamp'].strftime('%Y-%m-%d')} | "
            f"{row['open']:>9.2f} | {row['high']:>9.2f} | {row['low']:>9.2f} | "
            f"{row['close']:>9.2f} | {row['volume']:>10.0f}\n"
        )

    # Market context
    ctx = compute_market_context(bars_df, day_idx)

    # Position info
    if position:
        hold_days = (current_bar["timestamp"] - position.entry_date).days
        if position.is_short:
            unreal_pnl = (position.entry_price - price) * position.qty
        else:
            unreal_pnl = (price - position.entry_price) * position.qty
        unreal_pct = unreal_pnl / (position.entry_price * position.qty) * 100
        pos_text = (
            f"CURRENT POSITION: {position.direction_str} {position.qty:.6f} {symbol}\n"
            f"  Entry price: ${position.entry_price:.2f}\n"
            f"  Current price: ${price:.2f}\n"
            f"  Unrealized P&L: ${unreal_pnl:.2f} ({unreal_pct:+.2f}%)\n"
            f"  Held for: {hold_days} days\n"
        )
    else:
        pos_text = "CURRENT POSITION: FLAT (no position)\n"

    # Portfolio state
    portfolio_value = state.cash
    for sym, pos in state.positions.items():
        sym_price = price if sym == symbol else 0  # simplified
        if pos.is_short:
            portfolio_value -= pos.qty * sym_price
        else:
            portfolio_value += pos.qty * sym_price
    port_text = f"CASH: ${state.cash:.2f} | PORTFOLIO VALUE: ~${portfolio_value:.2f}\n"

    # Forecast
    if forecast:
        fc_text = (
            f"CHRONOS2 24H FORECAST:\n"
            f"  Predicted close: ${forecast['predicted_close']:.2f}\n"
            f"  Range: ${forecast.get('predicted_low', 0):.2f} - ${forecast.get('predicted_high', 0):.2f}\n"
            f"  Confidence band: ${forecast.get('predicted_close_p10', 0):.2f} (p10) to ${forecast.get('predicted_close_p90', 0):.2f} (p90)\n"
            f"  Implied move: {((forecast['predicted_close'] - price) / price * 100):+.2f}%\n"
        )
    else:
        fc_text = "CHRONOS2 FORECAST: Not available for this date\n"

    # Previous reasoning
    prev_reason = state.prev_reasoning.get(symbol, "")
    prev_text = ""
    if prev_reason:
        prev_text = f"YOUR PREVIOUS REASONING: {prev_reason}\n"

    # Recent decisions history
    recent_decisions = [d for d in state.prev_decisions[-5:] if d.get("symbol") == symbol]
    if recent_decisions:
        prev_text += "RECENT DECISIONS:\n"
        for d in recent_decisions:
            prev_text += f"  {d['date']}: alloc={d['allocation']:+.1f}x, result={d.get('result', '?')}\n"

    # Fee info
    fee_text = f"TRADING FEES: {fee_rate * 100:.2f}% per side (maker)"
    if fee_rate == 0:
        fee_text = "TRADING FEES: 0% (FDUSD zero-fee pair)"
    fee_text += f"\nSHORT BORROW: {SHORT_BORROW_APR * 100:.2f}% annual ({SHORT_BORROW_APR / 365 * 100:.4f}%/day)"

    prompt = f"""You are an expert crypto trader managing a daily portfolio on Binance.

DATE: {date_str}
SYMBOL: {symbol}
CURRENT PRICE: ${price:.2f}

{port_text}
{pos_text}
MARKET REGIME: {ctx["regime"]} (trend: {ctx["trend_pct"]:+.1f}%, daily vol: {ctx["volatility_daily"]:.1f}%)
SMA-5: ${ctx["sma5"]:.2f} | SMA-20: ${ctx["sma20"]:.2f}

RECENT PRICE HISTORY (14 days):
{bars_text}
{fc_text}
{fee_text}

{prev_text}

DECIDE your position for TODAY. You control the leverage and direction.

Respond with JSON:
{{
  "allocation": <float from -5.0 to 5.0>,
  "buy_price": <limit buy price for today, or 0>,
  "sell_price": <limit sell price for today, or 0>,
  "confidence": <0.0 to 1.0>,
  "reasoning": "<brief 1-2 sentence explanation>"
}}

ALLOCATION GUIDE:
  -5.0 = maximum short (5x leveraged short)
  -1.0 = 1x short (sell borrowed shares)
   0.0 = flat (close all positions, hold cash)
  +1.0 = 1x long (buy with available cash)
  +3.0 = 3x leveraged long
  +5.0 = maximum long (5x leveraged long)

PRICE GUIDE:
  - buy_price: Set a limit price BELOW current price to buy on dips. 0 = no buy.
  - sell_price: Set a limit price ABOVE current price to take profit. 0 = no sell.
  - If you want to change direction (e.g., from long to short), set both prices.
  - Prices must be within today's realistic range.

OBJECTIVES: Maximize risk-adjusted returns (Sortino ratio). Keep max drawdown under 10%.
Think about position sizing, trend following, and when to stay flat."""

    return prompt


def call_llm_daily(prompt: str, model: str = "gemini-3.1-flash-lite-preview") -> DailyDecision:
    """Call LLM and parse daily decision."""
    from llm_hourly_trader.providers import _normalize_confidence, call_gemini

    # Use Gemini with thinking
    try:
        plan = call_gemini(prompt, model=model, thinking_level="HIGH")
        # Parse allocation from direction + confidence
        direction = plan.direction.lower().strip()
        allocation = 0.0
        if hasattr(plan, "allocation_pct") and plan.allocation_pct:
            allocation = float(plan.allocation_pct) / 100.0 * 5.0  # normalize to -5..5
        elif direction == "long":
            allocation = min(plan.confidence * 3.0, 5.0)
        elif direction == "short":
            allocation = max(-plan.confidence * 3.0, -5.0)

        return DailyDecision(
            symbol="",  # filled by caller
            allocation=allocation,
            buy_price=plan.buy_price,
            sell_price=plan.sell_price,
            confidence=_normalize_confidence(plan.confidence),
            reasoning=plan.reasoning,
        )
    except Exception as e:
        print(f"  LLM error: {e}")
        return DailyDecision(
            symbol="", allocation=0.0, buy_price=0, sell_price=0, confidence=0, reasoning=f"error: {e}"
        )


def call_llm_daily_structured(prompt: str, model: str = "gemini-3.1-flash-lite-preview") -> dict:
    """Call LLM with structured output for allocation decisions."""
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
                "allocation": genai.types.Schema(
                    type=genai.types.Type.STRING,
                    description="Float from -5.0 to 5.0. Negative=short, positive=long, 0=flat",
                ),
                "buy_price": genai.types.Schema(
                    type=genai.types.Type.STRING,
                    description="Limit buy price as number string, or 0",
                ),
                "sell_price": genai.types.Schema(
                    type=genai.types.Type.STRING,
                    description="Limit sell price as number string, or 0",
                ),
                "confidence": genai.types.Schema(
                    type=genai.types.Type.STRING,
                    description="Confidence 0.0-1.0",
                ),
                "reasoning": genai.types.Schema(
                    type=genai.types.Type.STRING,
                    description="Brief 1-2 sentence explanation",
                ),
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
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )
                result = json.loads(response.text)
                # Parse and validate
                parsed = {
                    "allocation": float(result.get("allocation", 0)),
                    "buy_price": float(result.get("buy_price", 0)),
                    "sell_price": float(result.get("sell_price", 0)),
                    "confidence": float(result.get("confidence", 0.5)),
                    "reasoning": str(result.get("reasoning", "")),
                }
                parsed["allocation"] = max(-5.0, min(5.0, parsed["allocation"]))

                # Log token usage
                if hasattr(response, "usage_metadata"):
                    um = response.usage_metadata
                    total = getattr(um, "total_token_count", 0)
                    prompt_tokens = getattr(um, "prompt_token_count", 0)
                    print(f"    tokens: {prompt_tokens} prompt + {total - prompt_tokens} output = {total} total")

                set_cached(model, prompt, parsed)
                return parsed

            except Exception as e:
                if "429" in str(e):
                    wait = min(30 * (attempt + 1), 90)
                    print(f"    Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise

    except Exception as e:
        print(f"  LLM structured call error: {e}")
        return {
            "allocation": 0.0,
            "buy_price": 0.0,
            "sell_price": 0.0,
            "confidence": 0.0,
            "reasoning": f"error: {e}",
        }


def execute_daily_decision(
    decision: dict,
    symbol: str,
    bar: pd.Series,
    state: BacktestState,
    fee_rate: float,
    slippage_bps: float = 5.0,
) -> None:
    """Execute a daily decision against the day's OHLCV bar."""
    allocation = decision["allocation"]
    buy_price = decision["buy_price"]
    sell_price = decision["sell_price"]
    low = bar["low"]
    high = bar["high"]
    open_price = bar["open"]
    close = bar["close"]
    date = bar["timestamp"]

    current_pos = state.positions.get(symbol)

    # Apply daily borrow fee for short positions
    if current_pos and current_pos.is_short:
        daily_borrow = current_pos.qty * close * SHORT_BORROW_APR / 365.0
        state.cash -= daily_borrow

    # Determine target direction and leverage
    target_direction = "long" if allocation > 0 else ("short" if allocation < 0 else "flat")
    target_leverage = abs(allocation)

    # Close position if direction changed or going flat
    if current_pos:
        should_close = False
        if (
            target_direction == "flat"
            or (target_direction == "long" and current_pos.is_short)
            or (target_direction == "short" and not current_pos.is_short)
        ):
            should_close = True

        if should_close:
            # Close at sell_price if it fills, otherwise at open
            if sell_price > 0 and low <= sell_price <= high:
                fill = sell_price * (1 - slippage_bps / 10000)
            elif buy_price > 0 and current_pos.is_short and low <= buy_price <= high:
                fill = buy_price * (1 + slippage_bps / 10000)
            else:
                fill = open_price  # close at market open

            if current_pos.is_short:
                cost = current_pos.qty * fill * (1 + fee_rate)
                state.cash -= cost
                pnl = (current_pos.entry_price - fill) * current_pos.qty
                side = "cover"
            else:
                proceeds = current_pos.qty * fill * (1 - fee_rate)
                state.cash += proceeds
                pnl = (fill - current_pos.entry_price) * current_pos.qty
                side = "sell"

            state.trades.append(
                TradeRecord(
                    date=date,
                    symbol=symbol,
                    side=side,
                    price=fill,
                    qty=current_pos.qty,
                    notional=current_pos.qty * fill,
                    fee=current_pos.qty * fill * fee_rate,
                    realized_pnl=pnl,
                    reason="direction_change" if target_direction != "flat" else "go_flat",
                )
            )
            del state.positions[symbol]
            current_pos = None

    # Open new position if allocation != 0 and not already positioned
    if target_direction != "flat" and symbol not in state.positions and target_leverage > 0:
        # Normalize leverage by number of active symbols to prevent over-allocation
        # E.g., 3 symbols each wanting 2x = 6x total, but we cap at max 5x overall
        n_symbols = max(len(state.positions) + 1, 1)  # include this new position
        max_per_symbol = min(target_leverage, 5.0 / n_symbols)  # cap total leverage at 5x
        effective_leverage = max_per_symbol

        # Determine entry price
        if target_direction == "long":
            if buy_price > 0 and low <= buy_price <= high:
                fill = buy_price * (1 + slippage_bps / 10000)
            elif buy_price > 0 and buy_price >= low:
                fill = min(buy_price, high) * (1 + slippage_bps / 10000)
            else:
                fill = open_price * (1 + slippage_bps / 10000)

            budget = max(state.cash, 0) * effective_leverage
            qty = budget / (fill * (1 + fee_rate))
            if qty > 0:
                cost = qty * fill * (1 + fee_rate)
                state.cash -= cost
                state.positions[symbol] = Position(
                    symbol=symbol,
                    qty=qty,
                    entry_price=fill,
                    entry_date=date,
                    is_short=False,
                )
                state.trades.append(
                    TradeRecord(
                        date=date,
                        symbol=symbol,
                        side="buy",
                        price=fill,
                        qty=qty,
                        notional=qty * fill,
                        fee=qty * fill * fee_rate,
                        realized_pnl=0,
                        reason=f"long_{target_leverage:.1f}x",
                    )
                )

        elif target_direction == "short":
            if sell_price > 0 and low <= sell_price <= high:
                fill = sell_price * (1 - slippage_bps / 10000)
            else:
                fill = open_price * (1 - slippage_bps / 10000)

            budget = max(state.cash, 0) * effective_leverage
            qty = budget / (fill * (1 + fee_rate))
            if qty > 0:
                proceeds = qty * fill * (1 - fee_rate)
                state.cash += proceeds
                state.positions[symbol] = Position(
                    symbol=symbol,
                    qty=qty,
                    entry_price=fill,
                    entry_date=date,
                    is_short=True,
                )
                state.trades.append(
                    TradeRecord(
                        date=date,
                        symbol=symbol,
                        side="short",
                        price=fill,
                        qty=qty,
                        notional=qty * fill,
                        fee=qty * fill * fee_rate,
                        realized_pnl=0,
                        reason=f"short_{target_leverage:.1f}x",
                    )
                )


def compute_equity(state: BacktestState, prices: dict) -> float:
    """Compute total portfolio equity."""
    equity = state.cash
    for sym, pos in state.positions.items():
        price = prices.get(sym, pos.entry_price)
        if pos.is_short:
            equity -= pos.qty * price  # cash already has short proceeds
        else:
            equity += pos.qty * price
    return equity


def run_backtest(
    symbols: list[str],
    start_date: str,
    end_date: str | None,
    model: str,
    fee_tier: str,
    initial_cash: float,
    slippage_bps: float,
    use_llm: bool = True,
) -> BacktestState:
    """Run the full daily LLM backtest."""
    fee_rate = FEE_TIERS[fee_tier]

    print(f"Loading data for {symbols}...")
    all_bars = {}
    all_forecasts = {}
    for sym in symbols:
        all_bars[sym] = load_daily_bars(sym)
        all_forecasts[sym] = load_forecast_h24(sym)
        print(f"  {sym}: {len(all_bars[sym])} bars, forecast={'yes' if all_forecasts[sym] is not None else 'no'}")

    # Find common date range
    start = pd.Timestamp(start_date, tz="UTC")
    if end_date:
        end = pd.Timestamp(end_date, tz="UTC")
    else:
        end = min(df["timestamp"].max() for df in all_bars.values())

    state = BacktestState(cash=initial_cash)

    # Get all trading dates
    dates = sorted(
        set().union(*(set(df[df["timestamp"].between(start, end)]["timestamp"].values) for df in all_bars.values()))
    )

    print(f"\nBacktest: {len(dates)} days from {dates[0]} to {dates[-1]}")
    print(f"  Model: {model}, Fee: {fee_tier} ({fee_rate * 100:.2f}%)")
    print(f"  Cash: ${initial_cash:,.0f}, Slippage: {slippage_bps}bps")
    print()

    for i, date in enumerate(dates):
        date = pd.Timestamp(date)
        if date.tz is None:
            date = date.tz_localize("UTC")
        prices = {}

        for sym in symbols:
            df = all_bars[sym]
            day_rows = df[df["timestamp"] == date]
            if day_rows.empty:
                continue

            bar = day_rows.iloc[0]
            day_idx = df.index[df["timestamp"] == date][0]
            prices[sym] = bar["close"]

            # Get forecast
            forecast = get_forecast_for_date(all_forecasts[sym], date)

            # Build prompt and get LLM decision
            if use_llm:
                prompt = build_daily_prompt(
                    symbol=sym,
                    bars_df=df,
                    day_idx=day_idx,
                    forecast=forecast,
                    position=state.positions.get(sym),
                    state=state,
                    fee_rate=fee_rate,
                )

                print(f"  [{date.strftime('%Y-%m-%d')}] {sym} @ ${bar['close']:.2f}", end="")
                decision = call_llm_daily_structured(prompt, model=model)
                print(
                    f" → alloc={decision['allocation']:+.1f}x, "
                    f"conf={decision['confidence']:.2f}, "
                    f"reason={decision['reasoning'][:60]}"
                )

                # Store for context
                state.prev_reasoning[sym] = decision["reasoning"]
                state.prev_decisions.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "symbol": sym,
                        "allocation": decision["allocation"],
                        "confidence": decision["confidence"],
                    }
                )
            else:
                # Simple buy-and-hold benchmark
                decision = {
                    "allocation": 1.0,
                    "buy_price": bar["open"],
                    "sell_price": 0,
                    "confidence": 1.0,
                    "reasoning": "buy_and_hold",
                }

            # Execute
            execute_daily_decision(decision, sym, bar, state, fee_rate, slippage_bps)

        # Record equity
        equity = compute_equity(state, prices)
        state.equity_curve.append({"date": date, "equity": equity})

        # Progress every 30 days
        if (i + 1) % 30 == 0:
            ret = (equity - initial_cash) / initial_cash * 100
            print(f"    Day {i + 1}/{len(dates)}: equity=${equity:,.2f} ({ret:+.1f}%)")

    return state


def print_results(state: BacktestState, initial_cash: float):
    """Print backtest results."""
    if not state.equity_curve:
        print("No equity data!")
        return

    equities = [float(e["equity"]) for e in state.equity_curve]
    final = equities[-1]
    total_ret = (final - initial_cash) / initial_cash

    returns = np.diff(equities) / np.clip(equities[:-1], 1e-8, None)
    neg_returns = returns[returns < 0]
    downside_std = neg_returns.std() if len(neg_returns) > 0 else 1e-8
    sortino = (returns.mean() / downside_std * np.sqrt(365)) if downside_std > 0 else 0

    peak = initial_cash
    max_dd = 0
    for eq in equities:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd

    n_days = len(equities)
    annualized = (1 + total_ret) ** (365 / max(n_days, 1)) - 1

    n_trades = len(state.trades)
    n_winners = sum(1 for t in state.trades if t.realized_pnl > 0)
    total_pnl = sum(t.realized_pnl for t in state.trades)
    total_fees = sum(t.fee for t in state.trades)

    print(f"\n{'=' * 60}")
    print(f"BACKTEST RESULTS ({n_days} days)")
    print(f"{'=' * 60}")
    print(f"  Final equity:     ${final:,.2f}")
    print(f"  Total return:     {total_ret * 100:+.2f}%")
    print(f"  Annualized:       {annualized * 100:+.1f}%")
    print(f"  Sortino:          {sortino:.2f}")
    print(f"  Max drawdown:     {max_dd * 100:.2f}%")
    print(f"  Trades:           {n_trades}")
    print(f"  Win rate:         {n_winners / max(n_trades, 1) * 100:.1f}%")
    print(f"  Total P&L:        ${total_pnl:,.2f}")
    print(f"  Total fees:       ${total_fees:,.2f}")
    print(f"  Fee drag:         {total_fees / initial_cash * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Daily LLM trading backtest")
    parser.add_argument("--symbols", default="BTCUSD,ETHUSD,SOLUSD")
    parser.add_argument("--start-date", default="2025-06-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    parser.add_argument("--fee-tier", choices=list(FEE_TIERS.keys()), default="fdusd")
    parser.add_argument("--initial-cash", type=float, default=10000)
    parser.add_argument("--slippage-bps", type=float, default=5)
    parser.add_argument("--no-llm", action="store_true", help="Buy-and-hold benchmark")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    state = run_backtest(
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        model=args.model,
        fee_tier=args.fee_tier,
        initial_cash=args.initial_cash,
        slippage_bps=args.slippage_bps,
        use_llm=not args.no_llm,
    )

    print_results(state, args.initial_cash)


if __name__ == "__main__":
    main()
