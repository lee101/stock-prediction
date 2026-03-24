from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from llm_hourly_trader.gemini_wrapper import TradePlan
from src.allocation_refiner import (
    AllocationRefinement,
    leverage_limits_for_asset,
    refine_allocation,
)
from unified_orchestrator.rl_gemini_bridge import RLSignal, build_portfolio_observation

DEFAULT_MIXED23_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA", "PLTR", "NET",
    "JPM", "V", "SPY", "QQQ", "NFLX", "AMD",
    "BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD", "AVAXUSD", "DOGEUSD", "LINKUSD", "AAVEUSD",
]

CRYPTO_SYMBOLS = {"BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD", "AVAXUSD", "DOGEUSD", "LINKUSD", "AAVEUSD"}

FORECAST_CACHE_CANDIDATES = [
    Path("binanceneural/forecast_cache"),
    Path("alpacanewccrosslearning/forecast_cache/crypto13_novol_20260208_lb4000"),
]

DEFAULT_STATE_FILE = Path("strategy_state/mixed_daily_hybrid_state.json")


@dataclass(frozen=True)
class DailyPosition:
    symbol: str
    direction: str
    qty: float
    entry_price: float
    entry_timestamp: str
    current_price: float = 0.0
    unrealized_pnl_pct: float = 0.0
    hold_days: int = 0


@dataclass(frozen=True)
class CandidatePlan:
    symbol: str
    asset_class: str
    current_price: float
    rl_signal: RLSignal
    plan: TradePlan
    score: float
    expected_edge_pct: float
    forecast_1h: Optional[dict[str, float]] = None
    forecast_24h: Optional[dict[str, float]] = None
    current_allocation: float = 0.0
    target_allocation: float = 0.0
    overnight_allocation: float = 0.0
    allocation_reason: str = ""


def asset_class_for_symbol(symbol: str) -> str:
    return "crypto" if str(symbol).upper() in CRYPTO_SYMBOLS else "stock"


def load_daily_bars(symbol: str) -> Optional[pd.DataFrame]:
    symbol = str(symbol).upper()
    for root in (Path("trainingdata/train"), Path("trainingdata")):
        for subdir in ("", "crypto", "stocks"):
            path = root / subdir / f"{symbol}.csv" if subdir else root / f"{symbol}.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            frame.columns = [str(col).lower() for col in frame.columns]
            ts_col = "timestamp" if "timestamp" in frame.columns else "date"
            if ts_col not in frame.columns:
                continue
            frame["timestamp"] = pd.to_datetime(frame[ts_col], utc=True)
            needed = ["timestamp", "open", "high", "low", "close", "volume"]
            missing = [col for col in needed if col not in frame.columns]
            if missing:
                continue
            for col in ("open", "high", "low", "close", "volume"):
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
            frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close"])
            frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
            return frame.loc[:, needed].reset_index(drop=True)
    return None


def load_forecast_frame(symbol: str, horizon: str) -> pd.DataFrame:
    symbol = str(symbol).upper()
    for root in FORECAST_CACHE_CANDIDATES:
        path = root / horizon / f"{symbol}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            frame = frame.sort_values("timestamp").reset_index(drop=True)
        else:
            frame.index = pd.to_datetime(frame.index, utc=True)
            frame = frame.sort_index().reset_index().rename(columns={"index": "timestamp"})
        return frame
    return pd.DataFrame()


def get_forecast_at(frame: pd.DataFrame, ts: pd.Timestamp) -> Optional[dict[str, float]]:
    if frame is None or frame.empty:
        return None
    timestamp = pd.Timestamp(ts)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    match = frame[frame["timestamp"] <= timestamp]
    if match.empty:
        return None
    row = match.iloc[-1]
    out = {}
    for col in (
        "predicted_close_p50",
        "predicted_close_p10",
        "predicted_close_p90",
        "predicted_high_p50",
        "predicted_low_p50",
    ):
        value = row.get(col)
        if value is None or pd.isna(value):
            continue
        out[col] = float(value)
    return out or None


def load_strategy_state(path: Path | str = DEFAULT_STATE_FILE) -> dict[str, Any]:
    state_path = Path(path)
    if not state_path.exists():
        return {"updated_at": None, "plans": {}, "trade_history": [], "forecast_refs": {}, "entry_times": {}}
    try:
        payload = json.loads(state_path.read_text())
    except Exception:
        return {"updated_at": None, "plans": {}, "trade_history": [], "forecast_refs": {}, "entry_times": {}}
    if not isinstance(payload, dict):
        return {"updated_at": None, "plans": {}, "trade_history": [], "forecast_refs": {}, "entry_times": {}}
    payload.setdefault("plans", {})
    payload.setdefault("trade_history", [])
    payload.setdefault("forecast_refs", {})
    payload.setdefault("entry_times", {})
    payload.setdefault("updated_at", None)
    return payload


def save_strategy_state(state: dict[str, Any], path: Path | str = DEFAULT_STATE_FILE) -> None:
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True))


def build_feature_matrix(
    daily_frames: dict[str, pd.DataFrame],
    symbols: list[str],
    analysis_ts: pd.Timestamp,
    feature_fn,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, float]] = []
    current_prices: dict[str, float] = {}
    for symbol in symbols:
        frame = daily_frames.get(symbol)
        if frame is None or frame.empty:
            rows.append({"symbol": symbol, **{f"f{i}": 0.0 for i in range(16)}})
            continue
        hist = frame[frame["timestamp"] < analysis_ts]
        if hist.empty:
            rows.append({"symbol": symbol, **{f"f{i}": 0.0 for i in range(16)}})
            continue
        current_prices[symbol] = float(hist["close"].iloc[-1])
        if len(hist) < 20:
            rows.append({"symbol": symbol, **{f"f{i}": 0.0 for i in range(16)}})
            continue
        feature_row = feature_fn(hist)
        rows.append({"symbol": symbol, **{f"f{i}": float(feature_row[i]) for i in range(len(feature_row))}})
    return pd.DataFrame(rows), current_prices


def build_rl_observation(
    feature_frame: pd.DataFrame,
    symbols: list[str],
    *,
    cash: float,
    equity: float,
    current_position: Optional[DailyPosition],
    step_idx: int,
    total_steps: int,
) -> Any:
    feature_cols = [col for col in feature_frame.columns if col.startswith("f")]
    ordered = feature_frame.set_index("symbol").reindex(symbols).fillna(0.0)
    features = ordered.loc[:, feature_cols].to_numpy(dtype="float32")

    cash_ratio = float(cash) / max(float(equity), 1e-6)
    position_value_ratio = 0.0
    unrealized_pnl_ratio = 0.0
    hold_fraction = 0.0
    position_symbol_idx = None
    position_direction = "long"

    if current_position is not None and current_position.symbol in symbols:
        ref_price = float(current_position.current_price or current_position.entry_price or 0.0)
        gross_value = float(current_position.qty) * ref_price
        if current_position.direction == "short":
            position_value_ratio = -gross_value / max(float(equity), 1e-6)
            position_direction = "short"
        else:
            position_value_ratio = gross_value / max(float(equity), 1e-6)
        unrealized_pnl_ratio = (
            position_value_ratio * float(current_position.unrealized_pnl_pct) / 100.0
        )
        hold_fraction = float(current_position.hold_days) / max(float(total_steps), 1.0)
        position_symbol_idx = symbols.index(current_position.symbol)

    return build_portfolio_observation(
        features,
        cash_ratio=cash_ratio,
        position_value_ratio=position_value_ratio,
        unrealized_pnl_ratio=unrealized_pnl_ratio,
        hold_fraction=hold_fraction,
        step_fraction=float(step_idx) / max(float(total_steps), 1.0),
        position_symbol_idx=position_symbol_idx,
        position_direction=position_direction,
    )


def _history_block(history_rows: list[dict[str, Any]], *, limit: int = 30) -> str:
    lines = []
    for row in history_rows[-limit:]:
        ts = pd.Timestamp(row["timestamp"]).strftime("%Y-%m-%d")
        lines.append(
            f"  {ts}  O:{float(row['open']):>10.2f}  H:{float(row['high']):>10.2f}  "
            f"L:{float(row['low']):>10.2f}  C:{float(row['close']):>10.2f}  "
            f"V:{float(row.get('volume', 0.0)):>12.0f}"
        )
    return "\n".join(lines) if lines else "  (no daily history available)"


def _forecast_block(label: str, forecast: Optional[dict[str, float]], current_price: float) -> str:
    if not forecast:
        return f"  {label}: unavailable"
    p50 = float(forecast.get("predicted_close_p50", 0.0) or 0.0)
    p10 = float(forecast.get("predicted_close_p10", 0.0) or 0.0)
    p90 = float(forecast.get("predicted_close_p90", 0.0) or 0.0)
    high = float(forecast.get("predicted_high_p50", 0.0) or 0.0)
    low = float(forecast.get("predicted_low_p50", 0.0) or 0.0)
    delta_pct = (p50 / current_price - 1.0) * 100.0 if current_price > 0 and p50 > 0 else 0.0
    return (
        f"  {label}: close={p50:.2f} ({delta_pct:+.2f}%) "
        f"range=[{p10:.2f}, {p90:.2f}] high={high:.2f} low={low:.2f}"
    )


def _recent_trades_block(trades: list[dict[str, Any]], symbol: str, *, limit: int = 5) -> str:
    relevant = [trade for trade in trades if str(trade.get("symbol", "")).upper() == str(symbol).upper()]
    if not relevant:
        return "  No previous trades recorded."
    lines = []
    for trade in relevant[-limit:]:
        lines.append(
            "  {timestamp} {side} @{price:.2f} pnl={pnl_pct:+.2f}% reason={reason}".format(
                timestamp=str(trade.get("timestamp", "")),
                side=str(trade.get("side", "")),
                price=float(trade.get("price", 0.0) or 0.0),
                pnl_pct=float(trade.get("pnl_pct", 0.0) or 0.0),
                reason=str(trade.get("reason", ""))[:80],
            )
        )
    return "\n".join(lines)


def _portfolio_positions_block(
    portfolio_positions: list[DailyPosition],
    *,
    symbol: str,
    limit: int = 8,
) -> str:
    if not portfolio_positions:
        return "  No open positions in the tracked daily portfolio."

    lines = []
    for position in portfolio_positions[:limit]:
        marker = "*" if str(position.symbol).upper() == str(symbol).upper() else "-"
        lines.append(
            f"  {marker} {position.direction.upper()} {position.symbol} "
            f"qty={position.qty:.6f} entry={position.entry_price:.2f} "
            f"current={position.current_price:.2f} pnl={position.unrealized_pnl_pct:+.2f}% "
            f"held={position.hold_days}d entry_time={position.entry_timestamp}"
        )
    return "\n".join(lines)


def previous_forecast_error_pct(
    state: dict[str, Any],
    symbol: str,
    current_price: float,
) -> Optional[float]:
    forecast_refs = state.get("forecast_refs", {})
    ref = forecast_refs.get(str(symbol).upper())
    if not isinstance(ref, dict):
        return None
    ref_price = float(ref.get("reference_price", 0.0) or 0.0)
    pred = float(ref.get("predicted_close_p50", 0.0) or 0.0)
    if ref_price <= 0.0 or pred <= 0.0 or current_price <= 0.0:
        return None
    actual_move_pct = (current_price / ref_price - 1.0) * 100.0
    predicted_move_pct = (pred / ref_price - 1.0) * 100.0
    return actual_move_pct - predicted_move_pct


def current_allocation_from_position(
    position: Optional[DailyPosition],
    *,
    equity: float,
) -> float:
    if position is None or equity <= 0.0:
        return 0.0
    reference_price = float(position.current_price or position.entry_price or 0.0)
    if reference_price <= 0.0:
        return 0.0
    notional = abs(float(position.qty)) * reference_price
    signed = notional / max(float(equity), 1e-6)
    if str(position.direction).lower() == "short":
        signed *= -1.0
    return float(signed)


def _format_signed_leverage(value: float) -> str:
    return f"{float(value):+.2f}x"


def build_daily_hybrid_prompt(
    *,
    symbol: str,
    asof: pd.Timestamp,
    asset_class: str,
    current_price: float,
    rl_signal: RLSignal,
    other_rl_signals: list[RLSignal],
    history_rows: list[dict[str, Any]],
    forecast_1h: Optional[dict[str, float]],
    forecast_24h: Optional[dict[str, float]],
    current_position: Optional[DailyPosition],
    portfolio_positions: Optional[list[DailyPosition]],
    previous_plan: Optional[dict[str, Any]],
    recent_trades: list[dict[str, Any]],
    previous_forecast_error: Optional[float],
    cash: float,
    equity: float,
    tracked_symbols: int,
    max_positions: int,
    allowed_directions: list[str],
    current_allocation: Optional[float] = None,
    refined_allocation: Optional[float] = None,
    overnight_allocation: Optional[float] = None,
    allocation_reason: Optional[str] = None,
) -> str:
    asof_ts = pd.Timestamp(asof)
    if asof_ts.tzinfo is None:
        asof_ts = asof_ts.tz_localize("UTC")
    allowed = ", ".join(direction.upper() for direction in allowed_directions)

    if current_position is None:
        current_position_block = "  CURRENT POSITION: FLAT"
    else:
        current_position_block = (
            f"  CURRENT POSITION: {current_position.direction.upper()} {current_position.symbol} "
            f"qty={current_position.qty:.6f} entry={current_position.entry_price:.2f} "
            f"current={current_position.current_price:.2f} "
            f"unrealized={current_position.unrealized_pnl_pct:+.2f}% "
            f"held={current_position.hold_days}d "
            f"entry_time={current_position.entry_timestamp}"
        )

    if previous_plan:
        previous_plan_block = (
            f"  Previous plan time={previous_plan.get('timestamp', '')} "
            f"direction={previous_plan.get('direction', 'hold')} "
            f"buy={float(previous_plan.get('buy_price', 0.0) or 0.0):.2f} "
            f"sell={float(previous_plan.get('sell_price', 0.0) or 0.0):.2f} "
            f"confidence={float(previous_plan.get('confidence', 0.0) or 0.0):.2f} "
            f"reason={str(previous_plan.get('reasoning', ''))[:120]}"
        )
    else:
        previous_plan_block = "  Previous plan: none"

    forecast_error_block = (
        f"  Previous Chronos2 forecast error vs realized move: {previous_forecast_error:+.2f}%"
        if previous_forecast_error is not None
        else "  Previous Chronos2 forecast error vs realized move: unavailable"
    )

    other_signals_block = ["  No other ranked RL signals."]
    if other_rl_signals:
        other_signals_block = [
            f"  {signal.symbol_name}: {signal.direction.upper()} "
            f"conf={signal.confidence:.2f} gap={signal.logit_gap:+.2f} alloc={signal.allocation_pct:.2f}"
            for signal in other_rl_signals[:5]
        ]

    leverage_limits = leverage_limits_for_asset(asset_class)
    if current_allocation is None:
        current_alloc_block = "  Current signed allocation: unavailable"
    else:
        current_alloc_block = f"  Current signed allocation: {_format_signed_leverage(current_allocation)}"

    if refined_allocation is None:
        refined_alloc_block = "  Refined target allocation: unavailable"
    else:
        refined_alloc_block = (
            f"  Refined target allocation: {_format_signed_leverage(refined_allocation)}"
        )

    if overnight_allocation is None:
        overnight_alloc_block = "  Overnight capped allocation: unavailable"
    else:
        overnight_alloc_block = (
            f"  Overnight capped allocation: {_format_signed_leverage(overnight_allocation)}"
        )

    alloc_reason_block = (
        f"  Allocation rationale: {allocation_reason}"
        if allocation_reason
        else "  Allocation rationale: use RL allocation as the prior, then adjust with forecasts and current holdings."
    )

    return f"""You are re-planning a daily multi-asset trading decision with structured output.

ANALYSIS TIME: {asof_ts.isoformat()}
SYMBOL: {symbol} ({asset_class})
CURRENT REFERENCE PRICE: {current_price:.2f}
ALLOWED DIRECTIONS: {allowed}

## RL Simulator Recommendation
The RL simulator already ranked this symbol and wants a fresh judgment.
Primary RL signal: {rl_signal.direction.upper()} conf={rl_signal.confidence:.2f} gap={rl_signal.logit_gap:+.2f} alloc={rl_signal.allocation_pct:.2f}
Other ranked RL signals:
{chr(10).join(other_signals_block)}

## Portfolio State
{current_position_block}
  Cash={cash:.2f} Equity={equity:.2f} Tracked symbols={tracked_symbols}
  Open positions={0 if not portfolio_positions else len(portfolio_positions)} / max {max_positions}
  Current tracked portfolio:
{_portfolio_positions_block(portfolio_positions or [], symbol=symbol)}

## Allocation And Leverage Envelope
{current_alloc_block}
{refined_alloc_block}
{overnight_alloc_block}
{alloc_reason_block}
  Stocks: long up to {leverage_limits.long_max_leverage:.1f}x intraday, short up to {leverage_limits.short_max_leverage:.1f}x, overnight gross cap {leverage_limits.overnight_max_gross:.1f}x.
  Crypto: long-only up to 1.0x and should not be shorted.

## Previous Plan And Outcome Context
{previous_plan_block}
{forecast_error_block}

## Previous Trades For This Symbol
{_recent_trades_block(recent_trades, symbol)}

## Current Chronos2 Forecasts
{_forecast_block("1h", forecast_1h, current_price)}
{_forecast_block("24h", forecast_24h, current_price)}

## Recent Daily Bars
{_history_block(history_rows)}

## Decision Rules
- The RL signal is a strong prior, but you may downgrade it if the forecasts or recent trade context conflict.
- Respect the leverage envelope and use the refined allocation as sizing guidance, not a guarantee.
- Use previous positions, previous forecasts, previous plan timing, and current Chronos2 forecasts explicitly in your judgment.
- If already in a position, focus on whether to keep the position and where to set a realistic exit.
- If FLAT and there is no clean edge after fees, return HOLD.
- Use LIMIT prices only.
- LONG: buy_price > 0 and buy_price <= current reference price, sell_price > buy_price.
- SHORT: only valid for stocks; sell_price >= current reference price, buy_price < sell_price.
- HOLD with an existing position should still set a realistic exit price in sell_price for longs or buy_price for shorts.
- Confidence must be a decimal between 0.0 and 1.0.

Return JSON with direction, buy_price, sell_price, confidence, reasoning."""


def build_fallback_trade_plan(
    *,
    rl_signal: RLSignal,
    current_price: float,
    asset_class: str,
    history_rows: list[dict[str, Any]],
    forecast_1h: Optional[dict[str, float]],
    forecast_24h: Optional[dict[str, float]],
    current_position: Optional[DailyPosition],
) -> TradePlan:
    ranges = [
        max(0.0, float(row["high"]) - float(row["low"]))
        for row in history_rows[-10:]
        if "high" in row and "low" in row
    ]
    atr_pct = 0.01
    if ranges and current_price > 0:
        atr_pct = max(0.002, min(0.08, sum(ranges) / len(ranges) / current_price))
    forecast_close = 0.0
    forecast_high = 0.0
    forecast_low = 0.0
    for forecast in (forecast_24h, forecast_1h):
        if forecast:
            forecast_close = max(forecast_close, float(forecast.get("predicted_close_p50", 0.0) or 0.0))
            forecast_high = max(forecast_high, float(forecast.get("predicted_high_p50", 0.0) or 0.0))
            low_val = float(forecast.get("predicted_low_p50", 0.0) or 0.0)
            forecast_low = low_val if forecast_low <= 0.0 else min(forecast_low, low_val)
    upside_pct = max(0.0, (max(forecast_close, forecast_high) / current_price - 1.0)) if current_price > 0 else 0.0
    downside_pct = max(0.0, (1.0 - forecast_low / current_price)) if current_price > 0 and forecast_low > 0 else 0.0
    fee_buffer = 0.0015 if asset_class == "crypto" else 0.0010

    if current_position is not None:
        if current_position.direction == "short":
            cover_pct = max(fee_buffer + 0.0010, 0.50 * atr_pct, 0.40 * downside_pct)
            buy_price = current_price * (1.0 - min(0.06, cover_pct))
            return TradePlan(
                direction="hold",
                buy_price=float(buy_price),
                sell_price=0.0,
                confidence=max(0.2, float(rl_signal.confidence)),
                reasoning="fallback_short_exit",
            )
        exit_pct = max(fee_buffer + 0.0010, 0.60 * atr_pct, 0.40 * upside_pct)
        sell_price = max(current_price * (1.0 + min(0.08, exit_pct)), current_position.entry_price * (1.0 + fee_buffer))
        return TradePlan(
            direction="hold",
            buy_price=0.0,
            sell_price=float(sell_price),
            confidence=max(0.2, float(rl_signal.confidence)),
            reasoning="fallback_long_exit",
        )

    if rl_signal.direction == "long":
        entry_pullback = max(0.0010, min(0.03, 0.35 * atr_pct - 0.15 * upside_pct))
        target_pct = max(fee_buffer + 0.0015, 0.80 * atr_pct, 0.50 * upside_pct)
        buy_price = current_price * (1.0 - entry_pullback)
        sell_price = max(current_price * (1.0 + min(0.10, target_pct)), buy_price * (1.0 + fee_buffer + 0.0010))
        return TradePlan(
            direction="long",
            buy_price=float(buy_price),
            sell_price=float(sell_price),
            confidence=float(max(0.2, rl_signal.confidence)),
            reasoning="fallback_long_entry",
        )
    if rl_signal.direction == "short" and asset_class == "stock":
        entry_markup = max(0.0010, min(0.03, 0.35 * atr_pct - 0.15 * downside_pct))
        target_pct = max(fee_buffer + 0.0015, 0.80 * atr_pct, 0.50 * downside_pct)
        sell_price = current_price * (1.0 + entry_markup)
        buy_price = min(current_price * (1.0 - min(0.10, target_pct)), sell_price * (1.0 - fee_buffer - 0.0010))
        return TradePlan(
            direction="short",
            buy_price=float(buy_price),
            sell_price=float(sell_price),
            confidence=float(max(0.2, rl_signal.confidence)),
            reasoning="fallback_short_entry",
        )
    return TradePlan(direction="hold", buy_price=0.0, sell_price=0.0, confidence=0.1, reasoning="fallback_hold")


def expected_edge_pct(
    plan: TradePlan,
    *,
    current_price: float,
    current_position: Optional[DailyPosition],
) -> float:
    if plan.direction == "long" and plan.buy_price > 0 and plan.sell_price > plan.buy_price:
        return (float(plan.sell_price) - float(plan.buy_price)) / float(plan.buy_price)
    if plan.direction == "short" and plan.sell_price > 0 and plan.buy_price > 0 and plan.sell_price > plan.buy_price:
        return (float(plan.sell_price) - float(plan.buy_price)) / float(plan.sell_price)
    if current_position is not None and current_position.direction == "long" and plan.sell_price > current_price > 0:
        return (float(plan.sell_price) - float(current_price)) / float(current_price)
    if current_position is not None and current_position.direction == "short" and current_price > 0 and plan.buy_price > 0:
        return (float(current_price) - float(plan.buy_price)) / float(current_price)
    return 0.0


def score_candidate_plan(
    plan: TradePlan,
    *,
    rl_signal: RLSignal,
    asset_class: str,
    current_price: float,
    current_position: Optional[DailyPosition],
    target_allocation: Optional[float] = None,
) -> float:
    edge_pct = expected_edge_pct(plan, current_price=current_price, current_position=current_position)
    fee_buffer = 0.0015 if asset_class == "crypto" else 0.0010
    score = max(0.0, edge_pct - fee_buffer) * max(float(plan.confidence), 0.0)
    if target_allocation is not None:
        notional_scale = max(0.35, min(2.0, math.sqrt(abs(float(target_allocation)))))
        score *= notional_scale
    if plan.direction != rl_signal.direction and plan.direction != "hold":
        score *= 0.65
    if asset_class == "crypto" and plan.direction == "short":
        return -1.0
    if current_position is not None and plan.direction == "hold":
        score += 1e-6
    return score


def build_candidate_plan(
    *,
    symbol: str,
    asset_class: str,
    current_price: float,
    rl_signal: RLSignal,
    plan: TradePlan,
    current_position: Optional[DailyPosition],
    equity: float,
    forecast_1h: Optional[dict[str, float]] = None,
    forecast_24h: Optional[dict[str, float]] = None,
    previous_forecast_error: Optional[float] = None,
) -> CandidatePlan:
    current_allocation = current_allocation_from_position(current_position, equity=equity)
    refinement: AllocationRefinement = refine_allocation(
        asset_class=asset_class,
        rl_direction=rl_signal.direction,
        rl_allocation_pct=rl_signal.allocation_pct,
        rl_confidence=rl_signal.confidence,
        rl_logit_gap=rl_signal.logit_gap,
        current_allocation=current_allocation,
        current_price=current_price,
        forecast_1h=forecast_1h,
        forecast_24h=forecast_24h,
        previous_forecast_error=previous_forecast_error,
    )
    edge_pct = expected_edge_pct(plan, current_price=current_price, current_position=current_position)
    score = score_candidate_plan(
        plan,
        rl_signal=rl_signal,
        asset_class=asset_class,
        current_price=current_price,
        current_position=current_position,
        target_allocation=refinement.target_allocation,
    )
    return CandidatePlan(
        symbol=symbol,
        asset_class=asset_class,
        current_price=float(current_price),
        rl_signal=rl_signal,
        plan=plan,
        score=float(score),
        expected_edge_pct=float(edge_pct),
        forecast_1h=forecast_1h,
        forecast_24h=forecast_24h,
        current_allocation=float(current_allocation),
        target_allocation=float(refinement.target_allocation),
        overnight_allocation=float(refinement.overnight_allocation),
        allocation_reason=str(refinement.reason),
    )


def select_best_candidate(candidates: list[CandidatePlan]) -> Optional[CandidatePlan]:
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda item: (
            item.score,
            abs(float(item.target_allocation)),
            item.plan.confidence,
            item.rl_signal.confidence,
            item.expected_edge_pct,
        ),
        reverse=True,
    )
    best = ranked[0]
    if best.score <= 0.0 and best.plan.direction == "hold" and best.plan.sell_price <= 0.0 and best.plan.buy_price <= 0.0:
        return None
    return best


def snapshot_plan(
    *,
    timestamp: pd.Timestamp,
    symbol: str,
    rl_signal: RLSignal,
    plan: TradePlan,
    current_price: float,
    forecast_1h: Optional[dict[str, float]],
    forecast_24h: Optional[dict[str, float]],
    score: float,
    target_allocation: Optional[float] = None,
    overnight_allocation: Optional[float] = None,
) -> dict[str, Any]:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return {
        "timestamp": ts.isoformat(),
        "symbol": str(symbol).upper(),
        "direction": plan.direction,
        "buy_price": float(plan.buy_price),
        "sell_price": float(plan.sell_price),
        "confidence": float(plan.confidence),
        "reasoning": str(plan.reasoning),
        "current_price": float(current_price),
        "rl_direction": rl_signal.direction,
        "rl_confidence": float(rl_signal.confidence),
        "rl_logit_gap": float(rl_signal.logit_gap),
        "score": float(score),
        "forecast_1h": forecast_1h,
        "forecast_24h": forecast_24h,
        "target_allocation": None if target_allocation is None else float(target_allocation),
        "overnight_allocation": None if overnight_allocation is None else float(overnight_allocation),
    }


def snapshot_position(position: Optional[DailyPosition]) -> Optional[dict[str, Any]]:
    if position is None:
        return None
    return asdict(position)
