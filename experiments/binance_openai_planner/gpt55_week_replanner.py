#!/usr/bin/env python3
"""Seven-day GPT-5.5/xhigh portfolio replanning simulation.

The experiment asks a model for one complete portfolio plan per day, then walks
the next 24 hourly Binance bars with limit entry, take-profit, stop, allocation,
fees, slippage, and daily forced flattening. It is intentionally small: 7 model
calls for 7 days.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.binance_transition_graph.graph_policy import (
    DEFAULT_SYMBOLS,
    HOURS_PER_MONTH,
    HOURS_PER_YEAR,
    close_matrix,
    load_hourly_panel,
)


DEFAULT_MODEL = "gpt-5.5"
DEFAULT_REASONING = "xhigh"
DEFAULT_INITIAL_CASH = 10_000.0


@dataclass(frozen=True)
class PlanLeg:
    symbol: str
    side: str
    allocation_pct: float
    buy_price: float
    sell_price: float
    stop_price: float
    confidence: float
    rationale: str


@dataclass(frozen=True)
class DailyPlan:
    day_index: int
    plan_timestamp: str
    cash_pct: float
    gross_exposure_pct: float
    risk_notes: str
    legs: list[PlanLeg]


@dataclass(frozen=True)
class TradeEvent:
    timestamp: str
    symbol: str
    side: str
    event: str
    price: float
    quantity: float
    notional: float
    fee: float
    pnl: float
    equity_after: float
    rationale: str


def plan_schema() -> dict[str, Any]:
    leg = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "symbol",
            "side",
            "allocation_pct",
            "buy_price",
            "sell_price",
            "stop_price",
            "confidence",
            "rationale",
        ],
        "properties": {
            "symbol": {"type": "string"},
            "side": {"type": "string", "enum": ["long", "short", "flat"]},
            "allocation_pct": {"type": "number"},
            "buy_price": {"type": "number"},
            "sell_price": {"type": "number"},
            "stop_price": {"type": "number"},
            "confidence": {"type": "number"},
            "rationale": {"type": "string"},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["day_index", "plan_timestamp", "cash_pct", "gross_exposure_pct", "risk_notes", "legs"],
        "properties": {
            "day_index": {"type": "integer"},
            "plan_timestamp": {"type": "string"},
            "cash_pct": {"type": "number"},
            "gross_exposure_pct": {"type": "number"},
            "risk_notes": {"type": "string"},
            "legs": {"type": "array", "minItems": 1, "items": leg},
        },
    }


def _load_env_real() -> None:
    try:
        import env_real  # type: ignore
    except Exception:
        return
    value = getattr(env_real, "OPENAI_API_KEY", None)
    if value and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = str(value)


def _json_schema_for_openai() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "binance_daily_portfolio_plan",
        "strict": True,
        "schema": plan_schema(),
    }


def _market_payload(
    frames: Mapping[str, pd.DataFrame],
    *,
    plan_i: int,
    day_start_i: int,
    history_hours: int,
    equity: float,
    feature_style: str,
    line_bars: int,
) -> dict[str, Any]:
    close = close_matrix(frames)
    hist_start = max(0, day_start_i - int(history_hours))
    hist_close = close.iloc[hist_start:day_start_i]
    returns = hist_close.pct_change().replace([np.inf, -np.inf], np.nan)
    per_symbol: dict[str, Any] = {}
    for symbol in close.columns:
        series = hist_close[symbol].dropna()
        if series.empty:
            continue
        row: dict[str, Any] = {
            "last_close": float(series.iloc[-1]),
            "ret_6h": _window_return(series, 6),
            "ret_24h": _window_return(series, 24),
            "ret_72h": _window_return(series, 72),
            "ret_168h": _window_return(series, 168),
            "vol_24h": float(returns[symbol].tail(24).std(ddof=0) or 0.0),
            "vol_168h": float(returns[symbol].tail(168).std(ddof=0) or 0.0),
            "drawdown_168h": 0.0,
            "last_12h_closes": [float(x) for x in series.tail(12).to_numpy(dtype=float)],
        }
        high_168 = float(series.tail(168).max())
        if high_168 > 0:
            row["drawdown_168h"] = float(series.iloc[-1] / high_168 - 1.0)
        per_symbol[symbol] = row
    payload = {
        "task": "Create one 24-hour crypto portfolio plan with buy/sell levels and allocation percentages.",
        "feature_style": str(feature_style),
        "day_index": int(plan_i),
        "plan_timestamp": close.index[day_start_i - 1].isoformat(),
        "execution_window": {
            "start": close.index[day_start_i].isoformat(),
            "end": close.index[min(day_start_i + 23, len(close) - 1)].isoformat(),
        },
        "portfolio": {
            "equity_usd": float(equity),
            "max_gross_exposure_pct": 150.0,
            "max_single_symbol_allocation_pct": 60.0,
            "cash_pct_minimum": 10.0,
            "shorts_allowed": True,
            "daily_forced_flatten": True,
        },
        "execution_assumptions": {
            "fee_bps": 10.0,
            "slippage_bps": 5.0,
            "decision_lag_hours": 2,
            "one_entry_per_symbol_per_day": True,
            "long_semantics": "buy_price is entry, sell_price is take-profit, stop_price is stop-loss",
            "short_semantics": "sell_price is entry, buy_price is take-profit cover, stop_price is stop-loss cover",
        },
        "symbols": list(close.columns),
        "features": per_symbol,
        "output_rules": [
            "Return one leg for each symbol.",
            "Use side=flat and allocation_pct=0 for symbols not traded.",
            "Keep gross_exposure_pct equal to the sum of non-flat allocation_pct values.",
            "Use plausible limit levels near recent prices, not impossible far-away levels.",
            "Each symbol can enter at most once during the 24-hour plan.",
            "Known failed pattern: mostly correlated crypto-beta longs with tight stops lost after fees/slippage.",
            "Prefer post-fee expectancy, cash in weak regimes, selective shorts/hedges, and stops wide enough for hourly noise.",
            "The objective is maximize 7-day PnL while keeping drawdown under 20%.",
        ],
    }
    if feature_style in {"line_pct", "ohlc_mid"}:
        payload["line_encoding"] = _line_encoding_description(feature_style)
        payload["line_tokens"] = _market_line_tokens(
            frames,
            day_start_i=day_start_i,
            bars=int(line_bars),
            feature_style=feature_style,
        )
    return payload


def _line_encoding_description(feature_style: str) -> dict[str, Any]:
    if feature_style == "ohlc_mid":
        return {
            "order": "oldest_to_newest",
            "unit": "basis_points",
            "token": "ret_close_prev/o_mid/h_mid/l_mid/c_mid/range_mid",
            "meaning": (
                "ret_close_prev is close vs previous close. o_mid/h_mid/l_mid/c_mid are OHLC percent offsets "
                "from the bar midpoint where midpoint=(high+low)/2. range_mid is (high-low)/midpoint."
            ),
        }
    return {
        "order": "oldest_to_newest",
        "unit": "basis_points",
        "token": "o/h/l/c",
        "meaning": "OHLC each expressed as percent move from previous close.",
    }


def _market_line_tokens(
    frames: Mapping[str, pd.DataFrame],
    *,
    day_start_i: int,
    bars: int,
    feature_style: str,
) -> dict[str, Any]:
    tokens: dict[str, Any] = {}
    start = max(1, int(day_start_i) - int(bars))
    for symbol, df in frames.items():
        rows: list[str] = []
        for i in range(start, int(day_start_i)):
            row = df.iloc[i]
            prev_close = float(df.iloc[i - 1]["close"])
            if prev_close <= 0:
                continue
            open_ = float(row["open"])
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])
            if feature_style == "ohlc_mid":
                mid = max((high + low) / 2.0, 1e-12)
                vals = [
                    (close / prev_close - 1.0) * 10_000.0,
                    (open_ / mid - 1.0) * 10_000.0,
                    (high / mid - 1.0) * 10_000.0,
                    (low / mid - 1.0) * 10_000.0,
                    (close / mid - 1.0) * 10_000.0,
                    ((high - low) / mid) * 10_000.0,
                ]
            else:
                vals = [
                    (open_ / prev_close - 1.0) * 10_000.0,
                    (high / prev_close - 1.0) * 10_000.0,
                    (low / prev_close - 1.0) * 10_000.0,
                    (close / prev_close - 1.0) * 10_000.0,
                ]
            rows.append(",".join(f"{x:.0f}" for x in vals))
        tokens[symbol] = {
            "last_close": float(df.iloc[day_start_i - 1]["close"]),
            "bars": rows,
        }
    return tokens


def _window_return(series: pd.Series, hours: int) -> float:
    if len(series) <= hours:
        return 0.0
    p0 = float(series.iloc[-hours - 1])
    p1 = float(series.iloc[-1])
    return float(p1 / p0 - 1.0) if p0 > 0 else 0.0


def call_openai_api(
    payload: Mapping[str, Any],
    *,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    prompt_mode: str,
) -> str:
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openai package is required for --provider openai") from exc
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": _system_text(prompt_mode),
                    }
                ],
            },
            {"role": "user", "content": [{"type": "input_text", "text": json.dumps(payload, indent=2)}]},
        ],
        text={"format": _json_schema_for_openai()},
        reasoning={"effort": reasoning_effort},
        max_output_tokens=int(max_output_tokens),
        store=False,
    )
    text = getattr(response, "output_text", None)
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("OpenAI response had no output_text")
    return text


def call_codex_cli(
    payload: Mapping[str, Any],
    *,
    model: str,
    reasoning_effort: str,
    timeout_s: int,
    prompt_mode: str,
) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".schema.json", delete=False) as schema_file:
        json.dump(plan_schema(), schema_file)
        schema_path = schema_file.name
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as output_file:
        output_path = output_file.name
    prompt = f"{_system_text(prompt_mode)}\n\n{json.dumps(payload, indent=2)}"
    try:
        result = subprocess.run(
            [
                "codex",
                "exec",
                "--model",
                str(model),
                "--ephemeral",
                "--output-schema",
                schema_path,
                "-o",
                output_path,
                "--dangerously-bypass-approvals-and-sandbox",
                "-c",
                f"model_reasoning_effort={reasoning_effort}",
                prompt,
            ],
            capture_output=True,
            text=True,
            timeout=int(timeout_s),
        )
        text = Path(output_path).read_text(encoding="utf-8").strip()
        if not text:
            raise RuntimeError(f"codex returned empty output rc={result.returncode} stderr={result.stderr[:500]}")
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        return text
    finally:
        for path in (schema_path, output_path):
            try:
                Path(path).unlink()
            except OSError:
                pass


def _system_text(prompt_mode: str) -> str:
    if prompt_mode == "no_think":
        return (
            "Valid JSON only. Use only given OHLC tokens. No analysis. No tools. No web. "
            "No code. No external facts. Pick robust levels after fees."
        )
    if prompt_mode == "minimal":
        return (
            "Return only valid JSON matching the schema. Use only the provided OHLC/feature data. "
            "No web search, no tools, no code, no external facts, no explanations. "
            "Avoid correlated beta longs with tight stops; prefer post-fee expectancy."
        )
    return (
        "You are a crypto portfolio planner. Return only the strict structured plan. "
        "Use only the provided market tokens and features. Do not use web search, tools, code interpreter, "
        "or external market narratives. Prior one-shot plans failed by overloading correlated crypto-beta longs "
        "with tight stops; avoid that pattern."
    )


def _parse_plan(raw_text: str, *, expected_day: int, allowed_symbols: Sequence[str]) -> DailyPlan:
    data = json.loads(raw_text)
    symbols = {symbol.upper() for symbol in allowed_symbols}
    legs: list[PlanLeg] = []
    for item in data.get("legs", []):
        symbol = str(item["symbol"]).upper()
        if symbol not in symbols:
            continue
        side = str(item["side"]).lower()
        allocation = max(0.0, min(100.0, float(item["allocation_pct"])))
        if side == "flat":
            allocation = 0.0
        legs.append(
            PlanLeg(
                symbol=symbol,
                side=side if side in {"long", "short", "flat"} else "flat",
                allocation_pct=allocation,
                buy_price=max(0.0, float(item["buy_price"])),
                sell_price=max(0.0, float(item["sell_price"])),
                stop_price=max(0.0, float(item["stop_price"])),
                confidence=max(0.0, min(1.0, float(item["confidence"]))),
                rationale=str(item["rationale"]),
            )
        )
    if not legs:
        raise ValueError("model returned no usable legs")
    gross = sum(leg.allocation_pct for leg in legs if leg.side != "flat")
    if gross > 150.0:
        scale = 150.0 / gross
        legs = [
            PlanLeg(**{**asdict(leg), "allocation_pct": leg.allocation_pct * scale})
            if leg.side != "flat"
            else leg
            for leg in legs
        ]
        gross = 150.0
    return DailyPlan(
        day_index=int(data.get("day_index", expected_day)),
        plan_timestamp=str(data.get("plan_timestamp", "")),
        cash_pct=max(0.0, float(data.get("cash_pct", 0.0))),
        gross_exposure_pct=float(data.get("gross_exposure_pct", gross)),
        risk_notes=str(data.get("risk_notes", "")),
        legs=legs,
    )


def _fee(notional: float, fee_bps: float, slippage_bps: float) -> float:
    return abs(float(notional)) * (float(fee_bps) + float(slippage_bps)) / 10_000.0


def simulate_day(
    frames: Mapping[str, pd.DataFrame],
    plan: DailyPlan,
    *,
    start_i: int,
    end_i: int,
    starting_equity: float,
    fee_bps: float,
    slippage_bps: float,
) -> tuple[float, list[TradeEvent], list[dict[str, Any]]]:
    equity = float(starting_equity)
    positions: dict[str, dict[str, Any]] = {}
    entered_symbols: set[str] = set()
    trades: list[TradeEvent] = []
    hourly_rows: list[dict[str, Any]] = []
    day_start_equity = equity

    for i in range(start_i, min(end_i, len(next(iter(frames.values()))))):
        ts = next(iter(frames.values())).index[i]

        # Exits first, conservatively taking stops before take-profits when both touch.
        for symbol, pos in list(positions.items()):
            row = frames[symbol].iloc[i]
            side = pos["side"]
            qty = float(pos["quantity"])
            entry = float(pos["entry_price"])
            leg: PlanLeg = pos["leg"]
            exit_price: float | None = None
            event = ""
            if side == "long":
                if float(row["low"]) <= leg.stop_price:
                    exit_price = leg.stop_price
                    event = "stop"
                elif float(row["high"]) >= leg.sell_price:
                    exit_price = leg.sell_price
                    event = "take_profit"
                if exit_price is not None:
                    notional = qty * exit_price
                    pnl = qty * (exit_price - entry)
                    fee = _fee(notional, fee_bps, slippage_bps)
                    equity += pnl - fee
                    trades.append(_trade_event(ts, symbol, side, event, exit_price, qty, notional, fee, pnl - fee, equity, leg))
                    positions.pop(symbol, None)
            elif side == "short":
                if float(row["high"]) >= leg.stop_price:
                    exit_price = leg.stop_price
                    event = "stop"
                elif float(row["low"]) <= leg.buy_price:
                    exit_price = leg.buy_price
                    event = "take_profit"
                if exit_price is not None:
                    notional = qty * exit_price
                    pnl = qty * (entry - exit_price)
                    fee = _fee(notional, fee_bps, slippage_bps)
                    equity += pnl - fee
                    trades.append(_trade_event(ts, symbol, side, event, exit_price, qty, notional, fee, pnl - fee, equity, leg))
                    positions.pop(symbol, None)

        # Entries use day-start equity so allocation sizing is stable within the plan.
        for leg in plan.legs:
            if leg.side == "flat" or leg.symbol in positions or leg.symbol in entered_symbols or leg.allocation_pct <= 0.0:
                continue
            row = frames[leg.symbol].iloc[i]
            notional = day_start_equity * leg.allocation_pct / 100.0
            if leg.side == "long" and leg.buy_price > 0 and float(row["low"]) <= leg.buy_price:
                qty = notional / leg.buy_price
                fee = _fee(notional, fee_bps, slippage_bps)
                equity -= fee
                positions[leg.symbol] = {"side": "long", "quantity": qty, "entry_price": leg.buy_price, "leg": leg}
                entered_symbols.add(leg.symbol)
                trades.append(_trade_event(ts, leg.symbol, "long", "entry", leg.buy_price, qty, notional, fee, -fee, equity, leg))
            elif leg.side == "short" and leg.sell_price > 0 and float(row["high"]) >= leg.sell_price:
                qty = notional / leg.sell_price
                fee = _fee(notional, fee_bps, slippage_bps)
                equity -= fee
                positions[leg.symbol] = {"side": "short", "quantity": qty, "entry_price": leg.sell_price, "leg": leg}
                entered_symbols.add(leg.symbol)
                trades.append(_trade_event(ts, leg.symbol, "short", "entry", leg.sell_price, qty, notional, fee, -fee, equity, leg))

        marked = equity
        for symbol, pos in positions.items():
            close = float(frames[symbol].iloc[i]["close"])
            if pos["side"] == "long":
                marked += float(pos["quantity"]) * (close - float(pos["entry_price"]))
            else:
                marked += float(pos["quantity"]) * (float(pos["entry_price"]) - close)
        hourly_rows.append({"timestamp": ts.isoformat(), "equity": marked, "cash_equity": equity, "open_positions": len(positions)})

    # Forced daily flatten at final close.
    final_i = min(end_i - 1, len(next(iter(frames.values()))) - 1)
    final_ts = next(iter(frames.values())).index[final_i]
    for symbol, pos in list(positions.items()):
        close = float(frames[symbol].iloc[final_i]["close"])
        qty = float(pos["quantity"])
        entry = float(pos["entry_price"])
        leg = pos["leg"]
        notional = qty * close
        pnl = qty * (close - entry) if pos["side"] == "long" else qty * (entry - close)
        fee = _fee(notional, fee_bps, slippage_bps)
        equity += pnl - fee
        trades.append(_trade_event(final_ts, symbol, pos["side"], "daily_flatten", close, qty, notional, fee, pnl - fee, equity, leg))
    if hourly_rows:
        hourly_rows[-1]["equity"] = equity
        hourly_rows[-1]["cash_equity"] = equity
        hourly_rows[-1]["open_positions"] = 0
    return equity, trades, hourly_rows


def _trade_event(
    ts: pd.Timestamp,
    symbol: str,
    side: str,
    event: str,
    price: float,
    quantity: float,
    notional: float,
    fee: float,
    pnl: float,
    equity: float,
    leg: PlanLeg,
) -> TradeEvent:
    return TradeEvent(
        timestamp=ts.isoformat(),
        symbol=symbol,
        side=side,
        event=event,
        price=float(price),
        quantity=float(quantity),
        notional=float(notional),
        fee=float(fee),
        pnl=float(pnl),
        equity_after=float(equity),
        rationale=leg.rationale,
    )


def _metrics(equity_rows: list[dict[str, Any]], initial_cash: float) -> dict[str, float]:
    if not equity_rows:
        return {"total_return": 0.0, "monthly_return": 0.0, "max_drawdown": 0.0, "sortino": 0.0}
    equity = pd.Series([float(row["equity"]) for row in equity_rows])
    total = float(equity.iloc[-1] / float(initial_cash) - 1.0)
    hours = max(1, len(equity) - 1)
    monthly = float(max(equity.iloc[-1] / float(initial_cash), 1e-12) ** (HOURS_PER_MONTH / hours) - 1.0)
    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    downside = returns[returns < 0]
    sortino = 0.0
    if len(downside) > 0:
        sortino = float(returns.mean() / max(downside.std(ddof=0), 1e-12) * math.sqrt(HOURS_PER_YEAR))
    elif len(returns) > 0 and returns.mean() > 0:
        sortino = 999.0
    max_dd = float(abs((equity / equity.cummax() - 1.0).min()))
    return {"total_return": total, "monthly_return": monthly, "max_drawdown": max_dd, "sortino": sortino}


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    if args.load_env_real:
        _load_env_real()
    total_hours = int(args.history_hours) + int(args.days) * 24 + int(args.decision_lag_hours)
    frames = load_hourly_panel(args.data_root, args.symbols, tail_hours=total_hours)
    common = close_matrix(frames).index
    start_i = int(args.history_hours) + int(args.decision_lag_hours)
    equity = float(args.initial_cash)
    all_trades: list[TradeEvent] = []
    all_equity_rows: list[dict[str, Any]] = []
    plans: list[DailyPlan] = []
    raw_paths: list[str] = []

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    for day in range(int(args.days)):
        day_start = start_i + day * 24
        day_end = min(day_start + 24, len(common))
        if day_end <= day_start:
            break
        payload = _market_payload(
            frames,
            plan_i=day,
            day_start_i=day_start,
            history_hours=int(args.history_hours),
            equity=equity,
            feature_style=str(args.feature_style),
            line_bars=int(args.line_bars),
        )
        if args.provider == "openai":
            raw = call_openai_api(
                payload,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                max_output_tokens=args.max_output_tokens,
                prompt_mode=str(args.prompt_mode),
            )
        elif args.provider == "codex":
            raw = call_codex_cli(
                payload,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                timeout_s=args.codex_timeout_s,
                prompt_mode=str(args.prompt_mode),
            )
        else:
            raise ValueError(f"Unsupported provider: {args.provider}")
        raw_path = output_dir / f"gpt55_week_raw_{stamp}_day{day}.json"
        raw_path.write_text(raw, encoding="utf-8")
        raw_paths.append(str(raw_path))
        plan = _parse_plan(raw, expected_day=day, allowed_symbols=args.symbols)
        plans.append(plan)
        equity, trades, rows = simulate_day(
            frames,
            plan,
            start_i=day_start,
            end_i=day_end,
            starting_equity=equity,
            fee_bps=float(args.fee_bps),
            slippage_bps=float(args.slippage_bps),
        )
        all_trades.extend(trades)
        all_equity_rows.extend(rows)

    metrics = _metrics(all_equity_rows, float(args.initial_cash))
    return {
        "provider": args.provider,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "symbols": list(args.symbols),
        "days": int(args.days),
        "history_hours": int(args.history_hours),
        "feature_style": str(args.feature_style),
        "line_bars": int(args.line_bars),
        "prompt_mode": str(args.prompt_mode),
        "initial_cash": float(args.initial_cash),
        "final_equity": float(equity),
        "metrics": metrics,
        "plans": [
            {
                **asdict(plan),
                "legs": [asdict(leg) for leg in plan.legs],
            }
            for plan in plans
        ],
        "trades": [asdict(trade) for trade in all_trades],
        "equity_curve": all_equity_rows,
        "raw_response_paths": raw_paths,
    }


def _write_report(path: Path, result: Mapping[str, Any]) -> None:
    metrics = result["metrics"]
    lines = [
        f"# {result['model']} {result['reasoning_effort']} {result['days']}-Day Binance Replanner",
        "",
        f"- Provider: `{result['provider']}`",
        f"- Model: `{result['model']}`",
        f"- Reasoning: `{result['reasoning_effort']}`",
        f"- Prompt mode: `{result.get('prompt_mode', 'standard')}`",
        f"- Feature style: `{result.get('feature_style', 'summary')}`",
        f"- Symbols: `{', '.join(result['symbols'])}`",
        f"- Initial cash: `${result['initial_cash']:,.2f}`",
        f"- Final equity: `${result['final_equity']:,.2f}`",
        f"- Total return: `{metrics['total_return'] * 100:+.2f}%`",
        f"- Monthly-equivalent return: `{metrics['monthly_return'] * 100:+.2f}%`",
        f"- Max drawdown: `{metrics['max_drawdown'] * 100:.2f}%`",
        f"- Sortino: `{metrics['sortino']:.2f}`",
        f"- Trades: `{len(result['trades'])}`",
        "",
        "## Daily Plans",
        "",
    ]
    for plan in result["plans"]:
        active = [leg for leg in plan["legs"] if leg["side"] != "flat" and leg["allocation_pct"] > 0]
        summary = ", ".join(f"{leg['side']} {leg['symbol']} {leg['allocation_pct']:.1f}%" for leg in active) or "flat"
        lines.append(f"- Day {plan['day_index']}: {summary}. Risk: {plan['risk_notes']}")
    lines.extend(["", "## Trades", ""])
    for trade in result["trades"]:
        lines.append(
            "- "
            f"{trade['timestamp']} {trade['event']} {trade['side']} {trade['symbol']} "
            f"notional=${trade['notional']:.2f} price={trade['price']:.6g} "
            f"pnl=${trade['pnl']:.2f} equity=${trade['equity_after']:.2f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["openai", "codex"], default="codex")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING)
    parser.add_argument("--data-root", default="trainingdatahourlybinance")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--history-hours", type=int, default=168)
    parser.add_argument("--feature-style", choices=["summary", "line_pct", "ohlc_mid"], default="summary")
    parser.add_argument("--line-bars", type=int, default=48)
    parser.add_argument("--prompt-mode", choices=["standard", "minimal", "no_think"], default="standard")
    parser.add_argument("--decision-lag-hours", type=int, default=2)
    parser.add_argument("--initial-cash", type=float, default=DEFAULT_INITIAL_CASH)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--codex-timeout-s", type=int, default=240)
    parser.add_argument("--load-env-real", action="store_true")
    parser.add_argument("--output-dir", default="experiments/binance_openai_planner/results")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"gpt55_week_replanner_{stamp}.json"
    md_path = output_dir / f"gpt55_week_replanner_{stamp}.md"
    try:
        result = run_experiment(args)
    except Exception as exc:
        error = {
            "provider": args.provider,
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "error": str(exc),
        }
        write = json.dumps(error, indent=2, sort_keys=True)
        json_path.write_text(write, encoding="utf-8")
        print(f"experiment failed: {exc}")
        print(f"wrote {json_path}")
        return 2
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(md_path, result)
    metrics = result["metrics"]
    print(
        f"final=${result['final_equity']:.2f} "
        f"return={metrics['total_return'] * 100:+.2f}% "
        f"monthly={metrics['monthly_return'] * 100:+.2f}% "
        f"dd={metrics['max_drawdown'] * 100:.2f}% "
        f"trades={len(result['trades'])}"
    )
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
