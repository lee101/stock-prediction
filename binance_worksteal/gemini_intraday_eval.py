#!/usr/bin/env python3
"""Evaluate Gemini overlays on the current hourly-executed work-steal strategy."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from binance_worksteal.backtest import FULL_UNIVERSE
from binance_worksteal.gemini_overlay import DailyTradePlan, build_daily_prompt, call_gemini_daily, load_forecast_daily
from binance_worksteal.robust_eval import EvaluationWindow, build_recent_windows, build_start_state_config
from binance_worksteal.strategy import (
    Position,
    TradeLog,
    WorkStealConfig,
    _apply_seeded_rebalance,
    _compute_rebalance_keep_symbols,
    _compute_starting_equity,
    _normalize_base_asset_symbol,
    _seed_initial_holdings,
    build_entry_candidates,
    compute_metrics,
    get_fee,
    load_daily_bars,
    load_hourly_bars,
    resolve_entry_regime,
    run_worksteal_backtest,
)
from binance_worksteal.trade_live import DEFAULT_CONFIG, _relative_bps_distance
from src.robust_trading_metrics import compute_pnl_smoothness_from_equity, summarize_scenario_results


GEMINI_CACHE_ROOT = Path("binance_worksteal/gemini_cache_eval")


@dataclass(frozen=True)
class GeminiPendingOrder:
    symbol: str
    direction: str
    score: float
    fill_price: float
    target_exit_price: float
    stop_price: float
    confidence: float
    source: str


def _sanitize_model_name(model: str) -> str:
    text = str(model or "").strip() or "unknown"
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)


def _cache_path(*, model: str, prompt: str) -> Path:
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
    safe_model = _sanitize_model_name(model)
    path = GEMINI_CACHE_ROOT / safe_model / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_cached_plan(*, model: str, prompt: str) -> Optional[DailyTradePlan]:
    path = _cache_path(model=model, prompt=prompt)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return DailyTradePlan(**data)
    except Exception:
        return None


def _store_cached_plan(*, model: str, prompt: str, plan: DailyTradePlan) -> None:
    path = _cache_path(model=model, prompt=prompt)
    path.write_text(
        json.dumps(
            {
                "action": plan.action,
                "buy_price": plan.buy_price,
                "sell_price": plan.sell_price,
                "stop_price": plan.stop_price,
                "confidence": plan.confidence,
                "reasoning": plan.reasoning,
            }
        )
    )


def _build_universe_summary(
    *,
    current_bars: Dict[str, pd.Series],
    history: Dict[str, pd.DataFrame],
    positions: Dict[str, Position],
) -> str:
    lines: list[str] = []
    for sym, bar in sorted(current_bars.items()):
        hist = history.get(sym)
        if hist is None or len(hist) < 2:
            continue
        close = float(bar["close"])
        prev = float(hist.iloc[-2]["close"])
        if prev <= 0.0:
            continue
        change_pct = (close - prev) / prev * 100.0
        marker = "HELD" if sym in positions else ""
        lines.append(f"  {sym:10s} ${close:>10.2f} {change_pct:+5.1f}% {marker}")
    return "\n".join(lines[:15])


def _gemini_entry_plan(
    *,
    symbol: str,
    score: float,
    fill_price: float,
    close: float,
    bars: pd.DataFrame,
    current_bars: Dict[str, pd.Series],
    history: Dict[str, pd.DataFrame],
    positions: Dict[str, Position],
    recent_trades: List[dict],
    config: WorkStealConfig,
    model: str,
    use_cache: bool,
    api_key: Optional[str],
) -> tuple[Optional[DailyTradePlan], bool]:
    forecast = load_forecast_daily(symbol)
    prompt = build_daily_prompt(
        symbol=symbol,
        bars=bars,
        current_price=close,
        rule_signal={"buy_target": fill_price, "dip_score": score, "ref_price": 0, "sma_ok": True},
        recent_trades=recent_trades[-5:],
        forecast_24h=forecast,
        universe_summary=_build_universe_summary(current_bars=current_bars, history=history, positions=positions),
        fee_bps=0 if symbol in {"BTCUSD", "ETHUSD"} else 10,
        entry_proximity_bps=config.entry_proximity_bps,
    )
    if use_cache:
        cached = _load_cached_plan(model=model, prompt=prompt)
        if cached is not None:
            return cached, False
    plan = call_gemini_daily(prompt, model=model, api_key=api_key)
    if plan is not None and use_cache:
        _store_cached_plan(model=model, prompt=prompt, plan=plan)
    return plan, True


def run_gemini_intraday_backtest(
    *,
    all_bars: Dict[str, pd.DataFrame],
    intraday_bars: Dict[str, pd.DataFrame],
    config: WorkStealConfig,
    start_date: str,
    end_date: str,
    model: str,
    use_cache: bool = True,
    api_key: Optional[str] = None,
) -> tuple[pd.DataFrame, List[TradeLog], Dict[str, float]]:
    all_dates = sorted(set().union(*[set(df["timestamp"].tolist()) for df in all_bars.values()]))
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC")
    all_dates = [d for d in all_dates if start_ts <= d <= end_ts]

    cash = config.initial_cash
    positions: Dict[str, Position] = {}
    trades: List[TradeLog] = []
    equity_rows: List[Dict[str, object]] = []
    last_exit: Dict[str, pd.Timestamp] = {}
    recent_trades: List[dict] = []
    initial_holdings_seeded = False
    seeded_positions_rebalanced = False
    starting_equity = float(config.initial_cash)
    llm_calls = 0
    llm_overrides = 0
    base_symbol = _normalize_base_asset_symbol(config)
    base_qty = 0.0

    for idx, date in enumerate(all_dates):
        current_bars: Dict[str, pd.Series] = {}
        history: Dict[str, pd.DataFrame] = {}
        for sym, df in all_bars.items():
            hist = df[df["timestamp"] <= date]
            if hist.empty:
                continue
            bar = hist.iloc[-1]
            if bar["timestamp"] != date:
                continue
            current_bars[sym] = bar
            history[sym] = hist
        if not current_bars:
            continue

        if not initial_holdings_seeded:
            base_qty += _seed_initial_holdings(
                date=date,
                current_bars=current_bars,
                config=config,
                positions=positions,
                base_symbol=base_symbol,
            )
            starting_equity = _compute_starting_equity(config, current_bars)
            initial_holdings_seeded = True

        current_prices = {sym: float(bar["close"]) for sym, bar in current_bars.items()}
        if initial_holdings_seeded and not seeded_positions_rebalanced:
            keep_symbols = _compute_rebalance_keep_symbols(
                date=date,
                current_bars=current_bars,
                history=history,
                last_exit=last_exit,
                config=config,
                base_symbol=base_symbol,
            )
            cash = _apply_seeded_rebalance(
                timestamp=date,
                current_prices=current_prices,
                positions=positions,
                trades=trades,
                last_exit=last_exit,
                cash=cash,
                config=config,
                keep_symbols=keep_symbols,
            )
            seeded_positions_rebalanced = True

        entry_regime = resolve_entry_regime(current_bars=current_bars, history=history, config=config)
        entry_config = entry_regime.config
        skip_entries = entry_regime.skip_entries
        daily_candidates = (
            build_entry_candidates(
                date=date,
                current_bars=current_bars,
                history=history,
                positions=positions,
                last_exit=last_exit,
                config=entry_config,
                base_symbol=None,
            )
            if not skip_entries and len(positions) < config.max_positions
            else []
        )

        next_date = all_dates[idx + 1] if idx + 1 < len(all_dates) else date + pd.Timedelta(days=1)
        interval_frames: Dict[str, pd.DataFrame] = {}
        interval_timestamps: set[pd.Timestamp] = set()
        for sym, df in intraday_bars.items():
            window = df[(df["timestamp"] > date) & (df["timestamp"] <= next_date)]
            if window.empty:
                continue
            window = window.set_index("timestamp", drop=False)
            interval_frames[sym] = window
            interval_timestamps.update(window.index.tolist())

        pending_orders: list[GeminiPendingOrder] = []
        slots = config.max_positions - len(positions)
        for sym, direction, score, fill_price, bar in daily_candidates[:slots]:
            if direction != "long" or sym in positions:
                continue
            close = float(bar["close"])
            buy_price = float(fill_price)
            sell_target = buy_price * (1.0 + config.profit_target_pct)
            stop_price = buy_price * (1.0 - config.stop_loss_pct)
            confidence = 1.0
            source = "rule"

            plan, used_api = _gemini_entry_plan(
                symbol=sym,
                score=float(score),
                fill_price=buy_price,
                close=close,
                bars=history[sym],
                current_bars=current_bars,
                history=history,
                positions=positions,
                recent_trades=recent_trades,
                config=config,
                model=model,
                use_cache=use_cache,
                api_key=api_key,
            )
            if used_api:
                llm_calls += 1

            if plan:
                if plan.action == "hold" and plan.confidence > 0.5:
                    continue
                if plan.action in ("buy", "adjust") and plan.confidence > 0.3:
                    if plan.buy_price > 0.0:
                        buy_price = float(plan.buy_price)
                    if plan.sell_price > 0.0:
                        sell_target = float(plan.sell_price)
                    if plan.stop_price > 0.0:
                        stop_price = float(plan.stop_price)
                    confidence = float(plan.confidence)
                    source = f"gemini(conf={confidence:.2f})"
                    llm_overrides += 1

            stage_frame = interval_frames.get(sym)
            if stage_frame is None or stage_frame.empty:
                continue
            stage_row = stage_frame.iloc[0]
            stage_price = float(stage_row.get("open", stage_row["close"]) or stage_row["close"])
            if _relative_bps_distance(stage_price, buy_price) > float(config.entry_proximity_bps):
                continue

            pending_orders.append(
                GeminiPendingOrder(
                    symbol=sym,
                    direction="long",
                    score=float(score),
                    fill_price=buy_price,
                    target_exit_price=sell_target,
                    stop_price=stop_price,
                    confidence=confidence,
                    source=source,
                )
            )

        ordered_timestamps = sorted(interval_timestamps)
        equity_rows.append(
            {
                "timestamp": date,
                "equity": cash + sum(
                    pos.quantity * float(current_prices.get(sym, pos.entry_price)) for sym, pos in positions.items()
                ),
                "cash": cash,
                "n_positions": len(positions),
            }
        )

        for ts in ordered_timestamps:
            hourly_bars: Dict[str, pd.Series] = {}
            for sym, frame in interval_frames.items():
                if ts not in frame.index:
                    continue
                row = frame.loc[ts]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]
                hourly_bars[sym] = row
                current_prices[sym] = float(row["close"])

            symbols_to_exit: list[tuple[str, float, str]] = []
            for sym, pos in list(positions.items()):
                row = hourly_bars.get(sym)
                if row is None:
                    continue
                close = float(row["close"])
                high = float(row["high"])
                low = float(row["low"])
                pos.peak_price = max(pos.peak_price, high)

                exit_price = None
                exit_reason = ""
                if high >= pos.target_exit_price:
                    exit_price = pos.target_exit_price
                    exit_reason = "profit_target"
                elif low <= pos.stop_price:
                    exit_price = pos.stop_price
                    exit_reason = "stop_loss"
                elif config.trailing_stop_pct > 0:
                    trail = pos.peak_price * (1.0 - config.trailing_stop_pct)
                    if low <= trail:
                        exit_price = trail
                        exit_reason = "trailing_stop"

                held_days = max(0.0, float((ts - pos.entry_date).total_seconds()) / 86_400.0)
                if exit_price is None and config.max_hold_days > 0 and held_days >= config.max_hold_days:
                    exit_price = close
                    exit_reason = "max_hold"

                if exit_price is not None:
                    symbols_to_exit.append((sym, float(exit_price), exit_reason))

            for sym, exit_price, reason in symbols_to_exit:
                pos = positions[sym]
                fee_rate = get_fee(sym, config)
                proceeds = pos.quantity * exit_price * (1.0 - fee_rate)
                pnl = proceeds - pos.cost_basis
                cash += proceeds
                trades.append(
                    TradeLog(
                        timestamp=ts,
                        symbol=sym,
                        side="sell",
                        price=exit_price,
                        quantity=pos.quantity,
                        notional=pos.quantity * exit_price,
                        fee=pos.quantity * exit_price * fee_rate,
                        pnl=pnl,
                        reason=reason,
                        direction="long",
                    )
                )
                recent_trades.append(
                    {
                        "timestamp": str(ts),
                        "symbol": sym,
                        "side": "sell",
                        "price": exit_price,
                        "pnl": pnl,
                        "reason": reason,
                    }
                )
                last_exit[sym] = ts
                del positions[sym]

            if pending_orders and len(positions) < config.max_positions:
                next_pending: list[GeminiPendingOrder] = []
                for order in pending_orders:
                    if len(positions) >= config.max_positions:
                        next_pending.append(order)
                        continue
                    if order.symbol in positions:
                        continue
                    row = hourly_bars.get(order.symbol)
                    if row is None:
                        next_pending.append(order)
                        continue
                    low = float(row["low"])
                    if low > order.fill_price:
                        next_pending.append(order)
                        continue
                    fee_rate = get_fee(order.symbol, config)
                    max_alloc = starting_equity * config.max_position_pct * config.max_leverage
                    alloc = min(max_alloc, cash) * min(order.confidence, 1.0)
                    if alloc <= 0.0:
                        next_pending.append(order)
                        continue
                    quantity = alloc / (order.fill_price * (1.0 + fee_rate))
                    if quantity <= 0.0:
                        next_pending.append(order)
                        continue
                    actual_cost = quantity * order.fill_price * (1.0 + fee_rate)
                    cash -= min(actual_cost, cash)
                    positions[order.symbol] = Position(
                        symbol=order.symbol,
                        direction="long",
                        entry_price=order.fill_price,
                        entry_date=ts,
                        quantity=quantity,
                        cost_basis=actual_cost,
                        peak_price=float(row["high"]),
                        target_exit_price=order.target_exit_price,
                        stop_price=order.stop_price,
                        margin_borrowed=0.0,
                        source="strategy",
                    )
                    trades.append(
                        TradeLog(
                            timestamp=ts,
                            symbol=order.symbol,
                            side="buy",
                            price=order.fill_price,
                            quantity=quantity,
                            notional=quantity * order.fill_price,
                            fee=quantity * order.fill_price * fee_rate,
                            reason=f"dip_buy({order.source})",
                            direction="long",
                        )
                    )
                    recent_trades.append(
                        {
                            "timestamp": str(ts),
                            "symbol": order.symbol,
                            "side": "buy",
                            "price": order.fill_price,
                            "pnl": 0.0,
                            "reason": order.source,
                        }
                    )
                pending_orders = next_pending

            equity = cash + sum(pos.quantity * float(current_prices.get(sym, pos.entry_price)) for sym, pos in positions.items())
            equity_rows.append(
                {
                    "timestamp": ts,
                    "equity": equity,
                    "cash": cash,
                    "n_positions": len(positions),
                }
            )

    equity_df = pd.DataFrame(equity_rows)
    metrics = compute_metrics(equity_df, config, trades)
    metrics["llm_calls"] = float(llm_calls)
    metrics["llm_overrides"] = float(llm_overrides)
    return equity_df, trades, metrics


def _scenario_row(
    *,
    candidate: str,
    model: str,
    window: EvaluationWindow,
    start_state: str,
    equity_df: pd.DataFrame,
    trades: List[TradeLog],
    metrics: Dict[str, float],
) -> dict:
    equity_values = equity_df["equity"].astype(float).to_numpy(copy=False) if not equity_df.empty else []
    n_days = float(metrics.get("n_days", 0.0) or 0.0)
    total_return = float(metrics.get("total_return", 0.0) or 0.0)
    annualized_return_pct = 0.0
    if n_days > 0.0:
        gross = 1.0 + total_return
        annualized_return_pct = -100.0 if gross <= 0.0 else float((gross ** (365.0 / max(n_days, 1.0)) - 1.0) * 100.0)
    return {
        "candidate": candidate,
        "model": model,
        "window_label": window.label,
        "start_state": start_state,
        "return_pct": float(metrics.get("total_return_pct", 0.0)),
        "annualized_return_pct": annualized_return_pct,
        "sortino": float(metrics.get("sortino", 0.0)),
        "max_drawdown_pct": abs(float(metrics.get("max_drawdown_pct", 0.0))),
        "pnl_smoothness": float(compute_pnl_smoothness_from_equity(equity_values)),
        "trade_count": float(len(trades)),
        "n_days": n_days,
        "llm_calls": float(metrics.get("llm_calls", 0.0)),
        "llm_overrides": float(metrics.get("llm_overrides", 0.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare current work-steal config with Gemini overlays on hourly replay.")
    parser.add_argument("--data-dir", default="trainingdatadailybinance")
    parser.add_argument("--hourly-data-dir", default="trainingdatahourly/crypto")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--window-count", type=int, default=1)
    parser.add_argument("--end-date", default="2026-03-14")
    parser.add_argument("--models", nargs="+", default=["gemini-3.1-flash-lite-preview", "gemini-3.1-pro-preview"])
    parser.add_argument("--start-states", nargs="+", default=["flat", "BTCUSD", "ETHUSD"])
    parser.add_argument("--symbols", nargs="+", default=FULL_UNIVERSE[:30])
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output-prefix", default="analysis/worksteal_hourly_20260318/gemini_intraday_compare")
    args = parser.parse_args()

    all_bars = load_daily_bars(args.data_dir, list(args.symbols))
    intraday_bars = load_hourly_bars(args.hourly_data_dir, list(args.symbols))
    windows = build_recent_windows(end_date=args.end_date, window_days=args.days, window_count=args.window_count)
    base_config = replace(DEFAULT_CONFIG)

    scenario_rows: list[dict] = []
    for window in windows:
        for raw_start_state in args.start_states:
            start_label, scenario_config = build_start_state_config(
                base_config=base_config,
                all_bars=all_bars,
                start_state=raw_start_state,
                start_date=window.start_date,
                end_date=window.end_date,
                starting_equity=base_config.initial_cash,
            )
            baseline_eq, baseline_trades, baseline_metrics = run_worksteal_backtest(
                {sym: bars.copy() for sym, bars in all_bars.items()},
                scenario_config,
                start_date=window.start_date,
                end_date=window.end_date,
                intraday_bars={sym: bars.copy() for sym, bars in intraday_bars.items()},
            )
            _ = (baseline_eq, baseline_trades)
            scenario_rows.append(
                _scenario_row(
                    candidate="rule_only",
                    model="rule_only",
                    window=window,
                    start_state=start_label,
                    equity_df=baseline_eq,
                    trades=baseline_trades,
                    metrics=baseline_metrics,
                )
            )
            for model in args.models:
                eq, trades, metrics = run_gemini_intraday_backtest(
                    all_bars={sym: bars.copy() for sym, bars in all_bars.items()},
                    intraday_bars={sym: bars.copy() for sym, bars in intraday_bars.items()},
                    config=scenario_config,
                    start_date=window.start_date,
                    end_date=window.end_date,
                    model=model,
                    use_cache=not args.no_cache,
                )
                _ = (eq, trades)
                scenario_rows.append(
                    _scenario_row(
                        candidate="gemini_overlay",
                        model=model,
                        window=window,
                        start_state=start_label,
                        equity_df=eq,
                        trades=trades,
                        metrics=metrics,
                    )
                )
                print(
                    f"{window.label} start={start_label} model={model} "
                    f"ret={metrics.get('total_return_pct', 0.0):+.2f}% "
                    f"sortino={metrics.get('sortino', 0.0):.2f} "
                    f"maxdd={metrics.get('max_drawdown_pct', 0.0):.2f}% "
                    f"llm_calls={int(metrics.get('llm_calls', 0.0))}"
                )

    scenario_df = pd.DataFrame(scenario_rows)
    summary_rows: list[dict] = []
    for (candidate, model), group in scenario_df.groupby(["candidate", "model"], sort=False):
        summary = summarize_scenario_results(group.to_dict("records"))
        summary_rows.append({"candidate": candidate, "model": model, **summary})

    summary_df = pd.DataFrame(summary_rows).sort_values("robust_score", ascending=False)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    scenario_path = output_prefix.with_name(f"{output_prefix.name}_scenarios.csv")
    summary_path = output_prefix.with_name(f"{output_prefix.name}_summary.csv")
    scenario_df.to_csv(scenario_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {scenario_path}")
    print(f"Wrote {summary_path}")
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
