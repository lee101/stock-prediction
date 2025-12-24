"""Backtest stockagent3 strategy over a date range."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import asyncio
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Mapping, Optional, Sequence

import pandas as pd
from loguru import logger
import anthropic

from stockagent.agentsimulator.market_data import fetch_latest_ohlc, MarketDataBundle
from stockagent.constants import TRADING_FEE, CRYPTO_TRADING_FEE
from src.symbol_utils import is_crypto_symbol

from .agent import TradePlan, TradePosition, async_generate_trade_plan
from .async_api import _get_api_key

LEVERAGE_ANNUAL_RATE = 0.065
LEVERAGE_DAILY_RATE = LEVERAGE_ANNUAL_RATE / 365


@dataclass
class PendingEntry:
    symbol: str
    direction: str
    entry_price: float
    target_alloc: float
    leverage: float
    entry_deadline: date
    hold_expiry_days: int
    exit_price: float
    stop_price: Optional[float]
    entry_mode: str


@dataclass
class OpenPosition:
    symbol: str
    direction: str
    qty: float
    entry_price: float
    entry_date: date
    exit_price: float
    stop_price: Optional[float]
    expiry_date: date
    leverage: float
    target_alloc: float


def _get_all_data(
    symbols: Sequence[str],
    end_date: date,
    lookback_days: int,
    *,
    allow_remote_download: bool,
) -> MarketDataBundle:
    as_of = datetime.combine(end_date, datetime.max.time()).replace(tzinfo=timezone.utc)
    return fetch_latest_ohlc(
        symbols=symbols,
        lookback_days=lookback_days,
        as_of=as_of,
        allow_remote_download=allow_remote_download,
    )


def _get_bar(df: pd.DataFrame, day: date) -> Optional[dict]:
    if df.empty or not hasattr(df.index, "date"):
        return None
    row = df[df.index.date == day]
    if row.empty:
        return None
    row = row.iloc[0]
    return {
        "open": float(row.get("open", row.get("close"))),
        "high": float(row.get("high", row.get("close"))),
        "low": float(row.get("low", row.get("close"))),
        "close": float(row.get("close", row.get("open"))),
    }


def _trading_calendar(df: pd.DataFrame) -> List[date]:
    if df.empty or not hasattr(df.index, "date"):
        return []
    return sorted({d for d in df.index.date})


def _advance_trading_date(calendar: List[date], start: date, days: int) -> date:
    if not calendar:
        return start
    try:
        idx = calendar.index(start)
    except ValueError:
        # pick nearest date after start
        later = [d for d in calendar if d >= start]
        if not later:
            return calendar[-1]
        idx = calendar.index(later[0])
    target_idx = min(idx + max(0, days), len(calendar) - 1)
    return calendar[target_idx]


def _fee_rate(symbol: str) -> float:
    return CRYPTO_TRADING_FEE if is_crypto_symbol(symbol) else TRADING_FEE


def _compute_leverage_cost(position: OpenPosition, exit_date: date) -> float:
    if is_crypto_symbol(position.symbol):
        return 0.0
    if position.leverage <= 1.0:
        return 0.0
    days_held = max(1, (exit_date - position.entry_date).days)
    borrowed = position.leverage - 1.0
    notional = abs(position.qty * position.entry_price)
    return notional * borrowed * LEVERAGE_DAILY_RATE * days_held


def _exit_position(position: OpenPosition, exit_price: float, exit_date: date) -> tuple[float, dict]:
    qty = position.qty
    direction = position.direction
    fee_rate = _fee_rate(position.symbol)
    entry_fee = abs(position.entry_price * qty) * fee_rate
    exit_fee = abs(exit_price * qty) * fee_rate

    if direction == "long":
        gross_pnl = (exit_price - position.entry_price) * qty
    else:
        gross_pnl = (position.entry_price - exit_price) * qty

    leverage_cost = _compute_leverage_cost(position, exit_date)
    net_pnl = gross_pnl - entry_fee - exit_fee - leverage_cost

    trade = {
        "symbol": position.symbol,
        "direction": direction,
        "qty": qty,
        "entry_price": position.entry_price,
        "exit_price": exit_price,
        "entry_date": position.entry_date.isoformat(),
        "exit_date": exit_date.isoformat(),
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "entry_fee": entry_fee,
        "exit_fee": exit_fee,
        "leverage_cost": leverage_cost,
    }
    return net_pnl, trade


def _build_current_positions(open_positions: Dict[str, OpenPosition], day: date) -> List[Mapping[str, object]]:
    snapshot = []
    for pos in open_positions.values():
        held_days = (day - pos.entry_date).days
        snapshot.append(
            {
                "symbol": pos.symbol,
                "side": pos.direction,
                "qty": pos.qty,
                "avg_price": pos.entry_price,
                "days_held": held_days,
                "expiry_date": pos.expiry_date.isoformat(),
            }
        )
    return snapshot


def _apply_plan(
    plan: TradePlan,
    *,
    trade_date: date,
    market_data: Mapping[str, pd.DataFrame],
    calendars: Mapping[str, List[date]],
    equity: float,
    open_positions: Dict[str, OpenPosition],
    pending_entries: List[PendingEntry],
) -> tuple[float, List[dict]]:
    realized = 0.0
    trades: List[dict] = []
    plan_map = {pos.symbol: pos for pos in plan.positions}
    plan_symbols = set(plan_map.keys())

    # Remove pending entries that are superseded by new plan
    pending_entries[:] = [entry for entry in pending_entries if entry.symbol not in plan_symbols]

    # Close positions no longer in plan
    for symbol in list(open_positions.keys()):
        if symbol in plan_symbols:
            if open_positions[symbol].direction == plan_map[symbol].direction:
                continue
        bar = _get_bar(market_data[symbol], trade_date)
        if not bar:
            continue
        net_pnl, trade = _exit_position(open_positions[symbol], bar["open"], trade_date)
        open_positions.pop(symbol, None)
        trade["exit_reason"] = "rebalanced"
        trade["net_pnl"] = net_pnl
        trades.append(trade)
        realized += net_pnl

    # Add new positions
    for position in plan.positions:
        symbol = position.symbol
        if symbol in open_positions:
            # Keep existing, update exit/expiry guidance
            existing = open_positions[symbol]
            existing.exit_price = position.exit_price
            existing.stop_price = position.stop_price
            existing.expiry_date = _advance_trading_date(
                calendars[symbol],
                trade_date,
                position.hold_expiry_days,
            )
            continue

        bar = _get_bar(market_data[symbol], trade_date)
        if not bar:
            continue

        notional = equity * position.target_alloc * position.leverage
        entry_price = bar["open"] if position.entry_mode in {"market", "ramp"} else position.entry_price
        qty = notional / entry_price if entry_price > 0 else 0.0
        if qty <= 0:
            continue

        if position.entry_mode in {"market", "ramp"}:
            expiry_date = _advance_trading_date(calendars[symbol], trade_date, position.hold_expiry_days)
            open_positions[symbol] = OpenPosition(
                symbol=symbol,
                direction=position.direction,
                qty=qty,
                entry_price=entry_price,
                entry_date=trade_date,
                exit_price=position.exit_price,
                stop_price=position.stop_price,
                expiry_date=expiry_date,
                leverage=position.leverage,
                target_alloc=position.target_alloc,
            )
        else:
            entry_deadline = _advance_trading_date(
                calendars[symbol],
                trade_date,
                max(0, position.entry_expiry_days - 1),
            )
            pending_entries.append(
                PendingEntry(
                    symbol=symbol,
                    direction=position.direction,
                    entry_price=position.entry_price,
                    target_alloc=position.target_alloc,
                    leverage=position.leverage,
                    entry_deadline=entry_deadline,
                    hold_expiry_days=position.hold_expiry_days,
                    exit_price=position.exit_price,
                    stop_price=position.stop_price,
                    entry_mode=position.entry_mode,
                )
            )
    return realized, trades


def _process_pending_entries(
    *,
    day: date,
    market_data: Mapping[str, pd.DataFrame],
    calendars: Mapping[str, List[date]],
    equity: float,
    pending_entries: List[PendingEntry],
    open_positions: Dict[str, OpenPosition],
) -> None:
    for entry in list(pending_entries):
        if day > entry.entry_deadline:
            pending_entries.remove(entry)
            continue
        bar = _get_bar(market_data[entry.symbol], day)
        if not bar:
            continue
        entry_price = entry.entry_price
        if entry.direction == "long":
            hit = bar["low"] <= entry_price <= bar["high"]
        else:
            hit = bar["low"] <= entry_price <= bar["high"]
        if not hit:
            continue

        notional = equity * entry.target_alloc * entry.leverage
        qty = notional / entry_price if entry_price > 0 else 0.0
        if qty <= 0:
            pending_entries.remove(entry)
            continue

        expiry_date = _advance_trading_date(calendars[entry.symbol], day, entry.hold_expiry_days)
        open_positions[entry.symbol] = OpenPosition(
            symbol=entry.symbol,
            direction=entry.direction,
            qty=qty,
            entry_price=entry_price,
            entry_date=day,
            exit_price=entry.exit_price,
            stop_price=entry.stop_price,
            expiry_date=expiry_date,
            leverage=entry.leverage,
            target_alloc=entry.target_alloc,
        )
        pending_entries.remove(entry)


def _process_open_positions(
    *,
    day: date,
    market_data: Mapping[str, pd.DataFrame],
    open_positions: Dict[str, OpenPosition],
    trade_log: List[dict],
) -> float:
    realized = 0.0
    for symbol in list(open_positions.keys()):
        position = open_positions[symbol]
        bar = _get_bar(market_data[symbol], day)
        if not bar:
            continue
        exit_reason = None
        exit_price = None

        if position.direction == "long":
            if position.stop_price is not None and bar["low"] <= position.stop_price:
                exit_reason = "stop"
                exit_price = position.stop_price
            elif bar["high"] >= position.exit_price:
                exit_reason = "target"
                exit_price = position.exit_price
        else:
            if position.stop_price is not None and bar["high"] >= position.stop_price:
                exit_reason = "stop"
                exit_price = position.stop_price
            elif bar["low"] <= position.exit_price:
                exit_reason = "target"
                exit_price = position.exit_price

        if exit_reason is None and day >= position.expiry_date:
            exit_reason = "expiry"
            exit_price = bar["close"]

        if exit_reason and exit_price is not None:
            net_pnl, trade = _exit_position(position, exit_price, day)
            trade["exit_reason"] = exit_reason
            trade["net_pnl"] = net_pnl
            trade_log.append(trade)
            realized += net_pnl
            open_positions.pop(symbol, None)

    return realized


def _compute_unrealized(
    *,
    day: date,
    market_data: Mapping[str, pd.DataFrame],
    open_positions: Dict[str, OpenPosition],
) -> float:
    unrealized = 0.0
    for symbol, position in open_positions.items():
        bar = _get_bar(market_data[symbol], day)
        if not bar:
            continue
        close_price = bar["close"]
        if position.direction == "long":
            unrealized += (close_price - position.entry_price) * position.qty
        else:
            unrealized += (position.entry_price - close_price) * position.qty
    return unrealized


async def _run_backtest_async(
    *,
    symbols: Sequence[str],
    start_date: date,
    end_date: date,
    initial_capital: float = 100_000.0,
    max_lines: int = 160,
    use_thinking: bool = False,
    min_confidence: float = 0.3,
    allow_remote_download: bool = False,
    fallback_days: Optional[int] = None,
) -> dict:
    symbols = [s.upper() for s in symbols]
    lookback = max(400, max_lines + 50)
    bundle = _get_all_data(
        symbols,
        end_date,
        lookback,
        allow_remote_download=allow_remote_download,
    )
    market_data = bundle.bars

    calendars = {symbol: _trading_calendar(df) for symbol, df in market_data.items()}
    trading_days = sorted(
        {
            d
            for calendar in calendars.values()
            for d in calendar
            if start_date <= d <= end_date
        }
    )
    if len(trading_days) < 2:
        all_dates = sorted({d for calendar in calendars.values() for d in calendar})
        if len(all_dates) >= 2:
            end_date = all_dates[-1]
            if fallback_days is None:
                fallback_days = 20
            start_idx = max(0, len(all_dates) - max(2, fallback_days))
            start_date = all_dates[start_idx]
            trading_days = [d for d in all_dates if start_date <= d <= end_date]
            logger.warning(
                "Adjusted backtest window to available data: {} -> {} ({} days)",
                start_date,
                end_date,
                len(trading_days),
            )
        if len(trading_days) < 2:
            raise RuntimeError("Not enough trading days in range")

    equity = initial_capital
    realized_pnl = 0.0
    trade_log: List[dict] = []
    equity_curve: List[dict] = []
    pending_entries: List[PendingEntry] = []
    open_positions: Dict[str, OpenPosition] = {}

    client = anthropic.AsyncAnthropic(api_key=_get_api_key())
    try:
        for day in trading_days:
            realized_today = _process_open_positions(
                day=day,
                market_data=market_data,
                open_positions=open_positions,
                trade_log=trade_log,
            )
            realized_pnl += realized_today
            equity += realized_today

            _process_pending_entries(
                day=day,
                market_data=market_data,
                calendars=calendars,
                equity=equity,
                pending_entries=pending_entries,
                open_positions=open_positions,
            )

            active_symbols = [sym for sym in symbols if day in calendars.get(sym, [])]
            if not active_symbols:
                unrealized = _compute_unrealized(day=day, market_data=market_data, open_positions=open_positions)
                equity = initial_capital + realized_pnl + unrealized
                equity_curve.append({"date": day.isoformat(), "equity": equity})
                continue
            current_positions = _build_current_positions(open_positions, day)
            filtered_data = {
                symbol: df[df.index.date <= day].copy()
                for symbol, df in market_data.items()
                if symbol in active_symbols
            }

            plan = await async_generate_trade_plan(
                symbols=active_symbols,
                market_data=filtered_data,
                portfolio_value=equity,
                current_positions=current_positions,
                max_lines=max_lines,
                use_thinking=use_thinking,
                client=client,
                close_client=False,
            )

            if plan.overall_confidence >= min_confidence:
                rebalance_realized, rebalance_trades = _apply_plan(
                    plan,
                    trade_date=day,
                    market_data=market_data,
                    calendars=calendars,
                    equity=equity,
                    open_positions=open_positions,
                    pending_entries=pending_entries,
                )
                if rebalance_trades:
                    trade_log.extend(rebalance_trades)
                realized_pnl += rebalance_realized
                equity += rebalance_realized
            else:
                logger.info(
                    "Skipping plan on {} due to low confidence {:.2f} < {:.2f}",
                    day,
                    plan.overall_confidence,
                    min_confidence,
                )

            unrealized = _compute_unrealized(day=day, market_data=market_data, open_positions=open_positions)
            equity = initial_capital + realized_pnl + unrealized
            equity_curve.append({"date": day.isoformat(), "equity": equity})
    finally:
        try:
            await client.close()
        except Exception as exc:  # pragma: no cover
            logger.debug("Failed closing Anthropic client: {}", exc)

    total_return = (equity - initial_capital) / initial_capital
    num_days = len(equity_curve)
    annualized_252 = (1 + total_return) ** (252 / num_days) - 1 if num_days > 0 else 0.0
    annualized_365 = (1 + total_return) ** (365 / num_days) - 1 if num_days > 0 else 0.0
    daily_returns: List[float] = []
    prev_equity = initial_capital
    for point in equity_curve:
        eq = float(point["equity"])
        if prev_equity > 0:
            daily_returns.append(eq / prev_equity - 1.0)
        prev_equity = eq
    avg_daily_return = sum(daily_returns) / len(daily_returns) if daily_returns else 0.0

    has_crypto = any(is_crypto_symbol(sym) for sym in symbols)
    has_stock = any(not is_crypto_symbol(sym) for sym in symbols)
    if has_crypto and not has_stock:
        annualized_primary = annualized_365
    else:
        annualized_primary = annualized_252

    summary = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "trading_days": num_days,
        "initial_capital": initial_capital,
        "final_equity": equity,
        "total_return": total_return,
        "avg_daily_return": avg_daily_return,
        "annualized_return": annualized_primary,
        "annualized_return_252": annualized_252,
        "annualized_return_365": annualized_365,
        "trade_count": len(trade_log),
        "trades": trade_log,
        "equity_curve": equity_curve,
    }
    return summary


def run_backtest(
    *,
    symbols: Sequence[str],
    start_date: date,
    end_date: date,
    initial_capital: float = 100_000.0,
    max_lines: int = 160,
    use_thinking: bool = False,
    min_confidence: float = 0.3,
    allow_remote_download: bool = False,
    fallback_days: Optional[int] = None,
) -> dict:
    return asyncio.run(
        _run_backtest_async(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            max_lines=max_lines,
            use_thinking=use_thinking,
            min_confidence=min_confidence,
            allow_remote_download=allow_remote_download,
            fallback_days=fallback_days,
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest stockagent3 strategy")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols")
    parser.add_argument("--days", type=int, default=20)
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--max-lines", type=int, default=160)
    parser.add_argument("--use-thinking", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--allow-remote", action="store_true", help="Allow remote OHLC downloads when local data is missing")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        raise SystemExit("No symbols provided")

    end_date = date.today()
    start_date = end_date - timedelta(days=max(1, args.days))

    result = run_backtest(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.initial_capital,
        max_lines=args.max_lines,
        use_thinking=args.use_thinking,
        min_confidence=args.min_confidence,
        allow_remote_download=args.allow_remote,
        fallback_days=args.days,
    )

    logger.info("Backtest complete: total_return={:.2f}%", result["total_return"] * 100)
    logger.info("Avg daily return: {:.3f}%", result["avg_daily_return"] * 100)
    logger.info("Annualized (primary) = {:.2f}%", result["annualized_return"] * 100)
    logger.info("Annualized (252d) = {:.2f}%", result["annualized_return_252"] * 100)
    logger.info("Annualized (365d) = {:.2f}%", result["annualized_return_365"] * 100)


if __name__ == "__main__":
    main()
