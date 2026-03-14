#!/usr/bin/env python3
"""Live Alpaca stock trader with per-symbol meta strategy selection."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import torch
from loguru import logger

import unified_hourly_experiment.trade_unified_hourly as live
from binanceneural.config import DatasetConfig
from binanceneural.data import BinanceHourlyDataModule
from binanceneural.inference import generate_actions_from_frame, generate_latest_action
from src.hourly_trader_utils import entry_intensity_fraction
from src.trade_directions import DEFAULT_ALPACA_LIVE8_STOCKS
from unified_hourly_experiment.marketsimulator import PortfolioConfig, run_portfolio_simulation
from unified_hourly_experiment.meta_live_runtime import choose_latest_winner, compute_symbol_edge
from unified_hourly_experiment.meta_selector import daily_returns_from_equity
from unified_hourly_experiment.sweep_meta_portfolio import load_strategy_model, parse_strategy_spec


@dataclass
class SymbolStrategyPayload:
    bars: pd.DataFrame
    actions: pd.DataFrame
    latest_action: dict


@dataclass
class MetaSelectionCache:
    selection_day: date | None = None
    winners: dict[str, str | None] | None = None


def parse_csv_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def apply_live_sizing_overrides(args: argparse.Namespace) -> None:
    live.TRADE_AMOUNT_SCALE = float(args.trade_amount_scale)
    live.MIN_BUY_AMOUNT = float(args.min_buy_amount)
    live.ENTRY_INTENSITY_POWER = float(args.entry_intensity_power)
    live.ENTRY_MIN_INTENSITY_FRACTION = float(args.entry_min_intensity_fraction)
    live.LONG_INTENSITY_MULTIPLIER = float(args.long_intensity_multiplier)
    live.SHORT_INTENSITY_MULTIPLIER = float(args.short_intensity_multiplier)


def _load_symbol_frame(
    *,
    symbol: str,
    strategy,
    data_root: Path,
    cache_root: Path,
    history_days: int,
) -> tuple[BinanceHourlyDataModule, pd.DataFrame] | None:
    cfg = DatasetConfig(
        symbol=symbol,
        data_root=str(data_root),
        forecast_cache_root=str(cache_root),
        forecast_horizons=list(strategy.horizons),
        sequence_length=int(strategy.sequence_length),
        min_history_hours=100,
        validation_days=30,
        cache_only=True,
    )
    try:
        dm = BinanceHourlyDataModule(cfg)
    except Exception as exc:
        logger.warning("{} {}: failed to load data module ({})", strategy.name, symbol, exc)
        return None

    frame = dm.frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["symbol"] = symbol
    if history_days > 0 and not frame.empty:
        cutoff = frame["timestamp"].max() - pd.Timedelta(days=int(history_days))
        frame = frame[frame["timestamp"] >= cutoff].reset_index(drop=True)

    if len(frame) < int(strategy.sequence_length):
        logger.warning(
            "{} {}: frame too short after history cutoff ({} < seq={})",
            strategy.name,
            symbol,
            len(frame),
            strategy.sequence_length,
        )
        return None
    return dm, frame


def load_symbol_payload(
    *,
    symbol: str,
    strategy,
    data_root: Path,
    cache_root: Path,
    device: torch.device,
    history_days: int,
) -> SymbolStrategyPayload | None:
    loaded = _load_symbol_frame(
        symbol=symbol,
        strategy=strategy,
        data_root=data_root,
        cache_root=cache_root,
        history_days=history_days,
    )
    if loaded is None:
        return None
    dm, frame = loaded

    normalizer = strategy.normalizer if strategy.normalizer is not None else dm.normalizer
    try:
        actions = generate_actions_from_frame(
            model=strategy.model,
            frame=frame,
            feature_columns=strategy.feature_columns,
            normalizer=normalizer,
            sequence_length=int(strategy.sequence_length),
            horizon=1,
            device=device,
        )
    except Exception as exc:
        logger.warning("{} {}: action generation failed ({})", strategy.name, symbol, exc)
        return None

    if actions.empty:
        logger.warning("{} {}: no generated actions", strategy.name, symbol)
        return None

    actions = actions.copy()
    actions["timestamp"] = pd.to_datetime(actions["timestamp"], utc=True)
    actions["symbol"] = actions["symbol"].astype(str).str.upper()

    latest_action = actions.iloc[-1].to_dict()
    last_bar = frame.iloc[-1]
    latest_action["predicted_high"] = float(last_bar.get("predicted_high_p50_h1", 0.0))
    latest_action["predicted_low"] = float(last_bar.get("predicted_low_p50_h1", 0.0))
    latest_action["predicted_close"] = float(last_bar.get("predicted_close_p50_h1", 0.0))
    latest_action["timestamp"] = last_bar["timestamp"]
    latest_action["symbol"] = symbol

    return SymbolStrategyPayload(bars=frame, actions=actions, latest_action=latest_action)


def load_symbol_latest_action(
    *,
    symbol: str,
    strategy,
    data_root: Path,
    cache_root: Path,
    device: torch.device,
    history_days: int,
) -> dict | None:
    loaded = _load_symbol_frame(
        symbol=symbol,
        strategy=strategy,
        data_root=data_root,
        cache_root=cache_root,
        history_days=history_days,
    )
    if loaded is None:
        return None
    dm, frame = loaded

    normalizer = strategy.normalizer if strategy.normalizer is not None else dm.normalizer
    try:
        action = generate_latest_action(
            model=strategy.model,
            frame=frame,
            feature_columns=strategy.feature_columns,
            normalizer=normalizer,
            sequence_length=int(strategy.sequence_length),
            horizon=1,
            device=device,
        )
    except Exception as exc:
        logger.warning("{} {}: latest action generation failed ({})", strategy.name, symbol, exc)
        return None

    action = dict(action)
    action["symbol"] = symbol
    return action


def simulate_symbol_daily_returns(
    *,
    symbol: str,
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.Series:
    cfg = PortfolioConfig(
        initial_cash=float(args.meta_sim_initial_cash),
        max_positions=1,
        min_edge=float(args.min_edge),
        max_hold_hours=int(args.max_hold_hours),
        enforce_market_hours=True,
        close_at_eod=not bool(args.no_close_at_eod),
        symbols=[symbol],
        decision_lag_bars=int(args.decision_lag_bars),
        entry_selection_mode=str(args.entry_selection_mode),
        market_order_entry=bool(args.market_order_entry),
        bar_margin=float(args.bar_margin),
        entry_order_ttl_hours=int(args.entry_order_ttl_hours),
        max_leverage=float(args.leverage),
        force_close_slippage=float(args.force_close_slippage),
        int_qty=not bool(args.no_int_qty),
        fee_by_symbol={symbol: float(args.fee_rate)},
        margin_annual_rate=float(args.margin_rate),
    )
    sim = run_portfolio_simulation(bars, actions, cfg, horizon=1)
    return daily_returns_from_equity(sim.equity_curve)


def collect_daily_returns(
    *,
    strategies: Sequence,
    symbols: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, dict[str, pd.Series]]:
    daily_returns: dict[str, dict[str, pd.Series]] = {s.name: {} for s in strategies}

    for strategy in strategies:
        for symbol in symbols:
            payload = load_symbol_payload(
                symbol=symbol,
                strategy=strategy,
                data_root=args.stock_data_root,
                cache_root=args.stock_cache_root,
                device=device,
                history_days=int(args.meta_history_days),
            )
            if payload is None:
                continue
            try:
                returns = simulate_symbol_daily_returns(
                    symbol=symbol,
                    bars=payload.bars,
                    actions=payload.actions,
                    args=args,
                )
            except Exception as exc:
                logger.warning("{} {}: simulation failed ({})", strategy.name, symbol, exc)
                continue
            if returns.empty:
                logger.warning("{} {}: empty daily return history", strategy.name, symbol)
                continue
            daily_returns[strategy.name][symbol] = returns
            live.log_event(
                "meta_daily_returns_computed",
                strategy=strategy.name,
                symbol=symbol,
                observations=int(len(returns)),
                latest_return=float(returns.iloc[-1]) if len(returns) else 0.0,
            )

    return daily_returns


def select_meta_winners(
    *,
    strategies: Sequence,
    symbols: Sequence[str],
    daily_returns: dict[str, dict[str, pd.Series]],
    args: argparse.Namespace,
) -> dict[str, str | None]:
    strategy_names = [s.name for s in strategies]
    tie_break_order = parse_csv_list(args.meta_tie_break_order) if args.meta_tie_break_order else strategy_names
    winners: dict[str, str | None] = {}

    for symbol in symbols:
        available = [name for name in strategy_names if symbol in daily_returns.get(name, {})]
        if not available:
            logger.info("{}: no usable strategies this cycle", symbol)
            winners[symbol] = None
            live.log_event("meta_winner_selected", symbol=symbol, winner=None, reason="no_usable_strategies")
            continue

        returns_for_symbol = {name: daily_returns[name][symbol] for name in available}
        local_tie_break = [name for name in tie_break_order if name in available]
        if not local_tie_break:
            local_tie_break = list(available)

        fallback = args.meta_default_strategy.strip() if args.meta_default_strategy else local_tie_break[0]
        if fallback not in available:
            fallback = local_tie_break[0]

        winner = choose_latest_winner(
            returns_for_symbol,
            lookback_days=int(args.meta_lookback_days),
            metric=args.meta_metric,
            fallback_strategy=fallback,
            tie_break_order=local_tie_break,
            sit_out_threshold=(float(args.sit_out_threshold) if args.sit_out_if_negative else None),
            selection_mode=args.meta_selection_mode,
            switch_margin=float(args.meta_switch_margin),
            min_score_gap=float(args.meta_min_score_gap),
            recency_halflife_days=(
                float(args.meta_recency_halflife_days)
                if float(args.meta_recency_halflife_days) > 0
                else None
            ),
        )
        winners[symbol] = winner
        live.log_event(
            "meta_winner_selected",
            symbol=symbol,
            winner=winner,
            available_strategies=available,
            fallback_strategy=fallback,
        )

    return winners


def build_meta_signals(
    *,
    strategies: Sequence,
    symbols: Sequence[str],
    winners_by_symbol: dict[str, str | None],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, dict]:
    signals: dict[str, dict] = {}
    short_only = set(live.SHORT_ONLY)
    strategies_by_name = {s.name: s for s in strategies}

    for symbol in symbols:
        winner = winners_by_symbol.get(symbol)
        if winner is None:
            logger.info("{}: winner=CASH (sit-out)", symbol)
            live.log_event("meta_signal_skipped", symbol=symbol, reason="winner_cash")
            continue

        strategy = strategies_by_name.get(winner)
        if strategy is None:
            logger.warning("{}: winner {} not found in loaded strategies", symbol, winner)
            live.log_event("meta_signal_skipped", symbol=symbol, reason="winner_not_loaded", winner=winner)
            continue

        action = load_symbol_latest_action(
            symbol=symbol,
            strategy=strategy,
            data_root=args.stock_data_root,
            cache_root=args.stock_cache_root,
            device=device,
            history_days=int(args.meta_history_days),
        )
        if action is None:
            live.log_event("meta_signal_skipped", symbol=symbol, reason="latest_action_missing", winner=winner)
            continue

        market_entry_reference_price, market_entry_reference_source = live.resolve_live_entry_reference_price(
            symbol,
            default_price=float(action.get("sell_price" if symbol in short_only else "buy_price", 0.0) or 0.0),
            is_short=(symbol in short_only),
            use_market_orders=bool(args.market_order_entry),
        )
        edge = compute_symbol_edge(
            symbol=symbol,
            action=action,
            fee_rate=float(args.fee_rate),
            short_only_symbols=short_only,
            entry_reference_price=market_entry_reference_price,
        )
        if edge < float(args.min_edge):
            logger.info("{}: winner={} edge={:.4f} below {:.4f}", symbol, winner, edge, args.min_edge)
            live.log_event(
                "meta_signal_skipped",
                symbol=symbol,
                winner=winner,
                reason="edge_below_threshold",
                edge=float(edge),
                min_edge=float(args.min_edge),
                market_entry_reference_price=float(market_entry_reference_price),
                market_entry_reference_source=str(market_entry_reference_source),
            )
            continue

        action["edge"] = float(edge)
        action["meta_strategy"] = winner
        action["market_entry_reference_price"] = float(market_entry_reference_price)
        action["market_entry_reference_source"] = str(market_entry_reference_source)
        signals[symbol] = action
        side = "short" if symbol in short_only else "long"
        _, intensity = entry_intensity_fraction(
            action,
            is_short=(symbol in short_only),
            trade_amount_scale=float(args.trade_amount_scale),
            intensity_power=float(args.entry_intensity_power),
            min_intensity_fraction=float(args.entry_min_intensity_fraction),
            side_multiplier=(
                float(args.short_intensity_multiplier)
                if (symbol in short_only)
                else float(args.long_intensity_multiplier)
            ),
        )
        logger.info(
            "{}: winner={} {} edge={:.4f} buy={:.2f} sell={:.2f} ref={:.2f} hold={:.1f}h int={:.3f}",
            symbol,
            winner,
            side,
            edge,
            float(action.get("buy_price", 0.0)),
            float(action.get("sell_price", 0.0)),
            float(market_entry_reference_price),
            float(action.get("hold_hours", args.max_hold_hours)),
            intensity,
        )
        live.log_event(
            "meta_signal_ready",
            symbol=symbol,
            winner=winner,
            side=side,
            edge=float(edge),
            buy_price=float(action.get("buy_price", 0.0)),
            sell_price=float(action.get("sell_price", 0.0)),
            market_entry_reference_price=float(market_entry_reference_price),
            market_entry_reference_source=str(market_entry_reference_source),
            hold_hours=float(action.get("hold_hours", args.max_hold_hours)),
            intensity=float(intensity),
        )

    return signals


def run_cycle(
    *,
    strategies: Sequence,
    symbols: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
    api,
    state: dict,
    selection_cache: MetaSelectionCache,
) -> None:
    active_set = set(symbols)
    market_open = live.is_market_open_now() or args.ignore_market_hours
    live.log_event(
        "meta_cycle_start",
        symbols=sorted(str(symbol) for symbol in symbols),
        market_open=bool(market_open),
        dry_run=bool(args.dry_run),
        reselect_frequency=args.meta_reselect_frequency,
        selection_cache_day=str(selection_cache.selection_day) if selection_cache.selection_day is not None else None,
    )

    if api is not None:
        live.poll_broker_events(api, state, reason="pre_cycle")

    if api is not None:
        num_pos = live.manage_positions(
            api,
            state,
            max_hold_hours=int(args.max_hold_hours),
            active_symbols=active_set,
        )
    else:
        num_pos = len(state.get("positions", {}))
    logger.info("Active positions: {} | Market: {}", num_pos, "OPEN" if market_open else "CLOSED")

    today_utc = datetime.now(UTC).date()
    should_reselect = (
        args.meta_reselect_frequency == "hourly"
        or selection_cache.selection_day != today_utc
        or not selection_cache.winners
    )
    if should_reselect:
        logger.info("Recomputing meta winners (frequency={} date={})", args.meta_reselect_frequency, today_utc)
        daily_returns = collect_daily_returns(
            strategies=strategies,
            symbols=symbols,
            args=args,
            device=device,
        )
        winners = select_meta_winners(
            strategies=strategies,
            symbols=symbols,
            daily_returns=daily_returns,
            args=args,
        )
        selection_cache.selection_day = today_utc
        selection_cache.winners = winners
        live.log_event(
            "meta_winner_cache_refreshed",
            selection_day=str(today_utc),
            winners=winners,
        )
    else:
        winners = dict(selection_cache.winners or {})
        logger.info("Using cached meta winners from {}", selection_cache.selection_day)
        live.log_event(
            "meta_winner_cache_used",
            selection_day=str(selection_cache.selection_day),
            winners=winners,
        )

    signals = build_meta_signals(
        strategies=strategies,
        symbols=symbols,
        winners_by_symbol=winners,
        args=args,
        device=device,
    )

    if market_open and not args.dry_run and signals:
        live.execute_trades(
            api,
            signals,
            state,
            max_positions=int(args.max_positions),
            market_order_entry=bool(args.market_order_entry),
            entry_order_ttl_hours=float(args.entry_order_ttl_hours),
            fee_rate=float(args.fee_rate),
        )
        execution_mode = "live_execute"
    elif not market_open and signals:
        logger.info("Market closed - {} signals ready, will trade when open", len(signals))
        execution_mode = "market_closed"
    elif not signals:
        logger.info("No meta signals above threshold")
        execution_mode = "no_signals"
    else:
        execution_mode = "dry_run"

    if api is not None:
        live.poll_broker_events(api, state, reason="post_cycle")
    live.save_state(state)
    live.log_event(
        "meta_cycle_complete",
        execution_mode=execution_mode,
        winners=winners,
        signal_symbols=sorted(str(symbol) for symbol in signals.keys()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        action="append",
        required=True,
        help="Meta strategy spec NAME=PATH or NAME=PATH:EPOCH (repeatable).",
    )
    parser.add_argument("--stock-symbols", default=",".join(DEFAULT_ALPACA_LIVE8_STOCKS))
    parser.add_argument("--stock-data-root", type=Path, default=Path("trainingdatahourly/stocks"))
    parser.add_argument("--stock-cache-root", type=Path, default=Path("unified_hourly_experiment/forecast_cache"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--ignore-market-hours", action="store_true")
    parser.add_argument("--min-edge", type=float, default=0.006)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-hold-hours", type=int, default=6)
    parser.add_argument("--trade-amount-scale", type=float, default=100.0)
    parser.add_argument("--min-buy-amount", type=float, default=0.0)
    parser.add_argument("--entry-intensity-power", type=float, default=1.0)
    parser.add_argument("--entry-min-intensity-fraction", type=float, default=0.0)
    parser.add_argument("--long-intensity-multiplier", type=float, default=1.0)
    parser.add_argument("--short-intensity-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--meta-metric",
        default="sharpe",
        choices=["return", "sortino", "sharpe", "calmar", "omega", "gain_pain", "p10", "median", "goodness"],
    )
    parser.add_argument("--meta-lookback-days", type=int, default=7)
    parser.add_argument("--meta-history-days", type=int, default=120)
    parser.add_argument("--meta-default-strategy", default="")
    parser.add_argument("--meta-tie-break-order", default="")
    parser.add_argument("--meta-selection-mode", choices=["winner", "winner_cash", "sticky"], default="winner")
    parser.add_argument("--meta-switch-margin", type=float, default=0.0)
    parser.add_argument("--meta-min-score-gap", type=float, default=0.0)
    parser.add_argument(
        "--meta-recency-halflife-days",
        type=float,
        default=0.0,
        help="Exponential recency half-life in days for meta scoring (<=0 disables weighting).",
    )
    parser.add_argument("--sit-out-if-negative", action="store_true")
    parser.add_argument("--sit-out-threshold", type=float, default=0.7)
    parser.add_argument("--meta-reselect-frequency", choices=["daily", "hourly"], default="daily")
    parser.add_argument("--meta-sim-initial-cash", type=float, default=10_000.0)
    parser.add_argument("--decision-lag-bars", type=int, default=1)
    parser.add_argument(
        "--entry-selection-mode",
        default="edge_rank",
        choices=["edge_rank", "first_trigger"],
        help="How selector simulations prioritize competing fillable entries for limit-style fills.",
    )
    parser.add_argument(
        "--market-order-entry",
        action="store_true",
        help="Use market-order entry fill assumption in selector simulations (recommended when live uses market entries).",
    )
    parser.add_argument("--bar-margin", type=float, default=0.0013)
    parser.add_argument(
        "--entry-order-ttl-hours",
        type=float,
        default=6.0,
        help="How many hourly bars non-filled entry orders stay pending in selector simulations (0 disables).",
    )
    parser.add_argument("--leverage", type=float, default=2.0)
    parser.add_argument("--force-close-slippage", type=float, default=0.003)
    parser.add_argument("--margin-rate", type=float, default=0.0625)
    parser.add_argument("--no-int-qty", action="store_true")
    parser.add_argument("--no-close-at-eod", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if len(args.strategy) < 2:
        raise ValueError("At least two --strategy specs are required.")
    if args.meta_lookback_days <= 0:
        raise ValueError("--meta-lookback-days must be > 0")
    if args.meta_switch_margin < 0:
        raise ValueError("--meta-switch-margin must be >= 0")
    if args.meta_min_score_gap < 0:
        raise ValueError("--meta-min-score-gap must be >= 0")
    if args.meta_recency_halflife_days < 0:
        raise ValueError("--meta-recency-halflife-days must be >= 0")
    if args.trade_amount_scale <= 0:
        raise ValueError("--trade-amount-scale must be > 0")
    if args.min_buy_amount < 0:
        raise ValueError("--min-buy-amount must be >= 0")
    if args.entry_intensity_power < 0:
        raise ValueError("--entry-intensity-power must be >= 0")
    if args.entry_min_intensity_fraction < 0:
        raise ValueError("--entry-min-intensity-fraction must be >= 0")
    if args.long_intensity_multiplier < 0:
        raise ValueError("--long-intensity-multiplier must be >= 0")
    if args.short_intensity_multiplier < 0:
        raise ValueError("--short-intensity-multiplier must be >= 0")

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but torch.cuda.is_available() is false.")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    strategies = []
    for spec in args.strategy:
        name, path, epoch = parse_strategy_spec(spec)
        strategies.append(load_strategy_model(name, path, epoch=epoch, device=device))

    names = [s.name for s in strategies]
    if len(names) != len(set(names)):
        raise ValueError(f"Strategy names must be unique. Got {names}")

    stocks = [s.strip().upper() for s in args.stock_symbols.split(",") if s.strip()]
    if not stocks:
        raise ValueError("No stock symbols were provided.")

    apply_live_sizing_overrides(args)

    paper = not args.live
    api = None if args.dry_run else live.get_alpaca_client(paper=paper)
    state = live.load_state()

    logger.info("=" * 60)
    logger.info("Meta Portfolio Stock Trading Bot")
    logger.info("=" * 60)
    logger.info("Strategies: {}", ", ".join(names))
    logger.info("Symbols: {}", ", ".join(stocks))
    logger.info(
        "Meta selector: metric={} lookback={}d mode={} switch_margin={:.4f} min_gap={:.4f} recency_halflife={} sitout={} threshold={} reselect={} entry_selection_mode={}",
        args.meta_metric,
        args.meta_lookback_days,
        args.meta_selection_mode,
        float(args.meta_switch_margin),
        float(args.meta_min_score_gap),
        float(args.meta_recency_halflife_days),
        bool(args.sit_out_if_negative),
        float(args.sit_out_threshold),
        args.meta_reselect_frequency,
        args.entry_selection_mode,
    )
    logger.info("Max positions: {}, Hold limit: {}h", args.max_positions, args.max_hold_hours)
    logger.info(
        "Sizing: scale={} min_buy={} power={} min_intensity={} long_mult={} short_mult={}",
        float(args.trade_amount_scale),
        float(args.min_buy_amount),
        float(args.entry_intensity_power),
        float(args.entry_min_intensity_fraction),
        float(args.long_intensity_multiplier),
        float(args.short_intensity_multiplier),
    )
    logger.info("Selector sim market-order-entry: {}", bool(args.market_order_entry))
    logger.info("Mode: {}", "DRY-RUN" if args.dry_run else ("LIVE" if not paper else "PAPER"))
    live.log_event(
        "meta_trader_started",
        strategies=names,
        symbols=stocks,
        paper=bool(paper),
        dry_run=bool(args.dry_run),
        loop=bool(args.loop),
        meta_metric=args.meta_metric,
        meta_lookback_days=int(args.meta_lookback_days),
        entry_selection_mode=str(args.entry_selection_mode),
    )

    if api is not None:
        account = live.get_account_info(api)
        logger.info("Equity: ${:.2f}, Buying power: ${:.2f}", account["equity"], account["buying_power"])

    selection_cache = MetaSelectionCache()
    run_cycle(
        strategies=strategies,
        symbols=stocks,
        args=args,
        device=device,
        api=api,
        state=state,
        selection_cache=selection_cache,
    )

    if not args.loop:
        return

    while True:
        now = datetime.now(UTC)
        next_hour = (now + timedelta(hours=1)).replace(minute=1, second=0, microsecond=0)
        wait_secs = (next_hour - now).total_seconds()
        logger.info("Sleeping {:.0f}s until next hour", wait_secs)
        time.sleep(wait_secs)
        run_cycle(
            strategies=strategies,
            symbols=stocks,
            args=args,
            device=device,
            api=api,
            state=state,
            selection_cache=selection_cache,
        )


if __name__ == "__main__":
    main()
