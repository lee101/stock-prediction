#!/usr/bin/env python3
"""
Walk-forward backtest for Bags.fm OHLC data using Chronos2 forecasts.

Defaults to the CODEX (Bags.fm) token and evaluates the most recent 5 days.
Optionally performs a simple threshold "training" step on a prior window.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from bagsfm import (
    BagsConfig,
    DataConfig,
    ForecastConfig,
    DataCollector,
    MarketSimulator,
    SimulationConfig,
    TokenConfig,
    TokenForecast,
    TokenForecaster,
    build_forecast_cache,
    forecast_threshold_strategy,
    daily_high_low_strategy,
)
from bagsfm.config import CODEX_MINT
from bagsfm.data_collector import OHLCBar


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid datetime: {value}") from exc


def _parse_thresholds(value: str) -> List[float]:
    thresholds = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        thresholds.append(float(part))
    if not thresholds:
        raise argparse.ArgumentTypeError("At least one threshold is required.")
    return thresholds


def _load_ohlc_bars(path: Path, token_mints: Optional[Iterable[str]] = None) -> Dict[str, List[OHLCBar]]:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Missing timestamp column in {path}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    mints = set(token_mints) if token_mints is not None else None
    bars: Dict[str, List[OHLCBar]] = {}

    for row in df.itertuples(index=False):
        mint = getattr(row, "token_mint")
        if mints is not None and mint not in mints:
            continue

        bar = OHLCBar(
            timestamp=getattr(row, "timestamp").to_pydatetime(),
            token_mint=mint,
            token_symbol=getattr(row, "token_symbol"),
            open=float(getattr(row, "open")),
            high=float(getattr(row, "high")),
            low=float(getattr(row, "low")),
            close=float(getattr(row, "close")),
            volume=float(getattr(row, "volume", 0.0) or 0.0),
            num_ticks=int(getattr(row, "num_ticks", 0) or 0),
        )
        bars.setdefault(mint, []).append(bar)

    for mint, mint_bars in bars.items():
        mint_bars.sort(key=lambda b: b.timestamp)

    return bars


def _filter_bars(
    bars: Dict[str, List[OHLCBar]],
    start: Optional[datetime],
    end: Optional[datetime],
) -> Dict[str, List[OHLCBar]]:
    if start is None and end is None:
        return {mint: list(mint_bars) for mint, mint_bars in bars.items()}

    filtered: Dict[str, List[OHLCBar]] = {}
    for mint, mint_bars in bars.items():
        kept = [
            bar
            for bar in mint_bars
            if (start is None or bar.timestamp >= start)
            and (end is None or bar.timestamp <= end)
        ]
        if kept:
            filtered[mint] = kept
    return filtered


def _infer_interval_minutes(bars: Sequence[OHLCBar]) -> Optional[float]:
    if len(bars) < 2:
        return None
    diffs = []
    for prev, curr in zip(bars, bars[1:]):
        delta = (curr.timestamp - prev.timestamp).total_seconds() / 60.0
        if delta > 0:
            diffs.append(delta)
        if len(diffs) >= 500:
            break
    if not diffs:
        return None
    return float(pd.Series(diffs).median())


def _filter_forecast_cache(
    cache: Dict[datetime, Dict[str, TokenForecast]],
    start: Optional[datetime],
    end: Optional[datetime],
) -> Dict[datetime, Dict[str, TokenForecast]]:
    if start is None and end is None:
        return dict(cache)
    filtered: Dict[datetime, Dict[str, TokenForecast]] = {}
    for ts, forecasts in cache.items():
        if start is not None and ts < start:
            continue
        if end is not None and ts > end:
            continue
        filtered[ts] = forecasts
    return filtered


def _build_token_configs(
    bars: Dict[str, List[OHLCBar]],
    decimals: int,
    entry_fee_bps: float = 0.0,
    exit_fee_bps: float = 0.0,
    spread_bps: float = 0.0,
    creator_rebate_bps: float = 0.0,
) -> Dict[str, TokenConfig]:
    tokens: Dict[str, TokenConfig] = {}
    for mint, mint_bars in bars.items():
        symbol = mint_bars[-1].token_symbol if mint_bars else mint[:6]
        tokens[mint] = TokenConfig(
            symbol=symbol,
            mint=mint,
            decimals=decimals,
            name=symbol,
            min_trade_amount=0.0,
            entry_fee_bps=entry_fee_bps,
            exit_fee_bps=exit_fee_bps,
            spread_bps=spread_bps,
            creator_rebate_bps=creator_rebate_bps,
        )
    return tokens


def _run_simulation(
    bars: Dict[str, List[OHLCBar]],
    tokens: Dict[str, TokenConfig],
    forecaster: Optional[TokenForecaster],
    forecast_cache: Optional[Dict[datetime, Dict[str, TokenForecast]]],
    threshold: float,
    context_bars: int,
    min_context_bars: int,
    sim_config: SimulationConfig,
) -> tuple[float, object]:
    simulator = MarketSimulator(sim_config)
    strategy = forecast_threshold_strategy(
        min_return=threshold, max_drawdown_return=-threshold
    )
    result = simulator.run_walk_forward_backtest(
        bars=bars,
        tokens=tokens,
        strategy_fn=strategy,
        forecaster=forecaster,
        forecast_cache=forecast_cache,
        context_bars=context_bars,
        min_context_bars=min_context_bars,
    )
    return result.total_return_pct, result


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Bags.fm Chronos2 strategy")
    parser.add_argument(
        "--ohlc",
        type=Path,
        default=Path("bagstraining") / "ohlc_data.csv",
        help="Path to OHLC CSV [default: bagstraining/ohlc_data.csv]",
    )
    parser.add_argument(
        "--mints",
        type=str,
        default=CODEX_MINT,
        help="Comma-separated token mint list [default: CODEX]",
    )
    parser.add_argument(
        "--days",
        type=float,
        default=5.0,
        help="Evaluation window in days [default: 5]",
    )
    parser.add_argument(
        "--train-days",
        type=float,
        default=30.0,
        help="Training window in days before eval [default: 30]",
    )
    parser.add_argument(
        "--thresholds",
        type=_parse_thresholds,
        default=_parse_thresholds("0.0025,0.005,0.01,0.02"),
        help="Comma-separated threshold grid for training",
    )
    parser.add_argument(
        "--context-bars",
        type=int,
        default=256,
        help="Context bars for forecasting [default: 256]",
    )
    parser.add_argument(
        "--min-context-bars",
        type=int,
        default=20,
        help="Minimum bars needed to forecast [default: 20]",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=9,
        help="Token decimals [default: 9]",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="forecast",
        choices=["forecast", "daily-range"],
        help="Strategy: forecast (Chronos2) or daily-range [default: forecast]",
    )
    parser.add_argument(
        "--min-range-bps",
        type=float,
        default=100.0,
        help="Minimum prior-day range (bps) to trade [default: 100]",
    )
    parser.add_argument(
        "--max-actions-per-day",
        type=int,
        default=2,
        help="Max actions per token per day for daily-range [default: 2]",
    )
    parser.add_argument(
        "--max-trades-per-day",
        type=int,
        default=None,
        help="Max entry trades per day (global) [default: None]",
    )
    parser.add_argument(
        "--entry-fee-bps",
        type=float,
        default=0.0,
        help="Entry fee bps applied in simulator [default: 0]",
    )
    parser.add_argument(
        "--exit-fee-bps",
        type=float,
        default=0.0,
        help="Exit fee bps applied in simulator [default: 0]",
    )
    parser.add_argument(
        "--spread-bps",
        type=float,
        default=0.0,
        help="Bid/ask spread bps applied in simulator [default: 0]",
    )
    parser.add_argument(
        "--creator-rebate-bps",
        type=float,
        default=0.0,
        help="Creator rebate bps credited in simulator [default: 0]",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Chronos2 device_map [default: cuda]",
    )
    parser.add_argument(
        "--end",
        type=_parse_datetime,
        default=None,
        help="End datetime (ISO format) [default: latest bar]",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level [default: INFO]",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    mint_list = [m.strip() for m in args.mints.split(",") if m.strip()]
    bars = _load_ohlc_bars(args.ohlc, token_mints=mint_list if mint_list else None)
    if not bars:
        raise SystemExit("No OHLC bars found for requested tokens.")

    latest_bar = max(bar.timestamp for mint_bars in bars.values() for bar in mint_bars)
    end_time = args.end or latest_bar
    eval_start = end_time - timedelta(days=args.days)

    train_end = eval_start
    train_start = train_end - timedelta(days=args.train_days) if args.train_days else None

    tokens = _build_token_configs(
        bars,
        decimals=args.decimals,
        entry_fee_bps=args.entry_fee_bps,
        exit_fee_bps=args.exit_fee_bps,
        spread_bps=args.spread_bps,
        creator_rebate_bps=args.creator_rebate_bps,
    )

    sim_config = SimulationConfig(
        initial_sol=1.0,
        max_trades_per_day=args.max_trades_per_day,
    )

    sample_bars = next(iter(bars.values()))
    interval_minutes = _infer_interval_minutes(sample_bars)
    warmup_start = None
    if interval_minutes:
        anchor = train_start if train_start is not None else eval_start
        warmup_start = anchor - timedelta(minutes=interval_minutes * args.context_bars)

    eval_bars = _filter_bars(bars, eval_start, end_time)
    level_bars = _filter_bars(bars, eval_start - timedelta(days=1), end_time)

    if args.strategy == "daily-range":
        strategy = daily_high_low_strategy(
            bars=level_bars,
            min_range_bps=args.min_range_bps,
            max_actions_per_day=args.max_actions_per_day,
            use_previous_day=True,
        )
        simulator = MarketSimulator(sim_config)
        logging.info("Evaluating daily-range from %s to %s", eval_start.isoformat(), end_time.isoformat())
        result = simulator.run_walk_forward_backtest(
            bars=eval_bars,
            tokens=tokens,
            strategy_fn=strategy,
            forecaster=None,
            forecast_cache=None,
            context_bars=None,
            min_context_bars=args.min_context_bars,
        )
        total_return = result.total_return_pct
        best_threshold = None
    else:
        forecast_config = ForecastConfig(
            context_length=args.context_bars,
            prediction_length=6,
            device_map=args.device,
        )
        data_config = DataConfig(
            tracked_tokens=list(tokens.values()),
            data_dir=args.ohlc.parent,
        )
        bags_config = BagsConfig()

        collector = DataCollector(bags_config, data_config)
        forecaster = TokenForecaster(
            data_collector=collector,
            config=forecast_config,
        )

        cache_bars = _filter_bars(bars, warmup_start, end_time)
        cache = build_forecast_cache(
            bars=cache_bars,
            tokens=tokens,
            forecaster=forecaster,
            context_bars=args.context_bars,
            min_context_bars=args.min_context_bars,
        )

        train_bars = _filter_bars(bars, train_start, train_end)
        train_cache = _filter_forecast_cache(cache, train_start, train_end)

        best_threshold = args.thresholds[0]
        best_return = float("-inf")

        if args.train_days and len(args.thresholds) > 1 and train_bars:
            logging.info(
                "Training thresholds on %s to %s",
                train_start.isoformat() if train_start else "start",
                train_end.isoformat(),
            )
            for threshold in args.thresholds:
                total_return, _ = _run_simulation(
                    bars=train_bars,
                    tokens=tokens,
                    forecaster=None,
                    forecast_cache=train_cache,
                    threshold=threshold,
                    context_bars=args.context_bars,
                    min_context_bars=args.min_context_bars,
                    sim_config=sim_config,
                )
                logging.info("Threshold %.4f -> return %.2f%%", threshold, total_return)
                if total_return > best_return:
                    best_return = total_return
                    best_threshold = threshold

            logging.info("Selected threshold %.4f", best_threshold)
        else:
            if args.train_days and not train_bars:
                logging.warning(
                    "No training bars found between %s and %s; skipping threshold training.",
                    train_start.isoformat() if train_start else "start",
                    train_end.isoformat(),
                )
            best_threshold = args.thresholds[0]

        eval_cache = _filter_forecast_cache(cache, eval_start, end_time)

        logging.info("Evaluating from %s to %s", eval_start.isoformat(), end_time.isoformat())
        total_return, result = _run_simulation(
            bars=eval_bars,
            tokens=tokens,
            forecaster=None,
            forecast_cache=eval_cache,
            threshold=best_threshold,
            context_bars=args.context_bars,
            min_context_bars=args.min_context_bars,
            sim_config=sim_config,
        )

    print("\n=== Backtest Summary ===")
    print(f"Token mints: {', '.join(tokens.keys())}")
    print(f"Eval window: {eval_start} -> {end_time}")
    if best_threshold is not None:
        print(f"Threshold: {best_threshold:.4f}")
    else:
        print("Strategy: daily-range")
        print(f"Min prior-day range: {args.min_range_bps:.1f} bps")
        print(f"Max actions per day: {args.max_actions_per_day}")
    print(f"Total return: {total_return:.2f}%")
    print(f"Sharpe: {result.sharpe_ratio:.2f} | Sortino: {result.sortino_ratio:.2f}")
    print(f"Max drawdown: {result.max_drawdown:.2%}")
    print(f"Trades: {result.total_trades} | Win rate: {result.win_rate:.2%}")


if __name__ == "__main__":
    main()
