#!/usr/bin/env python3
"""Hourly market simulation runner."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import List

import pandas as pd

from .config import DataConfigHourly, ForecastConfigHourly, SimulationConfigHourly
from .simulator import run_simulation, SimulationResultHourly

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_date(date_str: str) -> date:
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def _clean_symbol(symbol: str) -> str:
    cleaned = symbol.strip().upper()
    if not cleaned:
        return ""
    if cleaned in {"CORRELATION_MATRIX", "DATA_SUMMARY", "VOLATILITY_METRICS"}:
        return ""
    if not cleaned.replace("-", "").replace("_", "").isalnum():
        return ""
    return cleaned


def _load_symbols_from_dir(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Symbol directory not found: {path}")
    symbols = sorted({_clean_symbol(p.stem) for p in path.glob("*.csv")})
    return [s for s in symbols if s]


def _load_symbols_from_file(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Symbols file not found: {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            candidates = data.get("available_symbols") or data.get("symbols") or []
        else:
            candidates = data
        cleaned = [_clean_symbol(str(s)) for s in candidates if str(s).strip()]
        return [s for s in cleaned if s]

    symbols: List[str] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        cleaned = _clean_symbol(stripped.strip("',\" "))
        if cleaned:
            symbols.append(cleaned)
    return symbols


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def print_simulation_summary(result: SimulationResultHourly) -> None:
    print("\n" + "=" * 60)
    print("HOURLY SIMULATION RESULTS")
    print("=" * 60)

    print(f"\nPeriod: {result.start_time} to {result.end_time}")
    print(f"Total Hours: {result.total_periods}")
    print(f"Total Trades: {result.total_trades}")

    print("\n--- Portfolio Performance ---")
    print(f"Initial Capital: {format_currency(result.initial_cash)}")
    print(f"Final Value: {format_currency(result.final_portfolio_value)}")
    print(f"Total Return: {format_pct(result.total_return)}")
    print(f"Annualized Return: {format_pct(result.annualized_return)}")

    print("\n--- Risk Metrics ---")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {result.sortino_ratio:.2f}")
    print(f"Max Drawdown: {format_pct(result.max_drawdown)}")
    print(f"Win Rate: {format_pct(result.win_rate)}")

    print("\n--- Hourly Statistics ---")
    print(f"Avg Hourly Return: {format_pct(result.avg_period_return)}")

    if result.total_margin_interest_paid or result.total_risk_penalty:
        print("\n--- Costs ---")
        print(f"Margin Interest Paid: {format_currency(result.total_margin_interest_paid)}")
        print(f"Risk Penalties: {format_currency(result.total_risk_penalty)}")

    print("\n" + "=" * 60)


def save_results(
    result: SimulationResultHourly,
    output_dir: Path,
    config_used: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    equity_path = output_dir / "equity_curve.csv"
    result.equity_curve.to_csv(equity_path, header=["portfolio_value"])
    logger.info("Saved equity curve to %s", equity_path)

    summary = {
        "start_time": str(result.start_time),
        "end_time": str(result.end_time),
        "initial_cash": result.initial_cash,
        "final_portfolio_value": result.final_portfolio_value,
        "total_return": result.total_return,
        "annualized_return": result.annualized_return,
        "sharpe_ratio": result.sharpe_ratio,
        "sortino_ratio": result.sortino_ratio,
        "max_drawdown": result.max_drawdown,
        "win_rate": result.win_rate,
        "total_trades": result.total_trades,
        "total_periods": result.total_periods,
        "margin_interest_paid": result.total_margin_interest_paid,
        "risk_penalties": result.total_risk_penalty,
        "config": config_used,
        "symbol_returns": result.symbol_returns,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Saved summary to %s", summary_path)

    if result.all_trades:
        trades_data = [
            {
                "timestamp": str(t.timestamp),
                "symbol": t.symbol,
                "side": t.side,
                "quantity": t.quantity,
                "price": t.price,
                "notional": t.notional,
                "fee": t.fee,
            }
            for t in result.all_trades
        ]
        trades_df = pd.DataFrame(trades_data)
        trades_path = output_dir / "trades.csv"
        trades_df.to_csv(trades_path, index=False)
        logger.info("Saved %d trades to %s", len(trades_data), trades_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Hourly market simulation with Chronos2 forecasting")
    parser.add_argument("--top-n", type=int, default=1)
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--start", type=str, default="2025-01-01")
    parser.add_argument("--end", type=str, default="2025-01-31")
    parser.add_argument("--symbols-dir", type=str, default="")
    parser.add_argument("--symbols-file", type=str, default="")
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--data-dir", type=str, default="trainingdatahourly")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--prediction-length", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cross-learning", action="store_true")
    parser.add_argument("--cross-learning-min-batch", type=int, default=2)
    parser.add_argument("--cross-learning-chunk", type=int, default=0)
    parser.add_argument("--cross-learning-no-group", action="store_true")
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--margin-rate", type=float, default=0.0625)
    parser.add_argument("--max-hold-hours", type=int, default=0)
    parser.add_argument("--leverage-soft-cap", type=float, default=0.0)
    parser.add_argument("--leverage-penalty-rate", type=float, default=0.0)
    parser.add_argument("--hold-penalty-start-hours", type=int, default=0)
    parser.add_argument("--hold-penalty-rate", type=float, default=0.0)
    parser.add_argument("--output-dir", type=str, default="reports/marketsimhourly")
    args = parser.parse_args()

    start_date = parse_date(args.start)
    end_date = parse_date(args.end)

    symbols_override: List[str] = []
    if args.symbols_file:
        symbols_override = _load_symbols_from_file(Path(args.symbols_file))
    elif args.symbols_dir:
        symbols_override = _load_symbols_from_dir(Path(args.symbols_dir))

    if args.max_symbols and args.max_symbols > 0:
        symbols_override = symbols_override[: args.max_symbols]

    if symbols_override:
        stock_symbols = tuple(sym for sym in symbols_override if not sym.endswith(("USD", "USDT", "USDC", "BTC", "ETH")))
        crypto_symbols = tuple(sym for sym in symbols_override if sym.endswith(("USD", "USDT", "USDC", "BTC", "ETH")))
    else:
        stock_symbols = DataConfigHourly().stock_symbols
        crypto_symbols = DataConfigHourly().crypto_symbols

    data_config = DataConfigHourly(
        stock_symbols=stock_symbols,
        crypto_symbols=crypto_symbols,
        data_root=Path(args.data_dir),
        start_date=start_date,
        end_date=end_date,
        context_hours=args.context_length,
    )

    forecast_config = ForecastConfigHourly(
        device_map=args.device,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        batch_size=args.batch_size,
        use_multivariate=True,
        use_cross_learning=bool(args.cross_learning),
        cross_learning_min_batch=max(2, int(args.cross_learning_min_batch)),
        cross_learning_group_by_asset_type=not args.cross_learning_no_group,
        cross_learning_chunk_size=int(args.cross_learning_chunk) if args.cross_learning_chunk > 0 else None,
    )

    sim_config = SimulationConfigHourly(
        top_n=args.top_n,
        initial_cash=args.initial_cash,
        leverage=args.leverage,
        margin_rate_annual=args.margin_rate,
        max_hold_hours=args.max_hold_hours,
        leverage_soft_cap=args.leverage_soft_cap,
        leverage_penalty_rate=args.leverage_penalty_rate,
        hold_penalty_start_hours=args.hold_penalty_start_hours,
        hold_penalty_rate=args.hold_penalty_rate,
    )

    result = run_simulation(data_config, forecast_config, sim_config)
    print_simulation_summary(result)

    config_used = {
        "top_n": args.top_n,
        "initial_cash": args.initial_cash,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "context_length": args.context_length,
        "prediction_length": args.prediction_length,
        "batch_size": args.batch_size,
        "cross_learning": bool(args.cross_learning),
        "leverage": args.leverage,
        "margin_rate": args.margin_rate,
        "max_hold_hours": args.max_hold_hours,
        "leverage_soft_cap": args.leverage_soft_cap,
        "leverage_penalty_rate": args.leverage_penalty_rate,
        "hold_penalty_start_hours": args.hold_penalty_start_hours,
        "hold_penalty_rate": args.hold_penalty_rate,
    }
    save_results(result, Path(args.output_dir), config_used)
    return 0


if __name__ == "__main__":
    sys.exit(main())
