#!/usr/bin/env python3
"""Run large-universe marketsimulator experiments with Chronos2 cross-learning."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from marketsimlong.config import DataConfigLong, ForecastConfigLong, SimulationConfigLong
from marketsimlong.data import DailyDataLoader, is_crypto_symbol
from marketsimlong.simulator import run_simulation, SimulationResult


_EXCLUDED_SYMBOLS = {"CORRELATION_MATRIX", "DATA_SUMMARY", "VOLATILITY_METRICS"}


@dataclass
class ExperimentSpec:
    name: str
    top_n: int
    context_length: int
    batch_size: int
    chunk_size: int


DEFAULT_EXPERIMENTS = [
    ExperimentSpec(name="cl_ctx512_bs128_chunk200_top5", top_n=5, context_length=512, batch_size=128, chunk_size=200),
    ExperimentSpec(name="cl_ctx512_bs256_chunk200_top5", top_n=5, context_length=512, batch_size=256, chunk_size=200),
    ExperimentSpec(name="cl_ctx1024_bs128_chunk200_top5", top_n=5, context_length=1024, batch_size=128, chunk_size=200),
]


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _clean_symbol(symbol: str) -> str:
    cleaned = symbol.strip().upper()
    if not cleaned:
        return ""
    if cleaned in _EXCLUDED_SYMBOLS:
        return ""
    cleaned = cleaned.replace("/", "").replace("-", "")
    if not cleaned.replace("_", "").isalnum():
        return ""
    return cleaned


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


def _load_symbols_from_dir(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Symbol directory not found: {path}")
    symbols = sorted({_clean_symbol(p.stem) for p in path.glob("*.csv")})
    return [s for s in symbols if s]


def _filter_symbols(
    loader: DailyDataLoader,
    symbols: Sequence[str],
    *,
    min_history: int,
    start_date: date,
) -> List[str]:
    eligible: List[str] = []
    for symbol in symbols:
        frame = loader._data_cache.get(symbol)
        if frame is None or frame.empty:
            continue
        context = loader.get_context_for_date(symbol, start_date, context_days=min_history)
        if len(context) < min_history:
            continue
        eligible.append(symbol)
    return eligible


def _save_result(
    result: SimulationResult,
    output_dir: Path,
    config: Dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.equity_curve.to_csv(output_dir / "equity_curve.csv", header=["portfolio_value"])

    summary = {
        "start_date": str(result.start_date),
        "end_date": str(result.end_date),
        "initial_cash": result.initial_cash,
        "final_portfolio_value": result.final_portfolio_value,
        "total_return": result.total_return,
        "annualized_return": result.annualized_return,
        "sharpe_ratio": result.sharpe_ratio,
        "sortino_ratio": result.sortino_ratio,
        "max_drawdown": result.max_drawdown,
        "win_rate": result.win_rate,
        "total_trades": result.total_trades,
        "total_days": result.total_days,
        "margin_interest_paid": result.total_margin_interest_paid,
        "risk_penalties": result.total_risk_penalty,
        "config": config,
        "symbol_returns": result.symbol_returns,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if result.all_trades:
        trades_data = [
            {
                "date": str(t.timestamp),
                "symbol": t.symbol,
                "side": t.side,
                "quantity": t.quantity,
                "price": t.price,
                "notional": t.notional,
                "fee": t.fee,
            }
            for t in result.all_trades
        ]
        pd.DataFrame(trades_data).to_csv(output_dir / "trades.csv", index=False)

    # Top picks by day (use forecasts stored in DayResult)
    picks_rows: List[Dict[str, object]] = []
    for day in result.daily_results:
        if day.forecasts_used is None:
            continue
        ranked = day.forecasts_used.get_ranked_symbols(metric="predicted_return", ascending=False)
        for rank, (symbol, score) in enumerate(ranked[: config["top_n"]], start=1):
            forecast = day.forecasts_used.forecasts.get(symbol)
            if forecast is None:
                continue
            expected_range = forecast.predicted_high - forecast.predicted_low
            range_pct = expected_range / forecast.current_close if forecast.current_close else 0.0
            picks_rows.append(
                {
                    "date": str(day.date),
                    "rank": rank,
                    "symbol": symbol,
                    "score": float(score),
                    "current_close": forecast.current_close,
                    "predicted_close": forecast.predicted_close,
                    "predicted_high": forecast.predicted_high,
                    "predicted_low": forecast.predicted_low,
                    "expected_range": expected_range,
                    "expected_range_pct": range_pct,
                }
            )
    if picks_rows:
        pd.DataFrame(picks_rows).to_csv(output_dir / "top_picks.csv", index=False)


def _load_experiments(path: Path) -> List[ExperimentSpec]:
    if not path.exists():
        raise FileNotFoundError(f"Experiment JSON not found: {path}")
    raw = json.loads(path.read_text())
    specs: List[ExperimentSpec] = []
    for entry in raw:
        specs.append(
            ExperimentSpec(
                name=str(entry["name"]),
                top_n=int(entry.get("top_n", 5)),
                context_length=int(entry.get("context_length", 512)),
                batch_size=int(entry.get("batch_size", 128)),
                chunk_size=int(entry.get("chunk_size", 200)),
            )
        )
    return specs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run large-universe Chronos2 cross-learning experiments.",
    )
    parser.add_argument("--start", type=str, default="2023-05-08", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2023-06-16", help="End date (YYYY-MM-DD)")
    parser.add_argument("--symbols-file", type=str, default="available_stocks_with_data.json")
    parser.add_argument("--symbols-dir", type=str, default="")
    parser.add_argument("--data-root", type=str, default="trainingdata/train")
    parser.add_argument("--min-history", type=int, default=60)
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="reports/marketsimulatorwide")
    parser.add_argument("--experiments-json", type=str, default="")
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--margin-rate", type=float, default=0.0625)
    parser.add_argument("--max-hold-days", type=int, default=5)
    parser.add_argument("--leverage-soft-cap", type=float, default=0.0)
    parser.add_argument("--leverage-penalty-rate", type=float, default=0.0)
    parser.add_argument("--hold-penalty-start-days", type=int, default=0)
    parser.add_argument("--hold-penalty-rate", type=float, default=0.0)
    args = parser.parse_args()

    try:
        import torch

        if args.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for marketsimulatorwide experiments.")
    except Exception as exc:
        raise RuntimeError("CUDA is required for marketsimulatorwide experiments.") from exc

    start_date = _parse_date(args.start)
    end_date = _parse_date(args.end)

    if args.symbols_dir:
        symbols = _load_symbols_from_dir(Path(args.symbols_dir))
    else:
        symbols = _load_symbols_from_file(Path(args.symbols_file))

    if args.max_symbols and args.max_symbols > 0:
        symbols = symbols[: args.max_symbols]

    stock_symbols = tuple(sym for sym in symbols if not is_crypto_symbol(sym))
    crypto_symbols = tuple(sym for sym in symbols if is_crypto_symbol(sym))

    # Pre-load data and filter symbols based on history
    base_config = DataConfigLong(
        stock_symbols=stock_symbols,
        crypto_symbols=crypto_symbols,
        data_root=Path(args.data_root),
        start_date=start_date,
        end_date=end_date,
        context_days=args.min_history,
    )
    loader = DailyDataLoader(base_config)
    loader.load_all_symbols()
    filtered = _filter_symbols(loader, symbols, min_history=args.min_history, start_date=start_date)
    if not filtered:
        raise RuntimeError("No symbols passed the history filter.")

    if args.max_symbols and args.max_symbols > 0:
        filtered = filtered[: args.max_symbols]

    print(f"Universe size (filtered): {len(filtered)}")

    experiments = DEFAULT_EXPERIMENTS
    if args.experiments_json:
        experiments = _load_experiments(Path(args.experiments_json))

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for spec in experiments:
        exp_dir = output_root / spec.name
        exp_dir.mkdir(parents=True, exist_ok=True)

        data_config = DataConfigLong(
            stock_symbols=tuple(sym for sym in filtered if not is_crypto_symbol(sym)),
            crypto_symbols=tuple(sym for sym in filtered if is_crypto_symbol(sym)),
            data_root=Path(args.data_root),
            start_date=start_date,
            end_date=end_date,
            context_days=spec.context_length,
        )
        forecast_config = ForecastConfigLong(
            device_map=args.device,
            context_length=spec.context_length,
            prediction_length=1,
            batch_size=spec.batch_size,
            use_multivariate=True,
            use_cross_learning=True,
            cross_learning_min_batch=2,
            cross_learning_group_by_asset_type=True,
            cross_learning_chunk_size=spec.chunk_size,
        )
        sim_config = SimulationConfigLong(
            top_n=spec.top_n,
            initial_cash=100_000.0,
            leverage=args.leverage,
            margin_rate_annual=args.margin_rate,
            max_hold_days=args.max_hold_days,
            leverage_soft_cap=args.leverage_soft_cap,
            leverage_penalty_rate=args.leverage_penalty_rate,
            hold_penalty_start_days=args.hold_penalty_start_days,
            hold_penalty_rate=args.hold_penalty_rate,
        )

        # Reset GPU peak memory stats
        try:
            import torch

            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

        print(f"\n=== Running {spec.name} ===")
        try:
            result = run_simulation(data_config, forecast_config, sim_config)
        except Exception as exc:
            import traceback

            err_path = exp_dir / "error.txt"
            err_path.write_text(f"{exc}\n\n{traceback.format_exc()}")
            print(f"Experiment {spec.name} failed: {exc}")
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
            continue

        peak_gb = None
        try:
            import torch

            peak_bytes = torch.cuda.max_memory_allocated()
            peak_gb = peak_bytes / (1024 ** 3)
        except Exception:
            pass

        config_payload = {
            "name": spec.name,
            "top_n": spec.top_n,
            "context_length": spec.context_length,
            "batch_size": spec.batch_size,
            "chunk_size": spec.chunk_size,
            "device": args.device,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "min_history": args.min_history,
            "peak_gpu_gb": peak_gb,
            "symbols": len(filtered),
            "leverage": args.leverage,
            "margin_rate": args.margin_rate,
            "max_hold_days": args.max_hold_days,
            "leverage_soft_cap": args.leverage_soft_cap,
            "leverage_penalty_rate": args.leverage_penalty_rate,
            "hold_penalty_start_days": args.hold_penalty_start_days,
            "hold_penalty_rate": args.hold_penalty_rate,
        }
        _save_result(result, exp_dir, config_payload)

        if peak_gb is not None:
            print(f"Peak GPU memory: {peak_gb:.2f} GB")
        print(f"Return: {result.total_return * 100:.2f}% | Sharpe: {result.sharpe_ratio:.2f}")

        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
