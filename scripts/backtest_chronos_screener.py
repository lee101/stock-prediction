#!/usr/bin/env python3
"""Historical backtest of the Chronos2 wide screener strategy.

For each day D in [train_end - backtest_days, train_end]:
  1. Load data up to day D for each symbol
  2. Run Chronos2 prediction to get predicted 1-day return
  3. Select top-K stocks
  4. Compute actual next-day return (open→close)
  5. Aggregate into daily P&L

Validates whether top predicted stocks actually outperform.
Uses trainingdata/train/ CSVs, so no future-data lookahead.

Usage:
  python scripts/backtest_chronos_screener.py \
      --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
      --backtest-days 30 \
      --top-k 5 \
      --context-length 128 \
      --backend cute_fp32
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

logger = logging.getLogger(__name__)


def _load_symbols_file(path: Path) -> list[str]:
    symbols: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        symbol = stripped.split("#", 1)[0].strip().upper()
        if symbol:
            symbols.append(symbol)
    return symbols


def _load_full_csv(symbol: str, data_root: Path) -> pd.DataFrame | None:
    """Load full daily CSV for symbol."""
    for sub in ("train", "stocks", ""):
        csv_path = data_root / sub / f"{symbol}.csv" if sub else data_root / f"{symbol}.csv"
        if csv_path.exists():
            break
    else:
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    rename_map = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=rename_map)
    if not {"open", "high", "low", "close"}.issubset(df.columns):
        return None
    ts_col = "timestamp" if "timestamp" in df.columns else None
    if ts_col:
        df["timestamp"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "close"])
    if len(df) < 60:
        return None
    return df.reset_index(drop=True)


def _predict_return(wrapper: object, df: pd.DataFrame, symbol: str, context_length: int) -> float | None:
    """Predict 1-day return using Chronos2."""
    if len(df) < 30:
        return None
    ctx = df.tail(context_length).copy()
    if "timestamp" not in ctx.columns:
        ctx["timestamp"] = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=len(ctx), freq="B")
    try:
        batch = wrapper.predict_ohlc(ctx, symbol=symbol, prediction_length=3, context_length=context_length)
    except Exception as exc:
        logger.debug("Predict failed for %s: %s", symbol, exc)
        return None
    med = batch.median
    if med is None or (hasattr(med, 'empty') and med.empty):
        return None
    if "close" not in med.columns:
        return None
    pred_close = float(med["close"].iloc[0])
    last_close = float(df["close"].iloc[-1])
    if last_close <= 0 or not np.isfinite(pred_close):
        return None
    return (pred_close - last_close) / last_close


def run_backtest(
    symbols: list[str],
    *,
    data_root: Path,
    backend: str,
    context_length: int,
    top_k: int,
    backtest_days: int,
    fee_bps: float = 10.0,
    fill_bps: float = 5.0,
    device_map: str = "cuda",
) -> dict:
    """Run Chronos screener backtest over historical data."""
    from scripts.chronos_wide_screener import _build_chronos_wrapper

    print(f"[backtest] Loading {len(symbols)} symbol CSVs...", flush=True)
    all_dfs: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = _load_full_csv(sym, data_root)
        if df is not None and "timestamp" in df.columns:
            all_dfs[sym] = df

    print(f"[backtest] {len(all_dfs)} symbols with data loaded", flush=True)
    if not all_dfs:
        raise ValueError("No symbol data loaded")

    # Find the latest common date and backtest range
    all_last = {sym: df["timestamp"].iloc[-1] for sym, df in all_dfs.items()}
    latest_date = max(all_last.values())
    latest_date = latest_date.floor("D")

    # Use last 60-day period for backtest (ensure all symbols have coverage)
    # Pick backtest_days trading days from the end
    # Get the benchmark (SPY or first symbol) dates
    _spy = all_dfs.get("SPY")
    spy_df = _spy if _spy is not None else all_dfs[next(iter(all_dfs))]
    spy_dates = spy_df["timestamp"].dt.floor("D").unique()
    spy_dates = sorted(spy_dates)
    # Use last backtest_days as the backtest period
    if len(spy_dates) <= backtest_days + context_length:
        raise ValueError(f"Not enough history for {backtest_days}-day backtest (have {len(spy_dates)} days)")
    backtest_date_range = spy_dates[-(backtest_days + 1):]  # +1 to get next-day return

    print(f"[backtest] Building Chronos2 wrapper (backend={backend})...", flush=True)
    wrapper = _build_chronos_wrapper(backend, device_map=device_map)
    print(f"[backtest] Wrapper ready. Warming up...", flush=True)

    # Warmup
    warmup_sym = "SPY" if "SPY" in all_dfs else next(iter(all_dfs))
    warmup_df = all_dfs[warmup_sym].copy()
    _predict_return(wrapper, warmup_df.head(-2), warmup_sym, context_length)
    print(f"[backtest] Warmup done", flush=True)

    daily_results = []
    fee_rate = fee_bps / 10_000.0
    fill_rate = fill_bps / 10_000.0
    total_cost = 2 * (fee_rate + fill_rate)  # round-trip

    t_start = time.perf_counter()
    for day_i, target_date in enumerate(backtest_date_range[:-1]):
        next_date = backtest_date_range[day_i + 1]

        # For each symbol, compute predicted return using data up to target_date
        predictions: dict[str, float] = {}
        for sym, df in all_dfs.items():
            # Use only data up to (and including) target_date
            mask = df["timestamp"].dt.floor("D") <= pd.Timestamp(target_date)
            hist_df = df[mask]
            if len(hist_df) < 30:
                continue
            ret = _predict_return(wrapper, hist_df, sym, context_length)
            if ret is not None:
                predictions[sym] = ret

        if not predictions:
            continue

        # Rank and take top-K
        ranked = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        selected = [sym for sym, _ in ranked[:top_k]]

        # Compute actual next-day return (buy at next-day open, sell at next-day close)
        actual_returns = []
        for sym in selected:
            df = all_dfs[sym]
            next_mask = df["timestamp"].dt.floor("D") == pd.Timestamp(next_date)
            next_rows = df[next_mask]
            if next_rows.empty:
                continue
            row = next_rows.iloc[0]
            entry = float(row["open"]) * (1.0 + fill_rate)
            exit_p = float(row["close"])
            if entry <= 0:
                continue
            ret = (exit_p - entry) / entry - total_cost
            actual_returns.append(ret)

        if actual_returns:
            day_return = float(np.mean(actual_returns))
            daily_results.append({
                "date": str(pd.Timestamp(target_date).date()),
                "selected_symbols": selected,
                "predicted_returns": {sym: float(v) for sym, v in ranked[:top_k]},
                "actual_returns": actual_returns,
                "day_return": day_return,
            })

        if day_i % 5 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  [{day_i+1}/{len(backtest_date_range)-1}] {str(pd.Timestamp(target_date).date())} "
                  f"selected={selected[:3]} day_ret={day_return*100:+.2f}% ({elapsed:.1f}s)", flush=True)

    # Compute summary
    returns = [r["day_return"] for r in daily_results]
    cum_return = float(np.prod([1.0 + r for r in returns])) - 1.0
    sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if returns and np.std(returns) > 0 else 0.0
    win_rate = float(np.mean([1 if r > 0 else 0 for r in returns])) if returns else 0.0

    return {
        "backend": backend,
        "context_length": context_length,
        "top_k": top_k,
        "backtest_days": len(daily_results),
        "symbols_screened": len(all_dfs),
        "fee_bps": fee_bps,
        "fill_bps": fill_bps,
        "cumulative_return_pct": cum_return * 100,
        "annualized_sharpe": sharpe,
        "win_rate_pct": win_rate * 100,
        "mean_daily_return_pct": float(np.mean(returns)) * 100 if returns else 0.0,
        "daily_results": daily_results,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols-file", type=Path, default=Path("symbol_lists/stocks_wide_1000_v1.txt"))
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--data-root", type=Path, default=Path("trainingdata"))
    parser.add_argument("--backend", default="cute_fp32",
                        choices=["cute_compiled_fp32", "cute_fp32", "compiled_fp32", "eager_fp32"])
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--backtest-days", type=int, default=20)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--fill-bps", type=float, default=5.0)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/chronos_screener_backtest"))
    parser.add_argument("--limit-symbols", type=int, default=None, help="Limit number of symbols for quick test")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = _load_symbols_file(args.symbols_file)

    if args.limit_symbols:
        symbols = symbols[: args.limit_symbols]

    print(f"[backtest] Chronos screener backtest: {len(symbols)} symbols, "
          f"top_k={args.top_k}, days={args.backtest_days}, backend={args.backend}", flush=True)

    result = run_backtest(
        symbols,
        data_root=args.data_root,
        backend=args.backend,
        context_length=args.context_length,
        top_k=args.top_k,
        backtest_days=args.backtest_days,
        fee_bps=args.fee_bps,
        fill_bps=args.fill_bps,
        device_map=args.device_map,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = args.output_dir / f"backtest_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\n=== BACKTEST RESULTS ===")
    print(f"Days:               {result['backtest_days']}")
    print(f"Symbols screened:   {result['symbols_screened']}")
    print(f"Cumulative return:  {result['cumulative_return_pct']:+.2f}%")
    print(f"Mean daily return:  {result['mean_daily_return_pct']:+.3f}%/day")
    print(f"Annualized Sharpe:  {result['annualized_sharpe']:.2f}")
    print(f"Win rate:           {result['win_rate_pct']:.1f}%")
    print(f"\nOutput: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
