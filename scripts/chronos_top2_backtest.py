#!/usr/bin/env python3
"""
Chronos2 Top-N daily open-to-close strategy backtest.

Strategy
--------
Each trading day D:
  1. Forecast next-day close return for ALL stocks in the universe using
     Chronos2, using ONLY data strictly before D (no lookahead).
  2. Rank stocks by predicted return, pick top N.
  3. Simulate:  buy at D's open,  sell at D's close.
  4. Apply realistic costs:
       entry cost = half_spread + commission
       exit  cost = half_spread + commission
       total = spread_bps + 2 * commission_bps

Spread model
------------
Corwin-Schultz (2012) high-low estimator on rolling 20-day window,
with volume-based fallback for illiquid / data-deficient names.

Caching
-------
Forecasts are cached per trading day under ``--cache-dir``.
Re-running with a different ``--top-n`` (or just to view results again)
takes seconds instead of re-running GPU inference.

Usage
-----
  python scripts/chronos_top2_backtest.py \\
      --start-date 2025-09-01 \\
      --end-date   2025-11-28 \\
      --top-n 2 \\
      --commission-bps 10 \\
      --backend cute_compiled_fp32 \\
      --context-length 128 \\
      --output-dir analysis/top2_backtest

  # Re-run analytics without re-forecasting:
  python scripts/chronos_top2_backtest.py \\
      --start-date 2025-09-01 \\
      --end-date   2025-11-28 \\
      --top-n 5 \\
      --load-cache analysis/top2_backtest/forecast_cache
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.spread_estimate import estimate_spread_bps

logger = logging.getLogger(__name__)

# ── backend → Chronos2OHLCWrapper kwargs ────────────────────────────────────
BACKEND_CONFIGS = {
    "cute_compiled_fp32": dict(pipeline_backend="cutechronos", torch_compile=True,  torch_dtype="float32"),
    "cute_fp32":          dict(pipeline_backend="cutechronos", torch_compile=False, torch_dtype="float32"),
    "compiled_fp32":      dict(pipeline_backend="chronos",     torch_compile=True,  torch_dtype="float32"),
    "compiled_bf16":      dict(pipeline_backend="chronos",     torch_compile=True,  torch_dtype="bfloat16"),
    "eager_fp32":         dict(pipeline_backend="chronos",     torch_compile=False, torch_dtype="float32"),
}


# ── data structures ──────────────────────────────────────────────────────────

@dataclass
class DayTrade:
    symbol: str
    predicted_return_pct: float
    actual_open: float
    actual_close: float
    gross_return_pct: float   # (close - open) / open * 100
    spread_bps: float         # Corwin-Schultz estimate
    commission_bps: float     # per side
    net_return_pct: float     # gross - (spread + 2*commission) in %

@dataclass
class DayResult:
    day: date
    trades: list[DayTrade]
    equity_start: float
    equity_end: float
    daily_return_pct: float   # portfolio-level (equal weight)
    n_candidates_screened: int

@dataclass
class BacktestResult:
    start_date: date
    end_date: date
    initial_cash: float
    final_equity: float
    total_return_pct: float
    annualized_return_pct: float
    monthly_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    total_days: int
    total_trades: int
    avg_spread_bps: float
    avg_commission_bps: float
    day_results: list[DayResult]


# ── CSV loading ──────────────────────────────────────────────────────────────

def _load_symbols_file(path: Path) -> list[str]:
    symbols: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        symbols.append(s.split("#", 1)[0].strip().upper())
    return symbols


def load_all_csvs(
    symbols: list[str],
    data_root: Path,
    min_rows: int = 50,
) -> dict[str, pd.DataFrame]:
    """Load full OHLCV history for each symbol.

    Returns dict {symbol: DataFrame} with columns
    [timestamp, open, high, low, close, volume].
    Symbols with < min_rows are excluded.
    """
    result: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        csv_path: Path | None = None
        for sub in ("train", "stocks", ""):
            c = data_root / sub / f"{symbol}.csv" if sub else data_root / f"{symbol}.csv"
            if c.exists():
                csv_path = c
                break
        if csv_path is None:
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        df.columns = df.columns.str.strip().str.lower()
        if not {"open", "high", "low", "close"}.issubset(df.columns):
            continue

        ts_col = next((c for c in ("timestamp", "date") if c in df.columns), df.columns[0])
        df["timestamp"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        df = df.drop_duplicates(subset=["timestamp"], keep="last")

        for col in ("open", "high", "low", "close"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"])

        if "volume" not in df.columns:
            df["volume"] = 0.0
        else:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

        if len(df) < min_rows:
            continue

        result[symbol] = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()

    return result


def get_trading_days(all_data: dict[str, pd.DataFrame], start: date, end: date) -> list[date]:
    """Find dates where ≥ 50% of stocks have a bar, within [start, end]."""
    day_counts: dict[date, int] = {}
    for df in all_data.values():
        for ts in df["timestamp"]:
            d = ts.date()
            if start <= d <= end:
                day_counts[d] = day_counts.get(d, 0) + 1

    threshold = max(1, len(all_data) // 2)
    days = sorted(d for d, cnt in day_counts.items() if cnt >= threshold)
    return days


# ── spread estimation ────────────────────────────────────────────────────────

def _spread_for_context(context_df: pd.DataFrame) -> float:
    """Return estimated spread in bps for a stock given its context window."""
    return estimate_spread_bps(context_df, window=20, cs_max_bps=200.0, fallback="volume")


# ── Chronos2 wrapper builder ─────────────────────────────────────────────────

def build_wrapper(backend: str, device_map: str = "cuda") -> object:
    cfg = BACKEND_CONFIGS[backend]
    from src.models.chronos2_wrapper import Chronos2OHLCWrapper
    return Chronos2OHLCWrapper.from_pretrained(
        device_map=device_map,
        pipeline_backend=cfg["pipeline_backend"],
        torch_compile=cfg.get("torch_compile", False),
        torch_dtype=cfg.get("torch_dtype", "float32"),
        default_context_length=8192,
    )


# ── per-day forecasting with caching ────────────────────────────────────────

def _context_for_day(
    df: pd.DataFrame,
    target_day: date,
    context_length: int,
) -> pd.DataFrame | None:
    """Return up to context_length rows strictly before target_day."""
    cutoff = pd.Timestamp(target_day, tz="UTC")
    ctx = df[df["timestamp"] < cutoff].tail(context_length)
    if len(ctx) < 30:
        return None
    return ctx.reset_index(drop=True)


def _actual_for_day(
    df: pd.DataFrame,
    target_day: date,
) -> dict[str, float] | None:
    """Return {'open': ..., 'close': ...} for target_day, or None."""
    day_start = pd.Timestamp(target_day, tz="UTC")
    day_end   = day_start + pd.Timedelta(days=1)
    rows = df[(df["timestamp"] >= day_start) & (df["timestamp"] < day_end)]
    if rows.empty:
        return None
    row = rows.iloc[0]
    o = float(row["open"])
    c = float(row["close"])
    if o <= 0 or c <= 0 or not (np.isfinite(o) and np.isfinite(c)):
        return None
    return {"open": o, "close": c}


def _cache_path(cache_dir: Path, target_day: date) -> Path:
    return cache_dir / f"{target_day.isoformat()}.json"


def load_cached_forecasts(cache_dir: Path, target_day: date) -> dict[str, dict] | None:
    """Load {symbol: {cc_return_pct, oc_return_pct, ...}} from cache, or None if missing.

    Also accepts the old format {symbol: float} and upgrades it on the fly.
    """
    path = _cache_path(cache_dir, target_day)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    # Upgrade old flat-float format
    if raw and isinstance(next(iter(raw.values())), (int, float)):
        return {sym: {"cc_return_pct": float(v), "oc_return_pct": None,
                      "pred_open": None, "pred_close": None, "last_close": None}
                for sym, v in raw.items()}
    return raw  # type: ignore[return-value]


def save_cached_forecasts(cache_dir: Path, target_day: date, forecasts: dict[str, dict]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    _cache_path(cache_dir, target_day).write_text(
        json.dumps(forecasts, indent=2), encoding="utf-8"
    )


def run_day_forecasts(
    wrapper: object,
    all_data: dict[str, pd.DataFrame],
    target_day: date,
    context_length: int,
    prediction_length: int = 3,
    verbose: bool = False,
) -> dict[str, dict]:
    """Run Chronos2 for all symbols on target_day.

    Returns {sym: {"cc_return_pct": float, "oc_return_pct": float,
                   "pred_open": float, "pred_close": float, "last_close": float}}

    cc_return_pct: (pred_close - last_close) / last_close * 100  (close-to-close trend)
    oc_return_pct: (pred_close - pred_open) / pred_open * 100    (predicted open-to-close)
    """
    results: dict[str, dict] = {}
    symbols = list(all_data.keys())
    t0 = time.perf_counter()

    for i, symbol in enumerate(symbols):
        df = all_data[symbol]
        ctx = _context_for_day(df, target_day, context_length)
        if ctx is None:
            continue

        last_close = float(ctx["close"].iloc[-1])
        if last_close <= 0:
            continue

        try:
            batch = wrapper.predict_ohlc(
                ctx,
                symbol=symbol,
                prediction_length=prediction_length,
                context_length=context_length,
            )
        except Exception as exc:
            logger.debug("Forecast failed %s on %s: %s", symbol, target_day, exc)
            continue

        med = getattr(batch, "median", None)
        if med is None or (hasattr(med, "empty") and med.empty):
            continue
        try:
            pred_close = float(med["close"].iloc[0])
            pred_open = float(med["open"].iloc[0]) if "open" in med.columns else None
        except Exception:
            continue

        if not np.isfinite(pred_close) or pred_close <= 0:
            continue

        cc_ret = (pred_close - last_close) / last_close * 100.0
        oc_ret: float | None = None
        if pred_open is not None and np.isfinite(pred_open) and pred_open > 0:
            oc_ret = (pred_close - pred_open) / pred_open * 100.0

        results[symbol] = {
            "cc_return_pct": cc_ret,
            "oc_return_pct": oc_ret,
            "pred_open": pred_open,
            "pred_close": pred_close,
            "last_close": last_close,
        }

        if verbose and (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            print(f"    [{i+1}/{len(symbols)}] {elapsed:.1f}s elapsed", flush=True)

    return results


# ── simulation ───────────────────────────────────────────────────────────────

def simulate_day(
    target_day: date,
    forecasts: dict[str, dict],
    all_data: dict[str, pd.DataFrame],
    equity: float,
    top_n: int,
    commission_bps: float,
    context_length: int,
    min_predicted_return_pct: float = -999.0,
    rank_by: str = "cc_return_pct",
) -> DayResult | None:
    """Simulate one trading day.  Returns None if no trades could be made.

    rank_by: "cc_return_pct"  — close-to-close predicted return (default, trend signal)
             "oc_return_pct"  — predicted open-to-close return (intraday signal)
    """
    def _score(info: dict) -> float:
        v = info.get(rank_by)
        if v is None:
            v = info.get("cc_return_pct", 0.0)  # fallback
        return v if v is not None else -999.0

    ranked = sorted(forecasts.items(), key=lambda kv: _score(kv[1]), reverse=True)
    ranked = [(sym, info) for sym, info in ranked
              if _score(info) >= min_predicted_return_pct]
    if not ranked:
        return None

    trades: list[DayTrade] = []
    for symbol, info in ranked:
        if len(trades) >= top_n:
            break

        actual = _actual_for_day(all_data[symbol], target_day) if symbol in all_data else None
        if actual is None:
            continue  # stock didn't trade that day; try next

        ctx = _context_for_day(all_data[symbol], target_day, context_length)
        spread_bps = _spread_for_context(ctx) if ctx is not None else 25.0
        if not np.isfinite(spread_bps) or spread_bps <= 0:
            spread_bps = 25.0

        o, c = actual["open"], actual["close"]
        gross_ret_pct = (c - o) / o * 100.0
        # Total round-trip cost in %: full spread + 2 × commission
        cost_pct = (spread_bps + 2.0 * commission_bps) / 100.0
        net_ret_pct = gross_ret_pct - cost_pct
        pred_ret = _score(info)

        trades.append(DayTrade(
            symbol=symbol,
            predicted_return_pct=pred_ret,
            actual_open=o,
            actual_close=c,
            gross_return_pct=gross_ret_pct,
            spread_bps=spread_bps,
            commission_bps=commission_bps,
            net_return_pct=net_ret_pct,
        ))

    if not trades:
        return None

    # Equal-weight portfolio return
    daily_ret_pct = float(np.mean([t.net_return_pct for t in trades]))
    new_equity = equity * (1.0 + daily_ret_pct / 100.0)

    return DayResult(
        day=target_day,
        trades=trades,
        equity_start=equity,
        equity_end=new_equity,
        daily_return_pct=daily_ret_pct,
        n_candidates_screened=len(forecasts),
    )


# ── metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(day_results: list[DayResult], initial_cash: float) -> BacktestResult:
    if not day_results:
        raise ValueError("No day results to compute metrics from")

    daily_returns = np.array([r.daily_return_pct / 100.0 for r in day_results])
    equity_curve = np.array([initial_cash] + [r.equity_end for r in day_results])

    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
    n_days = len(day_results)
    # Annualise assuming 252 trading days/year
    annualised = (1.0 + total_return) ** (252.0 / n_days) - 1.0
    # Monthly (21 trading days)
    monthly = (1.0 + total_return) ** (21.0 / n_days) - 1.0

    mean_r = float(np.mean(daily_returns))
    std_r  = float(np.std(daily_returns, ddof=1)) if n_days > 1 else 0.0
    sharpe = (mean_r / std_r * np.sqrt(252.0)) if std_r > 0 else 0.0

    down = daily_returns[daily_returns < 0]
    down_std = float(np.std(down, ddof=1)) if len(down) > 1 else 0.0
    sortino = (mean_r / down_std * np.sqrt(252.0)) if down_std > 0 else 0.0

    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    max_dd = float(abs(np.min(drawdowns)))

    win_days = int(np.sum(daily_returns > 0))
    win_rate = win_days / n_days * 100.0 if n_days > 0 else 0.0

    all_trades = [t for r in day_results for t in r.trades]
    all_spreads = [t.spread_bps for t in all_trades]
    all_commissions = [t.commission_bps for t in all_trades]

    return BacktestResult(
        start_date=day_results[0].day,
        end_date=day_results[-1].day,
        initial_cash=initial_cash,
        final_equity=float(equity_curve[-1]),
        total_return_pct=total_return * 100.0,
        annualized_return_pct=annualised * 100.0,
        monthly_return_pct=monthly * 100.0,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown_pct=max_dd * 100.0,
        win_rate_pct=win_rate,
        total_days=n_days,
        total_trades=len(all_trades),
        avg_spread_bps=float(np.mean(all_spreads)) if all_spreads else 0.0,
        avg_commission_bps=float(np.mean(all_commissions)) if all_commissions else 0.0,
        day_results=day_results,
    )


# ── printing / saving ─────────────────────────────────────────────────────────

def _print_day(r: DayResult) -> None:
    picks = ", ".join(
        f"{t.symbol}(pred={t.predicted_return_pct:+.2f}% actual_gross={t.gross_return_pct:+.2f}% net={t.net_return_pct:+.2f}%)"
        for t in r.trades
    )
    print(
        f"  {r.day}  day_ret={r.daily_return_pct:+.2f}%  eq=${r.equity_end:,.0f} "
        f"  screened={r.n_candidates_screened}  picks=[{picks}]",
        flush=True,
    )


def _print_summary(res: BacktestResult) -> None:
    print("\n" + "=" * 70)
    print(f"  BACKTEST SUMMARY  {res.start_date} → {res.end_date}")
    print("=" * 70)
    print(f"  Initial cash      : ${res.initial_cash:,.0f}")
    print(f"  Final equity      : ${res.final_equity:,.2f}")
    print(f"  Total return      : {res.total_return_pct:+.2f}%")
    print(f"  Monthly return    : {res.monthly_return_pct:+.2f}%  (21-day equiv)")
    print(f"  Ann. return       : {res.annualized_return_pct:+.2f}%")
    print(f"  Sharpe (ann.)     : {res.sharpe_ratio:.3f}")
    print(f"  Sortino (ann.)    : {res.sortino_ratio:.3f}")
    print(f"  Max drawdown      : {res.max_drawdown_pct:.2f}%")
    print(f"  Win rate          : {res.win_rate_pct:.1f}%")
    print(f"  Trading days      : {res.total_days}")
    print(f"  Total trades      : {res.total_trades}")
    print(f"  Avg spread        : {res.avg_spread_bps:.1f} bps")
    print(f"  Avg commission    : {res.avg_commission_bps:.1f} bps/side")
    print("=" * 70)


def save_results(res: BacktestResult, output_dir: Path, tag: str = "") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""

    # CSV: per-trade
    rows = []
    for dr in res.day_results:
        for t in dr.trades:
            rows.append({
                "date": dr.day.isoformat(),
                "equity_start": dr.equity_start,
                "equity_end": dr.equity_end,
                "day_return_pct": dr.daily_return_pct,
                "symbol": t.symbol,
                "predicted_return_pct": t.predicted_return_pct,
                "actual_open": t.actual_open,
                "actual_close": t.actual_close,
                "gross_return_pct": t.gross_return_pct,
                "spread_bps": t.spread_bps,
                "commission_bps": t.commission_bps,
                "net_return_pct": t.net_return_pct,
            })

    csv_path = output_dir / f"backtest_trades{suffix}_{ts}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # JSON: summary
    summary = {
        "start_date": res.start_date.isoformat(),
        "end_date": res.end_date.isoformat(),
        "initial_cash": res.initial_cash,
        "final_equity": res.final_equity,
        "total_return_pct": res.total_return_pct,
        "monthly_return_pct": res.monthly_return_pct,
        "annualized_return_pct": res.annualized_return_pct,
        "sharpe_ratio": res.sharpe_ratio,
        "sortino_ratio": res.sortino_ratio,
        "max_drawdown_pct": res.max_drawdown_pct,
        "win_rate_pct": res.win_rate_pct,
        "total_days": res.total_days,
        "total_trades": res.total_trades,
        "avg_spread_bps": res.avg_spread_bps,
        "avg_commission_bps": res.avg_commission_bps,
        "equity_curve": [
            {"date": r.day.isoformat(), "equity": r.equity_end}
            for r in res.day_results
        ],
    }
    json_path = output_dir / f"backtest_summary{suffix}_{ts}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # Also write a "latest" link
    (output_dir / "backtest_latest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n  Saved trades  → {csv_path}")
    print(f"  Saved summary → {json_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--symbols-file", type=Path,
                   default=Path("symbol_lists/stocks_wide_1000_v1.txt"))
    p.add_argument("--symbols", default=None, help="Comma-separated symbol override")
    p.add_argument("--data-root", type=Path, default=Path("trainingdata"))
    p.add_argument("--start-date", default=None,
                   help="Backtest start date YYYY-MM-DD (default: 60 trading days before end)")
    p.add_argument("--end-date", default=None,
                   help="Backtest end date YYYY-MM-DD (default: last date in data)")
    p.add_argument("--top-n", type=int, default=2,
                   help="Number of top picks per day (default: 2)")
    p.add_argument("--commission-bps", type=float, default=10.0,
                   help="Commission per side in bps (default: 10)")
    p.add_argument("--min-predicted-return-pct", type=float, default=-999.0,
                   help="Minimum predicted return %% to be eligible for selection (default: no filter)")
    p.add_argument("--initial-cash", type=float, default=10_000.0)
    p.add_argument("--backend", default="cute_compiled_fp32",
                   choices=list(BACKEND_CONFIGS))
    p.add_argument("--context-length", type=int, default=128,
                   help="Chronos2 context window in bars (default: 128)")
    p.add_argument("--device-map", default="cuda")
    p.add_argument("--output-dir", type=Path, default=Path("analysis/top2_backtest"))
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="Forecast cache directory (default: --output-dir/forecast_cache)")
    p.add_argument("--load-cache", type=Path, default=None,
                   help="Load forecasts only from this cache dir (skip GPU inference)")
    p.add_argument("--n-trading-days", type=int, default=60,
                   help="Number of most-recent trading days to backtest (default: 60)")
    p.add_argument("--rank-by", default="cc_return_pct",
                   choices=["cc_return_pct", "oc_return_pct"],
                   help="Signal to rank stocks by. "
                        "cc_return_pct = predicted close vs prev close (trend, default). "
                        "oc_return_pct = predicted open-to-close (intraday signal).")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    # ── load symbols ──────────────────────────────────────────────────────────
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = _load_symbols_file(args.symbols_file)
    print(f"[backtest] Loading {len(symbols)} symbols from {args.data_root} ...", flush=True)

    all_data = load_all_csvs(symbols, args.data_root)
    print(f"[backtest] Loaded {len(all_data)} symbols with sufficient history.", flush=True)

    if not all_data:
        print("ERROR: No usable stock data found.", file=sys.stderr)
        return 1

    # ── determine backtest date range ─────────────────────────────────────────
    # Find latest common date
    if args.end_date:
        end_dt = date.fromisoformat(args.end_date)
    else:
        latest = max(
            df["timestamp"].iloc[-1].date()
            for df in all_data.values()
        )
        end_dt = latest

    if args.start_date:
        start_dt = date.fromisoformat(args.start_date)
    else:
        # Back up n_trading_days from end
        # Approximate: 1.4× calendar days to get N trading days
        approx_start = end_dt - timedelta(days=int(args.n_trading_days * 1.5))
        start_dt = approx_start

    print(f"[backtest] Date range: {start_dt} → {end_dt}", flush=True)

    trading_days = get_trading_days(all_data, start_dt, end_dt)
    if not trading_days:
        print("ERROR: No trading days found in date range.", file=sys.stderr)
        return 1

    # Apply n_trading_days cap from the end
    if len(trading_days) > args.n_trading_days:
        trading_days = trading_days[-args.n_trading_days:]
    print(f"[backtest] {len(trading_days)} trading days: {trading_days[0]} → {trading_days[-1]}", flush=True)

    # ── forecast cache setup ──────────────────────────────────────────────────
    cache_dir = args.load_cache or args.cache_dir or (args.output_dir / "forecast_cache")
    load_only = args.load_cache is not None

    # ── load or build Chronos2 wrapper ────────────────────────────────────────
    wrapper = None
    if not load_only:
        print(f"[backtest] Building Chronos2 wrapper (backend={args.backend}) ...", flush=True)
        wrapper = build_wrapper(args.backend, device_map=args.device_map)
        print("[backtest] Wrapper ready.", flush=True)

    # ── main backtest loop ────────────────────────────────────────────────────
    day_results: list[DayResult] = []
    equity = args.initial_cash

    for day_idx, target_day in enumerate(trading_days):
        # Try to load from cache
        forecasts = load_cached_forecasts(cache_dir, target_day)

        if forecasts is None:
            if load_only:
                print(f"  [{day_idx+1}/{len(trading_days)}] {target_day}: no cache, skipping (--load-cache mode)", flush=True)
                continue
            if wrapper is None:
                print("ERROR: wrapper not built but cache miss.", file=sys.stderr)
                return 1

            print(f"  [{day_idx+1}/{len(trading_days)}] {target_day}: running inference ...", flush=True)
            t0 = time.perf_counter()
            forecasts = run_day_forecasts(
                wrapper, all_data, target_day,
                context_length=args.context_length,
                verbose=args.verbose,
            )
            elapsed = time.perf_counter() - t0
            print(f"    → {len(forecasts)} forecasts in {elapsed:.1f}s", flush=True)
            save_cached_forecasts(cache_dir, target_day, forecasts)

        if not forecasts:
            print(f"  [{day_idx+1}/{len(trading_days)}] {target_day}: no forecasts, skipping", flush=True)
            continue

        dr = simulate_day(
            target_day=target_day,
            forecasts=forecasts,
            all_data=all_data,
            equity=equity,
            top_n=args.top_n,
            commission_bps=args.commission_bps,
            context_length=args.context_length,
            min_predicted_return_pct=args.min_predicted_return_pct,
            rank_by=args.rank_by,
        )

        if dr is None:
            print(f"  [{day_idx+1}/{len(trading_days)}] {target_day}: no tradable picks, skipping", flush=True)
            continue

        equity = dr.equity_end
        day_results.append(dr)
        _print_day(dr)

    if not day_results:
        print("ERROR: No trading days simulated.", file=sys.stderr)
        return 1

    # ── compute and display metrics ───────────────────────────────────────────
    result = compute_metrics(day_results, initial_cash=args.initial_cash)
    _print_summary(result)
    save_results(result, args.output_dir, tag=f"top{args.top_n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
