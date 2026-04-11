#!/usr/bin/env python3
"""Wide Chronos2 daily stock screener.

Loads the 846-stock universe, runs compiled Chronos2 (cute_fp32 backend)
on every symbol, and ranks by predicted next-day close return.

Output:
  - JSON: screener results with predicted return per symbol
  - CSV: top-N buy signals with predicted return, confidence

Usage:
  python scripts/chronos_wide_screener.py \
      --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
      --top-n 20 \
      --output-dir analysis/chronos_screener \
      --context-length 256 \
      --backend cute_compiled_fp32

Backends:
  cute_compiled_fp32  - CuteChronos compiled (fastest, ~warmup needed)
  cute_fp32           - CuteChronos eager
  compiled_fp32       - standard Chronos compiled
  eager_fp32          - standard Chronos eager (slowest)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

logger = logging.getLogger(__name__)


BACKEND_CONFIGS = {
    "cute_compiled_fp32": dict(pipeline_backend="cutechronos", torch_compile=True, torch_dtype="float32"),
    "cute_fp32": dict(pipeline_backend="cutechronos", torch_compile=False, torch_dtype="float32"),
    "compiled_fp32": dict(pipeline_backend="chronos", torch_compile=True, torch_dtype="float32"),
    "compiled_bf16": dict(pipeline_backend="chronos", torch_compile=True, torch_dtype="bfloat16"),
    "eager_fp32": dict(pipeline_backend="chronos", torch_compile=False, torch_dtype="float32"),
}


@dataclass
class ScreenerResult:
    symbol: str
    last_close: float
    predicted_close: float
    predicted_return_pct: float
    predicted_high: float
    predicted_low: float
    context_rows: int
    inference_ms: float
    lora_applied: bool
    preaug_strategy: str | None


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


def _load_daily_csv(
    symbol: str,
    data_root: Path,
    context_length: int,
    *,
    max_staleness_days: int = 45,
) -> pd.DataFrame | None:
    """Load last context_length rows of daily OHLC CSV for symbol.

    Search order: train/ > stocks/ > root (prefer subdir with more recent data).
    """
    csv_path = None
    for sub in ("train", "stocks", ""):
        candidate = data_root / sub / f"{symbol}.csv" if sub else data_root / f"{symbol}.csv"
        if candidate.exists():
            csv_path = candidate
            break
    if csv_path is None:
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        logger.debug("Failed to read %s: %s", csv_path, exc)
        return None

    rename_map = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=rename_map)
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        return None

    # Handle timestamp
    ts_col = "timestamp" if "timestamp" in df.columns else (df.columns[0] if df.columns[0] not in required else None)
    if ts_col is not None and ts_col in df.columns:
        df["timestamp"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    else:
        df = df.reset_index(drop=True)

    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"])
    if len(df) < 30:
        return None

    # Staleness check: skip symbols whose last bar is too old
    if max_staleness_days > 0 and "timestamp" in df.columns and not df["timestamp"].empty:
        last_ts = df["timestamp"].iloc[-1]
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=max_staleness_days)
        if last_ts < cutoff:
            logger.debug("Skipping %s: last bar %s is stale (>%dd)", symbol, last_ts.date(), max_staleness_days)
            return None

    # Take last context_length rows
    df = df.tail(context_length).reset_index(drop=True)
    if "timestamp" not in df.columns:
        # Create a synthetic timestamp index
        df["timestamp"] = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=len(df), freq="B")
    return df[["timestamp", "open", "high", "low", "close"]]


def _build_chronos_wrapper(backend: str, device_map: str = "cuda") -> object:
    """Build and return a Chronos2OHLCWrapper with the specified backend."""
    cfg = BACKEND_CONFIGS.get(backend)
    if cfg is None:
        raise ValueError(f"Unknown backend {backend!r}. Choose from: {list(BACKEND_CONFIGS)}")

    from src.models.chronos2_wrapper import Chronos2OHLCWrapper

    pipeline_backend = cfg["pipeline_backend"]
    torch_compile = cfg.get("torch_compile", False)
    torch_dtype_str = cfg.get("torch_dtype", "float32")

    # Use Chronos2OHLCWrapper.from_pretrained which handles cutechronos internally
    wrapper = Chronos2OHLCWrapper.from_pretrained(
        device_map=device_map,
        pipeline_backend=pipeline_backend,
        torch_compile=torch_compile,
        torch_dtype=torch_dtype_str,
        default_context_length=8192,
    )
    return wrapper


def _predict_next_day(
    wrapper: object,
    df: pd.DataFrame,
    symbol: str,
    *,
    context_length: int = 256,
    prediction_length: int = 3,
) -> tuple[float, float, float, float] | None:
    """Run Chronos2 inference; return (predicted_close, predicted_high, predicted_low, inference_ms).

    Uses predict_ohlc which returns a Chronos2PredictionBatch with .median DataFrame.
    We take the first prediction step's close/high/low.
    """
    t0 = time.perf_counter()
    try:
        batch = wrapper.predict_ohlc(
            df,
            symbol=symbol,
            prediction_length=prediction_length,
            context_length=context_length,
        )
    except Exception as exc:
        logger.debug("Predict failed for %s: %s", symbol, exc)
        return None
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    med = batch.median
    if med is None or (hasattr(med, 'empty') and med.empty):
        return None

    try:
        pred_close = float(med["close"].iloc[0]) if "close" in med.columns else None
        pred_high = float(med["high"].iloc[0]) if "high" in med.columns else pred_close
        pred_low = float(med["low"].iloc[0]) if "low" in med.columns else pred_close
    except Exception:
        return None

    if pred_close is None:
        return None
    if not all(map(np.isfinite, [pred_close, pred_high or pred_close, pred_low or pred_close])):
        return None

    return pred_close, pred_high or pred_close, pred_low or pred_close, elapsed_ms


def run_screener(
    symbols: list[str],
    *,
    data_root: Path,
    backend: str,
    context_length: int,
    device_map: str,
    use_multivariate: bool = True,
    warmup_symbol: str | None = None,
    verbose: bool = False,
    max_staleness_days: int = 45,
) -> list[ScreenerResult]:
    """Run Chronos2 inference on all symbols and return ScreenerResult list."""
    print(f"[screener] Building Chronos2 wrapper (backend={backend}) ...", flush=True)
    wrapper = _build_chronos_wrapper(backend, device_map=device_map)
    print(f"[screener] Wrapper ready. Starting warmup...", flush=True)

    # Warmup pass (first inference is slow due to JIT/compile)
    warmup_sym = warmup_symbol or symbols[0]
    warmup_df = _load_daily_csv(warmup_sym, data_root, context_length)
    if warmup_df is not None:
        r = _predict_next_day(wrapper, warmup_df, warmup_sym,
                              context_length=context_length)
        if r is not None:
            print(f"[screener] Warmup done ({r[3]:.0f}ms for {warmup_sym})", flush=True)

    results: list[ScreenerResult] = []
    t_start = time.perf_counter()

    for i, symbol in enumerate(symbols, 1):
        df = _load_daily_csv(symbol, data_root, context_length, max_staleness_days=max_staleness_days)
        if df is None or len(df) < 30:
            if verbose:
                logger.debug("[%d/%d] %s: no data", i, len(symbols), symbol)
            continue

        last_close = float(df["close"].iloc[-1])
        if last_close <= 0:
            continue

        pred = _predict_next_day(wrapper, df, symbol,
                                 context_length=context_length)
        if pred is None:
            continue

        pred_close, pred_high, pred_low, inf_ms = pred
        ret_pct = (pred_close - last_close) / last_close * 100.0

        result = ScreenerResult(
            symbol=symbol,
            last_close=last_close,
            predicted_close=pred_close,
            predicted_return_pct=ret_pct,
            predicted_high=pred_high,
            predicted_low=pred_low,
            context_rows=len(df),
            inference_ms=inf_ms,
            lora_applied=False,
            preaug_strategy=None,
        )
        results.append(result)

        if verbose or (i % 50 == 0):
            elapsed = time.perf_counter() - t_start
            rate = i / elapsed if elapsed > 0 else 0
            eta_s = (len(symbols) - i) / rate if rate > 0 else 0
            print(f"  [{i}/{len(symbols)}] {symbol}: pred_ret={ret_pct:+.2f}% "
                  f"(inf={inf_ms:.0f}ms) | rate={rate:.1f}/s ETA={eta_s:.0f}s", flush=True)

    elapsed_total = time.perf_counter() - t_start
    print(f"[screener] Done: {len(results)}/{len(symbols)} symbols in {elapsed_total:.1f}s "
          f"({len(results)/elapsed_total:.1f} sym/s)", flush=True)
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols-file", type=Path, default=Path("symbol_lists/stocks_wide_1000_v1.txt"))
    parser.add_argument("--symbols", default=None, help="Comma-separated symbol override")
    parser.add_argument("--data-root", type=Path, default=Path("trainingdata"))
    parser.add_argument("--backend", default="cute_compiled_fp32", choices=list(BACKEND_CONFIGS))
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-predicted-return-pct", type=float, default=0.0,
                        help="Minimum predicted return %% to include in output")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/chronos_screener"))
    parser.add_argument("--no-multivariate", action="store_true", help="Unused; multivariate handled by wrapper config")
    parser.add_argument("--max-staleness-days", type=int, default=45,
                        help="Skip stocks whose latest bar is older than this many days (0=disable)")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")

    # Load symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = _load_symbols_file(args.symbols_file)

    if not symbols:
        print("ERROR: No symbols loaded", file=sys.stderr)
        return 1

    print(f"[screener] {len(symbols)} symbols, backend={args.backend}, context={args.context_length}", flush=True)

    results = run_screener(
        symbols,
        data_root=args.data_root,
        backend=args.backend,
        context_length=args.context_length,
        device_map=args.device_map,
        use_multivariate=not args.no_multivariate,
        verbose=args.verbose,
        max_staleness_days=args.max_staleness_days,
    )

    if not results:
        print("ERROR: No results produced", file=sys.stderr)
        return 1

    # Sort by predicted return descending
    results.sort(key=lambda r: r.predicted_return_pct, reverse=True)

    # Filter and take top-N
    filtered = [r for r in results if r.predicted_return_pct >= args.min_predicted_return_pct]
    top_n = filtered[: args.top_n]

    # Output
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = args.output_dir / f"screener_{ts}.json"
    csv_path = args.output_dir / f"screener_{ts}.csv"
    latest_json = args.output_dir / "screener_latest.json"

    payload = {
        "generated_at": ts,
        "backend": args.backend,
        "context_length": args.context_length,
        "symbols_screened": len(results),
        "symbols_file": str(args.symbols_file),
        "top_n": args.top_n,
        "top_candidates": [asdict(r) for r in top_n],
        "all_results": [asdict(r) for r in results],
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    latest_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Write CSV
    rows = [asdict(r) for r in results]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # Print summary
    print(f"\n=== TOP {args.top_n} BUY SIGNALS ===")
    print(f"{'Rank':<4} {'Symbol':<8} {'LastClose':>10} {'PredClose':>10} {'PredRet%':>9} {'PredHigh':>10} {'PredLow':>10}")
    print("-" * 65)
    for rank, r in enumerate(top_n, 1):
        print(f"{rank:<4} {r.symbol:<8} {r.last_close:>10.2f} {r.predicted_close:>10.2f} "
              f"{r.predicted_return_pct:>+9.2f}% {r.predicted_high:>10.2f} {r.predicted_low:>10.2f}")
    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
