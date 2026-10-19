#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xgbnew.artifacts import write_json_atomic

from src.tradinglib.hourly_level_optimizer import (
    HourlyLevelSearchConfig,
    summarize_walk_forward_results,
    validate_hourly_level_search_config,
    walk_forward_hourly_level_search,
)


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    values: list[float] = []
    for item in raw.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        try:
            values.append(float(stripped))
        except ValueError as exc:
            raise ValueError(f"grid contains non-numeric value: {stripped}") from exc
    return tuple(values)


def _read_symbols(args: argparse.Namespace) -> list[str]:
    symbols: list[str] = []
    if args.symbols:
        symbols.extend(x.strip().upper() for x in args.symbols.split(",") if x.strip())
    if args.symbols_file:
        symbols.extend(
            line.strip().upper()
            for line in args.symbols_file.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    seen: set[str] = set()
    out: list[str] = []
    for symbol in symbols:
        if symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


def _find_hourly_csv(data_root: Path, symbol: str) -> Path | None:
    for rel in ("stocks", "crypto", "crypto_new", ""):
        path = data_root / rel / f"{symbol}.csv"
        if path.exists():
            return path
    return None


def _parse_optional_utc_timestamp(raw: str, *, name: str) -> pd.Timestamp | None:
    if not raw:
        return None
    try:
        ts = pd.Timestamp(raw)
    except Exception as exc:
        raise ValueError(f"{name} must be a valid timestamp") from exc
    if pd.isna(ts):
        raise ValueError(f"{name} must be a valid timestamp")
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _timestamp_iso(ts: pd.Timestamp | None) -> str | None:
    return None if ts is None else ts.isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate rolling torch hourly level optimizer.")
    parser.add_argument("--data-root", type=Path, default=Path("trainingdatahourly"))
    parser.add_argument("--symbols", type=str, default="AAPL,MSFT,NVDA,SPY,QQQ")
    parser.add_argument("--symbols-file", type=Path, default=None)
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--start", type=str, default="")
    parser.add_argument("--end", type=str, default="")
    parser.add_argument("--lookback-bars", type=int, default=48)
    parser.add_argument("--forward-bars", type=int, default=48)
    parser.add_argument("--max-hold-bars", type=int, default=12)
    parser.add_argument("--entry-bps-grid", type=str, default="5,10,20,30,50,75,100,150")
    parser.add_argument("--take-profit-bps-grid", type=str, default="10,20,30,50,75,100,150,200")
    parser.add_argument("--fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--stop-loss-bps", type=float, default=0.0)
    parser.add_argument("--min-train-trades", type=int, default=1)
    parser.add_argument("--min-deploy-train-return-pct", type=float, default=-1e9)
    parser.add_argument("--min-deploy-train-win-rate-pct", type=float, default=0.0)
    parser.add_argument("--side-mode", choices=("long", "short", "both"), default="long")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/hourly_level_optimizer"))
    args = parser.parse_args()

    symbols = _read_symbols(args)
    if args.max_symbols > 0:
        symbols = symbols[: int(args.max_symbols)]
    if not symbols:
        raise SystemExit("No symbols provided")

    try:
        cfg = HourlyLevelSearchConfig(
            lookback_bars=int(args.lookback_bars),
            forward_bars=int(args.forward_bars),
            entry_bps_grid=_parse_float_tuple(args.entry_bps_grid),
            take_profit_bps_grid=_parse_float_tuple(args.take_profit_bps_grid),
            fill_buffer_bps=float(args.fill_buffer_bps),
            fee_bps=float(args.fee_bps),
            max_hold_bars=int(args.max_hold_bars),
            stop_loss_bps=float(args.stop_loss_bps),
            min_train_trades=int(args.min_train_trades),
            min_deploy_train_return_pct=float(args.min_deploy_train_return_pct),
            min_deploy_train_win_rate_pct=float(args.min_deploy_train_win_rate_pct),
            side_mode=str(args.side_mode),
            device=str(args.device),
        )
        validate_hourly_level_search_config(cfg)
        start_ts = _parse_optional_utc_timestamp(args.start, name="start")
        end_ts = _parse_optional_utc_timestamp(args.end, name="end")
    except ValueError as exc:
        print(f"[hourly-level] invalid config: {exc}", file=sys.stderr)
        return 2
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        print("[hourly-level] invalid config: start must be <= end", file=sys.stderr)
        return 2

    results = []
    skipped = []
    for symbol in symbols:
        path = _find_hourly_csv(args.data_root, symbol)
        if path is None:
            skipped.append({"symbol": symbol, "reason": "missing_csv"})
            continue
        frame = pd.read_csv(path)
        if start_ts is not None or end_ts is not None:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            if start_ts is not None:
                frame = frame[frame["timestamp"] >= start_ts]
            if end_ts is not None:
                frame = frame[frame["timestamp"] <= end_ts]
        result = walk_forward_hourly_level_search(frame, symbol=symbol, config=cfg)
        results.append(result)
        print(
            f"{symbol:>8s} windows={len(result.windows):4d} "
            f"total={result.total_return_pct:+8.2f}% "
            f"median_window={result.median_window_return_pct:+7.3f}%",
            flush=True,
        )

    summary = summarize_walk_forward_results(results)
    rows = []
    for result in results:
        rows.append(
            {
                "symbol": result.symbol,
                "n_windows": len(result.windows),
                "total_return_pct": result.total_return_pct,
                "median_window_return_pct": result.median_window_return_pct,
                "windows": [
                    {
                        "start": w.start_timestamp.isoformat(),
                        "end": w.end_timestamp.isoformat(),
                        "entry_bps": w.entry_bps,
                        "take_profit_bps": w.take_profit_bps,
                        "side": w.side,
                        "train_return_pct": w.train_return_pct,
                        "forward_return_pct": w.forward_return_pct,
                        "forward_trades": w.forward_trades,
                        "forward_win_rate_pct": w.forward_win_rate_pct,
                    }
                    for w in result.windows
                ],
            }
        )
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "data_root": str(args.data_root),
            "symbols": symbols,
            "max_symbols": int(args.max_symbols),
            "start": _timestamp_iso(start_ts),
            "end": _timestamp_iso(end_ts),
        },
        "config": cfg.__dict__,
        "summary": summary,
        "results": rows,
        "skipped": skipped,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"hourly_level_optimizer_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    write_json_atomic(out, payload, default=str, sort_keys=True)
    print(f"[hourly-level] wrote {out}")
    print(f"[hourly-level] summary {json.dumps(summary, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
