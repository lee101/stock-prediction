#!/usr/bin/env python3
"""Evaluate a Migas-style narrative overlay on top of existing Binance Chronos caches."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from binanceneural.narrative_forecasts import apply_narrative_overlay


def _load_history(data_root: Path, symbol: str) -> pd.DataFrame:
    candidates = [
        Path(data_root) / f"{symbol.upper()}.csv",
        REPO_ROOT / "trainingdatahourlybinance" / f"{symbol.upper()}.csv",
        REPO_ROOT / "trainingdatahourly" / "crypto" / f"{symbol.upper()}.csv",
    ]
    for path in candidates:
        if path.exists():
            frame = pd.read_csv(path, low_memory=False)
            frame.columns = [col.lower() for col in frame.columns]
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            return frame
    raise FileNotFoundError(f"Unable to find hourly history for {symbol} under {data_root}")


def _load_forecast(cache_root: Path, horizon: int, symbol: str) -> pd.DataFrame:
    path = Path(cache_root) / f"h{int(horizon)}" / f"{symbol.upper()}.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    for col in ("timestamp", "issued_at", "target_timestamp"):
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return frame


def _write_forecast(cache_root: Path, horizon: int, symbol: str, frame: pd.DataFrame) -> Path:
    path = Path(cache_root) / f"h{int(horizon)}" / f"{symbol.upper()}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values("timestamp").reset_index(drop=True).to_parquet(path, index=False)
    return path


def _score_forecast_frame(history: pd.DataFrame, forecast: pd.DataFrame) -> dict[str, float]:
    if forecast.empty:
        return {"count": 0.0}
    actual = history.copy()
    actual["timestamp"] = pd.to_datetime(actual["timestamp"], utc=True, errors="coerce")
    actual = actual.dropna(subset=["timestamp"]).set_index("timestamp")

    rows: list[dict[str, float]] = []
    for item in forecast.to_dict(orient="records"):
        target_ts = pd.to_datetime(item.get("target_timestamp") or item.get("timestamp"), utc=True, errors="coerce")
        if pd.isna(target_ts) or target_ts not in actual.index:
            continue
        actual_row = actual.loc[target_ts]
        close_pred = float(item.get("predicted_close_p50", np.nan))
        high_pred = float(item.get("predicted_high_p50", np.nan))
        low_pred = float(item.get("predicted_low_p50", np.nan))
        if not np.isfinite(close_pred) or not np.isfinite(high_pred) or not np.isfinite(low_pred):
            continue
        rows.append(
            {
                "close_abs": abs(close_pred - float(actual_row["close"])),
                "high_abs": abs(high_pred - float(actual_row["high"])),
                "low_abs": abs(low_pred - float(actual_row["low"])),
                "close_scale": abs(float(actual_row["close"])),
                "high_scale": abs(float(actual_row["high"])),
                "low_scale": abs(float(actual_row["low"])),
            }
        )
    if not rows:
        return {"count": 0.0}
    frame = pd.DataFrame(rows)
    return {
        "count": float(len(frame)),
        "close_mae": float(frame["close_abs"].mean()),
        "close_mae_pct": float((frame["close_abs"] / frame["close_scale"].clip(lower=1e-8)).mean() * 100.0),
        "high_mae": float(frame["high_abs"].mean()),
        "high_mae_pct": float((frame["high_abs"] / frame["high_scale"].clip(lower=1e-8)).mean() * 100.0),
        "low_mae": float(frame["low_abs"].mean()),
        "low_mae_pct": float((frame["low_abs"] / frame["low_scale"].clip(lower=1e-8)).mean() * 100.0),
    }


def build_overlay_cache(
    *,
    symbols: list[str],
    horizons: list[int],
    data_root: Path,
    raw_cache_root: Path,
    overlay_cache_root: Path,
    narrative_backend: str,
    narrative_model: str | None,
    narrative_context_hours: int,
    force_rebuild: bool,
    tail_rows: int,
) -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for symbol in symbols:
        history = _load_history(data_root, symbol)
        symbol_summary: dict[str, dict[str, float]] = {}
        for horizon in horizons:
            raw_full_frame = _load_forecast(raw_cache_root, horizon, symbol)
            if raw_full_frame.empty:
                continue
            raw_frame = raw_full_frame
            if int(tail_rows) > 0 and len(raw_frame) > int(tail_rows):
                raw_frame = raw_frame.tail(int(tail_rows)).reset_index(drop=True)
            overlay_frame = apply_narrative_overlay(
                raw_frame,
                symbol=symbol,
                history=history,
                backend=narrative_backend,
                model=narrative_model,
                forecast_cache_dir=Path(overlay_cache_root) / f"h{int(horizon)}",
                summary_cache_dir=Path(overlay_cache_root) / "_narrative_summaries" / f"h{int(horizon)}",
                context_hours=int(narrative_context_hours),
                force_rebuild=force_rebuild,
            )
            merged_overlay = raw_full_frame
            if len(overlay_frame) != len(raw_full_frame):
                merged_overlay = (
                    pd.concat([raw_full_frame, overlay_frame], ignore_index=True)
                    .drop_duplicates(subset=["timestamp", "symbol"], keep="last")
                    .sort_values("timestamp")
                    .reset_index(drop=True)
                )
            else:
                merged_overlay = overlay_frame.sort_values("timestamp").reset_index(drop=True)
            _write_forecast(overlay_cache_root, horizon, symbol, merged_overlay)
            raw_metrics = _score_forecast_frame(history, raw_frame)
            overlay_metrics = _score_forecast_frame(history, overlay_frame)
            symbol_summary[f"h{int(horizon)}"] = {
                "raw_close_mae_pct": float(raw_metrics.get("close_mae_pct", np.nan)),
                "overlay_close_mae_pct": float(overlay_metrics.get("close_mae_pct", np.nan)),
                "raw_high_mae_pct": float(raw_metrics.get("high_mae_pct", np.nan)),
                "overlay_high_mae_pct": float(overlay_metrics.get("high_mae_pct", np.nan)),
                "raw_low_mae_pct": float(raw_metrics.get("low_mae_pct", np.nan)),
                "overlay_low_mae_pct": float(overlay_metrics.get("low_mae_pct", np.nan)),
                "count": float(overlay_metrics.get("count", raw_metrics.get("count", 0.0))),
                "close_mae_delta_pct": float(overlay_metrics.get("close_mae_pct", np.nan) - raw_metrics.get("close_mae_pct", np.nan)),
            }
        summary[symbol] = symbol_summary
    return summary


def _run_backtest(cache_root: Path, *, days: int, output_json: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "binanceleveragesui" / "backtest_trade_margin_meta.py"),
        "--days",
        str(int(days)),
        "--forecast-cache",
        str(cache_root),
        "--output-json",
        str(output_json),
    ]
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    return json.loads(output_json.read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate narrative-conditioned Chronos overlay for Binance.")
    parser.add_argument("--symbols", default="DOGEUSD,AAVEUSD", help="Comma-separated data symbols.")
    parser.add_argument("--horizons", default="1,4,24", help="Comma-separated forecast horizons.")
    parser.add_argument("--data-root", default=str(REPO_ROOT / "trainingdatahourlybinance"))
    parser.add_argument("--raw-cache-root", default=str(REPO_ROOT / "binanceneural" / "forecast_cache"))
    parser.add_argument("--overlay-cache-root", default=str(REPO_ROOT / "binanceneural" / "forecast_cache_migas_overlay"))
    parser.add_argument("--narrative-backend", default="heuristic")
    parser.add_argument("--narrative-model", default=None)
    parser.add_argument("--narrative-context-hours", type=int, default=24 * 7)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--tail-rows", type=int, default=0, help="Limit evaluation/build to the most recent N forecast rows per symbol/horizon.")
    parser.add_argument("--run-backtest", action="store_true", help="Also run the current Binance meta backtest with raw vs overlay caches.")
    parser.add_argument("--backtest-days", type=int, default=30)
    parser.add_argument("--output-json", default=str(REPO_ROOT / "binanceleveragesui" / "migas_overlay_eval.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = [item.strip().upper() for item in str(args.symbols).split(",") if item.strip()]
    horizons = [int(item.strip()) for item in str(args.horizons).split(",") if item.strip()]
    data_root = Path(args.data_root)
    raw_cache_root = Path(args.raw_cache_root)
    overlay_cache_root = Path(args.overlay_cache_root)
    output_json = Path(args.output_json)

    mae_summary = build_overlay_cache(
        symbols=symbols,
        horizons=horizons,
        data_root=data_root,
        raw_cache_root=raw_cache_root,
        overlay_cache_root=overlay_cache_root,
        narrative_backend=str(args.narrative_backend),
        narrative_model=args.narrative_model,
        narrative_context_hours=int(args.narrative_context_hours),
        force_rebuild=bool(args.force_rebuild),
        tail_rows=int(args.tail_rows),
    )

    payload: dict[str, Any] = {
        "config": {
            "symbols": symbols,
            "horizons": horizons,
            "data_root": str(data_root.resolve()),
            "raw_cache_root": str(raw_cache_root.resolve()),
            "overlay_cache_root": str(overlay_cache_root.resolve()),
            "narrative_backend": str(args.narrative_backend),
            "narrative_model": args.narrative_model,
            "narrative_context_hours": int(args.narrative_context_hours),
            "tail_rows": int(args.tail_rows),
        },
        "mae": mae_summary,
    }

    if args.run_backtest:
        raw_report_path = output_json.with_name(output_json.stem + "_raw_backtest.json")
        overlay_report_path = output_json.with_name(output_json.stem + "_overlay_backtest.json")
        raw_report = _run_backtest(raw_cache_root, days=int(args.backtest_days), output_json=raw_report_path)
        overlay_report = _run_backtest(overlay_cache_root, days=int(args.backtest_days), output_json=overlay_report_path)
        payload["backtest"] = {
            "days": int(args.backtest_days),
            "raw": raw_report,
            "overlay": overlay_report,
            "comparison": {
                "meta_sortino_delta": float(overlay_report["meta"].get("sortino_ratio") or 0.0)
                - float(raw_report["meta"].get("sortino_ratio") or 0.0),
                "meta_return_pct_delta": float(overlay_report["meta"].get("return_pct") or 0.0)
                - float(raw_report["meta"].get("return_pct") or 0.0),
            },
        }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
