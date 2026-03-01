#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return values


def _parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one ETH checkpoint across multiple windows and fill buffers.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--symbol", default="ETHUSD")
    parser.add_argument("--windows-hours", default="24,168,720", help="Comma-separated lookback windows in hours.")
    parser.add_argument("--fill-buffers-bps", default="0,5,10", help="Comma-separated fill buffer bps values.")
    parser.add_argument("--forecast-cache-root", default="binanceneural/forecast_cache")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-only", action="store_true", help="Pass --cache-only to simulator (recommended).")
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    windows = _parse_int_list(args.windows_hours)
    buffers = _parse_float_list(args.fill_buffers_bps)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    for hours in windows:
        for fill_buffer_bps in buffers:
            run_dir = args.output_dir / f"h{hours}_fb{fill_buffer_bps:g}"
            run_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                "-m",
                "newnanoalpacahourlyexp.run_hourly_trader_sim",
                "--symbols",
                args.symbol,
                "--checkpoint",
                str(checkpoint),
                "--eval-hours",
                str(hours),
                "--fill-buffer-bps",
                str(fill_buffer_bps),
                "--forecast-cache-root",
                str(args.forecast_cache_root),
                "--output-dir",
                str(run_dir),
            ]
            if args.cache_only:
                cmd.append("--cache-only")

            completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
            metrics_path = run_dir / "metrics.json"
            fills_path = run_dir / "fills.csv"
            metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
            fills = pd.read_csv(fills_path) if fills_path.exists() else pd.DataFrame()

            rows.append(
                {
                    "window_hours": int(hours),
                    "fill_buffer_bps": float(fill_buffer_bps),
                    "total_return": float(metrics.get("total_return", 0.0)),
                    "sortino": float(metrics.get("sortino", 0.0)),
                    "mean_hourly_return": float(metrics.get("mean_hourly_return", 0.0)),
                    "fills_total": int(len(fills)),
                    "fills_buy": int((fills["side"] == "buy").sum()) if not fills.empty and "side" in fills.columns else 0,
                    "fills_sell": int((fills["side"] == "sell").sum()) if not fills.empty and "side" in fills.columns else 0,
                    "sim_dir": str(run_dir),
                    "stdout_tail": completed.stdout[-500:],
                }
            )

    summary = pd.DataFrame(rows).sort_values(["window_hours", "fill_buffer_bps"]).reset_index(drop=True)
    summary_csv = args.output_dir / "summary.csv"
    summary_json = args.output_dir / "summary.json"
    summary.to_csv(summary_csv, index=False)
    summary_json.write_text(summary.to_json(orient="records", indent=2))

    print(summary.to_string(index=False))
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {summary_json}")


if __name__ == "__main__":
    main()

