#!/usr/bin/env python3
"""List symbols with positive trend signals for potential onboarding."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple


def parse_threshold_map(raw: str | None) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    if not raw:
        return thresholds
    for item in raw.split(","):
        entry = item.strip()
        if not entry or ":" not in entry:
            continue
        key, value = entry.split(":", 1)
        try:
            thresholds[key.strip().upper()] = float(value)
        except ValueError:
            continue
    return thresholds


def load_summary(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Trend summary not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def evaluate_status(
    symbol: str,
    pnl: float,
    suspend_map: Dict[str, float],
    resume_map: Dict[str, float],
) -> str:
    key = symbol.upper()
    suspend_threshold = suspend_map.get(key)
    resume_threshold = resume_map.get(key)
    if suspend_threshold is not None and pnl <= suspend_threshold:
        return "suspended"
    if resume_threshold is not None and pnl > resume_threshold:
        return "resume_ready"
    return "neutral"


def main() -> None:
    parser = argparse.ArgumentParser(description="Trend candidate screener")
    parser.add_argument(
        "summary",
        type=Path,
        nargs="?",
        default=Path("marketsimulator/run_logs/trend_summary.json"),
        help="Path to trend_summary.json (default: marketsimulator/run_logs/trend_summary.json)",
    )
    parser.add_argument(
        "--sma-threshold",
        type=float,
        default=0.0,
        help="Minimum SMA required to surface a candidate (default: 0.0)",
    )
    parser.add_argument(
        "--auto-threshold",
        action="store_true",
        help="Derive SMA threshold from current trend summary (mean of positive SMA values).",
    )
    args = parser.parse_args()

    summary = load_summary(args.summary)
    suspend_map = parse_threshold_map(os.getenv("MARKETSIM_TREND_PNL_SUSPEND_MAP"))
    resume_map = parse_threshold_map(os.getenv("MARKETSIM_TREND_PNL_RESUME_MAP"))

    sma_threshold = args.sma_threshold
    if args.auto_threshold:
        positive_smas = [
            float(stats.get("sma", 0.0))
            for symbol, stats in summary.items()
            if symbol.upper() != "__OVERALL__" and float(stats.get("sma", 0.0)) > 0
        ]
        if positive_smas:
            auto_value = sum(positive_smas) / len(positive_smas)
            sma_threshold = max(sma_threshold, auto_value)
            print(f"[info] Auto SMA threshold={auto_value:.2f}, using {sma_threshold:.2f}")
        else:
            print("[info] Auto SMA threshold unavailable (no positive SMA values); using manual threshold.")

    candidates: list[Tuple[str, float, float, str]] = []
    for symbol, stats in summary.items():
        if symbol.upper() == "__OVERALL__":
            continue
        sma = float(stats.get("sma", 0.0))
        pnl = float(stats.get("pnl", 0.0))
        status = evaluate_status(symbol, pnl, suspend_map, resume_map)
        if sma >= sma_threshold:
            candidates.append((symbol, sma, pnl, status))

    candidates.sort(key=lambda item: item[1], reverse=True)

    if not candidates:
        print("[info] No symbols met the SMA threshold.")
        return

    print("Symbol  | SMA      | Trend PnL | Status")
    print("--------|----------|-----------|----------------")
    for symbol, sma, pnl, status in candidates:
        print(f"{symbol:>6} | {sma:>8.2f} | {pnl:>9.2f} | {status}")


if __name__ == "__main__":
    main()
