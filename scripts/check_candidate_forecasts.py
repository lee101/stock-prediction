#!/usr/bin/env python3
"""Run short simulations for leading ETF candidates to capture gate failure reasons."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


def load_latest_candidates(history_path: Path, min_sma: float, min_pct: float) -> List[str]:
    if not history_path.exists():
        raise FileNotFoundError(f"Candidate readiness history not found: {history_path}")
    with history_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader]
    if not rows:
        return []
    rows_by_ts: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_ts[row["timestamp"]].append(row)
    latest_ts = max(rows_by_ts.keys())
    latest_rows = rows_by_ts[latest_ts]
    candidates = []
    for row in latest_rows:
        sma = float(row["sma"])
        pct = float(row["pct_change"])
        symbol = row["symbol"].upper()
        if sma >= min_sma and pct >= min_pct:
            candidates.append(symbol)
    return candidates


def run_probe(symbol: str, steps: int, env_overrides: Dict[str, str]) -> Tuple[int, str, str]:
    cmd = [
        "python",
        "marketsimulator/run_trade_loop.py",
        "--symbols",
        symbol,
        "--steps",
        str(steps),
        "--step-size",
        "1",
        "--initial-cash",
        "100000",
        "--kronos-only",
        "--flatten-end",
        "--compact-logs",
    ]
    env = os.environ.copy()
    env.update(env_overrides)
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
    )
    return proc.returncode, proc.stdout, proc.stderr


def analyze_output(symbol: str, stdout: str) -> Counter:
    counter: Counter = Counter()
    for line in stdout.splitlines():
        if f"Skipping {symbol}" in line:
            counter[line.strip()] += 1
    if not counter:
        counter["no_skip_detected"] += 1
    return counter


def build_env(symbol: str, min_strategy_return: float, min_pred_move: float) -> Dict[str, str]:
    return {
        "MARKETSIM_SYMBOL_MIN_STRATEGY_RETURN_MAP": f"{symbol}:{min_strategy_return}",
        "MARKETSIM_SYMBOL_MIN_PREDICTED_MOVE_MAP": f"{symbol}:{min_pred_move}",
        "MARKETSIM_SYMBOL_MAX_ENTRIES_MAP": f"{symbol}:6",
        "MARKETSIM_SYMBOL_MAX_HOLD_SECONDS_MAP": f"{symbol}:10800",
    }


def render_report(results: Dict[str, Counter], output_path: Path) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    lines: List[str] = []
    lines.append(f"# Candidate Forecast Gate Report — {timestamp}\n")
    for symbol, counter in results.items():
        lines.append(f"## {symbol}")
        total = sum(counter.values())
        for reason, count in counter.most_common():
            percent = (count / total) * 100 if total else 0
            lines.append(f"- {reason} ({count} occurrences, {percent:.1f}%)")
        lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[info] Forecast gate report written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose gate failures for ETF candidates.")
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("marketsimulator/run_logs/candidate_readiness_history.csv"),
        help="Path to candidate readiness history CSV.",
    )
    parser.add_argument(
        "--min-sma",
        type=float,
        default=200.0,
        help="Minimum SMA to consider (default 200).",
    )
    parser.add_argument(
        "--min-pct",
        type=float,
        default=0.0,
        help="Minimum fractional percent change to consider (default 0).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Number of steps for the probe simulation (default 1).",
    )
    parser.add_argument(
        "--min-strategy-return",
        type=float,
        default=0.015,
        help="Strategy return gate to test (default 0.015).",
    )
    parser.add_argument(
        "--min-predicted-move",
        type=float,
        default=0.01,
        help="Predicted move gate to test (default 0.01).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("marketsimulator/run_logs/candidate_forecast_gate_report.md"),
        help="Markdown report destination.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional CSV path to append raw shortfall data.",
    )
    args = parser.parse_args()

    candidates = load_latest_candidates(args.history, args.min_sma, args.min_pct)
    if not candidates:
        print("[info] No candidates meet SMA/% change thresholds.")
        return

    results: Dict[str, Counter] = {}
    for symbol in candidates:
        env = build_env(symbol, args.min_strategy_return, args.min_predicted_move)
        code, stdout, stderr = run_probe(symbol, args.steps, env)
        if code != 0:
            counter = Counter()
            counter[f"probe_failed (exit {code})"] += 1
            results[symbol] = counter
            print(f"[warn] Probe for {symbol} failed (exit {code}): {stderr.strip()}")
            continue
        counter = analyze_output(symbol, stdout)
        results[symbol] = counter
        print(f"[info] Probe completed for {symbol} ({sum(counter.values())} entries).")

    render_report(results, args.output)

    if args.csv_output:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        write_header = not args.csv_output.exists()
        with args.csv_output.open("a", encoding="utf-8") as handle:
            if write_header:
                handle.write("timestamp,symbol,reason,count\n")
            for symbol, counter in results.items():
                for reason, count in counter.items():
                    handle.write(f"{datetime.now(timezone.utc).isoformat()},{symbol},{reason},{count}\n")


if __name__ == "__main__":
    main()
