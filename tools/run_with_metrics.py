#!/usr/bin/env python3
"""Execute the trading loop, capture its log, and emit structured metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Sequence

from tools.extract_metrics import extract_metrics


def run_trade_loop(cmd: Sequence[str], log_path: Path) -> int:
    """Run the given command and store combined stdout/stderr in log_path."""
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return completed.returncode


def write_summary(log_path: Path, summary_path: Path) -> None:
    metrics = extract_metrics(log_path.read_text(encoding="utf-8", errors="ignore"))
    serialisable = {
        key: (None if value is None else round(float(value), 10))
        for key, value in metrics.items()
    }
    summary_path.write_text(
        json.dumps(serialisable, indent=2, sort_keys=True), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        type=Path,
        required=True,
        help="Destination file for captured stdout/stderr.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Path to write the JSON metrics summary.",
    )
    parser.add_argument(
        "trade_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to python -m marketsimulator.run_trade_loop.",
    )
    args = parser.parse_args()

    trade_args = args.trade_args
    if trade_args and trade_args[0] == "--":
        trade_args = trade_args[1:]

    if not trade_args:
        raise SystemExit("No trade loop arguments provided.")

    cmd = ["python", "-m", "marketsimulator.run_trade_loop", *trade_args]
    return_code = run_trade_loop(cmd, args.log)
    write_summary(args.log, args.summary)

    if return_code != 0:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
