#!/usr/bin/env python3
"""
Run guard-confirm backtests for the configured symbol list and refresh summaries.

Usage:
    python evaltests/run_guard_backtests.py            # run all targets
    python evaltests/run_guard_backtests.py --symbols AAPL GOOG
    python evaltests/run_guard_backtests.py --dry-run  # show commands only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS_PATH = REPO_ROOT / "evaltests" / "guard_backtest_targets.json"
SUMMARY_SCRIPTS = [
    ["python", "evaltests/guard_metrics_summary.py"],
    ["python", "evaltests/summarise_mock_backtests.py"],
    ["python", "evaltests/summarise_guard_vs_baseline.py"],
]


def load_targets(path: Path) -> list[Mapping[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Target configuration not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def filter_targets(targets: Iterable[Mapping[str, object]], symbols: set[str] | None) -> list[Mapping[str, object]]:
    if not symbols:
        return list(targets)
    filtered = [t for t in targets if str(t.get("symbol")).upper() in symbols]
    missing = symbols - {str(t.get("symbol")).upper() for t in filtered}
    if missing:
        raise ValueError(f"Requested symbols not in configuration: {', '.join(sorted(missing))}")
    return filtered


def run_backtest(target: Mapping[str, object], dry_run: bool = False) -> None:
    symbol = str(target.get("symbol"))
    output_json = target.get("output_json")
    if not output_json:
        raise ValueError(f"Target {symbol} missing 'output_json'")

    cmd = [
        sys.executable,
        "backtest_test3_inline.py",
        symbol,
        "--output-json",
        str(output_json),
    ]

    output_label = target.get("output_label")
    if output_label:
        cmd.extend(["--output-label", str(output_label)])

    extra_args = target.get("args")
    if isinstance(extra_args, list):
        cmd.extend(str(arg) for arg in extra_args)

    env = os.environ.copy()
    target_env = target.get("env", {})
    if isinstance(target_env, Mapping):
        env.update({str(k): str(v) for k, v in target_env.items()})

    if dry_run:
        print(f"[dry-run] {' '.join(cmd)}")
        return

    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


def refresh_summaries(dry_run: bool = False) -> None:
    for command in SUMMARY_SCRIPTS:
        if dry_run:
            print(f"[dry-run] {' '.join(command)}")
            continue
        print(f"[refresh] {' '.join(command)}")
        subprocess.run(command, cwd=str(REPO_ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run guard-confirm backtests across configured symbols.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_TARGETS_PATH,
        help="Path to guard backtest configuration (default: evaltests/guard_backtest_targets.json).",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Subset of symbols to run (default: all configured symbols).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Skip refreshing guard summary artefacts after runs.",
    )
    args = parser.parse_args()

    targets = filter_targets(
        load_targets(args.config),
        {s.upper() for s in args.symbols} if args.symbols else None,
    )
    for target in targets:
        run_backtest(target, dry_run=args.dry_run)

    if not args.skip_summary:
        refresh_summaries(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
