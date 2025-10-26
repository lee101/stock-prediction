from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import extract_metrics

DEFAULT_COMMAND = ["python", "-m", "marketsimulator.run_trade_loop"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a trading simulation and extract structured metrics from its log output."
    )
    parser.add_argument(
        "--log",
        required=True,
        type=Path,
        help="Path to write the combined stdout/stderr log from the simulation run.",
    )
    parser.add_argument(
        "--summary",
        required=True,
        type=Path,
        help="Where to write the extracted metrics JSON payload.",
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        default=None,
        help="Optional working directory for the simulation command.",
    )
    parser.add_argument(
        "trade_args",
        nargs=argparse.REMAINDER,
        help=(
            "Command to execute (defaults to %(default)s). "
            "Prefix with '--' to pass only flags (e.g. '-- --stub-config')."
        ),
        default=[],
    )
    return parser.parse_args(argv)


def build_command(args: argparse.Namespace) -> list[str]:
    trade_args = list(args.trade_args)
    if not trade_args:
        return DEFAULT_COMMAND.copy()

    if trade_args[0] == "--":
        trade_args = trade_args[1:]

    if not trade_args or trade_args[0].startswith("--"):
        return DEFAULT_COMMAND + trade_args

    return trade_args


def run_with_metrics(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    command = build_command(args)

    log_path = args.log
    summary_path = args.summary
    cwd = args.cwd

    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=cwd,
    )

    log_content = "\n".join(
        [
            f"$ {' '.join(command)}",
            proc.stdout.strip(),
            proc.stderr.strip(),
        ]
    ).strip() + "\n"
    log_path.write_text(log_content, encoding="utf-8")

    metrics = extract_metrics.extract_metrics(log_content)
    metrics["command"] = command
    metrics["returncode"] = proc.returncode
    summary_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    return proc.returncode


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(run_with_metrics(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
