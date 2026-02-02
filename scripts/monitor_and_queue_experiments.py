#!/usr/bin/env python3
"""Monitor running experiments and queue follow-up commands when they finish."""

from __future__ import annotations

import argparse
import datetime as dt
import shlex
import subprocess
import time
from pathlib import Path
from typing import List


def _now() -> str:
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()


def _pgrep(pattern: str) -> bool:
    """Return True if any process matches the pattern."""
    result = subprocess.run(
        ["pgrep", "-f", pattern],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _run_command(command: str, log_path: Path) -> int:
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[{_now()}] START: {command}\n")
        log.flush()
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=log,
            stderr=log,
            executable="/bin/bash",
        )
        code = proc.wait()
        log.write(f"[{_now()}] END: {command} (exit={code})\n")
        log.flush()
        return code


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor processes and queue commands.")
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Process search pattern to wait for (repeatable).",
    )
    parser.add_argument(
        "--command",
        action="append",
        default=[],
        help="Command to run once all patterns are gone (repeatable).",
    )
    parser.add_argument("--interval", type=int, default=300, help="Polling interval in seconds.")
    parser.add_argument(
        "--log-file",
        type=str,
        default="reports/monitor_queue.log",
        help="Log file for queued command output.",
    )
    args = parser.parse_args()

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    patterns: List[str] = list(args.pattern)
    commands: List[str] = list(args.command)

    if not commands:
        raise SystemExit("No --command provided.")

    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[{_now()}] Monitoring patterns: {patterns}\n")
        log.write(f"[{_now()}] Queued commands: {commands}\n")
        log.flush()

    while True:
        active = [pat for pat in patterns if _pgrep(pat)]
        if not active:
            break
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"[{_now()}] Active patterns: {active}\n")
            log.flush()
        time.sleep(max(1, int(args.interval)))

    for command in commands:
        _run_command(command, log_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
