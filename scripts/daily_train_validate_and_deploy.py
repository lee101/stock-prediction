#!/usr/bin/env python3
"""
One-shot pipeline:
1) Refresh daily data
2) Train 1 epoch (fit-all-data)
3) Run 10-day simulator
4) Compare vs baseline checkpoint; deploy if better

Intended to run once per day near market open. Safe to re-run; uses timestamped
output dirs and only swaps when the new model wins.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from zoneinfo import ZoneInfo


def _is_market_open_window(start: str = "09:30", end: str = "10:30") -> bool:
    now_et = datetime.now(timezone.utc).astimezone(ZoneInfo("US/Eastern")).time()
    h1, m1 = map(int, start.split(":"))
    h2, m2 = map(int, end.split(":"))
    return (h1, m1) <= (now_et.hour, now_et.minute) <= (h2, m2)


def run_cmd(cmd: list[str]) -> str:
    logger.info("Running: %s", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(Path.cwd()))
    out = subprocess.check_output(cmd, text=True, env=env)
    return out.strip()


def simulate(checkpoint: Path) -> float:
    cmd = [
        sys.executable,
        "neuraldailymarketsimulator/simulator.py",
        "--checkpoint",
        str(checkpoint),
        "--start-date",
        "2025-11-11",
        "--days",
        "10",
        "--stock-fee",
        "0.0005",
        "--crypto-fee",
        "0.0008",
        "--ignore-non-tradable",
        "--confidence-threshold",
        "0.2",
        "--stocks-closed",
    ]
    out = run_cmd(cmd)
    # Parse final equity line
    equity = None
    for line in out.splitlines():
        if line.lower().startswith("final equity"):
            equity = float(line.split(":")[1].strip())
            break
    if equity is None:
        raise RuntimeError("Could not parse final equity from simulator output.")
    return equity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Baseline checkpoint to beat.")
    parser.add_argument("--force", action="store_true", help="Bypass market-open window check.")
    parser.add_argument("--out-root", default="neuraldailytraining/checkpoints/auto_daily")
    args = parser.parse_args()

    if not args.force and not _is_market_open_window():
        logger.info("Outside market-open window; skipping.")
        return

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / f"run_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Refresh data (use current interpreter + repo PYTHONPATH)
    run_cmd([sys.executable, "update_daily_data.py"])

    # 2) Train fit-all-data 1 epoch
    run_cmd(
        [
            sys.executable,
            "neural_trade_stock_e2e.py",
            "--mode",
            "train",
            "--fit-all-data",
            "--epochs",
            "1",
            "--batch-size",
            "32",
            "--sequence-length",
            "128",
            "--learning-rate",
            "3e-4",
            "--checkpoint-root",
            str(out_dir),
            "--run-name",
            f"auto_daily_{ts}",
        ]
    )
    new_ckpt = next(out_dir.rglob("epoch_0001.pt"))

    # 3) Simulate both
    new_equity = simulate(new_ckpt)
    base_equity = simulate(Path(args.baseline))
    logger.info("Baseline equity=%.4f | New equity=%.4f", base_equity, new_equity)

    # 4) Deploy if better
    if new_equity > base_equity:
        target = Path("neuraldailytraining/checkpoints/active_latest.pt")
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(new_ckpt.resolve())
        logger.info("Deployed new checkpoint: %s", new_ckpt)
    else:
        logger.info("Keeping baseline; new model not better.")


if __name__ == "__main__":
    main()
