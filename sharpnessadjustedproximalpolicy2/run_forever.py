#!/usr/bin/env python3
"""Auto-research forever loop for SAP experiments.

Cycles through symbols and experiment configs, logging results to JSON/markdown.
Stops individual experiments early when val score degrades.
Saves top-K checkpoints.

Usage:
    python -m sharpnessadjustedproximalpolicy2.run_forever
    python -m sharpnessadjustedproximalpolicy2.run_forever --symbols DOGEUSD BTCUSD --max-rounds 50
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .config import DEFAULT_TRAINING_OVERRIDES, EXPERIMENTS, SAPConfig, SYMBOLS_CORE
from .sweep import build_configs, run_single_experiment

LOG_DIR = Path("sharpnessadjustedproximalpolicy2") / "analysis"
PROGRESS_DIR = Path("sharpnessadjustedproximalpolicy2")
CHECKPOINT_DIR = Path("sharpnessadjustedproximalpolicy2") / "checkpoints"


def load_leaderboard(path: Path) -> list[dict]:
    if path.exists():
        return json.loads(path.read_text())
    return []


def save_leaderboard(path: Path, data: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def append_log(path: Path, entry: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def write_progress(result: dict, round_num: int, is_best: bool):
    if is_best:
        idx = 1
        while (PROGRESS_DIR / f"sap2_progress{idx}.md").exists():
            idx += 1
        path = PROGRESS_DIR / f"sap2_progress{idx}.md"
        content = f"""# SAP Progress #{idx} - {time.strftime('%Y-%m-%d %H:%M')}

## New Best: {result['name']} on {result['symbol']}
- Val Sortino: {result.get('best_val_sortino', 0):.3f}
- Val Return: {result.get('best_val_return', 0):.4f}
- Best Epoch: {result.get('best_epoch', 0)}
- SAM Mode: {result.get('sam_mode', 'none')}
- Rho: {result.get('rho', 0)}
- Final Sharpness EMA: {result.get('final_sharpness_ema', 0):.4f}
- Step Scale: {result.get('final_step_scale', result.get('final_lr_scale', 1)):.3f}
- Checkpoint: {result.get('checkpoint_dir', '')}
- Wall Time: {result.get('wall_time_s', 0):.1f}s
"""
        path.write_text(content)
        print(f"Progress written to {path}")
    else:
        path = PROGRESS_DIR / "sap2_progress_failed.md"
        with open(path, "a") as f:
            f.write(
                f"\n## Round {round_num}: {result['name']} on {result['symbol']} "
                f"({time.strftime('%Y-%m-%d %H:%M')})\n"
                f"- Sort: {result.get('best_val_sortino', 0):.3f} "
                f"Ret: {result.get('best_val_return', 0):.4f} "
                f"Sharp: {result.get('final_sharpness_ema', 0):.4f}\n"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS_CORE)
    parser.add_argument("--max-rounds", type=int, default=0, help="0=forever")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--experiments", type=str, default=None, help="comma-sep filter")
    args = parser.parse_args()

    exps = EXPERIMENTS
    if args.experiments:
        names = set(args.experiments.split(","))
        exps = [e for e in EXPERIMENTS if e["name"] in names]

    leaderboard_path = LOG_DIR / "sap2_leaderboard.json"
    log_path = LOG_DIR / "sap2_forever_log.jsonl"
    leaderboard = load_leaderboard(leaderboard_path)

    best_score_by_symbol: dict[str, float] = {}
    for entry in leaderboard:
        sym = entry.get("symbol", "")
        sc = entry.get("best_val_score", float("-inf"))
        if sc > best_score_by_symbol.get(sym, float("-inf")):
            best_score_by_symbol[sym] = sc

    combos = list(itertools.product(exps, args.symbols))
    round_num = 0

    while True:
        if args.shuffle:
            random.shuffle(combos)

        for exp, symbol in combos:
            round_num += 1
            if args.max_rounds and round_num > args.max_rounds:
                print(f"Reached max rounds ({args.max_rounds}), exiting")
                return

            overrides = {"epochs": args.epochs}
            print(f"\n[Round {round_num}] {exp['name']} / {symbol}")

            result = run_single_experiment(exp, symbol, overrides)
            result["round"] = round_num
            result["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            append_log(log_path, result)

            if result.get("error"):
                write_progress(result, round_num, is_best=False)
                continue

            score = result.get("best_val_score", float("-inf"))
            prev_best = best_score_by_symbol.get(symbol, float("-inf"))
            is_best = score > prev_best

            if is_best:
                best_score_by_symbol[symbol] = score
                print(f"NEW BEST for {symbol}: {score:.4f} (was {prev_best:.4f})")

            leaderboard.append(result)
            save_leaderboard(leaderboard_path, leaderboard)
            write_progress(result, round_num, is_best=is_best)

        if args.max_rounds and round_num >= args.max_rounds:
            break

    print("Forever loop finished")


if __name__ == "__main__":
    main()
