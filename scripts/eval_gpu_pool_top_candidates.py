#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


REPO = Path(__file__).resolve().parents[1]


def load_leaderboard(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def pick_top_rows(
    rows: Sequence[dict[str, str]],
    *,
    sort_by: str,
    top_k: int,
) -> list[dict[str, str]]:
    ranked = sorted(rows, key=lambda row: float(row.get(sort_by, "-inf")), reverse=True)
    return ranked[: max(0, int(top_k))]


def resolve_checkpoint_path(checkpoint_root: Path, description: str, gpu_id: str) -> Path:
    trial_dir = checkpoint_root / f"gpu{gpu_id}" / description
    for name in ("best.pt", "val_best.pt", "final.pt"):
        candidate = trial_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No checkpoint found under {trial_dir}")


def build_eval_command(
    *,
    checkpoint_path: Path,
    val_data: Path,
    out_dir: Path,
    hourly_data_root: Path,
    daily_start_date: str,
) -> list[str]:
    return [
        sys.executable,
        "-u",
        "scripts/eval_100d.py",
        "--checkpoint",
        str(checkpoint_path),
        "--val-data",
        str(val_data),
        "--out-dir",
        str(out_dir),
        "--fail-fast-max-dd",
        "0.20",
        "--monthly-target",
        "0.27",
        "--execution-granularity",
        "hourly_intrabar",
        "--hourly-data-root",
        str(hourly_data_root),
        "--daily-start-date",
        str(daily_start_date),
    ]


def run_eval(cmd: Sequence[str], *, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(shlex.quote(str(part)) for part in cmd) + "\n\n")
        handle.flush()
        proc = subprocess.run(list(cmd), cwd=str(REPO), check=False, stdout=handle, stderr=subprocess.STDOUT, text=True)
    return int(proc.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run hourly eval_100d on top gpu_pool leaderboard candidates.")
    parser.add_argument("--leaderboard", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--hourly-data-root", type=Path, required=True)
    parser.add_argument("--daily-start-date", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--sort-by", default="val_return")
    args = parser.parse_args(list(argv) if argv is not None else None)

    rows = load_leaderboard(Path(args.leaderboard))
    winners = pick_top_rows(rows, sort_by=str(args.sort_by), top_k=int(args.top_k))
    summary: list[dict[str, Any]] = []
    for row in winners:
        description = str(row["description"])
        gpu_id = str(row["gpu_id"])
        checkpoint_path = resolve_checkpoint_path(Path(args.checkpoint_root), description, gpu_id)
        out_dir = Path(args.out_root) / description
        eval_cmd = build_eval_command(
            checkpoint_path=checkpoint_path,
            val_data=Path(args.val_data),
            out_dir=out_dir,
            hourly_data_root=Path(args.hourly_data_root),
            daily_start_date=str(args.daily_start_date),
        )
        rc = run_eval(eval_cmd, log_path=out_dir / "eval100d.log")
        summary.append(
            {
                "description": description,
                "gpu_id": gpu_id,
                "checkpoint": str(checkpoint_path),
                "returncode": int(rc),
                "eval_dir": str(out_dir),
            }
        )
    summary_path = Path(args.out_root) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(f"{json.dumps(summary, indent=2, sort_keys=True)}\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
