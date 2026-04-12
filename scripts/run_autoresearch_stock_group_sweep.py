from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


REPO = Path(__file__).resolve().parents[1]


EXPERIMENT_FLAG_SETS: dict[str, tuple[str, ...]] = {
    "base": (),
    "dynamic_score_floor": ("--dynamic-score-floor",),
    "soft_rank_sizing": ("--soft-rank-sizing",),
    "timestamp_budget_head": ("--timestamp-budget-head",),
    "budget_guided_keep_count": ("--budget-guided-keep-count",),
    "continuous_budget_thresholds": ("--continuous-budget-thresholds",),
    "budget_entropy_confidence": ("--budget-entropy-confidence",),
    "budget_consensus_dispersion": ("--budget-consensus-dispersion",),
}

_METRIC_PATTERNS = {
    "robust_score": re.compile(r"^robust_score:\s+([+-]?\d+(?:\.\d+)?)\s*$", re.MULTILINE),
    "val_loss": re.compile(r"^val_loss:\s+([+-]?\d+(?:\.\d+)?)\s*$", re.MULTILINE),
    "training_seconds": re.compile(r"^training_seconds:\s+([+-]?\d+(?:\.\d+)?)\s*$", re.MULTILINE),
    "checkpoint_dir": re.compile(r"^checkpoint_dir:\s+(.+?)\s*$", re.MULTILINE),
    "saved_checkpoint": re.compile(r"^saved_checkpoint:\s+(.+?)\s*$", re.MULTILINE),
    "best_checkpoint": re.compile(r"^best_checkpoint:\s*(.*?)\s*$", re.MULTILINE),
}


@dataclass(frozen=True)
class GroupSweepSpec:
    symbols: tuple[str, ...]
    experiment: str


def discover_symbols(data_root: Path, *, limit: int | None = None) -> list[str]:
    symbols = sorted(path.stem.upper() for path in data_root.glob("*.csv"))
    return symbols[: max(int(limit or 0), 0)] if limit is not None else symbols


def build_symbol_groups(
    symbols: Sequence[str],
    *,
    group_size: int,
    max_groups: int | None = None,
) -> list[tuple[str, ...]]:
    cleaned = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    groups = list(itertools.combinations(cleaned, int(group_size)))
    if max_groups is not None:
        return groups[: max(int(max_groups), 0)]
    return groups


def parse_train_stdout(stdout: str) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for key, pattern in _METRIC_PATTERNS.items():
        match = pattern.search(stdout or "")
        if not match:
            continue
        raw_value = match.group(1).strip()
        if key in {"checkpoint_dir", "saved_checkpoint", "best_checkpoint"}:
            metrics[key] = raw_value
        else:
            metrics[key] = float(raw_value)
    return metrics


def build_train_command(
    *,
    python_executable: str,
    data_root: Path,
    symbols: Sequence[str],
    frequency: str,
    experiment: str,
    hold_bars: int,
    eval_windows: str,
    max_positions: int,
    sequence_length: int | None,
    batch_size: int | None,
    eval_batch_size: int | None,
    hidden_size: int | None,
    layers: int | None,
    checkpoint_dir: Path,
    dashboard_db: str,
    disable_auto_lr_find: bool,
    device: str,
    extra_args: Sequence[str],
) -> list[str]:
    command = [
        str(python_executable),
        "-m",
        "autoresearch_stock.train",
        "--frequency",
        str(frequency),
        "--data-root",
        str(data_root),
        "--symbols",
        ",".join(symbols),
        "--hold-bars",
        str(int(hold_bars)),
        "--eval-windows",
        str(eval_windows),
        "--max-positions",
        str(int(max_positions)),
        "--dashboard-db",
        str(dashboard_db),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--device",
        str(device),
    ]
    if sequence_length is not None:
        command.extend(["--sequence-length", str(int(sequence_length))])
    if batch_size is not None:
        command.extend(["--batch-size", str(int(batch_size))])
    if eval_batch_size is not None:
        command.extend(["--eval-batch-size", str(int(eval_batch_size))])
    if hidden_size is not None:
        command.extend(["--hidden-size", str(int(hidden_size))])
    if layers is not None:
        command.extend(["--layers", str(int(layers))])
    if disable_auto_lr_find:
        command.append("--disable-auto-lr-find")
    command.extend(EXPERIMENT_FLAG_SETS.get(str(experiment), ()))
    command.extend(str(arg) for arg in extra_args)
    return command


def _python_has_cuda_torch(python_executable: str) -> bool:
    probe = [
        str(python_executable),
        "-c",
        (
            "import json; "
            "import torch; "
            "print(json.dumps({'cuda': bool(torch.cuda.is_available())}))"
        ),
    ]
    try:
        completed = subprocess.run(
            probe,
            cwd=str(REPO),
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
    except Exception:
        return False
    if completed.returncode != 0:
        return False
    try:
        payload = json.loads((completed.stdout or "").strip())
    except json.JSONDecodeError:
        return False
    return bool(payload.get("cuda"))


def resolve_python_executable(requested: str | None) -> str:
    if requested and str(requested).strip():
        return str(requested)
    candidates = [
        REPO / ".venv" / "bin" / "python",
        REPO / ".venv312" / "bin" / "python",
        REPO / ".venv313" / "bin" / "python",
        Path(sys.executable),
    ]
    for candidate in candidates:
        if not Path(candidate).exists():
            continue
        if _python_has_cuda_torch(str(candidate)):
            return str(candidate)
    return str(sys.executable)


def _default_output_path() -> Path:
    return REPO / "trade_stock_wide" / "rl_group_sweep_latest.csv"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run bounded autoresearch_stock sweeps for single-name or pair RL experiments")
    ap.add_argument("--symbols", default=None, help="Comma-separated symbols. Defaults to data-root discovery.")
    ap.add_argument("--data-root", default="trainingdatahourly/stocks")
    ap.add_argument("--frequency", default="hourly", choices=("hourly", "daily"))
    ap.add_argument("--group-size", type=int, default=1)
    ap.add_argument("--limit-symbols", type=int, default=8)
    ap.add_argument("--max-groups", type=int, default=8)
    ap.add_argument("--experiments", default="base,dynamic_score_floor,soft_rank_sizing,budget_entropy_confidence")
    ap.add_argument("--time-budget-seconds", type=int, default=60)
    ap.add_argument("--hold-bars", type=int, default=3)
    ap.add_argument("--eval-windows", default="8,16")
    ap.add_argument("--max-positions", type=int, default=2)
    ap.add_argument("--sequence-length", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--eval-batch-size", type=int, default=128)
    ap.add_argument("--hidden-size", type=int, default=32)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dashboard-db", default="dashboards/metrics.db")
    ap.add_argument("--python-executable", default=None)
    ap.add_argument("--checkpoint-root", default="checkpoints/autoresearch_stock/group_sweep")
    ap.add_argument("--output", default=str(_default_output_path()))
    ap.add_argument("--disable-auto-lr-find", action="store_true")
    ap.add_argument("--check-inputs-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--extra-arg", action="append", default=[], help="Additional arg passed through to autoresearch_stock.train")
    args = ap.parse_args(list(argv) if argv is not None else None)

    data_root = Path(args.data_root)
    if args.symbols:
        symbols = [item.strip().upper() for item in str(args.symbols).split(",") if item.strip()]
    else:
        symbols = discover_symbols(data_root, limit=args.limit_symbols)
    if not symbols:
        raise SystemExit("No symbols resolved for autoresearch_stock group sweep")

    experiments = [item.strip() for item in str(args.experiments).split(",") if item.strip()]
    unknown_experiments = sorted(set(experiments) - set(EXPERIMENT_FLAG_SETS))
    if unknown_experiments:
        raise SystemExit(f"Unknown experiments: {unknown_experiments}")

    groups = build_symbol_groups(symbols, group_size=args.group_size, max_groups=args.max_groups)
    if not groups:
        raise SystemExit("No groups resolved for autoresearch_stock group sweep")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    python_executable = resolve_python_executable(args.python_executable)
    print(f"python_executable={python_executable}")
    for group_index, group in enumerate(groups, start=1):
        for experiment in experiments:
            group_key = "-".join(group)
            checkpoint_dir = Path(args.checkpoint_root) / f"{group_key}__{experiment}"
            command = build_train_command(
                python_executable=python_executable,
                data_root=data_root,
                symbols=group,
                frequency=args.frequency,
                experiment=experiment,
                hold_bars=args.hold_bars,
                eval_windows=args.eval_windows,
                max_positions=args.max_positions,
                sequence_length=args.sequence_length,
                batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,
                hidden_size=args.hidden_size,
                layers=args.layers,
                checkpoint_dir=checkpoint_dir,
                dashboard_db=args.dashboard_db,
                disable_auto_lr_find=bool(args.disable_auto_lr_find),
                device=args.device,
                extra_args=(["--check-inputs-text"] if args.check_inputs_only else []) + list(args.extra_arg),
            )
            print(f"[{group_index}/{len(groups)}] experiment={experiment} symbols={','.join(group)}")
            print("  " + " ".join(command))
            row: dict[str, object] = {
                "symbols": ",".join(group),
                "group_size": int(args.group_size),
                "experiment": experiment,
                "command": " ".join(command),
                "checkpoint_dir": str(checkpoint_dir),
                "status": "dry_run" if args.dry_run else "pending",
            }
            if not args.dry_run:
                env = os.environ.copy()
                env["AUTORESEARCH_STOCK_TIME_BUDGET_SECONDS"] = str(int(args.time_budget_seconds))
                process = subprocess.run(
                    command,
                    cwd=str(REPO),
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                row["returncode"] = int(process.returncode)
                row["status"] = "ok" if process.returncode == 0 else "error"
                row["stdout"] = process.stdout
                row["stderr"] = process.stderr
                row.update(parse_train_stdout(process.stdout))
            rows.append(row)

    fieldnames = [
        "symbols",
        "group_size",
        "experiment",
        "status",
        "returncode",
        "robust_score",
        "val_loss",
        "training_seconds",
        "checkpoint_dir",
        "saved_checkpoint",
        "best_checkpoint",
        "command",
        "stdout",
        "stderr",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print(f"\noutput_csv={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
