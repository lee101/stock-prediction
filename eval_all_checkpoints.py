#!/usr/bin/env python3
"""Evaluate all trained checkpoints against the latest validation data.

Discovers all checkpoint directories, runs evaluation on each with the latest
val data, and produces a comprehensive leaderboard showing which checkpoints
work best on the most recent data.

Usage:
  python scripts/eval_all_checkpoints.py --dry-run
  python scripts/eval_all_checkpoints.py --data-path pufferlib_market/data/mixed23_latest_val_20250922_20260320.bin
  python scripts/eval_all_checkpoints.py --checkpoints-dir pufferlib_market/checkpoints/ --output leaderboard_latest.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime
import struct
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Default data paths to try in order
DEFAULT_VAL_DATA_CANDIDATES = [
    "pufferlib_market/data/mixed23_latest_val_20250922_20260320.bin",
    "pufferlib_market/data/mixed23_fresh_val.bin",
    "pufferlib_market/data/mixed23_daily_val.bin",
]

DEFAULT_CHECKPOINTS_DIR = "pufferlib_market/checkpoints"

# Matches evaluate_multiperiod.PERIODS — kept here to avoid circular import at module level
_KNOWN_PERIODS: dict[str, int] = {"1d": 24, "7d": 168, "30d": 720}


def _parse_periods(periods_str: str) -> dict[str, int]:
    """Parse a comma-separated periods string into a name->hours dict."""
    period_map: dict[str, int] = {}
    for p in periods_str.split(","):
        p = p.strip()
        if p in _KNOWN_PERIODS:
            period_map[p] = _KNOWN_PERIODS[p]
        elif p.endswith("d") and p[:-1].isdigit():
            period_map[p] = int(p[:-1]) * 24
        elif p.endswith("h") and p[:-1].isdigit():
            period_map[p] = int(p[:-1])
    return period_map or {"30d": 720}


def discover_best_checkpoints(checkpoints_dir: Path) -> list[Path]:
    """Find all best.pt files under checkpoints_dir (2 levels deep)."""
    found: list[Path] = []
    for pt in checkpoints_dir.rglob("best.pt"):
        found.append(pt)
    # Also include final.pt if no best.pt in same dir
    best_dirs = {p.parent for p in found}
    for pt in checkpoints_dir.rglob("final.pt"):
        if pt.parent not in best_dirs:
            found.append(pt)
    return sorted(found)


def checkpoint_label(pt_path: Path, checkpoints_dir: Path) -> str:
    """Build a short human-readable label for a checkpoint path."""
    try:
        rel = pt_path.relative_to(checkpoints_dir)
    except ValueError:
        rel = pt_path
    return str(rel)


def infer_num_symbols(data_path: Path) -> Optional[int]:
    """Read symbol count from MKTD binary header."""
    try:
        with data_path.open("rb") as f:
            header = f.read(24)
        if len(header) < 24:
            return None
        if header[:4] != b"MKTD":
            return None
        _, _, num_symbols, _, _, _ = struct.unpack("<4sIIIII", header[:24])
        return int(num_symbols)
    except Exception:
        return None


def run_evaluation(
    checkpoint_path: Path,
    data_path: Path,
    *,
    periods: str = "30d",
    fee_rate: float = 0.001,
    fill_buffer_bps: float = 5.0,
    max_leverage: float = 1.0,
    deterministic: bool = True,
    device: str = "cpu",
) -> Optional[dict]:
    """Evaluate a single checkpoint. Returns dict of metrics or None on error."""
    from pufferlib_market.evaluate_multiperiod import evaluate_checkpoint

    period_map = _parse_periods(periods)

    try:
        results = evaluate_checkpoint(
            str(checkpoint_path),
            str(data_path),
            period_map,
            fee_rate=fee_rate,
            fill_buffer_bps=fill_buffer_bps,
            max_leverage=max_leverage,
            deterministic=deterministic,
            device_str=device,
        )
    except Exception as e:
        return {"error": str(e)}

    # Flatten results for the primary period (30d if available, else first)
    primary_period = "30d" if "30d" in period_map else list(period_map.keys())[0]
    primary = next((r for r in results if r.get("period") == primary_period), None)
    if primary is None and results:
        primary = results[0]
    if primary is None:
        return {"error": "no results returned"}

    row: dict = {
        "checkpoint": str(checkpoint_path),
        "period": primary.get("period", ""),
        "total_return_pct": round(primary.get("total_return", 0.0) * 100, 4),
        "annualized_return_pct": round(primary.get("annualized_return", 0.0) * 100, 2),
        "sortino": round(primary.get("sortino", 0.0), 4),
        "max_drawdown_pct": round(primary.get("max_drawdown", 0.0) * 100, 4),
        "num_trades": primary.get("num_trades", 0),
        "win_rate": round(primary.get("win_rate", 0.0), 4),
        "avg_hold_steps": round(primary.get("avg_hold_steps", 0.0), 2),
        "error": primary.get("error", ""),
    }
    # Include additional periods as extra columns
    for r in results:
        p = r.get("period", "")
        if p and p != primary_period:
            suffix = f"_{p}"
            row[f"return{suffix}_pct"] = round(r.get("total_return", 0.0) * 100, 4)
            row[f"sortino{suffix}"] = round(r.get("sortino", 0.0), 4)
    return row


def print_leaderboard(rows: list[dict]) -> None:
    """Print a formatted leaderboard table to stdout."""
    if not rows:
        print("No results to display.")
        return
    sorted_rows = sorted(rows, key=lambda r: r.get("sortino", float("-inf")), reverse=True)
    header = f"{'Rank':<5} {'Checkpoint':<60} {'Return%':>9} {'Sortino':>8} {'MaxDD%':>7} {'Trades':>7} {'WinRate':>8}"
    print(header)
    print("-" * len(header))
    for rank, row in enumerate(sorted_rows, 1):
        label = str(row.get("checkpoint", ""))
        # Shorten label for display
        if len(label) > 58:
            label = "..." + label[-55:]
        ret = row.get("total_return_pct", 0.0)
        sortino = row.get("sortino", 0.0)
        dd = row.get("max_drawdown_pct", 0.0)
        trades = row.get("num_trades", 0)
        wr = row.get("win_rate", 0.0) * 100
        err = row.get("error", "")
        if err:
            print(f"{rank:<5} {label:<60}  ERROR: {err[:40]}")
        else:
            print(f"{rank:<5} {label:<60} {ret:>+8.2f}% {sortino:>8.2f} {dd:>6.2f}% {trades:>7d} {wr:>7.1f}%")


def save_leaderboard(rows: list[dict], output_path: Path) -> None:
    """Save leaderboard to CSV sorted by sortino descending."""
    if not rows:
        return
    sorted_rows = sorted(rows, key=lambda r: r.get("sortino", float("-inf")), reverse=True)
    # Collect all keys across all rows
    all_keys: list[str] = []
    seen: set[str] = set()
    priority = ["checkpoint", "period", "total_return_pct", "annualized_return_pct", "sortino",
                "max_drawdown_pct", "num_trades", "win_rate", "avg_hold_steps", "error"]
    for k in priority:
        if k not in seen:
            all_keys.append(k)
            seen.add(k)
    for row in sorted_rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted_rows)
    print(f"\nLeaderboard saved to: {output_path}")


def resolve_default_data_path() -> Optional[Path]:
    for candidate in DEFAULT_VAL_DATA_CANDIDATES:
        p = ROOT / candidate
        if p.exists():
            return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate all trained checkpoints against the latest validation data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoints-dir",
        default=DEFAULT_CHECKPOINTS_DIR,
        help="Root directory containing checkpoint subdirs (default: pufferlib_market/checkpoints/)",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to MKTD .bin validation file. Defaults to the latest mixed23 val file.",
    )
    parser.add_argument(
        "--output",
        default="leaderboard_latest.csv",
        help="Output CSV path (default: leaderboard_latest.csv)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List discovered checkpoints without running evaluation",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="Only evaluate the N most recently modified checkpoints (0=all)",
    )
    parser.add_argument(
        "--periods",
        default="30d",
        help="Comma-separated evaluation periods e.g. '1d,7d,30d' (default: 30d)",
    )
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--max-leverage", type=float, default=1.0)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--checkpoints-list",
        default=None,
        help="Comma-separated list of specific checkpoint .pt paths to evaluate (skips discovery)",
    )
    args = parser.parse_args()

    # Resolve paths relative to repo root
    checkpoints_dir = Path(args.checkpoints_dir)
    if not checkpoints_dir.is_absolute():
        checkpoints_dir = ROOT / checkpoints_dir

    if args.data_path:
        data_path = Path(args.data_path)
        if not data_path.is_absolute():
            data_path = ROOT / data_path
    else:
        data_path = resolve_default_data_path()
        if data_path is None:
            print("ERROR: Could not find a default val data file. Specify --data-path.", file=sys.stderr)
            sys.exit(1)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    # Discover checkpoints
    if args.checkpoints_list:
        checkpoint_paths = [
            Path(p.strip()) if Path(p.strip()).is_absolute() else ROOT / p.strip()
            for p in args.checkpoints_list.split(",")
            if p.strip()
        ]
    else:
        if not checkpoints_dir.exists():
            print(f"ERROR: Checkpoints directory not found: {checkpoints_dir}", file=sys.stderr)
            sys.exit(1)
        checkpoint_paths = discover_best_checkpoints(checkpoints_dir)

    if not checkpoint_paths:
        print("No checkpoints found.")
        sys.exit(0)

    # Apply --top-n filter: sort by modification time descending
    if args.top_n > 0:
        checkpoint_paths = sorted(checkpoint_paths, key=lambda p: p.stat().st_mtime, reverse=True)
        checkpoint_paths = checkpoint_paths[: args.top_n]

    num_symbols = infer_num_symbols(data_path)
    print(f"Data file: {data_path}")
    if num_symbols is not None:
        print(f"  Symbols: {num_symbols}")
    print(f"Checkpoints dir: {checkpoints_dir}")
    print(f"Checkpoints found: {len(checkpoint_paths)}")
    print(f"Periods: {args.periods}")
    print()

    if args.dry_run:
        print("DRY RUN — discovered checkpoints:")
        for i, pt in enumerate(checkpoint_paths, 1):
            label = checkpoint_label(pt, checkpoints_dir)
            mtime = pt.stat().st_mtime if pt.exists() else 0
            ts = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            print(f"  {i:>4}. [{ts}] {label}")
        return

    # Run evaluation
    rows: list[dict] = []
    t0_total = time.time()

    for idx, pt_path in enumerate(checkpoint_paths, 1):
        label = checkpoint_label(pt_path, checkpoints_dir)
        print(f"[{idx}/{len(checkpoint_paths)}] {label} ...", end=" ", flush=True)
        t0 = time.time()
        row = run_evaluation(
            pt_path,
            data_path,
            periods=args.periods,
            fee_rate=args.fee_rate,
            fill_buffer_bps=args.fill_buffer_bps,
            max_leverage=args.max_leverage,
            deterministic=args.deterministic,
            device=args.device,
        )
        elapsed = time.time() - t0
        if row is None:
            print(f"SKIP (None returned) [{elapsed:.1f}s]")
            continue
        row["checkpoint"] = label
        err = row.get("error", "")
        if err:
            print(f"ERROR: {err[:60]} [{elapsed:.1f}s]")
        else:
            print(
                f"return={row.get('total_return_pct',0):+.2f}%  "
                f"sortino={row.get('sortino',0):.2f}  "
                f"dd={row.get('max_drawdown_pct',0):.2f}%  "
                f"[{elapsed:.1f}s]"
            )
        rows.append(row)

    total_elapsed = time.time() - t0_total
    print(f"\nTotal time: {total_elapsed:.1f}s")
    print(f"\n{'='*80}")
    print("LEADERBOARD (sorted by Sortino, 30d tail)")
    print(f"{'='*80}")
    print_leaderboard(rows)
    save_leaderboard(rows, output_path)


if __name__ == "__main__":
    main()
