#!/usr/bin/env python3
"""Pull best checkpoint from remote training machine and optionally run local inference.

Usage:
  # Dry run — print what would happen without doing it
  python pull_best_checkpoint.py --dry-run

  # Pull best known checkpoint (random_mut_2272) by run-id
  python pull_best_checkpoint.py --run-id random_mut_2272

  # Pull using latest manifest file
  python pull_best_checkpoint.py --manifest manifest_stocks_*.json

  # List available remote checkpoints
  python pull_best_checkpoint.py --list-remote-checkpoints

  # Pull leaderboard and print top 5
  python pull_best_checkpoint.py --leaderboard

  # Pull and evaluate locally
  python pull_best_checkpoint.py --run-id random_mut_2272 --eval-after-pull
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent

DEFAULT_REMOTE_HOST = "administrator@93.127.141.100"
DEFAULT_REMOTE_DIR = "/nvme0n1-disk/code/stock-prediction"
DEFAULT_REMOTE_CHECKPOINT_ROOT = "pufferlib_market/checkpoints/autoresearch_stock"
DEFAULT_REMOTE_LEADERBOARD = "autoresearch_stock_daily_leaderboard.csv"
DEFAULT_LOCAL_DIR = str(REPO / "pufferlib_market" / "checkpoints" / "pulled")

# Candidate data filenames for evaluation (checked under REPO and main repo)
_CANDIDATE_DATA_FILENAMES = [
    "stocks12_daily_val.bin",
    "stocks15_daily_val.bin",
    "alpaca_daily_val.bin",
]

def _build_candidate_data_paths() -> list[Path]:
    """Return candidate val data paths, checking both worktree and main repo."""
    search_roots = [REPO]
    # Also check the main repo when running from a git worktree
    main_repo_candidate = REPO.parent.parent.parent
    if main_repo_candidate != REPO and main_repo_candidate.is_dir():
        search_roots.append(main_repo_candidate)
    paths: list[Path] = []
    for root in search_roots:
        for fname in _CANDIDATE_DATA_FILENAMES:
            paths.append(root / "pufferlib_market" / "data" / fname)
    return paths

CANDIDATE_DATA_PATHS = _build_candidate_data_paths()

SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_latest_manifest() -> Path | None:
    """Return the most-recently modified manifest_stocks_*.json file.

    Searches the script's directory first, then the git worktree root (handles
    the case where the script runs inside a git worktree but manifests live in
    the main repo checkout).
    """
    search_dirs = [REPO]
    # When running inside a git worktree (.claude/worktrees/<id>/), manifests
    # live in the main repo three levels up.
    main_repo_candidate = REPO.parent.parent.parent
    if main_repo_candidate != REPO and main_repo_candidate.is_dir():
        search_dirs.append(main_repo_candidate)

    matches: list[str] = []
    for search_dir in search_dirs:
        matches.extend(glob.glob(str(search_dir / "manifest_stocks_*.json")))

    if not matches:
        return None
    return Path(max(matches, key=os.path.getmtime))


def find_local_data_path() -> Path | None:
    """Return the first existing candidate data path for evaluation."""
    for path in CANDIDATE_DATA_PATHS:
        if path.exists():
            return path
    return None


def _run(cmd: list[str], *, dry_run: bool, capture: bool = False) -> subprocess.CompletedProcess | None:
    """Run a subprocess command, printing it first. Skips execution on dry_run."""
    print(f"  $ {' '.join(cmd)}")
    if dry_run:
        return None
    if capture:
        return subprocess.run(cmd, capture_output=True, text=True)
    return subprocess.run(cmd, check=True)




# ---------------------------------------------------------------------------
# Mode: list remote checkpoints
# ---------------------------------------------------------------------------

def list_remote_checkpoints(remote_host: str, remote_dir: str, dry_run: bool) -> None:
    """SSH to remote and list checkpoint dirs with sizes and dates."""
    remote_ckpt_root = f"{remote_dir}/{DEFAULT_REMOTE_CHECKPOINT_ROOT}"
    cmd = ["ssh"] + SSH_OPTS + [remote_host, f"ls -la {remote_ckpt_root}/"]
    print(f"\n[list-remote-checkpoints] listing {remote_host}:{remote_ckpt_root}/")
    result = _run(cmd, dry_run=dry_run, capture=True)
    if result is None:
        print("  (dry-run: would list remote checkpoints)")
        return
    if result.returncode != 0:
        print(f"  [error] SSH failed (returncode={result.returncode})", file=sys.stderr)
        if result.stderr:
            print(f"  stderr: {result.stderr.strip()}", file=sys.stderr)
        return
    print(result.stdout)

    # Also show file sizes of best.pt files
    cmd2 = ["ssh"] + SSH_OPTS + [
        remote_host,
        f"find {remote_ckpt_root} -name 'best.pt' -exec ls -lh {{}} \\;",
    ]
    print(f"\n[list-remote-checkpoints] best.pt sizes:")
    result2 = _run(cmd2, dry_run=dry_run, capture=True)
    if result2 is not None and result2.returncode == 0:
        print(result2.stdout)


# ---------------------------------------------------------------------------
# Mode: pull from manifest
# ---------------------------------------------------------------------------

def pull_from_manifest(manifest_path: Path, *, dry_run: bool, local_dir: str) -> list[Path]:
    """Execute the pull_checkpoints command from a manifest JSON. Returns pulled checkpoint paths."""
    print(f"\n[manifest] loading {manifest_path}")
    data = json.loads(manifest_path.read_text())

    pull_cmd = data.get("commands", {}).get("pull_checkpoints")
    if not pull_cmd:
        print("  [error] manifest has no commands.pull_checkpoints entry", file=sys.stderr)
        return []

    # Override destination to our local_dir
    modified_cmd = list(pull_cmd)
    # Last element of rsync command is the destination; replace it
    modified_cmd[-1] = local_dir.rstrip("/") + "/"
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    print(f"  $ {' '.join(modified_cmd)}")
    if not dry_run:
        # Track pre-existing files so we return only newly pulled ones on success.
        pre_existing = set(Path(local_dir).rglob("best.pt"))
        result = subprocess.run(modified_cmd)
        if result.returncode != 0:
            print(f"  [error] rsync failed (exit {result.returncode})", file=sys.stderr)
            return []  # don't return stale files on rsync failure
        newly_pulled = [p for p in Path(local_dir).rglob("best.pt") if p not in pre_existing]
        return newly_pulled

    # Dry-run: return empty list (nothing was actually pulled)
    return []


# ---------------------------------------------------------------------------
# Mode: pull single run-id
# ---------------------------------------------------------------------------

def pull_run_id(
    run_id: str,
    *,
    remote_host: str,
    remote_dir: str,
    local_dir: str,
    dry_run: bool,
) -> Path | None:
    """Pull pufferlib_market/checkpoints/autoresearch_stock/RUN_ID/best.pt from remote."""
    remote_src = (
        f"{remote_host}:{remote_dir}/{DEFAULT_REMOTE_CHECKPOINT_ROOT}/{run_id}/best.pt"
    )
    local_dst = Path(local_dir) / run_id / "best.pt"
    local_dst.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[pull-run-id] pulling {run_id}/best.pt → {local_dst}")
    cmd = ["scp"] + SSH_OPTS + [remote_src, str(local_dst)]
    print(f"  $ {' '.join(cmd)}")
    if dry_run:
        return local_dst
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  [error] scp failed (returncode={result.returncode})", file=sys.stderr)
        return None

    size_bytes = local_dst.stat().st_size
    print(f"  pulled: {local_dst} ({size_bytes / 1024:.1f} KB)")
    return local_dst


# ---------------------------------------------------------------------------
# Mode: pull leaderboard
# ---------------------------------------------------------------------------

def pull_leaderboard(
    *,
    remote_host: str,
    remote_dir: str,
    local_path: str,
    dry_run: bool,
) -> None:
    """Pull remote leaderboard CSV and print top 5 rows."""
    remote_src = f"{remote_host}:{remote_dir}/{DEFAULT_REMOTE_LEADERBOARD}"
    print(f"\n[leaderboard] pulling {DEFAULT_REMOTE_LEADERBOARD} from {remote_host}")
    cmd = ["scp"] + SSH_OPTS + [remote_src, local_path]
    print(f"  $ {' '.join(cmd)}")
    if dry_run:
        print("  (dry-run: would print top 5 rows)")
        return
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  [error] scp failed (returncode={result.returncode})", file=sys.stderr)
        return

    # Print top 5 rows
    try:
        import csv
        with open(local_path, newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        print(f"\n  Top 5 rows from {DEFAULT_REMOTE_LEADERBOARD}:")
        if not rows:
            print("  (empty)")
            return
        headers = list(rows[0].keys())
        col_widths = [max(len(h), max((len(str(r.get(h, ""))) for r in rows[:5]), default=0)) for h in headers]
        header_line = "  " + "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        print(header_line)
        print("  " + "-" * (len(header_line) - 2))
        for row in rows[:5]:
            print("  " + "  ".join(str(row.get(h, "")).ljust(w) for h, w in zip(headers, col_widths)))
    except Exception as exc:
        print(f"  [warn] could not parse CSV: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Mode: evaluate after pull
# ---------------------------------------------------------------------------

def evaluate_checkpoint(checkpoint_path: Path, *, data_path: Path, dry_run: bool) -> dict | None:
    """Run pufferlib_market.evaluate_fast on the pulled checkpoint. Returns result dict."""
    print(f"\n[eval] evaluating {checkpoint_path}")
    print(f"  data: {data_path}")

    cmd = [
        sys.executable, "-m", "pufferlib_market.evaluate_fast",
        "--checkpoint", str(checkpoint_path),
        "--data-path", str(data_path),
        "--deterministic",
        "--n-windows", "20",
        "--verbose",
    ]
    print(f"  $ {' '.join(cmd)}")
    if dry_run:
        print("  (dry-run: would run evaluate_fast)")
        return None

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    if result.returncode != 0:
        print(f"  [error] evaluate_fast returned non-zero exit code {result.returncode}", file=sys.stderr)
        if result.stderr:
            print(f"  stderr:\n{result.stderr[-2000:]}", file=sys.stderr)
        return None

    output = result.stdout.strip()
    print(f"\n  eval output:\n{output}")

    # Try to parse the JSON summary from the last JSON block in stdout
    try:
        # evaluate_fast prints JSON summary first, then elapsed line
        lines = output.splitlines()
        json_lines: list[str] = []
        in_json = False
        for line in lines:
            if line.strip().startswith("{"):
                in_json = True
            if in_json:
                json_lines.append(line)
            if in_json and line.strip() == "}":
                break
        if json_lines:
            return json.loads("\n".join(json_lines))
    except Exception:
        pass
    return None


def _print_eval_summary(checkpoint_path: Path, run_id: str, result: dict | None) -> None:
    """Print a concise summary of the evaluation result."""
    print(f"\n{'=' * 60}")
    print(f"  checkpoint : {checkpoint_path}")
    print(f"  run_id     : {run_id}")
    if checkpoint_path.exists():
        size_kb = checkpoint_path.stat().st_size / 1024
        print(f"  size       : {size_kb:.1f} KB")
    if result:
        ret = result.get("median_total_return")
        sortino = result.get("median_sortino")
        wr = result.get("win_rate")
        if ret is not None:
            print(f"  return     : {ret:+.4f} ({ret * 100:+.2f}%)")
        if sortino is not None:
            print(f"  sortino    : {sortino:.3f}")
        if wr is not None:
            print(f"  win_rate   : {wr:.3f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull best checkpoint from remote training machine and optionally run local inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--manifest", metavar="PATH",
        help="Path to manifest_stocks_*.json; use its pull_checkpoints command",
    )
    parser.add_argument(
        "--run-id", metavar="RUN_ID",
        help="Run ID to pull (e.g. random_mut_2272); pulls best.pt from remote",
    )
    parser.add_argument(
        "--remote-host", default=DEFAULT_REMOTE_HOST,
        help=f"Remote SSH host (default: {DEFAULT_REMOTE_HOST})",
    )
    parser.add_argument(
        "--remote-dir", default=DEFAULT_REMOTE_DIR,
        help=f"Remote repo root dir (default: {DEFAULT_REMOTE_DIR})",
    )
    parser.add_argument(
        "--local-dir", default=DEFAULT_LOCAL_DIR,
        help=f"Local directory to save pulled checkpoints (default: {DEFAULT_LOCAL_DIR})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without executing anything",
    )
    parser.add_argument(
        "--list-remote-checkpoints", action="store_true",
        help="SSH to remote and list available checkpoint dirs and sizes",
    )
    parser.add_argument(
        "--eval-after-pull", action="store_true",
        help="Run pufferlib_market.evaluate_fast after pulling checkpoint(s)",
    )
    parser.add_argument(
        "--leaderboard", action="store_true",
        help="Pull remote leaderboard CSV and print top 5 rows",
    )
    parser.add_argument(
        "--data-path", metavar="PATH",
        help="Override data path for evaluation (default: auto-detect stocks12_daily_val.bin)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.dry_run:
        print("[dry-run] no files will be transferred or commands executed")

    pulled_checkpoints: list[Path] = []

    # --- list-remote-checkpoints mode ---
    if args.list_remote_checkpoints:
        list_remote_checkpoints(args.remote_host, args.remote_dir, args.dry_run)
        return 0

    # --- leaderboard mode ---
    if args.leaderboard:
        local_lb_path = str(REPO / "autoresearch_stock_daily_leaderboard_pulled.csv")
        pull_leaderboard(
            remote_host=args.remote_host,
            remote_dir=args.remote_dir,
            local_path=local_lb_path,
            dry_run=args.dry_run,
        )
        return 0

    # --- manifest mode ---
    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists() and not args.dry_run:
            print(f"[error] manifest not found: {manifest_path}", file=sys.stderr)
            return 1
        pulled_checkpoints = pull_from_manifest(manifest_path, dry_run=args.dry_run, local_dir=args.local_dir)

    # --- run-id mode ---
    elif args.run_id:
        ckpt = pull_run_id(
            args.run_id,
            remote_host=args.remote_host,
            remote_dir=args.remote_dir,
            local_dir=args.local_dir,
            dry_run=args.dry_run,
        )
        if ckpt is not None:
            pulled_checkpoints = [ckpt]

    # --- default: find latest manifest and use it ---
    else:
        manifest_path = find_latest_manifest()
        if manifest_path is None:
            print("[error] no manifest_stocks_*.json found; use --manifest, --run-id, or --list-remote-checkpoints", file=sys.stderr)
            return 1
        print(f"[default] using latest manifest: {manifest_path}")
        pulled_checkpoints = pull_from_manifest(manifest_path, dry_run=args.dry_run, local_dir=args.local_dir)

    if not pulled_checkpoints:
        if args.dry_run:
            print("\n[dry-run] would have pulled checkpoints (none available locally yet)")
            return 0
        else:
            print("[error] no checkpoints were pulled", file=sys.stderr)
            return 1

    # Print summary of pulled checkpoints
    print(f"\n[pulled] {len(pulled_checkpoints)} checkpoint(s):")
    for ckpt in pulled_checkpoints:
        size_kb = ckpt.stat().st_size / 1024 if ckpt.exists() else 0.0
        run_id = ckpt.parent.name
        print(f"  {ckpt}  ({size_kb:.1f} KB, run_id={run_id})")

    # --- eval mode ---
    if args.eval_after_pull:
        data_path: Path | None = None
        if args.data_path:
            data_path = Path(args.data_path)
            if not data_path.exists():
                print(f"[error] --data-path not found: {data_path}", file=sys.stderr)
                return 1
        else:
            data_path = find_local_data_path()
            if data_path is None:
                print(
                    "[error] no local val data found; checked:\n  "
                    + "\n  ".join(str(p) for p in CANDIDATE_DATA_PATHS),
                    file=sys.stderr,
                )
                return 1

        for ckpt in pulled_checkpoints:
            run_id = ckpt.parent.name
            result = evaluate_checkpoint(ckpt, data_path=data_path, dry_run=args.dry_run)
            _print_eval_summary(ckpt, run_id, result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
