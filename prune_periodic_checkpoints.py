#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.checkpoint_manager import prune_periodic_checkpoints


def _iter_dirs(root: Path) -> list[Path]:
    directories = [root]
    directories.extend(path for path in root.rglob("*") if path.is_dir())
    return directories


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prune periodic update checkpoints while preserving manifest-selected and latest files.",
    )
    parser.add_argument("roots", nargs="+", help="Checkpoint root directories to scan.")
    parser.add_argument("--pattern", default="update_*.pt")
    parser.add_argument("--keep-latest", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    total_scanned = 0
    total_removed = 0
    total_reclaimed = 0

    for raw_root in args.roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.exists():
            print(f"missing root: {root}")
            continue
        for checkpoint_dir in _iter_dirs(root):
            matches = sorted(checkpoint_dir.glob(args.pattern))
            if not matches:
                continue
            if args.dry_run:
                summary = prune_periodic_checkpoints(
                    checkpoint_dir,
                    max_keep_latest=args.keep_latest,
                    pattern=args.pattern,
                    dry_run=True,
                )
            else:
                summary = prune_periodic_checkpoints(
                    checkpoint_dir,
                    max_keep_latest=args.keep_latest,
                    pattern=args.pattern,
                )
            if summary.removed <= 0:
                continue
            total_scanned += summary.scanned
            total_removed += summary.removed
            total_reclaimed += summary.reclaimed_bytes
            print(
                f"{checkpoint_dir}: removed={summary.removed} kept={summary.kept} "
                f"reclaimed_gb={summary.reclaimed_bytes / (1024 ** 3):.2f}"
            )

    print(
        f"total_removed={total_removed} total_scanned={total_scanned} "
        f"reclaimed_gb={total_reclaimed / (1024 ** 3):.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
