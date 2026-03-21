"""Top-k checkpoint manager. Keeps only the best k checkpoints by a metric."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
_UPDATE_CHECKPOINT_RE = re.compile(r"update_(\d+)\.pt$")


@dataclass(frozen=True)
class CheckpointPruneSummary:
    scanned: int
    kept: int
    removed: int
    reclaimed_bytes: int


def _load_manifest_entries(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(payload, list):
        return []
    entries: list[dict] = []
    seen: set[str] = set()
    for row in payload:
        if not isinstance(row, dict):
            continue
        checkpoint = str(row.get("path") or "").strip()
        if not checkpoint or checkpoint in seen:
            continue
        if not Path(checkpoint).exists():
            continue
        seen.add(checkpoint)
        entries.append({
            "path": checkpoint,
            "metric": float(row.get("metric", 0.0) or 0.0),
            "epoch": row.get("epoch"),
        })
    return entries


def _update_checkpoint_sort_key(path: Path) -> tuple[int, float, str]:
    match = _UPDATE_CHECKPOINT_RE.fullmatch(path.name)
    if match:
        return (int(match.group(1)), path.stat().st_mtime, path.name)
    return (-1, path.stat().st_mtime, path.name)


def prune_periodic_checkpoints(
    checkpoint_dir: Path | str,
    *,
    max_keep_latest: int = 3,
    pattern: str = "update_*.pt",
    keep_manifest_paths: bool = True,
    dry_run: bool = False,
) -> CheckpointPruneSummary:
    checkpoint_root = Path(checkpoint_dir)
    candidates = sorted(checkpoint_root.glob(pattern), key=_update_checkpoint_sort_key, reverse=True)
    if not candidates:
        return CheckpointPruneSummary(scanned=0, kept=0, removed=0, reclaimed_bytes=0)

    keep_paths = set(candidates[: max(0, int(max_keep_latest))])
    if keep_manifest_paths:
        manifest_path = checkpoint_root / ".topk_manifest.json"
        for entry in _load_manifest_entries(manifest_path):
            keep_paths.add(Path(entry["path"]))

    removed = 0
    reclaimed = 0
    for candidate in candidates:
        if candidate in keep_paths:
            continue
        try:
            size = candidate.stat().st_size
        except OSError:
            size = 0
        if dry_run:
            removed += 1
            reclaimed += size
            continue
        try:
            candidate.unlink()
            removed += 1
            reclaimed += size
        except OSError:
            logger.warning("Failed to prune checkpoint %s", candidate)

    return CheckpointPruneSummary(
        scanned=len(candidates),
        kept=(len(candidates) - removed) if dry_run else len([path for path in candidates if path.exists()]),
        removed=removed,
        reclaimed_bytes=reclaimed,
    )


class TopKCheckpointManager:
    """Manages a directory of checkpoints, keeping only the top-k by a metric.

    Usage:
        mgr = TopKCheckpointManager(checkpoint_dir, max_keep=10, mode="max")
        # After saving a checkpoint:
        mgr.register(path, metric_value)
        # Or use the convenience method:
        mgr.maybe_prune()
    """

    def __init__(
        self,
        checkpoint_dir: Path | str,
        max_keep: int = 10,
        mode: str = "max",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_keep = max_keep
        self.mode = mode  # "max" = higher is better, "min" = lower is better
        self._entries: list[dict] = []
        self._manifest_path = self.checkpoint_dir / ".topk_manifest.json"
        self._load_manifest()

    def _load_manifest(self):
        self._entries = _load_manifest_entries(self._manifest_path)

    def _save_manifest(self):
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._manifest_path.write_text(json.dumps(self._entries, indent=2))

    def register(self, path: Path | str, metric: float, epoch: Optional[int] = None):
        """Register a checkpoint and prune if over max_keep."""
        path = Path(path)
        self._entries = [entry for entry in self._entries if Path(entry["path"]) != path]
        self._entries.append({
            "path": str(path),
            "metric": metric,
            "epoch": epoch,
        })
        self._prune()
        self._save_manifest()

    def _prune(self):
        if len(self._entries) <= self.max_keep:
            return
        reverse = self.mode == "max"
        self._entries.sort(key=lambda e: e["metric"], reverse=reverse)
        to_remove = self._entries[self.max_keep:]
        self._entries = self._entries[:self.max_keep]
        for entry in to_remove:
            p = Path(entry["path"])
            if p.exists():
                p.unlink()
                logger.info(f"Pruned checkpoint: {p.name} (metric={entry['metric']:.4f})")


__all__ = [
    "CheckpointPruneSummary",
    "TopKCheckpointManager",
    "prune_periodic_checkpoints",
]
