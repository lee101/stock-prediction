"""Top-k checkpoint manager. Keeps only the best k checkpoints by a metric."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


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
        if self._manifest_path.exists():
            try:
                self._entries = json.loads(self._manifest_path.read_text())
            except (json.JSONDecodeError, KeyError):
                self._entries = []

    def _save_manifest(self):
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._manifest_path.write_text(json.dumps(self._entries, indent=2))

    def register(self, path: Path | str, metric: float, epoch: Optional[int] = None):
        """Register a checkpoint and prune if over max_keep."""
        path = Path(path)
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
