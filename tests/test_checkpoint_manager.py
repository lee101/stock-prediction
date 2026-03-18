"""Tests for src.checkpoint_manager.TopKCheckpointManager."""

import json
import tempfile
from pathlib import Path

from src.checkpoint_manager import TopKCheckpointManager, prune_periodic_checkpoints


def test_keeps_top_k():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        mgr = TopKCheckpointManager(d, max_keep=3, mode="max")
        paths = []
        for i in range(5):
            p = d / f"epoch_{i:03d}.pt"
            p.write_text(f"data{i}")
            paths.append(p)
            mgr.register(p, metric=float(i), epoch=i)

        # Should keep top 3: metrics 4, 3, 2
        existing = sorted(d.glob("epoch_*.pt"))
        assert len(existing) == 3
        names = {p.name for p in existing}
        assert names == {"epoch_002.pt", "epoch_003.pt", "epoch_004.pt"}


def test_mode_min():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        mgr = TopKCheckpointManager(d, max_keep=2, mode="min")
        for i in range(4):
            p = d / f"epoch_{i:03d}.pt"
            p.write_text(f"data{i}")
            mgr.register(p, metric=float(i), epoch=i)

        existing = sorted(d.glob("epoch_*.pt"))
        assert len(existing) == 2
        names = {p.name for p in existing}
        assert names == {"epoch_000.pt", "epoch_001.pt"}


def test_manifest_persists():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        mgr = TopKCheckpointManager(d, max_keep=5, mode="max")
        p = d / "test.pt"
        p.write_text("x")
        mgr.register(p, 1.0, epoch=1)
        manifest = d / ".topk_manifest.json"
        assert manifest.exists()
        data = json.loads(manifest.read_text())
        assert len(data) == 1
        assert data[0]["metric"] == 1.0


def test_no_prune_under_limit():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        mgr = TopKCheckpointManager(d, max_keep=10, mode="max")
        for i in range(5):
            p = d / f"epoch_{i:03d}.pt"
            p.write_text(f"data{i}")
            mgr.register(p, float(i))
        existing = sorted(d.glob("epoch_*.pt"))
        assert len(existing) == 5


def test_prune_periodic_checkpoints_keeps_latest_and_manifest_entries():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        update_paths = []
        for i in range(1, 6):
            path = d / f"update_{i:06d}.pt"
            path.write_text(f"data{i}")
            update_paths.append(path)
        manifest = d / ".topk_manifest.json"
        manifest.write_text(
            json.dumps(
                [
                    {"path": str(update_paths[1]), "metric": 0.9, "epoch": 2},
                    {"path": str(update_paths[3]), "metric": 1.1, "epoch": 4},
                ]
            )
        )

        summary = prune_periodic_checkpoints(d, max_keep_latest=1)

        remaining = sorted(path.name for path in d.glob("update_*.pt"))
        assert remaining == ["update_000002.pt", "update_000004.pt", "update_000005.pt"]
        assert summary.scanned == 5
        assert summary.removed == 2
