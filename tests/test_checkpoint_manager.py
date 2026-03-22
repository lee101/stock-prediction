"""Tests for src.checkpoint_manager.TopKCheckpointManager."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

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


def test_r2_upload_called_on_register_when_r2_prefix_set():
    """Verify upload_to_r2 is called when r2_prefix is set.

    Uses patch.object on the method rather than patch("src.r2_client.R2Client")
    to avoid triggering the real boto3 import at module scope in src/r2_client.py.
    """
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        mgr = TopKCheckpointManager(d, max_keep=5, mode="max", r2_prefix="test-prefix")
        p = d / "epoch_001.pt"
        p.write_text("data")

        with patch.object(TopKCheckpointManager, "upload_to_r2") as mock_upload:
            mgr.register(p, metric=1.0, epoch=1)
            mock_upload.assert_called_once_with(p)


def test_r2_upload_not_called_when_r2_prefix_not_set():
    """Verify upload_to_r2 is NOT called when r2_prefix is None (default)."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        mgr = TopKCheckpointManager(d, max_keep=5, mode="max")
        p = d / "epoch_001.pt"
        p.write_text("data")

        with patch.object(TopKCheckpointManager, "upload_to_r2") as mock_upload:
            mgr.register(p, metric=1.0, epoch=1)
            mock_upload.assert_not_called()


def test_r2_upload_not_called_for_pruned_checkpoint():
    """Verify upload_to_r2 is NOT called for a checkpoint that gets pruned out of top-k."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        mgr = TopKCheckpointManager(d, max_keep=1, mode="max", r2_prefix="test-prefix")

        # Register a good checkpoint first so it holds the top-1 slot
        p_best = d / "epoch_002.pt"
        p_best.write_text("best")
        with patch.object(TopKCheckpointManager, "upload_to_r2") as mock_upload:
            mgr.register(p_best, metric=2.0, epoch=2)
            mock_upload.assert_called_once_with(p_best)

        # Register a worse checkpoint — it gets pruned immediately and should NOT be uploaded
        p_worse = d / "epoch_001.pt"
        p_worse.write_text("worse")
        with patch.object(TopKCheckpointManager, "upload_to_r2") as mock_upload:
            mgr.register(p_worse, metric=0.5, epoch=1)
            mock_upload.assert_not_called()
