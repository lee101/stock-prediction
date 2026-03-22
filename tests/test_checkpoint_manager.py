"""Tests for src.checkpoint_manager.TopKCheckpointManager."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

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


# ---------------------------------------------------------------------------
# R2 backup behaviour
# ---------------------------------------------------------------------------

def test_r2_upload_called_when_prefix_set():
    """register() should upload to R2 when r2_prefix is configured."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        mock_client = MagicMock()
        # R2Client is imported lazily inside src.r2_client, so patch there.
        with patch("src.r2_client.R2Client", return_value=mock_client) as mock_cls:
            mgr = TopKCheckpointManager(d, max_keep=5, mode="max", r2_prefix="rl-checkpoints/run-001")
            p = d / "epoch_001.pt"
            p.write_text("weights")
            mgr.register(p, 1.0)

        mock_cls.assert_called_once()
        mock_client.upload_file.assert_called_once_with(
            str(p), "rl-checkpoints/run-001/epoch_001.pt"
        )


def test_r2_upload_not_called_when_prefix_none():
    """register() must NOT call R2Client at all when r2_prefix is None."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        with patch("src.r2_client.R2Client") as mock_cls:
            mgr = TopKCheckpointManager(d, max_keep=5, mode="max", r2_prefix=None)
            p = d / "epoch_001.pt"
            p.write_text("weights")
            mgr.register(p, 1.0)

        mock_cls.assert_not_called()


def test_r2_upload_only_when_checkpoint_kept_at_registration():
    """A checkpoint is uploaded only if it is in the top-k immediately after registration.

    When a new better checkpoint arrives and evicts an older one, the older one
    was already uploaded (it was kept when it was registered). A checkpoint that
    is immediately displaced at registration time (i.e. lower metric than all
    current top-k entries) is NOT uploaded.
    """
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        uploaded_keys = []

        def fake_upload(local_path, r2_key):
            uploaded_keys.append(r2_key)

        mock_client = MagicMock()
        mock_client.upload_file.side_effect = fake_upload

        with patch("src.r2_client.R2Client", return_value=mock_client):
            mgr = TopKCheckpointManager(d, max_keep=2, mode="max", r2_prefix="run")
            # Register top-2 worth of good checkpoints first.
            for i in [5, 4]:
                p = d / f"epoch_{i:03d}.pt"
                p.write_text(f"data{i}")
                mgr.register(p, float(i))
            # Now register a checkpoint with a metric BELOW the current bottom of top-k
            # (metric=3 < min(4,5)=4). It should be immediately pruned and NOT uploaded.
            low_p = d / "epoch_003.pt"
            low_p.write_text("low")
            mgr.register(low_p, 3.0)

        assert "run/epoch_005.pt" in uploaded_keys
        assert "run/epoch_004.pt" in uploaded_keys
        assert "run/epoch_003.pt" not in uploaded_keys


def test_r2_upload_failure_does_not_crash():
    """An R2 upload exception must be caught and logged, not re-raised."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        mock_client = MagicMock()
        mock_client.upload_file.side_effect = RuntimeError("network error")

        with patch("src.r2_client.R2Client", return_value=mock_client):
            mgr = TopKCheckpointManager(d, max_keep=5, mode="max", r2_prefix="run")
            p = d / "epoch_001.pt"
            p.write_text("weights")
            # Must not raise.
            mgr.register(p, 1.0)

        # Checkpoint was still kept locally.
        assert p.exists()


def test_r2_upload_uses_correct_key_format():
    """R2 key must be <r2_prefix>/<filename>."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        mock_client = MagicMock()
        with patch("src.r2_client.R2Client", return_value=mock_client):
            mgr = TopKCheckpointManager(d, max_keep=5, mode="max", r2_prefix="models/ppo/run-42")
            p = d / "update_000050.pt"
            p.write_text("w")
            mgr.register(p, 0.5)

        mock_client.upload_file.assert_called_once_with(
            str(p), "models/ppo/run-42/update_000050.pt"
        )
