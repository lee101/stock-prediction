from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.r2_client import R2Client, get_r2_client

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENDPOINT = "https://test.r2.cloudflarestorage.com"
_BUCKET = "test-bucket"
_ACCESS = "access-key"
_SECRET = "secret-key"


def _make_client(**overrides) -> tuple[R2Client, MagicMock]:
    """Return (R2Client, mock_s3_client) with boto3.client patched."""
    mock_s3 = MagicMock()
    with patch("boto3.client", return_value=mock_s3):
        kwargs = dict(
            endpoint=_ENDPOINT,
            bucket=_BUCKET,
            access_key=_ACCESS,
            secret_key=_SECRET,
        )
        kwargs.update(overrides)
        client = R2Client(**kwargs)
    return client, mock_s3


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


def test_raises_if_endpoint_missing(monkeypatch):
    monkeypatch.delenv("R2_ENDPOINT", raising=False)
    monkeypatch.delenv("R2_BUCKET", raising=False)
    with pytest.raises(ValueError, match="R2_ENDPOINT"):
        with patch("boto3.client"):
            R2Client(bucket=_BUCKET)


def test_raises_if_bucket_missing(monkeypatch):
    monkeypatch.delenv("R2_BUCKET", raising=False)
    with pytest.raises(ValueError, match="R2_BUCKET"):
        with patch("boto3.client"):
            R2Client(endpoint=_ENDPOINT)


def test_reads_config_from_env(monkeypatch):
    monkeypatch.setenv("R2_ENDPOINT", _ENDPOINT)
    monkeypatch.setenv("R2_BUCKET", _BUCKET)
    monkeypatch.setenv("R2_ACCESS_KEY", _ACCESS)
    monkeypatch.setenv("R2_SECRET_KEY", _SECRET)
    mock_s3 = MagicMock()
    with patch("boto3.client", return_value=mock_s3):
        client = R2Client()
    assert client.endpoint == _ENDPOINT
    assert client.bucket == _BUCKET


def test_get_r2_client_convenience(monkeypatch):
    monkeypatch.setenv("R2_ENDPOINT", _ENDPOINT)
    monkeypatch.setenv("R2_BUCKET", _BUCKET)
    monkeypatch.delenv("R2_ACCESS_KEY", raising=False)
    monkeypatch.delenv("R2_SECRET_KEY", raising=False)
    with patch("boto3.client", return_value=MagicMock()):
        client = get_r2_client()
    assert isinstance(client, R2Client)


# ---------------------------------------------------------------------------
# upload_file
# ---------------------------------------------------------------------------


def test_upload_file_calls_s3(tmp_path):
    local = tmp_path / "model.pt"
    local.write_bytes(b"weights")
    client, mock_s3 = _make_client()
    client.upload_file(local, "checkpoints/model.pt")
    mock_s3.upload_file.assert_called_once_with(str(local), _BUCKET, "checkpoints/model.pt")


def test_upload_file_accepts_str_path(tmp_path):
    local = tmp_path / "model.pt"
    local.write_bytes(b"weights")
    client, mock_s3 = _make_client()
    client.upload_file(str(local), "checkpoints/model.pt")
    mock_s3.upload_file.assert_called_once_with(str(local), _BUCKET, "checkpoints/model.pt")


# ---------------------------------------------------------------------------
# download_file
# ---------------------------------------------------------------------------


def test_download_file_calls_s3_and_creates_parents(tmp_path):
    dest = tmp_path / "deep" / "dir" / "model.pt"
    client, mock_s3 = _make_client()
    client.download_file("checkpoints/model.pt", dest)
    mock_s3.download_file.assert_called_once_with(_BUCKET, "checkpoints/model.pt", str(dest))
    assert dest.parent.exists()


def test_download_file_accepts_str_path(tmp_path):
    dest = tmp_path / "model.pt"
    client, mock_s3 = _make_client()
    client.download_file("checkpoints/model.pt", str(dest))
    mock_s3.download_file.assert_called_once_with(_BUCKET, "checkpoints/model.pt", str(dest))


# ---------------------------------------------------------------------------
# list_keys (pagination)
# ---------------------------------------------------------------------------


def test_list_keys_no_results():
    client, mock_s3 = _make_client()
    mock_paginator = MagicMock()
    mock_s3.get_paginator.return_value = mock_paginator
    mock_paginator.paginate.return_value = [{"Contents": []}]
    result = client.list_keys("prefix/")
    assert result == []
    mock_paginator.paginate.assert_called_once_with(Bucket=_BUCKET, Prefix="prefix/")


def test_list_keys_single_page():
    client, mock_s3 = _make_client()
    mock_paginator = MagicMock()
    mock_s3.get_paginator.return_value = mock_paginator
    mock_paginator.paginate.return_value = [
        {"Contents": [{"Key": "a/1.pt"}, {"Key": "a/2.pt"}]}
    ]
    assert client.list_keys("a/") == ["a/1.pt", "a/2.pt"]


def test_list_keys_multiple_pages():
    client, mock_s3 = _make_client()
    mock_paginator = MagicMock()
    mock_s3.get_paginator.return_value = mock_paginator
    mock_paginator.paginate.return_value = [
        {"Contents": [{"Key": "a/1.pt"}, {"Key": "a/2.pt"}]},
        {"Contents": [{"Key": "a/3.pt"}]},
        {},  # page with no Contents key
    ]
    assert client.list_keys("a/") == ["a/1.pt", "a/2.pt", "a/3.pt"]


def test_list_keys_empty_prefix():
    client, mock_s3 = _make_client()
    mock_paginator = MagicMock()
    mock_s3.get_paginator.return_value = mock_paginator
    mock_paginator.paginate.return_value = [{"Contents": [{"Key": "root.pt"}]}]
    assert client.list_keys() == ["root.pt"]
    mock_paginator.paginate.assert_called_once_with(Bucket=_BUCKET, Prefix="")


# ---------------------------------------------------------------------------
# delete_key
# ---------------------------------------------------------------------------


def test_delete_key():
    client, mock_s3 = _make_client()
    client.delete_key("old/checkpoint.pt")
    mock_s3.delete_object.assert_called_once_with(Bucket=_BUCKET, Key="old/checkpoint.pt")


# ---------------------------------------------------------------------------
# sync_dir_to_r2
# ---------------------------------------------------------------------------


def test_sync_dir_to_r2_uploads_all_files(tmp_path):
    (tmp_path / "a.pt").write_bytes(b"a")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.pt").write_bytes(b"b")
    client, mock_s3 = _make_client()

    uploaded = client.sync_dir_to_r2(tmp_path, "ckpts")

    assert sorted(uploaded) == ["ckpts/a.pt", "ckpts/sub/b.pt"]
    # Verify s3.upload_file was called for each file
    assert mock_s3.upload_file.call_count == 2


def test_sync_dir_to_r2_returns_correct_keys(tmp_path):
    (tmp_path / "epoch_001.pt").write_bytes(b"w")
    (tmp_path / "epoch_002.pt").write_bytes(b"w")
    client, mock_s3 = _make_client()

    uploaded = client.sync_dir_to_r2(tmp_path, "run/checkpoints")

    assert set(uploaded) == {"run/checkpoints/epoch_001.pt", "run/checkpoints/epoch_002.pt"}


def test_sync_dir_to_r2_skip_existing(tmp_path):
    (tmp_path / "old.pt").write_bytes(b"o")
    (tmp_path / "new.pt").write_bytes(b"n")
    client, mock_s3 = _make_client()

    mock_paginator = MagicMock()
    mock_s3.get_paginator.return_value = mock_paginator
    mock_paginator.paginate.return_value = [{"Contents": [{"Key": "ckpts/old.pt"}]}]

    uploaded = client.sync_dir_to_r2(tmp_path, "ckpts", skip_existing=True)

    assert uploaded == ["ckpts/new.pt"]
    mock_s3.upload_file.assert_called_once()


def test_sync_dir_to_r2_glob_pattern(tmp_path):
    (tmp_path / "model.pt").write_bytes(b"m")
    (tmp_path / "config.json").write_bytes(b"{}")
    client, mock_s3 = _make_client()

    uploaded = client.sync_dir_to_r2(tmp_path, "run", glob_pattern="*.pt")

    assert uploaded == ["run/model.pt"]


def test_sync_dir_to_r2_skips_directories(tmp_path):
    (tmp_path / "subdir").mkdir()
    (tmp_path / "file.pt").write_bytes(b"f")
    client, mock_s3 = _make_client()

    uploaded = client.sync_dir_to_r2(tmp_path, "run", glob_pattern="*")

    assert uploaded == ["run/file.pt"]


# ---------------------------------------------------------------------------
# sync_dir_from_r2
# ---------------------------------------------------------------------------


def test_sync_dir_from_r2_downloads_all(tmp_path):
    client, mock_s3 = _make_client()

    mock_paginator = MagicMock()
    mock_s3.get_paginator.return_value = mock_paginator
    mock_paginator.paginate.return_value = [
        {"Contents": [{"Key": "run/a.pt"}, {"Key": "run/sub/b.pt"}]}
    ]

    local_paths = client.sync_dir_from_r2("run", tmp_path)

    assert sorted(local_paths) == sorted(
        [str(tmp_path / "a.pt"), str(tmp_path / "sub" / "b.pt")]
    )
    assert mock_s3.download_file.call_count == 2


def test_sync_dir_from_r2_strips_prefix(tmp_path):
    client, mock_s3 = _make_client()

    mock_paginator = MagicMock()
    mock_s3.get_paginator.return_value = mock_paginator
    mock_paginator.paginate.return_value = [
        {"Contents": [{"Key": "prefix/epoch_001.pt"}]}
    ]

    local_paths = client.sync_dir_from_r2("prefix", tmp_path)

    assert local_paths == [str(tmp_path / "epoch_001.pt")]
    mock_s3.download_file.assert_called_once_with(
        _BUCKET, "prefix/epoch_001.pt", str(tmp_path / "epoch_001.pt")
    )


def test_sync_dir_from_r2_skips_directory_markers(tmp_path):
    """Keys that equal the prefix exactly (directory markers) should be skipped."""
    client, mock_s3 = _make_client()

    mock_paginator = MagicMock()
    mock_s3.get_paginator.return_value = mock_paginator
    mock_paginator.paginate.return_value = [
        {"Contents": [{"Key": "run/"}, {"Key": "run/file.pt"}]}
    ]

    local_paths = client.sync_dir_from_r2("run", tmp_path)

    assert local_paths == [str(tmp_path / "file.pt")]


# ---------------------------------------------------------------------------
# upload_checkpoint_topk
# ---------------------------------------------------------------------------


def test_upload_checkpoint_topk_uploads_top_k(tmp_path):
    # Create 5 checkpoint files
    for i in range(1, 6):
        (tmp_path / f"epoch_{i:03d}.pt").write_bytes(b"w")

    manifest = {
        "top_k": [
            "epoch_005.pt",
            "epoch_004.pt",
            "epoch_003.pt",
            "epoch_002.pt",
            "epoch_001.pt",
        ]
    }
    (tmp_path / ".topk_manifest.json").write_text(json.dumps(manifest))

    client, mock_s3 = _make_client()
    uploaded = client.upload_checkpoint_topk(tmp_path, "run/ckpts", k=3)

    assert uploaded == [
        "run/ckpts/epoch_005.pt",
        "run/ckpts/epoch_004.pt",
        "run/ckpts/epoch_003.pt",
    ]
    assert mock_s3.upload_file.call_count == 3


def test_upload_checkpoint_topk_fewer_than_k(tmp_path):
    (tmp_path / "epoch_001.pt").write_bytes(b"w")
    manifest = {"top_k": ["epoch_001.pt"]}
    (tmp_path / ".topk_manifest.json").write_text(json.dumps(manifest))

    client, mock_s3 = _make_client()
    uploaded = client.upload_checkpoint_topk(tmp_path, "run/ckpts", k=5)

    assert uploaded == ["run/ckpts/epoch_001.pt"]
    assert mock_s3.upload_file.call_count == 1


def test_upload_checkpoint_topk_missing_manifest(tmp_path):
    client, mock_s3 = _make_client()
    uploaded = client.upload_checkpoint_topk(tmp_path, "run/ckpts")

    assert uploaded == []
    mock_s3.upload_file.assert_not_called()


def test_upload_checkpoint_topk_invalid_json(tmp_path):
    (tmp_path / ".topk_manifest.json").write_text("not valid json {{{")
    client, mock_s3 = _make_client()
    uploaded = client.upload_checkpoint_topk(tmp_path, "run/ckpts")

    assert uploaded == []
    mock_s3.upload_file.assert_not_called()


def test_upload_checkpoint_topk_missing_file_skipped(tmp_path):
    (tmp_path / "epoch_002.pt").write_bytes(b"w")
    manifest = {"top_k": ["epoch_001.pt", "epoch_002.pt"]}
    (tmp_path / ".topk_manifest.json").write_text(json.dumps(manifest))

    client, mock_s3 = _make_client()
    uploaded = client.upload_checkpoint_topk(tmp_path, "run/ckpts", k=5)

    # epoch_001.pt doesn't exist — only epoch_002.pt should be uploaded
    assert uploaded == ["run/ckpts/epoch_002.pt"]
    assert mock_s3.upload_file.call_count == 1


def test_upload_checkpoint_topk_custom_manifest_name(tmp_path):
    (tmp_path / "best.pt").write_bytes(b"w")
    manifest = {"top_k": ["best.pt"]}
    (tmp_path / "my_manifest.json").write_text(json.dumps(manifest))

    client, mock_s3 = _make_client()
    uploaded = client.upload_checkpoint_topk(
        tmp_path, "run/ckpts", manifest_file="my_manifest.json"
    )

    assert uploaded == ["run/ckpts/best.pt"]


def test_upload_checkpoint_topk_absolute_paths_in_manifest(tmp_path):
    ckpt = tmp_path / "epoch_001.pt"
    ckpt.write_bytes(b"w")
    manifest = {"top_k": [str(ckpt)]}
    (tmp_path / ".topk_manifest.json").write_text(json.dumps(manifest))

    client, mock_s3 = _make_client()
    uploaded = client.upload_checkpoint_topk(tmp_path, "run/ckpts", k=1)

    assert uploaded == ["run/ckpts/epoch_001.pt"]


def test_upload_checkpoint_topk_malformed_top_k(tmp_path):
    manifest = {"top_k": "not-a-list"}
    (tmp_path / ".topk_manifest.json").write_text(json.dumps(manifest))

    client, mock_s3 = _make_client()
    uploaded = client.upload_checkpoint_topk(tmp_path, "run/ckpts")

    assert uploaded == []
    mock_s3.upload_file.assert_not_called()
