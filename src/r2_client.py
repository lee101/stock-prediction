from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import boto3
from botocore.client import Config

logger = logging.getLogger(__name__)


class R2Client:
    """Thin client for Cloudflare R2 (S3-compatible) object storage.

    All configuration is read from environment variables if not provided
    directly:
      - R2_ENDPOINT  : e.g. https://<account-id>.r2.cloudflarestorage.com
      - R2_BUCKET    : bucket name
      - R2_ACCESS_KEY: access key id
      - R2_SECRET_KEY: secret access key
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        bucket: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ) -> None:
        self.endpoint = endpoint or os.environ.get("R2_ENDPOINT", "")
        self.bucket = bucket or os.environ.get("R2_BUCKET", "")
        _access_key = access_key or os.environ.get("R2_ACCESS_KEY", "")
        _secret_key = secret_key or os.environ.get("R2_SECRET_KEY", "")

        if not self.endpoint:
            raise ValueError(
                "R2_ENDPOINT is required. Pass endpoint= or set the R2_ENDPOINT env var."
            )
        if not self.bucket:
            raise ValueError(
                "R2_BUCKET is required. Pass bucket= or set the R2_BUCKET env var."
            )

        self._s3 = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=_access_key or None,
            aws_secret_access_key=_secret_key or None,
            config=Config(signature_version="s3v4"),
        )

    # ------------------------------------------------------------------
    # Core single-object operations
    # ------------------------------------------------------------------

    def upload_file(self, local_path: str | Path, r2_key: str) -> None:
        """Upload a single file to R2."""
        self._s3.upload_file(str(local_path), self.bucket, r2_key)

    def download_file(self, r2_key: str, local_path: str | Path) -> None:
        """Download a single object from R2, creating parent directories."""
        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._s3.download_file(self.bucket, r2_key, str(dest))

    def list_keys(self, prefix: str = "") -> list[str]:
        """Return all object keys under *prefix* (handles pagination)."""
        paginator = self._s3.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    def delete_key(self, r2_key: str) -> None:
        """Delete a single object."""
        self._s3.delete_object(Bucket=self.bucket, Key=r2_key)

    # ------------------------------------------------------------------
    # Directory-level sync helpers
    # ------------------------------------------------------------------

    def sync_dir_to_r2(
        self,
        local_dir: str | Path,
        r2_prefix: str,
        glob_pattern: str = "**/*",
        skip_existing: bool = False,
    ) -> list[str]:
        """Upload all files matching *glob_pattern* under *local_dir*.

        The R2 key for each file is ``<r2_prefix>/<relative-path>``.
        Returns the list of uploaded R2 keys.
        """
        base = Path(local_dir)
        existing: set[str] = set()
        if skip_existing:
            existing = set(self.list_keys(prefix=r2_prefix))

        uploaded: list[str] = []
        for local_file in sorted(base.glob(glob_pattern)):
            if not local_file.is_file():
                continue
            rel = local_file.relative_to(base)
            r2_key = f"{r2_prefix}/{rel.as_posix()}" if r2_prefix else rel.as_posix()
            if skip_existing and r2_key in existing:
                logger.debug("sync_dir_to_r2: skipping existing key %s", r2_key)
                continue
            self.upload_file(local_file, r2_key)
            uploaded.append(r2_key)
        return uploaded

    def sync_dir_from_r2(
        self,
        r2_prefix: str,
        local_dir: str | Path,
    ) -> list[str]:
        """Download all objects under *r2_prefix* into *local_dir*.

        The local filename mirrors the suffix after *r2_prefix*.
        Returns the list of local file paths written.
        """
        base = Path(local_dir)
        keys = self.list_keys(prefix=r2_prefix)
        local_paths: list[str] = []
        for key in keys:
            # Strip the prefix (and any leading slash) to get the relative path.
            rel = key[len(r2_prefix):].lstrip("/")
            if not rel:
                # Key is exactly the prefix — skip (it's a "directory" marker).
                continue
            dest = base / rel
            self.download_file(key, dest)
            local_paths.append(str(dest))
        return local_paths

    # ------------------------------------------------------------------
    # Checkpoint-specific helper
    # ------------------------------------------------------------------

    def upload_checkpoint_topk(
        self,
        ckpt_dir: str | Path,
        r2_prefix: str,
        k: int = 5,
        manifest_file: str = ".topk_manifest.json",
    ) -> list[str]:
        """Upload the top-k checkpoint files listed in *manifest_file*.

        The manifest JSON must contain a key ``"top_k"`` whose value is a list
        of file paths (absolute or relative to *ckpt_dir*) ordered best-first.
        Only the first *k* entries are uploaded.

        Returns the list of uploaded R2 keys, or an empty list with a warning
        if the manifest is missing or malformed.
        """
        base = Path(ckpt_dir)
        manifest_path = base / manifest_file

        if not manifest_path.exists():
            logger.warning(
                "upload_checkpoint_topk: manifest not found at %s — nothing uploaded.",
                manifest_path,
            )
            return []

        try:
            data = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            logger.warning(
                "upload_checkpoint_topk: failed to parse manifest %s: %s — nothing uploaded.",
                manifest_path,
                exc,
            )
            return []

        entries: list[str] = data.get("top_k", [])
        if not isinstance(entries, list):
            logger.warning(
                "upload_checkpoint_topk: 'top_k' in manifest is not a list — nothing uploaded."
            )
            return []

        uploaded: list[str] = []
        for entry in entries[:k]:
            entry_path = Path(entry)
            local_file = entry_path if entry_path.is_absolute() else base / entry
            if not local_file.exists():
                logger.warning(
                    "upload_checkpoint_topk: file not found, skipping: %s", local_file
                )
                continue
            rel = local_file.relative_to(base)
            r2_key = f"{r2_prefix}/{rel.as_posix()}" if r2_prefix else rel.as_posix()
            self.upload_file(local_file, r2_key)
            uploaded.append(r2_key)

        return uploaded


def get_r2_client() -> R2Client:
    """Convenience factory that reads all config from environment variables."""
    return R2Client()
