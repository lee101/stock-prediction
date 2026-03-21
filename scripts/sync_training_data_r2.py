"""Sync MKTD training .bin files between local disk and Cloudflare R2.

Usage:
    python scripts/sync_training_data_r2.py upload   [--pattern GLOB] [--prefix R2_PREFIX] [--dry-run]
    python scripts/sync_training_data_r2.py download [--files A,B] [--prefix R2_PREFIX] [--output-dir DIR] [--dry-run] [--force]
    python scripts/sync_training_data_r2.py list     [--prefix R2_PREFIX] [--dry-run]

R2 credentials are read from environment variables:
    R2_ENDPOINT   https://<account-id>.r2.cloudflarestorage.com
    R2_BUCKET     bucket name (e.g. stock-training)
    R2_ACCESS_KEY access key id
    R2_SECRET_KEY secret access key
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# R2 client — use shared implementation when available, otherwise inline boto3
# ---------------------------------------------------------------------------

try:
    from src.r2_client import R2Client as _BaseR2Client  # type: ignore[import]

    class R2Client(_BaseR2Client):
        """Extends the shared client with a size-aware listing method."""

        def list_objects(self, prefix: str = "") -> list[tuple[str, int]]:
            """Return (key, size_bytes) pairs under *prefix*."""
            paginator = self._s3.get_paginator("list_objects_v2")
            results: list[tuple[str, int]] = []
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    results.append((obj["Key"], obj["Size"]))
            return results

except ImportError:
    import boto3  # type: ignore[import]
    from botocore.client import Config  # type: ignore[import]

    class R2Client:  # type: ignore[no-redef]
        """Minimal inline R2 client using boto3 (fallback when src.r2_client is absent)."""

        def __init__(self) -> None:
            endpoint = os.environ.get("R2_ENDPOINT", "")
            bucket = os.environ.get("R2_BUCKET", "")
            if not endpoint or not bucket:
                raise ValueError("R2_ENDPOINT and R2_BUCKET must be set")
            self.bucket = bucket
            self._s3 = boto3.client(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=os.environ.get("R2_ACCESS_KEY"),
                aws_secret_access_key=os.environ.get("R2_SECRET_KEY"),
                config=Config(signature_version="s3v4"),
            )

        def upload_file(self, local_path: str | Path, r2_key: str) -> None:
            self._s3.upload_file(str(local_path), self.bucket, r2_key)

        def download_file(self, r2_key: str, local_path: str | Path) -> None:
            dest = Path(local_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            self._s3.download_file(self.bucket, r2_key, str(dest))

        def list_objects(self, prefix: str = "") -> list[tuple[str, int]]:
            paginator = self._s3.get_paginator("list_objects_v2")
            results: list[tuple[str, int]] = []
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    results.append((obj["Key"], obj["Size"]))
            return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CREDS_HELP = """\
Error: R2_ENDPOINT and R2_BUCKET environment variables must be set.
Set them in your environment or .env file:
  export R2_ENDPOINT=https://<account-id>.r2.cloudflarestorage.com
  export R2_BUCKET=stock-training
  export R2_ACCESS_KEY=...
  export R2_SECRET_KEY=...
"""

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _fmt_size(num_bytes: int) -> str:
    """Human-readable file size (e.g. '1.1 MB')."""
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes //= 1024
    return f"{num_bytes:.1f} TB"


def _build_client() -> R2Client:
    """Instantiate R2Client, printing a helpful message on failure."""
    try:
        return R2Client()
    except Exception:
        print(_CREDS_HELP, file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def cmd_upload(args: argparse.Namespace) -> int:
    """Upload local .bin files to R2."""
    pattern = args.pattern
    prefix = args.prefix.rstrip("/")
    dry_run = args.dry_run

    # Glob is relative to repo root so the user can pass bare relative globs.
    matched = sorted(_REPO_ROOT.glob(pattern))
    bin_files = [p for p in matched if p.is_file()]

    if not bin_files:
        print(f"No files matched pattern: {pattern}")
        return 0

    if dry_run:
        print(f"[dry-run] Would upload {len(bin_files)} file(s) to R2 prefix '{prefix}/':")
        for path in bin_files:
            print(f"  {path.name}  ({_fmt_size(path.stat().st_size)})")
        return 0

    client = _build_client()
    for path in bin_files:
        size_str = _fmt_size(path.stat().st_size)
        r2_key = f"{prefix}/{path.name}"
        print(f"Uploading {path.name} ({size_str})...", flush=True)
        client.upload_file(path, r2_key)

    print(f"Done. Uploaded {len(bin_files)} file(s).")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    """Download files from R2 to local disk."""
    prefix = args.prefix.rstrip("/")
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = _REPO_ROOT / output_dir
    dry_run = args.dry_run
    force = args.force

    requested: list[str] | None = None
    if args.files:
        requested = [f.strip() for f in args.files.split(",") if f.strip()]

    client = _build_client()
    objects = client.list_objects(prefix=f"{prefix}/")

    if not objects:
        print(f"No objects found under R2 prefix '{prefix}/'.")
        return 0

    # Filter to requested filenames when --files is given
    if requested:
        req_set = set(requested)
        objects = [(key, size) for key, size in objects if Path(key).name in req_set]
        if not objects:
            print(f"None of the requested files found under prefix '{prefix}/'.")
            return 1

    if dry_run:
        print(f"[dry-run] Would download {len(objects)} file(s) to '{output_dir}':")
        for key, size in objects:
            fname = Path(key).name
            local = output_dir / fname
            status = " (already exists)" if local.exists() and not force else ""
            print(f"  {fname} → {local}{status}  ({_fmt_size(size)})")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    skipped = 0
    for key, remote_size in objects:
        fname = Path(key).name
        local = output_dir / fname
        if local.exists() and not force:
            if local.stat().st_size == remote_size:
                skipped += 1
                continue
        print(f"Downloading {fname} → {local}", flush=True)
        client.download_file(key, local)
        downloaded += 1

    print(f"Done. Downloaded {downloaded} file(s), skipped {skipped} (already up to date).")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List objects stored in R2 under the given prefix."""
    prefix = args.prefix.rstrip("/")
    dry_run = args.dry_run
    tag = "[dry-run] " if dry_run else ""

    client = _build_client()
    objects = client.list_objects(prefix=f"{prefix}/")

    if not objects:
        print(f"{tag}No objects found under R2 prefix '{prefix}/'.")
        return 0

    for key, size in sorted(objects):
        print(f"{tag}{key}  {_fmt_size(size)}")
    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sync MKTD training .bin files to/from Cloudflare R2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- upload --
    p_up = sub.add_parser("upload", help="Upload local .bin files to R2.")
    p_up.add_argument(
        "--pattern",
        default="pufferlib_market/data/*_daily_*.bin",
        help="Glob pattern relative to repo root (default: pufferlib_market/data/*_daily_*.bin)",
    )
    p_up.add_argument(
        "--prefix",
        default="training_data",
        help="R2 key prefix (default: training_data)",
    )
    p_up.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without uploading.",
    )

    # -- download --
    p_dl = sub.add_parser("download", help="Download .bin files from R2.")
    p_dl.add_argument(
        "--files",
        default="",
        help="Comma-separated filenames to download. Omit to download all under prefix.",
    )
    p_dl.add_argument(
        "--prefix",
        default="training_data",
        help="R2 key prefix (default: training_data)",
    )
    p_dl.add_argument(
        "--output-dir",
        default="pufferlib_market/data",
        help="Local directory to write files (default: pufferlib_market/data)",
    )
    p_dl.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without downloading.",
    )
    p_dl.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if local file exists and matches remote size.",
    )

    # -- list --
    p_ls = sub.add_parser("list", help="List objects in R2 under the prefix.")
    p_ls.add_argument(
        "--prefix",
        default="training_data",
        help="R2 key prefix (default: training_data)",
    )
    p_ls.add_argument(
        "--dry-run",
        action="store_true",
        help="Same as list but prefixes each line with '[dry-run]'.",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    dispatch = {
        "upload": cmd_upload,
        "download": cmd_download,
        "list": cmd_list,
    }
    sys.exit(dispatch[args.command](args))


if __name__ == "__main__":
    main()
