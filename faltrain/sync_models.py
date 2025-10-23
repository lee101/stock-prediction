#!/usr/bin/env python3
"""
CLI helper to sync model/checkpoint artifacts with R2 storage.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:  # pragma: no cover - script execution path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from faltrain.artifacts import REPO_ROOT, load_artifact_specs, sync_artifacts  # type: ignore
    from faltrain.logger_utils import configure_stdout_logging  # type: ignore
else:  # pragma: no cover - module execution path
    from .artifacts import REPO_ROOT, load_artifact_specs, sync_artifacts
    from .logger_utils import configure_stdout_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sync model checkpoints to/from R2 using the shared manifest.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path to a manifest TOML file (defaults to faltrain/model_manifest.toml).",
    )
    parser.add_argument(
        "--direction",
        choices=("upload", "download"),
        default="upload",
        help="Upload pushes local artifacts to R2. Download pulls from R2 into local_root.",
    )
    parser.add_argument(
        "--bucket",
        default=os.getenv("R2_BUCKET", "models"),
        help="R2 bucket name (defaults to R2_BUCKET env or 'models').",
    )
    parser.add_argument(
        "--endpoint",
        default=os.getenv("R2_ENDPOINT"),
        help="R2 endpoint URL (defaults to R2_ENDPOINT env var).",
    )
    parser.add_argument(
        "--local-root",
        type=Path,
        default=REPO_ROOT,
        help="Base directory for local artifacts (defaults to repository root).",
    )
    parser.add_argument(
        "--remote-prefix",
        default="",
        help="Optional prefix prepended to every remote key.",
    )
    parser.add_argument(
        "--aws-cli",
        default=os.getenv("AWS_CLI", "aws"),
        help="AWS CLI executable name (defaults to AWS_CLI env var or 'aws').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log planned AWS commands without executing them.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List resolved artifacts and exit without syncing.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip downloads when the destination already contains files (ignored for uploads).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    configure_stdout_logging(level="INFO", fmt="%(message)s")

    if not args.endpoint:
        parser.error("R2 endpoint missing. Pass --endpoint or set R2_ENDPOINT.")

    specs = load_artifact_specs(manifest_path=args.manifest, repo_root=REPO_ROOT)
    if not specs:
        parser.error("No artifacts resolved. Check the manifest configuration.")

    if args.list_only:
        for spec in specs:
            direction = "dir" if spec.recursive else "file"
            print(f"[{direction}] {spec.relative_path} -> {spec.remote_key}")
        return

    sync_artifacts(
        specs,
        direction=args.direction,
        bucket=args.bucket,
        endpoint_url=args.endpoint,
        local_root=args.local_root,
        remote_prefix=args.remote_prefix,
        dry_run=args.dry_run,
        aws_cli=args.aws_cli,
        skip_existing=args.skip_existing if args.direction == "download" else False,
    )


if __name__ == "__main__":
    main()
