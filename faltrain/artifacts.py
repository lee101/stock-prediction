#!/usr/bin/env python3
"""
Artifact manifest loading and sync helpers shared across CLI and FAL apps.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - fallback for older runtimes
    import tomli as tomllib  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path(__file__).with_name("model_manifest.toml")

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArtifactSpec:
    """Resolved artifact ready for upload/download."""

    name: str
    relative_path: Path
    remote_key: str
    recursive: bool = False
    required: bool = False

    def local_path(self, root: Path) -> Path:
        return (root / self.relative_path).resolve()


def _load_manifest_data(manifest_path: Optional[Path]) -> dict:
    path = manifest_path or DEFAULT_MANIFEST
    if not path.exists():
        LOG.debug("Manifest %s missing, will use built-in defaults", path)
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _normalise_remote(key: str) -> str:
    stripped = key.strip().lstrip("/")
    return stripped


def _resolve_artifacts_from_patterns(
    patterns: Iterable[dict],
    *,
    repo_root: Path,
) -> List[ArtifactSpec]:
    resolved: List[ArtifactSpec] = []
    for entry in patterns:
        name = entry.get("name", "pattern")
        base = entry.get("base")
        glob_expr = entry.get("glob")
        remote_prefix = entry.get("remote_prefix", "")
        required = bool(entry.get("required", False))
        recursive_flag = entry.get("recursive")

        if not base or not glob_expr:
            LOG.warning("Skipping pattern %s (missing base/glob)", name)
            continue

        base_path = (repo_root / base).resolve()
        matches = list(base_path.glob(glob_expr))

        if not matches:
            if required:
                LOG.warning("Required pattern '%s' under %s matched nothing", name, base_path)
            continue

        for match in matches:
            rel_path = match.relative_to(repo_root)
            remote_key = Path(remote_prefix.strip("/")) / match.relative_to(base_path)
            recursive = bool(recursive_flag) if recursive_flag is not None else match.is_dir()
            resolved.append(
                ArtifactSpec(
                    name=f"{name}: {match.name}",
                    relative_path=rel_path,
                    remote_key=_normalise_remote(remote_key.as_posix()),
                    recursive=recursive,
                    required=required,
                )
            )
    return resolved


def _resolve_direct_artifacts(
    artifacts: Iterable[dict],
    *,
    repo_root: Path,
) -> List[ArtifactSpec]:
    resolved: List[ArtifactSpec] = []
    for entry in artifacts:
        name = entry.get("name", "artifact")
        source = entry.get("source")
        remote = entry.get("remote")
        if not source or not remote:
            LOG.warning("Skipping artifact %s (missing source/remote)", name)
            continue
        rel_path = Path(source)
        remote_key = _normalise_remote(remote)
        recursive = bool(entry.get("recursive", False))
        required = bool(entry.get("required", False))
        resolved.append(
            ArtifactSpec(
                name=name,
                relative_path=rel_path,
                remote_key=remote_key,
                recursive=recursive,
                required=required,
            )
        )
    return resolved


def load_artifact_specs(
    *,
    manifest_path: Optional[Path] = None,
    repo_root: Path = REPO_ROOT,
) -> List[ArtifactSpec]:
    """Return resolved artifact specs from manifest."""
    data = _load_manifest_data(manifest_path)
    patterns = data.get("pattern", [])
    artifacts = data.get("artifact", [])

    resolved: List[ArtifactSpec] = []
    resolved.extend(_resolve_artifacts_from_patterns(patterns, repo_root=repo_root))
    resolved.extend(_resolve_direct_artifacts(artifacts, repo_root=repo_root))
    return resolved


def _build_remote_uri(bucket: str, remote_key: str, remote_prefix: str) -> str:
    bucket = bucket.strip()
    if remote_prefix:
        prefix = Path(remote_prefix.strip("/"))
        remote_key = (prefix / remote_key).as_posix()
    return f"s3://{bucket.rstrip('/')}/{remote_key}"


def _run_aws(cmd: list[str], *, dry_run: bool) -> None:
    LOG.info("• %s", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def sync_artifacts(
    specs: Iterable[ArtifactSpec],
    *,
    direction: str,
    bucket: str,
    endpoint_url: str,
    local_root: Path,
    remote_prefix: str = "",
    dry_run: bool = False,
    aws_cli: str = "aws",
    skip_existing: bool = False,
) -> None:
    """
    Sync artifacts either uploading (local->R2) or downloading (R2->local_root).
    """
    direction = direction.lower()
    if direction not in {"upload", "download"}:
        raise ValueError(f"Unsupported direction: {direction}")

    local_root = local_root.resolve()
    for spec in specs:
        local_path = spec.local_path(local_root)
        remote_uri = _build_remote_uri(bucket, spec.remote_key, remote_prefix)
        endpoint_args = ["--endpoint-url", endpoint_url]

        if direction == "upload":
            if not local_path.exists():
                message = f"Missing local artifact '{spec.name}' at {local_path}"
                if spec.required:
                    raise FileNotFoundError(message)
                LOG.warning("%s (skipping)", message)
                continue
            if spec.recursive:
                if not local_path.is_dir():
                    raise ValueError(
                        f"Artifact '{spec.name}' expected directory at {local_path}, found file"
                    )
                dest = remote_uri.rstrip("/") + "/"
                cmd = [aws_cli, "s3", "cp", str(local_path), dest, "--recursive", *endpoint_args]
            else:
                if local_path.is_dir():
                    raise ValueError(
                        f"Artifact '{spec.name}' expected file at {local_path}, found directory"
                    )
                cmd = [aws_cli, "s3", "cp", str(local_path), remote_uri, *endpoint_args]
        else:  # download
            if skip_existing:
                if spec.recursive:
                    if local_path.exists() and local_path.is_dir() and any(local_path.iterdir()):
                        LOG.info("Skipping existing directory '%s'", local_path)
                        continue
                else:
                    if local_path.exists() and local_path.is_file():
                        LOG.info("Skipping existing file '%s'", local_path)
                        continue
            if spec.recursive:
                dest_dir = local_path
                dest_dir.mkdir(parents=True, exist_ok=True)
                src = remote_uri.rstrip("/") + "/"
                cmd = [aws_cli, "s3", "cp", src, str(dest_dir), "--recursive", *endpoint_args]
            else:
                dest_dir = local_path.parent
                dest_dir.mkdir(parents=True, exist_ok=True)
                cmd = [aws_cli, "s3", "cp", remote_uri, str(local_path), *endpoint_args]

        try:
            _run_aws(cmd, dry_run=dry_run)
        except subprocess.CalledProcessError as exc:
            message = f"AWS CLI failed for '{spec.name}': {exc}"
            if spec.required:
                raise RuntimeError(message) from exc
            LOG.warning("%s (ignored)", message)
