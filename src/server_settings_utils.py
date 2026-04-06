"""Shared settings resolution helpers for trading-server and binance-trading-server."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Resolution result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PathResolution:
    """Resolved file-system path with provenance metadata."""

    path: Path
    source: str = "default"

    def as_dict(self) -> dict[str, object]:
        return {"path": str(self.path), "source": self.source}


@dataclass(frozen=True)
class IntResolution:
    """Resolved integer value with provenance metadata."""

    value: int
    source: str = "default"

    def as_dict(self) -> dict[str, object]:
        return {"value": self.value, "source": self.source}


@dataclass(frozen=True)
class SecretResolution:
    """Resolved optional secret with provenance metadata."""

    secret: str | None
    source: str = "default"

    def as_dict(self) -> dict[str, object]:
        # Never leak the actual secret value into logs.
        return {
            "present": self.secret is not None,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Resolver functions
# ---------------------------------------------------------------------------


def resolve_repo_relative_path(
    path: str | Path | None,
    *,
    repo_root: Path,
    env_name: str,
    default_path: Path,
    explicit_label: str,
) -> PathResolution:
    """Resolve a file-system path from an explicit value, env-var, or default."""
    if path is not None:
        return PathResolution(path=Path(path), source=f"explicit({explicit_label})")

    env_val = os.getenv(env_name)
    if env_val is not None:
        return PathResolution(path=Path(env_val), source=f"env({env_name})")

    return PathResolution(path=default_path, source="default")


def resolve_explicit_or_env_int_resolution(
    value: int | None,
    *,
    env_name: str,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
    explicit_label: str,
) -> IntResolution:
    """Resolve an integer from an explicit value, env-var, or default."""
    if value is not None:
        resolved = value
        source = f"explicit({explicit_label})"
    else:
        env_val = os.getenv(env_name)
        if env_val is not None:
            try:
                resolved = int(env_val)
            except ValueError:
                resolved = default
                source = "default"
            else:
                source = f"env({env_name})"
        else:
            resolved = default
            source = "default"

    if minimum is not None and resolved < minimum:
        resolved = minimum
    if maximum is not None and resolved > maximum:
        resolved = maximum

    return IntResolution(value=resolved, source=source)


def resolve_optional_secret(
    value: str | None,
    *,
    env_name: str,
    explicit_label: str,
) -> str | None:
    """Convenience wrapper: resolve a secret and return the raw string (or None)."""
    return resolve_optional_secret_resolution(value, env_name=env_name, explicit_label=explicit_label).secret


def resolve_optional_secret_resolution(
    value: str | None,
    *,
    env_name: str,
    explicit_label: str,
) -> SecretResolution:
    """Resolve an optional secret from an explicit value or env-var."""
    if value is not None:
        return SecretResolution(secret=value, source=f"explicit({explicit_label})")

    env_val = os.getenv(env_name)
    if env_val is not None:
        return SecretResolution(secret=env_val, source=f"env({env_name})")

    return SecretResolution(secret=None, source="default")
