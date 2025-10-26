"""
Environment-backed configuration helpers for GPU infrastructure providers.

The provisioning clients rely on API keys supplied via environment variables to
avoid persisting secrets inside the repository.  This module centralises the
logic so the rest of the codebase can remain focused on API concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
import os


def _get_env(key: str, *, required: bool = True, default: str | None = None) -> str:
    """Fetch ``key`` from the environment, optionally enforcing its presence."""
    value = os.getenv(key, default)
    if required and not value:
        raise RuntimeError(
            f"Environment variable {key!r} must be set for provisioning commands."
        )
    return value or ""


@dataclass(slots=True)
class VastSettings:
    """Authentication and behavioural settings for Vast.ai API calls."""

    api_key: str
    base_url: str = "https://console.vast.ai/api/v0"

    @classmethod
    def from_env(cls) -> "VastSettings":
        return cls(api_key=_get_env("VAST_API_KEY"))


@dataclass(slots=True)
class RunPodSettings:
    """Authentication and behavioural settings for RunPod API calls."""

    api_key: str
    rest_base_url: str = "https://rest.runpod.io/v1"
    queue_base_url: str = "https://api.runpod.ai/v2"
    graphql_url: str = "https://api.runpod.io/graphql"

    @classmethod
    def from_env(cls) -> "RunPodSettings":
        return cls(api_key=_get_env("RUNPOD_API_KEY"))
