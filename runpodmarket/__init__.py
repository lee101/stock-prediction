"""RunPod serverless entrypoint for the market simulator."""

from __future__ import annotations

from .handler import handle_job, handler

__all__ = ["handle_job", "handler"]
