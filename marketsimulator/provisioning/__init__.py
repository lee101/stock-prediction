"""
Provisioning helpers for the market simulator infrastructure.

This subpackage exposes typed clients for Vast.ai and RunPod along with a
Typer-powered command line interface that can be invoked via
``python -m marketsimulator.provisioning``.
"""

from .cli import app

__all__ = ["app"]
