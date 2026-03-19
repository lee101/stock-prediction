"""Shared CLI helpers for portfolio simulation/runtime flags."""

from __future__ import annotations

import argparse


def add_close_at_eod_args(
    parser: argparse.ArgumentParser,
    *,
    default: bool = False,
) -> None:
    """Add mutually-exclusive close-at-EOD flags with a shared default."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--close-at-eod",
        dest="close_at_eod",
        action="store_true",
        help="Force stock positions to close at the end of each trading day in simulations.",
    )
    group.add_argument(
        "--no-close-at-eod",
        dest="close_at_eod",
        action="store_false",
        help="Allow positions to remain open overnight in simulations.",
    )
    parser.set_defaults(close_at_eod=bool(default))


def resolve_close_at_eod(args: argparse.Namespace, *, default: bool = False) -> bool:
    """Resolve close-at-EOD from new or legacy argparse namespaces."""
    if hasattr(args, "close_at_eod"):
        return bool(getattr(args, "close_at_eod"))
    if hasattr(args, "no_close_at_eod"):
        return not bool(getattr(args, "no_close_at_eod"))
    return bool(default)
