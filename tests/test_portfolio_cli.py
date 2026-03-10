from __future__ import annotations

import argparse
from types import SimpleNamespace

from src.portfolio_cli import add_close_at_eod_args, resolve_close_at_eod
from unified_hourly_experiment.run_stock_sortino_lag_robust import build_parser


def test_add_close_at_eod_args_defaults_to_overnight_positions() -> None:
    parser = argparse.ArgumentParser()
    add_close_at_eod_args(parser, default=False)

    args = parser.parse_args([])

    assert args.close_at_eod is False


def test_add_close_at_eod_args_supports_both_explicit_flags() -> None:
    parser = argparse.ArgumentParser()
    add_close_at_eod_args(parser, default=False)

    assert parser.parse_args(["--close-at-eod"]).close_at_eod is True
    assert parser.parse_args(["--no-close-at-eod"]).close_at_eod is False


def test_resolve_close_at_eod_supports_legacy_namespaces() -> None:
    assert resolve_close_at_eod(SimpleNamespace(close_at_eod=False), default=True) is False
    assert resolve_close_at_eod(SimpleNamespace(no_close_at_eod=False), default=False) is True
    assert resolve_close_at_eod(SimpleNamespace(no_close_at_eod=True), default=True) is False


def test_run_stock_sortino_parser_defaults_to_overnight_positions() -> None:
    args = build_parser().parse_args([])

    assert args.close_at_eod is False
