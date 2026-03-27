#!/usr/bin/env python3
"""Tests for 120-day backtest evaluation wrapper scripts."""
from __future__ import annotations

import pytest
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


@pytest.mark.unit
class TestHybridSpotEval:

    def test_default_args(self):
        from scripts.run_120d_hybrid_spot_eval import parse_args, DEFAULT_SYMBOLS, DEFAULT_START, DEFAULT_END
        args = parse_args([])
        assert args.symbols == DEFAULT_SYMBOLS
        assert args.start_date == DEFAULT_START
        assert args.end_date == DEFAULT_END
        assert args.mode == "hybrid"
        assert args.cash == 10000.0
        assert args.use_cache is True

    def test_custom_symbols(self):
        from scripts.run_120d_hybrid_spot_eval import parse_args
        args = parse_args(["--symbols", "BTCUSD,ETHUSD"])
        assert args.symbols == "BTCUSD,ETHUSD"

    def test_custom_dates(self):
        from scripts.run_120d_hybrid_spot_eval import parse_args
        args = parse_args(["--start-date", "2025-12-01", "--end-date", "2026-02-01"])
        assert args.start_date == "2025-12-01"
        assert args.end_date == "2026-02-01"

    def test_rl_only_mode(self):
        from scripts.run_120d_hybrid_spot_eval import parse_args
        args = parse_args(["--mode", "rl_only"])
        assert args.mode == "rl_only"

    def test_dry_run_flag(self):
        from scripts.run_120d_hybrid_spot_eval import parse_args
        args = parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_custom_report_path(self):
        from scripts.run_120d_hybrid_spot_eval import parse_args
        args = parse_args(["--report", "reports/custom.json"])
        assert args.report == "reports/custom.json"

    def test_no_cache_flag(self):
        from scripts.run_120d_hybrid_spot_eval import parse_args
        args = parse_args(["--no-cache"])
        assert args.no_cache is True

    def test_cash_override(self):
        from scripts.run_120d_hybrid_spot_eval import parse_args
        args = parse_args(["--cash", "50000"])
        assert args.cash == 50000.0

    def test_default_symbols_use_auto_universe(self):
        from scripts.run_120d_hybrid_spot_eval import DEFAULT_SYMBOLS
        assert DEFAULT_SYMBOLS == "auto"

    def test_build_parser_returns_parser(self):
        from scripts.run_120d_hybrid_spot_eval import build_parser
        import argparse
        p = build_parser()
        assert isinstance(p, argparse.ArgumentParser)


@pytest.mark.unit
class TestWorkstealEval:

    def test_default_args(self):
        from scripts.run_120d_worksteal_eval import parse_args, DEFAULT_START, DEFAULT_END
        args = parse_args([])
        assert args.start_date == DEFAULT_START
        assert args.end_date == DEFAULT_END
        assert args.rule_only is False
        assert args.use_cache is True

    def test_rule_only_flag(self):
        from scripts.run_120d_worksteal_eval import parse_args
        args = parse_args(["--rule-only"])
        assert args.rule_only is True

    def test_custom_dates(self):
        from scripts.run_120d_worksteal_eval import parse_args
        args = parse_args(["--start-date", "2025-12-01", "--end-date", "2026-02-01"])
        assert args.start_date == "2025-12-01"
        assert args.end_date == "2026-02-01"

    def test_dry_run_flag(self):
        from scripts.run_120d_worksteal_eval import parse_args
        args = parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_custom_report_path(self):
        from scripts.run_120d_worksteal_eval import parse_args
        args = parse_args(["--report", "reports/custom_ws.json"])
        assert args.report == "reports/custom_ws.json"

    def test_no_cache_flag(self):
        from scripts.run_120d_worksteal_eval import parse_args
        args = parse_args(["--no-cache"])
        assert args.no_cache is True

    def test_deployed_config_values(self):
        from scripts.run_120d_worksteal_eval import DEPLOYED_CONFIG
        assert DEPLOYED_CONFIG["dip_pct"] == 0.18
        assert DEPLOYED_CONFIG["profit_target_pct"] == 0.20
        assert DEPLOYED_CONFIG["stop_loss_pct"] == 0.15
        assert DEPLOYED_CONFIG["sma_filter_period"] == 20
        assert DEPLOYED_CONFIG["trailing_stop_pct"] == 0.03
        assert DEPLOYED_CONFIG["max_positions"] == 5
        assert DEPLOYED_CONFIG["max_hold_days"] == 14

    def test_custom_data_dir(self):
        from scripts.run_120d_worksteal_eval import parse_args
        args = parse_args(["--data-dir", "/tmp/data"])
        assert args.data_dir == "/tmp/data"

    def test_custom_model(self):
        from scripts.run_120d_worksteal_eval import parse_args
        args = parse_args(["--model", "gemini-2.0-flash"])
        assert args.model == "gemini-2.0-flash"

    def test_build_parser_returns_parser(self):
        from scripts.run_120d_worksteal_eval import build_parser
        import argparse
        p = build_parser()
        assert isinstance(p, argparse.ArgumentParser)
