"""Tests for monitoring/health_check.py — unit tests for each check function."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "monitoring"))

from monitoring.health_check import (
    CheckResult,
    run_cmd,
    check_portfolio_state,
    auto_fix,
    run_all_checks,
)


class TestCheckResult:
    def test_ok_result(self):
        r = CheckResult("test", "ok", "all good")
        assert r.status == "ok"
        assert r.name == "test"

    def test_fail_result(self):
        r = CheckResult("test", "fail", "broken", {"error": True})
        assert r.status == "fail"
        assert r.details["error"] is True


class TestRunCmd:
    def test_echo(self):
        rc, out, err = run_cmd("echo hello")
        assert rc == 0
        assert out == "hello"

    def test_false_returns_nonzero(self):
        rc, _, _ = run_cmd("false")
        assert rc != 0

    def test_timeout_returns_negative(self):
        rc, _, err = run_cmd("sleep 30", timeout=1)
        assert rc == -1
        assert "timeout" in err


class TestCheckPortfolioState:
    def test_clean_state(self, tmp_path, monkeypatch):
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps({
            "positions": {},
            "pending_close": [],
        }))
        monkeypatch.setattr("monitoring.health_check.STATE_FILE", state_file)

        result = check_portfolio_state()
        assert result.status == "ok"
        assert "0 open positions" in result.message

    def test_stale_pending_close(self, tmp_path, monkeypatch):
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps({
            "positions": {},
            "pending_close": ["ETHUSD"],
        }))
        monkeypatch.setattr("monitoring.health_check.STATE_FILE", state_file)

        result = check_portfolio_state()
        assert result.status == "warn"
        assert "stale" in result.message

    def test_valid_pending_close(self, tmp_path, monkeypatch):
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps({
            "positions": {"ETHUSD": {"qty": 1.0}},
            "pending_close": ["ETHUSD"],
        }))
        monkeypatch.setattr("monitoring.health_check.STATE_FILE", state_file)

        result = check_portfolio_state()
        assert result.status == "ok"

    def test_missing_state_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("monitoring.health_check.STATE_FILE", tmp_path / "nope.json")
        result = check_portfolio_state()
        assert result.status == "warn"

    def test_corrupt_state_file(self, tmp_path, monkeypatch):
        state_file = tmp_path / "state.json"
        state_file.write_text("not json{{{")
        monkeypatch.setattr("monitoring.health_check.STATE_FILE", state_file)

        result = check_portfolio_state()
        assert result.status == "fail"


class TestAutoFix:
    def test_cleans_stale_pending_close(self, tmp_path, monkeypatch):
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps({
            "positions": {},
            "pending_close": ["ETHUSD", "BTCUSD"],
        }))
        monkeypatch.setattr("monitoring.health_check.STATE_FILE", state_file)

        results = [CheckResult("portfolio-state", "warn", "stale pending_close")]
        actions = auto_fix(results)

        assert len(actions) == 1
        assert "Cleaned" in actions[0]

        # Verify file was cleaned
        state = json.loads(state_file.read_text())
        assert state["pending_close"] == []

    def test_preserves_valid_pending_close(self, tmp_path, monkeypatch):
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps({
            "positions": {"BTCUSD": {"qty": 1.0}},
            "pending_close": ["BTCUSD", "ETHUSD"],
        }))
        monkeypatch.setattr("monitoring.health_check.STATE_FILE", state_file)

        results = [CheckResult("portfolio-state", "warn", "stale pending_close")]
        actions = auto_fix(results)

        state = json.loads(state_file.read_text())
        assert state["pending_close"] == ["BTCUSD"]
