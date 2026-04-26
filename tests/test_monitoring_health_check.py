"""Tests for monitoring/health_check.py — unit tests for each check function."""

import json
import sys
from pathlib import Path

import pytest

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "monitoring"))

import monitoring.health_check as health_check
from monitoring.health_check import (
    CheckResult,
    run_cmd,
    check_portfolio_state,
    check_xgb_spy_provenance,
    auto_fix,
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
        rc, out, _err = run_cmd("echo hello")
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


class TestCheckXgbSpyProvenance:
    def test_clean_analyzer_output_is_ok(self, monkeypatch):
        payload = {
            "overall": {
                "n_sessions": 2,
                "n_spy_session_hashes": 1,
                "n_spy_provenance_warning_sessions": 0,
                "spy_provenance_warning_sessions": [],
            },
            "sessions": [],
        }

        def fake_run_cmd(cmd: str, timeout: int = 15):
            assert "--fail-on-spy-provenance-warning" in cmd
            assert timeout == 30
            return 0, json.dumps(payload), ""

        monkeypatch.setattr("monitoring.health_check.run_cmd", fake_run_cmd)

        result = check_xgb_spy_provenance()

        assert result.status == "ok"
        assert "no SPY provenance warnings" in result.message
        assert result.details["overall"]["n_spy_session_hashes"] == 1

    def test_provenance_warning_is_fail(self, monkeypatch):
        payload = {
            "overall": {
                "n_sessions": 2,
                "n_spy_provenance_warning_sessions": 1,
                "spy_provenance_warning_sessions": ["2026-04-21"],
            },
            "sessions": [],
        }

        monkeypatch.setattr(
            "monitoring.health_check.run_cmd",
            lambda cmd, timeout=15: (3, json.dumps(payload), ""),
        )

        result = check_xgb_spy_provenance()

        assert result.status == "fail"
        assert "2026-04-21" in result.message

    def test_no_trade_logs_is_ok_because_recent_activity_checks_staleness(self, monkeypatch):
        monkeypatch.setattr(
            "monitoring.health_check.run_cmd",
            lambda cmd, timeout=15: (0, "", "[analyze] no *.jsonl files"),
        )

        result = check_xgb_spy_provenance()

        assert result.status == "ok"
        assert "recent-activity handles staleness" in result.message

    def test_non_json_output_is_warn(self, monkeypatch):
        monkeypatch.setattr(
            "monitoring.health_check.run_cmd",
            lambda cmd, timeout=15: (0, "not json", ""),
        )

        result = check_xgb_spy_provenance()

        assert result.status == "warn"
        assert "non-JSON" in result.message


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
        auto_fix(results)

        state = json.loads(state_file.read_text())
        assert state["pending_close"] == ["BTCUSD"]


class TestMain:
    def test_json_mode_exits_nonzero_on_fail(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(sys, "argv", ["health_check.py", "--json"])
        monkeypatch.setattr(health_check, "LOG_DIR", tmp_path)
        monkeypatch.setattr(
            health_check,
            "run_all_checks",
            lambda: [CheckResult("xgb-spy-provenance", "fail", "bad")],
        )

        with pytest.raises(SystemExit) as exc:
            health_check.main()

        assert exc.value.code == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["results"][0]["status"] == "fail"

    def test_json_mode_exits_zero_without_failures(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(sys, "argv", ["health_check.py", "--json"])
        monkeypatch.setattr(health_check, "LOG_DIR", tmp_path)
        monkeypatch.setattr(
            health_check,
            "run_all_checks",
            lambda: [
                CheckResult("portfolio-state", "ok", "clean"),
                CheckResult("llm-stock-trader", "warn", "optional"),
            ],
        )

        with pytest.raises(SystemExit) as exc:
            health_check.main()

        assert exc.value.code == 0
        payload = json.loads(capsys.readouterr().out)
        assert [r["status"] for r in payload["results"]] == ["ok", "warn"]
