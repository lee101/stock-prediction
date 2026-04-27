"""Tests for monitoring/health_check.py — unit tests for each check function."""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
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
    check_alpaca_monitor_timer,
    check_monitor_agent_status,
    check_portfolio_state,
    check_scheduled_audit_status,
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


class TestMainLogging:
    def test_script_runs_from_outside_repo(self, tmp_path):
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "monitoring" / "health_check.py"), "--help"],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            check=False,
        )

        assert proc.returncode == 0
        assert "Alpaca trading health check" in proc.stdout
        assert "ModuleNotFoundError" not in proc.stderr

    def test_main_uses_locked_jsonl_append(self, tmp_path, monkeypatch, capsys):
        calls = []

        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        monkeypatch.setattr(
            "monitoring.health_check.run_all_checks",
            lambda: [CheckResult("scheduled-audits", "ok", "healthy")],
        )

        def fake_append_jsonl_row(path, payload, *, default=None, sort_keys=False):
            calls.append(
                {
                    "path": path,
                    "payload": payload,
                    "default": default,
                    "sort_keys": sort_keys,
                },
            )

        monkeypatch.setattr(
            "monitoring.health_check.append_jsonl_row",
            fake_append_jsonl_row,
        )
        monkeypatch.setattr(sys, "argv", ["health_check.py", "--json"])

        with pytest.raises(SystemExit) as exc:
            health_check.main()

        assert exc.value.code == 0
        assert len(calls) == 1
        assert calls[0]["path"].parent == tmp_path
        assert calls[0]["path"].name.startswith("health_")
        assert calls[0]["path"].suffix == ".jsonl"
        assert calls[0]["default"] is str
        assert calls[0]["sort_keys"] is False
        assert calls[0]["payload"]["results"][0]["name"] == "scheduled-audits"
        assert '"scheduled-audits"' in capsys.readouterr().out


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


class TestCheckAlpacaMonitorTimer:
    def test_active_enabled_timer_is_ok(self, monkeypatch):
        def fake_run_cmd(cmd: str, timeout: int = 15):
            if cmd == "systemctl is-active alpaca-monitor.timer":
                return 0, "active", ""
            if cmd == "systemctl is-enabled alpaca-monitor.timer":
                return 0, "enabled", ""
            if cmd == "systemctl is-failed alpaca-monitor.service":
                return 1, "inactive", ""
            raise AssertionError(cmd)

        monkeypatch.setattr("monitoring.health_check.run_cmd", fake_run_cmd)

        result = check_alpaca_monitor_timer()

        assert result.status == "ok"
        assert "active and enabled" in result.message

    def test_inactive_timer_is_fail(self, monkeypatch):
        def fake_run_cmd(cmd: str, timeout: int = 15):
            if cmd == "systemctl is-active alpaca-monitor.timer":
                return 3, "inactive", ""
            if cmd == "systemctl is-enabled alpaca-monitor.timer":
                return 0, "enabled", ""
            if cmd == "systemctl is-failed alpaca-monitor.service":
                return 1, "inactive", ""
            raise AssertionError(cmd)

        monkeypatch.setattr("monitoring.health_check.run_cmd", fake_run_cmd)

        result = check_alpaca_monitor_timer()

        assert result.status == "fail"
        assert "not active" in result.message

    def test_failed_monitor_service_is_fail(self, monkeypatch):
        def fake_run_cmd(cmd: str, timeout: int = 15):
            if cmd == "systemctl is-active alpaca-monitor.timer":
                return 0, "active", ""
            if cmd == "systemctl is-enabled alpaca-monitor.timer":
                return 0, "enabled", ""
            if cmd == "systemctl is-failed alpaca-monitor.service":
                return 0, "failed", ""
            raise AssertionError(cmd)

        monkeypatch.setattr("monitoring.health_check.run_cmd", fake_run_cmd)

        result = check_alpaca_monitor_timer()

        assert result.status == "fail"
        assert "service is failed" in result.message

    def test_active_disabled_timer_is_warn(self, monkeypatch):
        def fake_run_cmd(cmd: str, timeout: int = 15):
            if cmd == "systemctl is-active alpaca-monitor.timer":
                return 0, "active", ""
            if cmd == "systemctl is-enabled alpaca-monitor.timer":
                return 1, "disabled", ""
            if cmd == "systemctl is-failed alpaca-monitor.service":
                return 1, "inactive", ""
            raise AssertionError(cmd)

        monkeypatch.setattr("monitoring.health_check.run_cmd", fake_run_cmd)

        result = check_alpaca_monitor_timer()

        assert result.status == "warn"
        assert "not enabled" in result.message


class TestCheckScheduledAuditStatus:
    def _sha256(self, path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _write_artifacts(self, tmp_path, *names):
        for name in names:
            (tmp_path / name).write_text("audit artifact\n")

    def _write_codex_artifacts(self, tmp_path, stem="codex_prod_20260426T000000Z"):
        self._write_artifacts(
            tmp_path,
            f"{stem}.log",
            f"{stem}.raw.jsonl",
            f"{stem}.last.txt",
        )
        return (
            f"log={stem}.log raw_log={stem}.raw.jsonl last_msg={stem}.last.txt "
            f"log_sha256={self._sha256(tmp_path / f'{stem}.log')} "
            f"raw_log_sha256={self._sha256(tmp_path / f'{stem}.raw.jsonl')} "
            f"last_msg_sha256={self._sha256(tmp_path / f'{stem}.last.txt')}"
        )

    def _write_hourly_artifacts(self, tmp_path, stem="hourly_prod_20260426T000000Z"):
        self._write_artifacts(tmp_path, f"{stem}.log", f"{stem}.raw.jsonl")
        return (
            f"log={stem}.log raw_log={stem}.raw.jsonl "
            f"log_sha256={self._sha256(tmp_path / f'{stem}.log')} "
            f"raw_log_sha256={self._sha256(tmp_path / f'{stem}.raw.jsonl')}"
        )

    def _codex_stem_from_timestamp(self, ts: str):
        stamp = datetime.fromisoformat(ts).strftime("%Y%m%dT%H%M%SZ")
        return f"codex_prod_{stamp}"

    def _hourly_stem_from_timestamp(self, ts: str):
        stamp = datetime.fromisoformat(ts).strftime("%Y%m%dT%H%M%SZ")
        return f"hourly_prod_{stamp}"

    def test_clean_current_status_files_are_ok(self, tmp_path, monkeypatch):
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        monkeypatch.setattr(
            "monitoring.health_check.time.time",
            lambda: datetime.fromisoformat("2026-04-26T01:00:00+00:00").timestamp(),
        )
        codex_fields = self._write_codex_artifacts(tmp_path)
        hourly_fields = self._write_hourly_artifacts(tmp_path)
        (tmp_path / "codex_current.log").write_text(
            f"2026-04-26T00:00:00+00:00 status=OK rc=0 {codex_fields}\n",
        )
        (tmp_path / "hourly_current.log").write_text(
            f"2026-04-26T00:00:00+00:00 status=OK rc=0 {hourly_fields}\n",
        )

        result = check_scheduled_audit_status()

        assert result.status == "ok"
        assert "codex" in result.message
        assert "hourly" in result.message

    def test_ok_current_status_requires_wrapper_artifact_names(self, tmp_path, monkeypatch):
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        monkeypatch.setattr(
            "monitoring.health_check.time.time",
            lambda: datetime.fromisoformat("2026-04-26T01:00:00+00:00").timestamp(),
        )
        self._write_artifacts(tmp_path, "wrong.log", "wrong.raw.jsonl", "wrong.last.txt")
        hourly_fields = self._write_hourly_artifacts(tmp_path)
        (tmp_path / "codex_current.log").write_text(
            "2026-04-26T00:00:00+00:00 "
            "status=OK rc=0 log=wrong.log raw_log=wrong.raw.jsonl last_msg=wrong.last.txt\n",
        )
        (tmp_path / "hourly_current.log").write_text(
            f"2026-04-26T00:00:00+00:00 status=OK rc=0 {hourly_fields}\n",
        )

        result = check_scheduled_audit_status()

        assert result.status == "fail"
        assert "invalid audit artifact" in result.message
        assert "codex" in result.message
        assert "log:bad_name" in result.message

    def test_ok_current_status_requires_matching_artifact_stems(self, tmp_path, monkeypatch):
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        monkeypatch.setattr(
            "monitoring.health_check.time.time",
            lambda: datetime.fromisoformat("2026-04-26T01:00:00+00:00").timestamp(),
        )
        self._write_artifacts(
            tmp_path,
            "codex_prod_20260426T000000Z.log",
            "codex_prod_20260426T010000Z.raw.jsonl",
            "codex_prod_20260426T000000Z.last.txt",
        )
        hourly_fields = self._write_hourly_artifacts(tmp_path)
        (tmp_path / "codex_current.log").write_text(
            "2026-04-26T00:00:00+00:00 status=OK rc=0 "
            "log=codex_prod_20260426T000000Z.log "
            "raw_log=codex_prod_20260426T010000Z.raw.jsonl "
            "last_msg=codex_prod_20260426T000000Z.last.txt\n",
        )
        (tmp_path / "hourly_current.log").write_text(
            f"2026-04-26T00:00:00+00:00 status=OK rc=0 {hourly_fields}\n",
        )

        result = check_scheduled_audit_status()

        assert result.status == "fail"
        assert "invalid audit artifact" in result.message
        assert "codex" in result.message
        assert "raw_log:bad_name" in result.message

    def test_ok_current_status_rejects_non_wrapper_artifact_stem_shape(
        self,
        tmp_path,
        monkeypatch,
    ):
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        monkeypatch.setattr(
            "monitoring.health_check.time.time",
            lambda: datetime.fromisoformat("2026-04-26T01:00:00+00:00").timestamp(),
        )
        stem = "codex_prod_manual_prod_20260426T000000Z"
        self._write_artifacts(
            tmp_path,
            f"{stem}.log",
            f"{stem}.raw.jsonl",
            f"{stem}.last.txt",
        )
        hourly_fields = self._write_hourly_artifacts(tmp_path)
        (tmp_path / "codex_current.log").write_text(
            f"2026-04-26T00:00:00+00:00 status=OK rc=0 "
            f"log={stem}.log raw_log={stem}.raw.jsonl last_msg={stem}.last.txt\n",
        )
        (tmp_path / "hourly_current.log").write_text(
            f"2026-04-26T00:00:00+00:00 status=OK rc=0 {hourly_fields}\n",
        )

        result = check_scheduled_audit_status()

        assert result.status == "fail"
        assert "invalid audit artifact" in result.message
        assert "log:bad_name" in result.message

    def test_ok_current_status_rejects_stale_reused_artifact_names(self, tmp_path, monkeypatch):
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        monkeypatch.setattr(
            "monitoring.health_check.time.time",
            lambda: datetime.fromisoformat("2026-04-27T17:00:00+00:00").timestamp(),
        )
        codex_fields = self._write_codex_artifacts(tmp_path, "codex_prod_20260426T000000Z")
        hourly_fields = self._write_hourly_artifacts(tmp_path, "hourly_prod_20260426T000000Z")
        (tmp_path / "codex_current.log").write_text(
            f"2026-04-27T16:30:00+00:00 status=OK rc=0 {codex_fields}\n",
        )
        (tmp_path / "hourly_current.log").write_text(
            f"2026-04-27T16:30:00+00:00 status=OK rc=0 {hourly_fields}\n",
        )

        result = check_scheduled_audit_status()

        assert result.status == "fail"
        assert "stale_artifact_timestamp" in result.message

    def test_ok_current_status_requires_referenced_artifacts(self, tmp_path, monkeypatch):
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        monkeypatch.setattr(
            "monitoring.health_check.time.time",
            lambda: datetime.fromisoformat("2026-04-26T01:00:00+00:00").timestamp(),
        )
        codex_fields = self._write_codex_artifacts(tmp_path)
        (tmp_path / "codex_current.log").write_text(
            f"2026-04-26T00:00:00+00:00 status=OK rc=0 {codex_fields}\n",
        )
        (tmp_path / "hourly_current.log").write_text(
            "2026-04-26T00:00:00+00:00 status=OK rc=0 "
            "log=hourly_prod_20260426T000000Z.log "
            "raw_log=hourly_prod_20260426T000000Z.raw.jsonl\n",
        )

        result = check_scheduled_audit_status()

        assert result.status == "fail"
        assert "invalid audit artifact" in result.message
        assert "hourly" in result.message
        assert "log:missing" in result.message
        assert "raw_log:missing" in result.message

    def test_ok_current_status_requires_non_empty_artifacts(self, tmp_path, monkeypatch):
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        monkeypatch.setattr(
            "monitoring.health_check.time.time",
            lambda: datetime.fromisoformat("2026-04-26T01:00:00+00:00").timestamp(),
        )
        self._write_artifacts(
            tmp_path,
            "codex_prod_20260426T000000Z.raw.jsonl",
            "codex_prod_20260426T000000Z.last.txt",
        )
        (tmp_path / "codex_prod_20260426T000000Z.log").write_text("")
        hourly_fields = self._write_hourly_artifacts(tmp_path)
        (tmp_path / "codex_current.log").write_text(
            "2026-04-26T00:00:00+00:00 status=OK rc=0 "
            "log=codex_prod_20260426T000000Z.log "
            "raw_log=codex_prod_20260426T000000Z.raw.jsonl "
            "last_msg=codex_prod_20260426T000000Z.last.txt\n",
        )
        (tmp_path / "hourly_current.log").write_text(
            f"2026-04-26T00:00:00+00:00 status=OK rc=0 {hourly_fields}\n",
        )

        result = check_scheduled_audit_status()

        assert result.status == "fail"
        assert "invalid audit artifact" in result.message
        assert "codex" in result.message
        assert "log:empty" in result.message

    def test_ok_current_status_requires_artifact_hashes(self, tmp_path, monkeypatch):
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        monkeypatch.setattr(
            "monitoring.health_check.time.time",
            lambda: datetime.fromisoformat("2026-04-26T01:00:00+00:00").timestamp(),
        )
        stem = "codex_prod_20260426T000000Z"
        self._write_artifacts(
            tmp_path,
            f"{stem}.log",
            f"{stem}.raw.jsonl",
            f"{stem}.last.txt",
        )
        hourly_fields = self._write_hourly_artifacts(tmp_path)
        (tmp_path / "codex_current.log").write_text(
            f"2026-04-26T00:00:00+00:00 status=OK rc=0 "
            f"log={stem}.log raw_log={stem}.raw.jsonl last_msg={stem}.last.txt\n",
        )
        (tmp_path / "hourly_current.log").write_text(
            f"2026-04-26T00:00:00+00:00 status=OK rc=0 {hourly_fields}\n",
        )

        result = check_scheduled_audit_status()

        assert result.status == "fail"
        assert "log_sha256:missing" in result.message
        assert "raw_log_sha256:missing" in result.message
        assert "last_msg_sha256:missing" in result.message

    def test_ok_current_status_rejects_artifact_hash_mismatch(self, tmp_path, monkeypatch):
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        monkeypatch.setattr(
            "monitoring.health_check.time.time",
            lambda: datetime.fromisoformat("2026-04-26T01:00:00+00:00").timestamp(),
        )
        codex_fields = self._write_codex_artifacts(tmp_path)
        hourly_fields = self._write_hourly_artifacts(tmp_path)
        (tmp_path / "codex_prod_20260426T000000Z.raw.jsonl").write_text(
            "changed after current status\n",
        )
        (tmp_path / "codex_current.log").write_text(
            f"2026-04-26T00:00:00+00:00 status=OK rc=0 {codex_fields}\n",
        )
        (tmp_path / "hourly_current.log").write_text(
            f"2026-04-26T00:00:00+00:00 status=OK rc=0 {hourly_fields}\n",
        )

        result = check_scheduled_audit_status()

        assert result.status == "fail"
        assert "raw_log_sha256:mismatch" in result.message

    def test_ok_current_status_rejects_artifacts_outside_log_dir(self, tmp_path, monkeypatch):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        outside = tmp_path / "codex_prod_20260426T000000Z.log"
        outside.write_text("not wrapper-owned evidence\n")
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", log_dir)
        monkeypatch.setattr(
            "monitoring.health_check.time.time",
            lambda: datetime.fromisoformat("2026-04-26T01:00:00+00:00").timestamp(),
        )
        self._write_codex_artifacts(log_dir)
        hourly_fields = self._write_hourly_artifacts(log_dir)
        (log_dir / "codex_current.log").write_text(
            "2026-04-26T00:00:00+00:00 "
            f"status=OK rc=0 log={outside} "
            "raw_log=codex_prod_20260426T000000Z.raw.jsonl "
            "last_msg=codex_prod_20260426T000000Z.last.txt\n",
        )
        (log_dir / "hourly_current.log").write_text(
            f"2026-04-26T00:00:00+00:00 status=OK rc=0 {hourly_fields}\n",
        )

        result = check_scheduled_audit_status()

        assert result.status == "fail"
        assert "invalid audit artifact" in result.message
        assert "codex" in result.message
        assert "log:outside_log_dir" in result.message

    def test_ok_current_status_rejects_relative_path_traversal(self, tmp_path, monkeypatch):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (tmp_path / "hourly_prod_20260426T000000Z.log").write_text(
            "not wrapper-owned evidence\n",
        )
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", log_dir)
        monkeypatch.setattr(
            "monitoring.health_check.time.time",
            lambda: datetime.fromisoformat("2026-04-26T01:00:00+00:00").timestamp(),
        )
        codex_fields = self._write_codex_artifacts(log_dir)
        self._write_hourly_artifacts(log_dir)
        (log_dir / "codex_current.log").write_text(
            f"2026-04-26T00:00:00+00:00 status=OK rc=0 {codex_fields}\n",
        )
        (log_dir / "hourly_current.log").write_text(
            "2026-04-26T00:00:00+00:00 "
            "status=OK rc=0 "
            "log=../hourly_prod_20260426T000000Z.log "
            "raw_log=hourly_prod_20260426T000000Z.raw.jsonl\n",
        )

        result = check_scheduled_audit_status()

        assert result.status == "fail"
        assert "invalid audit artifact" in result.message
        assert "hourly" in result.message
        assert "log:outside_log_dir" in result.message

    def test_lock_skipped_current_status_is_warn(self, tmp_path, monkeypatch):
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        monkeypatch.setattr(
            "monitoring.health_check.time.time",
            lambda: datetime.fromisoformat("2026-04-26T01:00:00+00:00").timestamp(),
        )
        hourly_fields = self._write_hourly_artifacts(tmp_path)
        (tmp_path / "codex_current.log").write_text(
            "2026-04-26T00:00:00+00:00 status=SKIPPED_LOCK rc=0 log=a raw_log=b\n",
        )
        (tmp_path / "hourly_current.log").write_text(
            f"2026-04-26T00:00:00+00:00 status=OK rc=0 {hourly_fields}\n",
        )

        result = check_scheduled_audit_status()

        assert result.status == "warn"
        assert "skipped" in result.message
        assert "codex" in result.message

    def test_dry_run_current_status_is_warn(self, tmp_path, monkeypatch):
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        monkeypatch.setattr(
            "monitoring.health_check.time.time",
            lambda: datetime.fromisoformat("2026-04-26T01:00:00+00:00").timestamp(),
        )
        hourly_fields = self._write_hourly_artifacts(tmp_path)
        (tmp_path / "codex_current.log").write_text(
            "2026-04-26T00:00:00+00:00 status=DRY_RUN rc=0 log=a raw_log=b\n",
        )
        (tmp_path / "hourly_current.log").write_text(
            f"2026-04-26T00:00:00+00:00 status=OK rc=0 {hourly_fields}\n",
        )

        result = check_scheduled_audit_status()

        assert result.status == "warn"
        assert "did not run normally" in result.message
        assert "codex" in result.message

    def test_unknown_current_status_is_fail(self, tmp_path, monkeypatch):
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        (tmp_path / "codex_current.log").write_text(
            "2026-04-26T00:00:00+00:00 status=MAYBE rc=0 log=a raw_log=b\n",
        )
        (tmp_path / "hourly_current.log").write_text(
            "2026-04-26T00:00:00+00:00 status=OK rc=0 log=d raw_log=e\n",
        )

        result = check_scheduled_audit_status()

        assert result.status == "fail"
        assert "unknown status=MAYBE" in result.message

    def test_failed_current_status_is_fail(self, tmp_path, monkeypatch):
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        (tmp_path / "codex_current.log").write_text(
            "2026-04-26T00:00:00+00:00 status=FAILED rc=7 log=a raw_log=b last_msg=c\n",
        )
        (tmp_path / "hourly_current.log").write_text(
            "2026-04-26T00:00:00+00:00 status=OK rc=0 log=d raw_log=e\n",
        )

        result = check_scheduled_audit_status()

        assert result.status == "fail"
        assert "codex" in result.message
        assert "rc=7" in result.message

    def test_malformed_current_status_is_fail(self, tmp_path, monkeypatch):
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        (tmp_path / "codex_current.log").write_text(
            "2026-04-26T00:00:00+00:00 status=OK log=a raw_log=b last_msg=c\n",
        )
        (tmp_path / "hourly_current.log").write_text(
            "2026-04-26T00:00:00+00:00 rc=0 log=d raw_log=e\n",
        )

        result = check_scheduled_audit_status()

        assert result.status == "fail"
        assert "malformed" in result.message

    def test_multi_line_current_status_is_fail(self, tmp_path, monkeypatch):
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        (tmp_path / "codex_current.log").write_text(
            "2026-04-26T00:00:00+00:00 status=OK rc=0 log=a raw_log=b\n"
            "[12:00:00Z] agent breadcrumb that should not be here\n",
        )
        (tmp_path / "hourly_current.log").write_text(
            "2026-04-26T00:00:00+00:00 status=OK rc=0 log=d raw_log=e\n",
        )

        result = check_scheduled_audit_status()

        assert result.status == "fail"
        assert "multi-line" in result.message

    def test_duplicate_current_status_field_is_fail(self, tmp_path, monkeypatch):
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        (tmp_path / "codex_current.log").write_text(
            "2026-04-26T00:00:00+00:00 status=FAILED status=OK rc=0 log=a raw_log=b\n",
        )
        (tmp_path / "hourly_current.log").write_text(
            "2026-04-26T00:00:00+00:00 status=OK rc=0 log=d raw_log=e\n",
        )

        result = check_scheduled_audit_status()

        assert result.status == "fail"
        assert "duplicate" in result.message

    def test_unexpected_current_status_token_is_fail(self, tmp_path, monkeypatch):
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        (tmp_path / "codex_current.log").write_text(
            "2026-04-26T00:00:00+00:00 status=OK rc=0 human-breadcrumb log=a raw_log=b\n",
        )
        (tmp_path / "hourly_current.log").write_text(
            "2026-04-26T00:00:00+00:00 status=OK rc=0 log=d raw_log=e\n",
        )

        result = check_scheduled_audit_status()

        assert result.status == "fail"
        assert "unexpected token" in result.message

    def test_stale_current_status_is_fail(self, tmp_path, monkeypatch):
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        codex = tmp_path / "codex_current.log"
        hourly = tmp_path / "hourly_current.log"
        old = 1_700_000_000
        old_ts = datetime.fromtimestamp(old, tz=timezone.utc).isoformat()
        codex_fields = self._write_codex_artifacts(
            tmp_path,
            self._codex_stem_from_timestamp(old_ts),
        )
        hourly_fields = self._write_hourly_artifacts(
            tmp_path,
            self._hourly_stem_from_timestamp(old_ts),
        )
        codex.write_text(f"{old_ts} status=OK rc=0 {codex_fields}\n")
        hourly.write_text(f"{old_ts} status=OK rc=0 {hourly_fields}\n")
        monkeypatch.setattr("monitoring.health_check.time.time", lambda: old + 49 * 3600)
        os_utime = __import__("os").utime
        os_utime(codex, (old, old))
        os_utime(hourly, (old, old))

        result = check_scheduled_audit_status()

        assert result.status == "fail"
        assert "stale" in result.message

    def test_hourly_current_status_uses_shorter_stale_window(self, tmp_path, monkeypatch):
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        now = datetime.fromisoformat("2026-04-27T17:00:00+00:00").timestamp()
        fresh_enough_for_codex = datetime.fromtimestamp(
            now - 4 * 3600,
            tz=timezone.utc,
        ).isoformat()
        codex_fields = self._write_codex_artifacts(
            tmp_path,
            self._codex_stem_from_timestamp(fresh_enough_for_codex),
        )
        hourly_stem = self._hourly_stem_from_timestamp(fresh_enough_for_codex)
        self._write_hourly_artifacts(tmp_path, hourly_stem)
        (tmp_path / "codex_current.log").write_text(
            f"{fresh_enough_for_codex} status=OK rc=0 {codex_fields}\n",
        )
        (tmp_path / "hourly_current.log").write_text(
            f"{fresh_enough_for_codex} status=OK rc=0 "
            f"log={hourly_stem}.log "
            f"raw_log={hourly_stem}.raw.jsonl\n",
        )
        monkeypatch.setattr("monitoring.health_check.time.time", lambda: now)

        result = check_scheduled_audit_status()

        assert result.status == "fail"
        assert "hourly" in result.message
        assert "codex" not in result.message
        assert result.details["hourly"]["stale_after_s"] == str(3 * 3600)
        assert result.details["codex"]["stale_after_s"] == str(48 * 3600)

    def test_hourly_staleness_is_suppressed_outside_market_hours(self, tmp_path, monkeypatch):
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        now = datetime.fromisoformat("2026-04-25T17:00:00+00:00").timestamp()
        stale_for_hourly = datetime.fromtimestamp(
            now - 10 * 3600,
            tz=timezone.utc,
        ).isoformat()
        codex_fields = self._write_codex_artifacts(
            tmp_path,
            self._codex_stem_from_timestamp(stale_for_hourly),
        )
        hourly_fields = self._write_hourly_artifacts(
            tmp_path,
            self._hourly_stem_from_timestamp(stale_for_hourly),
        )
        (tmp_path / "codex_current.log").write_text(
            f"{stale_for_hourly} status=OK rc=0 {codex_fields}\n",
        )
        (tmp_path / "hourly_current.log").write_text(
            f"{stale_for_hourly} status=OK rc=0 {hourly_fields}\n",
        )
        monkeypatch.setattr("monitoring.health_check.time.time", lambda: now)

        result = check_scheduled_audit_status()

        assert result.status == "ok"
        assert result.details["hourly"]["stale_after_s"] == "off_hours"

    def test_hourly_staleness_still_checked_at_2200_monitor_run(self, tmp_path, monkeypatch):
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        now = datetime.fromisoformat("2026-04-27T22:00:00+00:00").timestamp()
        stale_hourly = datetime.fromtimestamp(
            now - 4 * 3600,
            tz=timezone.utc,
        ).isoformat()
        codex_fields = self._write_codex_artifacts(
            tmp_path,
            self._codex_stem_from_timestamp(stale_hourly),
        )
        hourly_stem = self._hourly_stem_from_timestamp(stale_hourly)
        self._write_hourly_artifacts(tmp_path, hourly_stem)
        (tmp_path / "codex_current.log").write_text(
            f"{stale_hourly} status=OK rc=0 {codex_fields}\n",
        )
        (tmp_path / "hourly_current.log").write_text(
            f"{stale_hourly} status=OK rc=0 "
            f"log={hourly_stem}.log "
            f"raw_log={hourly_stem}.raw.jsonl\n",
        )
        monkeypatch.setattr("monitoring.health_check.time.time", lambda: now)

        result = check_scheduled_audit_status()

        assert result.status == "fail"
        assert "hourly" in result.message
        assert result.details["hourly"]["stale_after_s"] == str(3 * 3600)

    def test_embedded_timestamp_controls_staleness_over_mtime(self, tmp_path, monkeypatch):
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        now = 1_700_000_000 + 72 * 3600
        stale_ts = datetime.fromtimestamp(
            now - 72 * 3600,
            tz=timezone.utc,
        ).isoformat()
        fresh_mtime = now
        codex = tmp_path / "codex_current.log"
        hourly = tmp_path / "hourly_current.log"
        codex.write_text(f"{stale_ts} status=OK rc=0 log=a raw_log=b\n")
        hourly.write_text(f"{stale_ts} status=OK rc=0 log=c raw_log=d\n")
        monkeypatch.setattr("monitoring.health_check.time.time", lambda: now)
        os_utime = __import__("os").utime
        os_utime(codex, (fresh_mtime, fresh_mtime))
        os_utime(hourly, (fresh_mtime, fresh_mtime))

        result = check_scheduled_audit_status()

        assert result.status == "fail"
        assert "stale" in result.message
        assert result.details["codex"]["age_source"] == "line_timestamp"

    def test_invalid_embedded_timestamp_is_fail(self, tmp_path, monkeypatch):
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        now = 1_700_000_000
        codex = tmp_path / "codex_current.log"
        hourly = tmp_path / "hourly_current.log"
        codex.write_text("not-a-time status=OK rc=0 log=a raw_log=b\n")
        hourly.write_text("not-a-time status=OK rc=0 log=c raw_log=d\n")
        monkeypatch.setattr("monitoring.health_check.time.time", lambda: now)
        os_utime = __import__("os").utime
        os_utime(codex, (now, now))
        os_utime(hourly, (now, now))

        result = check_scheduled_audit_status()

        assert result.status == "fail"
        assert "invalid current status timestamp" in result.message

    def test_future_dated_current_status_is_fail(self, tmp_path, monkeypatch):
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        now = 1_700_000_000
        future_ts = datetime.fromtimestamp(
            now + 3600,
            tz=timezone.utc,
        ).isoformat()
        (tmp_path / "codex_current.log").write_text(
            f"{future_ts} status=OK rc=0 log=a raw_log=b\n",
        )
        (tmp_path / "hourly_current.log").write_text(
            f"{future_ts} status=OK rc=0 log=c raw_log=d\n",
        )
        monkeypatch.setattr("monitoring.health_check.time.time", lambda: now)

        result = check_scheduled_audit_status()

        assert result.status == "fail"
        assert "future-dated" in result.message

    def test_missing_current_status_is_warn(self, tmp_path, monkeypatch):
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)

        result = check_scheduled_audit_status()

        assert result.status == "warn"
        assert "missing" in result.message


class TestMonitorAgentStatus:
    def _sha256(self, path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _write_monitor_artifact(self, tmp_path, stem="monitor_20260426T000000"):
        (tmp_path / f"{stem}.log").write_text("monitor output\n", encoding="utf-8")
        return f"log={stem}.log log_sha256={self._sha256(tmp_path / f'{stem}.log')}"

    def _monitor_stem_from_timestamp(self, ts: str):
        stamp = datetime.fromisoformat(ts).strftime("%Y%m%dT%H%M%S")
        return f"monitor_{stamp}"

    def test_skip_env_avoids_self_referential_monitor_failure(self, monkeypatch):
        monkeypatch.setenv("HEALTH_CHECK_SKIP_MONITOR_CURRENT", "1")

        result = check_monitor_agent_status()

        assert result.status == "ok"
        assert "skipped" in result.message

    def test_clean_monitor_current_status_is_ok(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HEALTH_CHECK_SKIP_MONITOR_CURRENT", raising=False)
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        monkeypatch.setattr(
            "monitoring.health_check.time.time",
            lambda: datetime.fromisoformat("2026-04-26T01:00:00+00:00").timestamp(),
        )
        fields = self._write_monitor_artifact(tmp_path)
        (tmp_path / "monitor_current.log").write_text(
            f"2026-04-26T00:00:00+00:00 status=RECOVERED rc=0 "
            f"initial_rc=1 final_rc=0 agent_rc=0 {fields}\n",
            encoding="utf-8",
        )

        result = check_monitor_agent_status()

        assert result.status == "ok"
        assert "RECOVERED" in result.message
        assert result.details["agent_rc"] == "0"

    def test_recovered_monitor_with_failed_agent_is_warn(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HEALTH_CHECK_SKIP_MONITOR_CURRENT", raising=False)
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        monkeypatch.setattr(
            "monitoring.health_check.time.time",
            lambda: datetime.fromisoformat("2026-04-26T01:00:00+00:00").timestamp(),
        )
        fields = self._write_monitor_artifact(tmp_path)
        (tmp_path / "monitor_current.log").write_text(
            f"2026-04-26T00:00:00+00:00 status=RECOVERED rc=0 "
            f"initial_rc=1 final_rc=0 agent_rc=124 {fields}\n",
            encoding="utf-8",
        )

        result = check_monitor_agent_status()

        assert result.status == "warn"
        assert "agent exited rc=124" in result.message

    def test_missing_monitor_current_status_is_warn(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HEALTH_CHECK_SKIP_MONITOR_CURRENT", raising=False)
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)

        result = check_monitor_agent_status()

        assert result.status == "warn"
        assert "missing monitor current status" in result.message

    def test_still_unhealthy_monitor_current_status_is_fail(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HEALTH_CHECK_SKIP_MONITOR_CURRENT", raising=False)
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        fields = self._write_monitor_artifact(tmp_path)
        (tmp_path / "monitor_current.log").write_text(
            f"2026-04-26T00:00:00+00:00 status=STILL_UNHEALTHY rc=1 "
            f"initial_rc=1 final_rc=1 agent_rc=124 {fields}\n",
            encoding="utf-8",
        )

        result = check_monitor_agent_status()

        assert result.status == "fail"
        assert "STILL_UNHEALTHY" in result.message

    def test_setup_failed_monitor_current_status_is_fail(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HEALTH_CHECK_SKIP_MONITOR_CURRENT", raising=False)
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        fields = self._write_monitor_artifact(tmp_path)
        (tmp_path / "monitor_current.log").write_text(
            f"2026-04-26T00:00:00+00:00 status=SETUP_FAILED rc=2 "
            f"initial_rc=NA final_rc=2 agent_rc=NA {fields}\n",
            encoding="utf-8",
        )

        result = check_monitor_agent_status()

        assert result.status == "fail"
        assert "SETUP_FAILED" in result.message

    def test_monitor_current_status_requires_contract_fields(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HEALTH_CHECK_SKIP_MONITOR_CURRENT", raising=False)
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        fields = self._write_monitor_artifact(tmp_path)
        (tmp_path / "monitor_current.log").write_text(
            f"2026-04-26T00:00:00+00:00 status=OK rc=0 {fields}\n",
            encoding="utf-8",
        )

        result = check_monitor_agent_status()

        assert result.status == "fail"
        assert "missing field(s): initial_rc, final_rc, agent_rc" in result.message

    def test_monitor_current_status_requires_rc_final_rc_match(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HEALTH_CHECK_SKIP_MONITOR_CURRENT", raising=False)
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        fields = self._write_monitor_artifact(tmp_path)
        (tmp_path / "monitor_current.log").write_text(
            f"2026-04-26T00:00:00+00:00 status=OK rc=0 "
            f"initial_rc=0 final_rc=1 agent_rc=NA {fields}\n",
            encoding="utf-8",
        )

        result = check_monitor_agent_status()

        assert result.status == "fail"
        assert "rc/final_rc mismatch" in result.message

    def test_monitor_current_status_rejects_impossible_ok_agent_rc(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HEALTH_CHECK_SKIP_MONITOR_CURRENT", raising=False)
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        fields = self._write_monitor_artifact(tmp_path)
        (tmp_path / "monitor_current.log").write_text(
            f"2026-04-26T00:00:00+00:00 status=OK rc=0 "
            f"initial_rc=0 final_rc=0 agent_rc=0 {fields}\n",
            encoding="utf-8",
        )

        result = check_monitor_agent_status()

        assert result.status == "fail"
        assert "OK status must have" in result.message

    def test_monitor_current_status_rejects_recovered_without_agent_rc(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HEALTH_CHECK_SKIP_MONITOR_CURRENT", raising=False)
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        fields = self._write_monitor_artifact(tmp_path)
        (tmp_path / "monitor_current.log").write_text(
            f"2026-04-26T00:00:00+00:00 status=RECOVERED rc=0 "
            f"initial_rc=1 final_rc=0 agent_rc=NA {fields}\n",
            encoding="utf-8",
        )

        result = check_monitor_agent_status()

        assert result.status == "fail"
        assert "RECOVERED status must have" in result.message

    def test_monitor_current_status_requires_log_artifact(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HEALTH_CHECK_SKIP_MONITOR_CURRENT", raising=False)
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        monkeypatch.setattr(
            "monitoring.health_check.time.time",
            lambda: datetime.fromisoformat("2026-04-26T01:00:00+00:00").timestamp(),
        )
        (tmp_path / "monitor_current.log").write_text(
            "2026-04-26T00:00:00+00:00 status=OK rc=0 "
            "initial_rc=0 final_rc=0 agent_rc=NA log=wrong.log\n",
            encoding="utf-8",
        )

        result = check_monitor_agent_status()

        assert result.status == "fail"
        assert "invalid audit artifact" in result.message
        assert "log:bad_name" in result.message

    def test_monitor_current_status_requires_log_hash(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HEALTH_CHECK_SKIP_MONITOR_CURRENT", raising=False)
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        monkeypatch.setattr(
            "monitoring.health_check.time.time",
            lambda: datetime.fromisoformat("2026-04-26T01:00:00+00:00").timestamp(),
        )
        (tmp_path / "monitor_20260426T000000.log").write_text("monitor output\n")
        (tmp_path / "monitor_current.log").write_text(
            "2026-04-26T00:00:00+00:00 status=OK rc=0 "
            "initial_rc=0 final_rc=0 agent_rc=NA log=monitor_20260426T000000.log\n",
            encoding="utf-8",
        )

        result = check_monitor_agent_status()

        assert result.status == "fail"
        assert "log_sha256:missing" in result.message

    def test_monitor_current_status_rejects_log_hash_mismatch(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HEALTH_CHECK_SKIP_MONITOR_CURRENT", raising=False)
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        monkeypatch.setattr(
            "monitoring.health_check.time.time",
            lambda: datetime.fromisoformat("2026-04-26T01:00:00+00:00").timestamp(),
        )
        fields = self._write_monitor_artifact(tmp_path)
        (tmp_path / "monitor_20260426T000000.log").write_text("changed\n")
        (tmp_path / "monitor_current.log").write_text(
            f"2026-04-26T00:00:00+00:00 status=OK rc=0 "
            f"initial_rc=0 final_rc=0 agent_rc=NA {fields}\n",
            encoding="utf-8",
        )

        result = check_monitor_agent_status()

        assert result.status == "fail"
        assert "log_sha256:mismatch" in result.message

    def test_monitor_current_status_rejects_non_wrapper_log_stem_shape(
        self,
        tmp_path,
        monkeypatch,
    ):
        monkeypatch.delenv("HEALTH_CHECK_SKIP_MONITOR_CURRENT", raising=False)
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        monkeypatch.setattr(
            "monitoring.health_check.time.time",
            lambda: datetime.fromisoformat("2026-04-26T01:00:00+00:00").timestamp(),
        )
        fields = self._write_monitor_artifact(tmp_path, "monitor_manual_20260426T000000")
        (tmp_path / "monitor_current.log").write_text(
            f"2026-04-26T00:00:00+00:00 status=OK rc=0 "
            f"initial_rc=0 final_rc=0 agent_rc=NA {fields}\n",
            encoding="utf-8",
        )

        result = check_monitor_agent_status()

        assert result.status == "fail"
        assert "invalid audit artifact" in result.message
        assert "log:bad_timestamp" in result.message

    def test_monitor_current_status_rejects_stale_reused_log_artifact(
        self,
        tmp_path,
        monkeypatch,
    ):
        monkeypatch.delenv("HEALTH_CHECK_SKIP_MONITOR_CURRENT", raising=False)
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        monkeypatch.setattr(
            "monitoring.health_check.time.time",
            lambda: datetime.fromisoformat("2026-04-27T17:00:00+00:00").timestamp(),
        )
        fields = self._write_monitor_artifact(tmp_path, "monitor_20260426T000000")
        (tmp_path / "monitor_current.log").write_text(
            f"2026-04-27T16:30:00+00:00 status=OK rc=0 "
            f"initial_rc=0 final_rc=0 agent_rc=NA {fields}\n",
            encoding="utf-8",
        )

        result = check_monitor_agent_status()

        assert result.status == "fail"
        assert "stale_artifact_timestamp" in result.message

    def test_stale_monitor_current_status_fails_during_market_window(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HEALTH_CHECK_SKIP_MONITOR_CURRENT", raising=False)
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        now = datetime.fromisoformat("2026-04-27T17:00:00+00:00").timestamp()
        stale_ts = datetime.fromtimestamp(now - 9 * 3600, tz=timezone.utc).isoformat()
        fields = self._write_monitor_artifact(tmp_path)
        (tmp_path / "monitor_current.log").write_text(
            f"{stale_ts} status=OK rc=0 initial_rc=0 final_rc=0 agent_rc=NA {fields}\n",
            encoding="utf-8",
        )
        monkeypatch.setattr("monitoring.health_check.time.time", lambda: now)

        result = check_monitor_agent_status()

        assert result.status == "fail"
        assert "stale monitor current status" in result.message

    def test_monitor_current_staleness_suppressed_off_hours(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HEALTH_CHECK_SKIP_MONITOR_CURRENT", raising=False)
        monkeypatch.setattr("monitoring.health_check.LOG_DIR", tmp_path)
        now = datetime.fromisoformat("2026-04-25T17:00:00+00:00").timestamp()
        stale_ts = datetime.fromtimestamp(now - 12 * 3600, tz=timezone.utc).isoformat()
        fields = self._write_monitor_artifact(
            tmp_path,
            self._monitor_stem_from_timestamp(stale_ts),
        )
        (tmp_path / "monitor_current.log").write_text(
            f"{stale_ts} status=OK rc=0 initial_rc=0 final_rc=0 agent_rc=NA {fields}\n",
            encoding="utf-8",
        )
        monkeypatch.setattr("monitoring.health_check.time.time", lambda: now)

        result = check_monitor_agent_status()

        assert result.status == "ok"
        assert result.details["stale_after_s"] == "off_hours"


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

    def test_enables_and_starts_inactive_monitor_timer(self, monkeypatch):
        commands = []

        def fake_run_cmd(cmd: str, timeout: int = 15):
            commands.append(cmd)
            return 0, "", ""

        monkeypatch.setattr("monitoring.health_check.run_cmd", fake_run_cmd)

        actions = auto_fix(
            [
                CheckResult(
                    "alpaca-monitor-timer",
                    "fail",
                    "timer inactive",
                    {"active": "inactive", "enabled": "disabled"},
                ),
            ],
        )

        assert commands == [
            "sudo systemctl enable alpaca-monitor.timer",
            "sudo systemctl start alpaca-monitor.timer",
        ]
        assert actions == [
            "Enabled alpaca-monitor.timer",
            "Started alpaca-monitor.timer",
        ]

    def test_does_not_reset_failed_monitor_service_automatically(self, monkeypatch):
        commands = []

        def fake_run_cmd(cmd: str, timeout: int = 15):
            commands.append(cmd)
            return 0, "", ""

        monkeypatch.setattr("monitoring.health_check.run_cmd", fake_run_cmd)

        actions = auto_fix(
            [
                CheckResult(
                    "alpaca-monitor-timer",
                    "fail",
                    "service failed",
                    {
                        "active": "active",
                        "enabled": "enabled",
                        "service_failed_state": "failed",
                    },
                ),
            ],
        )

        assert commands == []
        assert actions == [
            "Skipped alpaca-monitor.timer auto-fix: alpaca-monitor.service is failed",
        ]

    def test_does_not_schedule_more_runs_when_monitor_service_failed(self, monkeypatch):
        commands = []

        def fake_run_cmd(cmd: str, timeout: int = 15):
            commands.append(cmd)
            return 0, "", ""

        monkeypatch.setattr("monitoring.health_check.run_cmd", fake_run_cmd)

        actions = auto_fix(
            [
                CheckResult(
                    "alpaca-monitor-timer",
                    "fail",
                    "timer inactive and service failed",
                    {
                        "active": "inactive",
                        "enabled": "disabled",
                        "service_failed_state": "failed",
                    },
                ),
            ],
        )

        assert commands == []
        assert actions == [
            "Skipped alpaca-monitor.timer auto-fix: alpaca-monitor.service is failed",
        ]

    def test_skips_monitor_timer_fix_with_incomplete_check_details(self, monkeypatch):
        commands = []

        def fake_run_cmd(cmd: str, timeout: int = 15):
            commands.append(cmd)
            return 0, "", ""

        monkeypatch.setattr("monitoring.health_check.run_cmd", fake_run_cmd)

        actions = auto_fix(
            [
                CheckResult(
                    "alpaca-monitor-timer",
                    "fail",
                    "malformed timer details",
                    {"active": "inactive"},
                ),
            ],
        )

        assert commands == []
        assert actions == [
            "Skipped alpaca-monitor.timer auto-fix: incomplete timer check details",
        ]


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
