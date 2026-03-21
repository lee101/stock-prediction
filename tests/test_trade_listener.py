"""Tests for trade_execution_listener.py — production audit suite.

Covers:
- Stale signal rejection
- NYT exclusion
- Dry-run mode (no actual processing)
- Error handling (bad JSON, bad payload, process_event failure)
- --check-config validation
- Structured JSON log output
- _run_stdin and _run_file mode paths
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import helpers — patch env_real before importing listener so it doesn't
# attempt real network connections.
# ---------------------------------------------------------------------------

import types

# Provide a minimal env_real stub so the import works without real secrets.
_env_real_stub = types.ModuleType("env_real")
_env_real_stub.ALP_KEY_ID_PROD = "TESTKEY123456"
_env_real_stub.ALP_SECRET_KEY_PROD = "TESTSECRET123456"
sys.modules.setdefault("env_real", _env_real_stub)


# ---------------------------------------------------------------------------
# Helper: collect structured log records emitted by _emit() via caplog
# ---------------------------------------------------------------------------

def _log_events(caplog_records) -> list[dict]:
    """Parse JSON log records captured via caplog into dicts."""
    events = []
    for record in caplog_records:
        msg = record.getMessage()
        try:
            parsed = json.loads(msg)
            events.append(parsed)
        except (json.JSONDecodeError, ValueError):
            pass
    return events


def _log_event_names(caplog_records) -> list[str]:
    return [e.get("event", "") for e in _log_events(caplog_records)]

import trade_execution_listener as tel  # noqa: E402  (must come after stub)
from trade_execution_listener import (  # noqa: E402
    EXCLUDED_SYMBOLS,
    SIGNAL_MAX_AGE_SECONDS,
    _check_config,
    _is_excluded,
    _is_stale,
    _parse_args,
    _process_event_with_logging,
    _run_file,
    _run_stdin,
    _should_process,
)
from src.trade_execution_monitor import TradeEvent, TradeExecutionMonitor  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_listener(tmp_path, monkeypatch):
    """TradeExecutionMonitor with state redirected to tmp_path."""
    def _fake_state_file(name: str, suffix: str | None = None, extension: str = ".json"):
        suffix_part = suffix or ""
        return tmp_path / f"{name}{suffix_part}{extension}"

    monkeypatch.setattr("src.trade_execution_monitor.get_state_file", _fake_state_file)
    return TradeExecutionMonitor(state_suffix="_test")


def _fresh_event(symbol="AAPL", side="buy", qty=1.0, price=100.0, age_seconds=0) -> TradeEvent:
    ts = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    return TradeEvent(symbol=symbol, side=side, quantity=qty, price=price, timestamp=ts)


# ---------------------------------------------------------------------------
# 1. Staleness check
# ---------------------------------------------------------------------------


class TestStalenessCheck:
    def test_fresh_signal_not_stale(self):
        event = _fresh_event(age_seconds=0)
        assert not _is_stale(event, max_age_seconds=300)

    def test_signal_at_boundary_not_stale(self):
        # Exactly at max age — should NOT be stale (strictly greater than threshold)
        event = _fresh_event(age_seconds=300)
        # Allow a small tolerance for clock jitter; use 305 s threshold
        assert not _is_stale(event, max_age_seconds=305)

    def test_signal_just_over_limit_is_stale(self):
        event = _fresh_event(age_seconds=310)
        assert _is_stale(event, max_age_seconds=300)

    def test_signal_5_min_old_is_stale(self):
        event = _fresh_event(age_seconds=301)
        assert _is_stale(event, max_age_seconds=300)

    def test_signal_1_hour_old_is_stale(self):
        event = _fresh_event(age_seconds=3600)
        assert _is_stale(event, max_age_seconds=300)

    def test_naive_timestamp_treated_as_utc(self):
        """A naive timestamp should be handled without raising."""
        naive_ts = datetime.utcnow()
        event = TradeEvent(symbol="AAPL", side="buy", quantity=1.0, price=100.0, timestamp=naive_ts)
        # Should not raise; stale or not depending on clock skew
        result = _is_stale(event, max_age_seconds=300)
        assert isinstance(result, bool)

    def test_stale_signal_rejected_by_should_process(self):
        event = _fresh_event(age_seconds=400)
        assert not _should_process(event)

    def test_fresh_signal_accepted_by_should_process(self):
        event = _fresh_event(age_seconds=0)
        assert _should_process(event)

    def test_stale_signal_not_processed(self, mock_listener):
        event = _fresh_event(age_seconds=400)
        processed = []
        mock_listener.process_event = lambda e: processed.append(e) or []
        _process_event_with_logging(mock_listener, event, dry_run=False)
        assert not processed, "Stale signal must not reach process_event"

    def test_fresh_signal_is_processed(self, mock_listener):
        event = _fresh_event(age_seconds=0)
        processed = []
        mock_listener.process_event = lambda e: processed.append(e) or []
        _process_event_with_logging(mock_listener, event, dry_run=False)
        assert processed, "Fresh signal must reach process_event"


# ---------------------------------------------------------------------------
# 2. NYT exclusion
# ---------------------------------------------------------------------------


class TestNYTExclusion:
    def test_nyt_in_excluded_symbols(self):
        assert "NYT" in EXCLUDED_SYMBOLS

    def test_nyt_is_excluded(self):
        event = _fresh_event(symbol="NYT")
        assert _is_excluded(event)

    def test_nyt_rejected_by_should_process(self):
        event = _fresh_event(symbol="NYT")
        assert not _should_process(event)

    def test_nyt_not_processed(self, mock_listener):
        event = _fresh_event(symbol="NYT", age_seconds=0)
        processed = []
        mock_listener.process_event = lambda e: processed.append(e) or []
        _process_event_with_logging(mock_listener, event, dry_run=False)
        assert not processed, "NYT must never reach process_event"

    def test_nyt_lowercase_rejected(self):
        ts = datetime.now(timezone.utc)
        event = TradeEvent(symbol="nyt", side="sell", quantity=1.0, price=50.0, timestamp=ts)
        # _is_excluded uses .upper() internally
        assert _is_excluded(event)

    def test_non_excluded_symbol_passes(self):
        event = _fresh_event(symbol="AAPL")
        assert not _is_excluded(event)

    def test_aapl_processed_when_fresh(self, mock_listener):
        event = _fresh_event(symbol="AAPL", age_seconds=0)
        processed = []
        mock_listener.process_event = lambda e: processed.append(e) or []
        _process_event_with_logging(mock_listener, event, dry_run=False)
        assert processed, "AAPL should be processed"


# ---------------------------------------------------------------------------
# 3. Dry-run mode
# ---------------------------------------------------------------------------


class TestDryRunMode:
    def test_dry_run_does_not_call_process_event(self, mock_listener):
        event = _fresh_event(age_seconds=0)
        called = []
        mock_listener.process_event = lambda e: called.append(e) or []
        _process_event_with_logging(mock_listener, event, dry_run=True)
        assert not called, "process_event must not be called in dry-run mode"

    def test_dry_run_still_logs_signal(self, mock_listener, caplog):
        event = _fresh_event(age_seconds=0)
        with caplog.at_level(logging.DEBUG, logger="trade_execution_listener"):
            _process_event_with_logging(mock_listener, event, dry_run=True)
        names = _log_event_names(caplog.records)
        assert "dry_run_would_process" in names or "signal_received" in names

    def test_stdin_dry_run_no_processing(self, mock_listener):
        event_line = json.dumps({
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 1.0,
            "price": 150.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        called = []
        mock_listener.process_event = lambda e: called.append(e) or []
        with patch("sys.stdin", StringIO(event_line + "\n")):
            _run_stdin(mock_listener, dry_run=True, heartbeat_interval=0)
        assert not called, "Dry-run stdin must not process events"

    def test_file_dry_run_no_processing(self, tmp_path, mock_listener):
        events_file = tmp_path / "events.jsonl"
        events_file.write_text(
            json.dumps({
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 1.0,
                "price": 150.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }) + "\n"
        )
        called = []
        mock_listener.process_event = lambda e: called.append(e) or []
        _run_file(mock_listener, events_file, dry_run=True)
        assert not called, "Dry-run file mode must not process events"


# ---------------------------------------------------------------------------
# 4. Error handling — bad JSON, bad payload, process_event failure
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_bad_json_in_stdin_does_not_crash(self, mock_listener, caplog):
        """A malformed JSON line should log an error and continue."""
        bad_line = "NOT_VALID_JSON\n"
        good_event = json.dumps({
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 1.0,
            "price": 150.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        stdin_content = bad_line + good_event + "\n"
        processed = []
        mock_listener.process_event = lambda e: processed.append(e) or []
        with caplog.at_level(logging.DEBUG, logger="trade_execution_listener"):
            with patch("sys.stdin", StringIO(stdin_content)):
                _run_stdin(mock_listener, dry_run=False, heartbeat_interval=0)
        names = _log_event_names(caplog.records)
        assert "json_parse_failed" in names, "Bad JSON must be logged"
        assert processed, "Good event after bad JSON must still be processed"

    def test_process_event_exception_logged_not_raised(self, mock_listener, caplog):
        """If process_event raises, the error must be logged, not propagated."""
        event = _fresh_event(age_seconds=0)
        mock_listener.process_event = MagicMock(side_effect=RuntimeError("boom"))
        with caplog.at_level(logging.DEBUG, logger="trade_execution_listener"):
            # Must not raise
            _process_event_with_logging(mock_listener, event, dry_run=False)
        names = _log_event_names(caplog.records)
        assert "process_event_failed" in names, "Exception must be logged as structured JSON"
        all_text = " ".join(r.getMessage() for r in caplog.records)
        assert "boom" in all_text

    def test_process_event_api_error_logged(self, mock_listener, caplog):
        """Simulate an Alpaca API error during processing."""
        event = _fresh_event(age_seconds=0)
        mock_listener.process_event = MagicMock(side_effect=ConnectionError("network error"))
        with caplog.at_level(logging.DEBUG, logger="trade_execution_listener"):
            _process_event_with_logging(mock_listener, event, dry_run=False)
        names = _log_event_names(caplog.records)
        assert "process_event_failed" in names
        all_text = " ".join(r.getMessage() for r in caplog.records)
        assert "network error" in all_text

    def test_bad_event_parse_in_stdin_continues(self, mock_listener, capsys):
        """A line that parses as JSON but is an invalid event should log an error."""
        bad_payload = json.dumps({"symbol": None, "side": "buy"})
        stdin_content = bad_payload + "\n"
        # Even if trade_event_from_dict doesn't raise, the flow must not crash
        with patch("sys.stdin", StringIO(stdin_content)):
            # Should not raise
            _run_stdin(mock_listener, dry_run=False, heartbeat_interval=0)


# ---------------------------------------------------------------------------
# 5. --check-config
# ---------------------------------------------------------------------------


class TestCheckConfig:
    def test_check_config_exits_0_with_valid_keys(self, monkeypatch):
        """Valid non-placeholder API keys and NYT in exclusions → exit 0."""
        monkeypatch.setattr(tel, "ALP_KEY_ID_PROD", "VALIDKEY1234567890")
        monkeypatch.setattr(tel, "ALP_SECRET_KEY_PROD", "VALIDSECRET1234567890")
        with pytest.raises(SystemExit) as exc_info:
            _check_config()
        assert exc_info.value.code == 0

    def test_check_config_exits_1_with_missing_key(self, monkeypatch):
        """Placeholder key → exit 1."""
        monkeypatch.setattr(tel, "ALP_KEY_ID_PROD", "AKAONQRN6CZJFGTHN3DWDFPIBA")
        monkeypatch.setattr(tel, "ALP_SECRET_KEY_PROD", "VALIDSECRET1234567890")
        with pytest.raises(SystemExit) as exc_info:
            _check_config()
        assert exc_info.value.code == 1

    def test_check_config_exits_1_with_missing_secret(self, monkeypatch):
        """Placeholder secret → exit 1."""
        monkeypatch.setattr(tel, "ALP_KEY_ID_PROD", "VALIDKEY1234567890")
        monkeypatch.setattr(tel, "ALP_SECRET_KEY_PROD", "GYwPufjn8TrKNHV4jwWoMs7cwmDPiP4U1Xsu8UHzXDz4")
        with pytest.raises(SystemExit) as exc_info:
            _check_config()
        assert exc_info.value.code == 1

    def test_check_config_exits_1_if_nyt_not_excluded(self, monkeypatch):
        """If NYT is somehow removed from EXCLUDED_SYMBOLS, exit 1."""
        monkeypatch.setattr(tel, "ALP_KEY_ID_PROD", "VALIDKEY1234567890")
        monkeypatch.setattr(tel, "ALP_SECRET_KEY_PROD", "VALIDSECRET1234567890")
        monkeypatch.setattr(tel, "EXCLUDED_SYMBOLS", frozenset())
        with pytest.raises(SystemExit) as exc_info:
            _check_config()
        assert exc_info.value.code == 1

    def test_main_check_config_flag_invokes_validation(self, monkeypatch):
        """--check-config CLI flag must trigger config validation."""
        monkeypatch.setattr(tel, "ALP_KEY_ID_PROD", "VALIDKEY1234567890")
        monkeypatch.setattr(tel, "ALP_SECRET_KEY_PROD", "VALIDSECRET1234567890")
        with pytest.raises(SystemExit) as exc_info:
            tel.main(["--check-config"])
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# 6. Structured JSON logging
# ---------------------------------------------------------------------------


class TestStructuredLogging:
    def test_signal_received_logged_as_json(self, mock_listener, caplog):
        event = _fresh_event(age_seconds=0)
        with caplog.at_level(logging.DEBUG, logger="trade_execution_listener"):
            _process_event_with_logging(mock_listener, event, dry_run=True)
        # All log messages should be valid JSON with required keys
        for record in caplog.records:
            msg = record.getMessage()
            parsed = json.loads(msg)
            assert "event" in parsed
            assert "ts" in parsed

    def test_stale_rejection_logged_as_json(self, mock_listener, caplog):
        event = _fresh_event(age_seconds=400)
        with caplog.at_level(logging.DEBUG, logger="trade_execution_listener"):
            _process_event_with_logging(mock_listener, event, dry_run=False)
        names = _log_event_names(caplog.records)
        assert "signal_rejected_stale" in names

    def test_exclusion_rejection_logged_as_json(self, mock_listener, caplog):
        event = _fresh_event(symbol="NYT", age_seconds=0)
        with caplog.at_level(logging.DEBUG, logger="trade_execution_listener"):
            _process_event_with_logging(mock_listener, event, dry_run=False)
        names = _log_event_names(caplog.records)
        assert "signal_excluded" in names


# ---------------------------------------------------------------------------
# 7. Argument parsing
# ---------------------------------------------------------------------------


class TestArgParsing:
    def test_default_mode_is_stdin(self):
        args = _parse_args([])
        assert args.mode == "stdin"

    def test_dry_run_flag(self):
        args = _parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_check_config_flag(self):
        args = _parse_args(["--check-config"])
        assert args.check_config is True

    def test_signal_max_age_override(self):
        args = _parse_args(["--signal-max-age", "120"])
        assert args.signal_max_age == 120

    def test_paper_flag(self):
        args = _parse_args(["--mode", "alpaca", "--paper"])
        assert args.paper is True

    def test_events_file_parsed(self, tmp_path):
        p = tmp_path / "events.jsonl"
        p.write_text("")
        args = _parse_args(["--mode", "file", "--events-file", str(p)])
        assert args.events_file == p


# ---------------------------------------------------------------------------
# 8. _run_file mode
# ---------------------------------------------------------------------------


class TestRunFile:
    def test_run_file_processes_valid_events(self, tmp_path, mock_listener):
        events_file = tmp_path / "events.jsonl"
        events_file.write_text(
            json.dumps({
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 2.0,
                "price": 150.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }) + "\n"
        )
        processed = []
        mock_listener.process_event = lambda e: processed.append(e) or []
        _run_file(mock_listener, events_file, dry_run=False)
        assert len(processed) == 1
        assert processed[0].symbol == "AAPL"

    def test_run_file_skips_nyt(self, tmp_path, mock_listener):
        events_file = tmp_path / "events.jsonl"
        events_file.write_text(
            json.dumps({
                "symbol": "NYT",
                "side": "sell",
                "quantity": 1.0,
                "price": 50.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }) + "\n"
        )
        processed = []
        mock_listener.process_event = lambda e: processed.append(e) or []
        _run_file(mock_listener, events_file, dry_run=False)
        assert not processed, "NYT must be skipped in file mode"

    def test_run_file_skips_stale(self, tmp_path, mock_listener):
        stale_ts = (datetime.now(timezone.utc) - timedelta(seconds=400)).isoformat()
        events_file = tmp_path / "events.jsonl"
        events_file.write_text(
            json.dumps({
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 1.0,
                "price": 150.0,
                "timestamp": stale_ts,
            }) + "\n"
        )
        processed = []
        mock_listener.process_event = lambda e: processed.append(e) or []
        _run_file(mock_listener, events_file, dry_run=False)
        assert not processed, "Stale event must be skipped in file mode"


# ---------------------------------------------------------------------------
# 9. _run_stdin heartbeat
# ---------------------------------------------------------------------------


class TestStdinHeartbeat:
    def test_heartbeat_emitted_when_interval_reached(self, mock_listener, caplog, monkeypatch):
        """Heartbeat should fire when monotonic time crosses the interval."""
        # Force time.monotonic to increment past heartbeat_interval
        call_count = [0]
        base = time.monotonic()

        def fake_monotonic():
            call_count[0] += 1
            # After first call (initialization), advance time by 65 seconds
            return base + (65.0 if call_count[0] > 1 else 0.0)

        monkeypatch.setattr(tel.time, "monotonic", fake_monotonic)

        event_line = json.dumps({
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 1.0,
            "price": 150.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        mock_listener.process_event = lambda e: []
        with caplog.at_level(logging.DEBUG, logger="trade_execution_listener"):
            with patch("sys.stdin", StringIO(event_line + "\n")):
                _run_stdin(mock_listener, dry_run=False, heartbeat_interval=60)
        names = _log_event_names(caplog.records)
        assert "heartbeat" in names
