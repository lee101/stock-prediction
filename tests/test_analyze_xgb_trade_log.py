"""Tests for scripts/analyze_xgb_trade_log.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "analyze_xgb_trade_log.py"


def _write_jsonl(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")


def _run(log_dir: Path, *extra: str) -> tuple[int, str, str]:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--log-dir", str(log_dir), *extra],
        capture_output=True, text=True, check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _single_session(slips: list[float], day: str = "2026-04-21") -> list[dict]:
    events = [
        {"ts": "x", "event": "session_start", "mode": "live",
         "paper": False, "equity_pre": 10_000.0},
    ]
    for i, s in enumerate(slips):
        events.append({"ts": "x", "event": "pick", "rank": i,
                       "symbol": f"SYM{i}", "score": 0.9})
        events.append({"ts": "x", "event": "buy_submitted",
                       "symbol": f"SYM{i}", "qty": 1, "expected_price": 100.0})
        events.append({
            "ts": "x", "event": "buy_filled",
            "symbol": f"SYM{i}", "fill_price": 100.0 * (1 + s / 10_000),
            "last_close": 100.0, "slippage_bps_vs_last_close": s,
        })
    events.append({"ts": "x", "event": "session_end",
                   "equity_pre": 10_000.0, "equity_post": 10_050.0,
                   "session_pnl_abs": 50.0, "session_pnl_pct": 0.5})
    return events


def test_empty_log_dir_exits_zero(tmp_path: Path):
    log_dir = tmp_path / "xgb_logs"
    log_dir.mkdir()
    rc, _out, err = _run(log_dir)
    assert rc == 0
    assert "no *.jsonl" in err


def test_missing_log_dir_exits_nonzero(tmp_path: Path):
    rc, _out, err = _run(tmp_path / "does_not_exist")
    assert rc == 2
    assert "not found" in err


def test_single_session_calibrated(tmp_path: Path):
    log_dir = tmp_path / "xgb_logs"
    _write_jsonl(log_dir / "2026-04-21.jsonl", _single_session([4.8, 5.2]))
    rc, out, err = _run(log_dir)
    assert rc == 0, err
    assert "sessions found       = 1" in out
    assert "CALIBRATED" in out
    # mean of [4.8, 5.2] = 5.00
    assert "mean     = +5.00" in out


def test_single_session_sim_undercosts(tmp_path: Path):
    log_dir = tmp_path / "xgb_logs"
    _write_jsonl(log_dir / "2026-04-21.jsonl",
                 _single_session([15.0, 20.0, 18.0]))
    rc, out, err = _run(log_dir)
    assert rc == 0, err
    assert "SIM UNDER-COSTS" in out


def test_single_session_sim_conservative(tmp_path: Path):
    log_dir = tmp_path / "xgb_logs"
    _write_jsonl(log_dir / "2026-04-21.jsonl",
                 _single_session([-1.0, 0.5, 1.0]))
    rc, out, err = _run(log_dir)
    assert rc == 0, err
    assert "SIM CONSERVATIVE" in out


def test_missing_slippage_field_is_tolerated(tmp_path: Path):
    log_dir = tmp_path / "xgb_logs"
    events = [
        {"event": "session_start", "mode": "live", "paper": False},
        {"event": "pick", "symbol": "A"},
        {"event": "buy_filled", "symbol": "A",
         "fill_price": 100.0, "last_close": 100.0},  # no slippage key
        {"event": "pick", "symbol": "B"},
        {"event": "buy_filled", "symbol": "B",
         "fill_price": 100.05, "last_close": 100.0,
         "slippage_bps_vs_last_close": 5.0},
        {"event": "session_end", "session_pnl_pct": 0.1},
    ]
    _write_jsonl(log_dir / "2026-04-21.jsonl", events)
    rc, out, err = _run(log_dir)
    assert rc == 0, err
    # Only 1 slippage sample collected from the 2 fills
    assert "mean     = +5.00" in out


def test_skipped_session_rendered(tmp_path: Path):
    log_dir = tmp_path / "xgb_logs"
    events = [
        {"event": "session_start", "mode": "live"},
        {"event": "session_skipped", "reason": "market_closed"},
    ]
    _write_jsonl(log_dir / "2026-04-21.jsonl", events)
    rc, out, err = _run(log_dir)
    assert rc == 0, err
    assert "skipped(market_closed)" in out


def test_failed_orders_counted(tmp_path: Path):
    log_dir = tmp_path / "xgb_logs"
    events = [
        {"event": "session_start", "mode": "live"},
        {"event": "pick", "symbol": "A"},
        {"event": "buy_submitted", "symbol": "A"},
        {"event": "buy_failed", "symbol": "A", "error": "alpaca 403"},
        {"event": "sell_submitted", "symbol": "B"},
        {"event": "sell_failed", "symbol": "B", "error": "alpaca 429"},
    ]
    _write_jsonl(log_dir / "2026-04-21.jsonl", events)
    rc, out, err = _run(log_dir)
    assert rc == 0, err
    assert "fail(1+1)" in out
    assert "total_failed" not in out  # human mode; only in --json


def test_json_output_is_machine_readable(tmp_path: Path):
    log_dir = tmp_path / "xgb_logs"
    _write_jsonl(log_dir / "2026-04-21.jsonl", _single_session([5.0, 6.0]))
    _write_jsonl(log_dir / "2026-04-22.jsonl", _single_session([4.0, 7.0]))
    rc, out, err = _run(log_dir, "--json")
    assert rc == 0, err
    payload = json.loads(out)
    assert "overall" in payload and "sessions" in payload
    assert payload["overall"]["n_sessions"] == 2
    assert payload["overall"]["n_slippage_samples"] == 4
    assert abs(payload["overall"]["mean_slip_bps"] - 5.5) < 1e-9
    # per-session slippages list is stripped from json output
    for s in payload["sessions"]:
        assert "slippages_bps" not in s


def test_spy_provenance_is_reported_in_json(tmp_path: Path):
    log_dir = tmp_path / "xgb_logs"
    _write_jsonl(
        log_dir / "2026-04-21.jsonl",
        [
            {
                "event": "session_start",
                "mode": "live",
                "spy_csv": "/repo/trainingdata/SPY.csv",
                "spy_csv_sha256": "a" * 64,
            },
            {
                "event": "spy_vol_target",
                "spy_csv": "/repo/trainingdata/SPY.csv",
                "spy_csv_sha256": "a" * 64,
                "scale": 0.75,
            },
        ],
    )

    rc, out, err = _run(log_dir, "--json")

    assert rc == 0, err
    payload = json.loads(out)
    assert payload["overall"]["n_spy_session_hashes"] == 1
    assert payload["overall"]["spy_session_sha256_values"] == ["a" * 64]
    assert payload["overall"]["n_spy_provenance_warning_sessions"] == 0
    session = payload["sessions"][0]
    assert session["spy_csv"] == "/repo/trainingdata/SPY.csv"
    assert session["spy_csv_sha256"] == "a" * 64
    assert session["spy_decision_event_count"] == 1
    assert session["spy_decision_missing_csv_count"] == 0
    assert session["spy_decision_missing_sha256_count"] == 0
    assert session["spy_decision_sha256_values"] == ["a" * 64]
    assert session["spy_provenance_warnings"] == []


def test_spy_provenance_mismatch_is_flagged(tmp_path: Path):
    log_dir = tmp_path / "xgb_logs"
    _write_jsonl(
        log_dir / "2026-04-21.jsonl",
        [
            {
                "event": "session_start",
                "mode": "live",
                "spy_csv": "/repo/trainingdata/SPY.csv",
                "spy_csv_sha256": "a" * 64,
            },
            {
                "event": "spy_regime_gate",
                "spy_csv": "/repo/trainingdata/SPY.csv",
                "spy_csv_sha256": "b" * 64,
                "closed": False,
            },
        ],
    )

    rc, out, err = _run(log_dir)

    assert rc == 0, err
    assert "spy_provenance_warning" in out
    assert "SPY provenance warnings = 1 session(s): 2026-04-21" in out


def test_spy_provenance_missing_decision_hash_is_flagged(tmp_path: Path):
    log_dir = tmp_path / "xgb_logs"
    _write_jsonl(
        log_dir / "2026-04-21.jsonl",
        [
            {
                "event": "session_start",
                "mode": "live",
                "spy_csv": "/repo/trainingdata/SPY.csv",
                "spy_csv_sha256": "a" * 64,
            },
            {
                "event": "spy_vol_target",
                "spy_csv": "/repo/trainingdata/SPY.csv",
                "scale": 0.75,
            },
        ],
    )

    rc, out, err = _run(log_dir, "--json")

    assert rc == 0, err
    payload = json.loads(out)
    session = payload["sessions"][0]
    assert session["spy_decision_missing_sha256_count"] == 1
    assert session["spy_provenance_warnings"] == ["spy_decision_hash_missing"]
    assert payload["overall"]["spy_provenance_warning_sessions"] == ["2026-04-21"]


def test_spy_provenance_missing_session_hash_is_flagged(tmp_path: Path):
    log_dir = tmp_path / "xgb_logs"
    _write_jsonl(
        log_dir / "2026-04-21.jsonl",
        [
            {"event": "session_start", "mode": "live"},
            {
                "event": "spy_regime_gate",
                "spy_csv": "/repo/trainingdata/SPY.csv",
                "spy_csv_sha256": "a" * 64,
                "closed": False,
            },
        ],
    )

    rc, out, err = _run(log_dir, "--json")

    assert rc == 0, err
    payload = json.loads(out)
    session = payload["sessions"][0]
    assert session["spy_provenance_warnings"] == ["spy_session_hash_missing"]


def test_fail_on_spy_provenance_warning_exits_nonzero(tmp_path: Path):
    log_dir = tmp_path / "xgb_logs"
    _write_jsonl(
        log_dir / "2026-04-21.jsonl",
        [
            {
                "event": "session_start",
                "mode": "live",
                "spy_csv": "/repo/trainingdata/SPY.csv",
                "spy_csv_sha256": "a" * 64,
            },
            {
                "event": "spy_regime_gate",
                "spy_csv": "/repo/trainingdata/SPY.csv",
                "spy_csv_sha256": "b" * 64,
                "closed": False,
            },
        ],
    )

    rc, out, err = _run(log_dir, "--fail-on-spy-provenance-warning")

    assert rc == 3
    assert err == ""
    assert "SPY provenance warnings = 1 session(s): 2026-04-21" in out


def test_fail_on_spy_provenance_warning_clean_session_exits_zero(tmp_path: Path):
    log_dir = tmp_path / "xgb_logs"
    _write_jsonl(
        log_dir / "2026-04-21.jsonl",
        [
            {
                "event": "session_start",
                "mode": "live",
                "spy_csv": "/repo/trainingdata/SPY.csv",
                "spy_csv_sha256": "a" * 64,
            },
            {
                "event": "spy_vol_target",
                "spy_csv": "/repo/trainingdata/SPY.csv",
                "spy_csv_sha256": "a" * 64,
                "scale": 0.75,
            },
        ],
    )

    rc, out, err = _run(log_dir, "--fail-on-spy-provenance-warning")

    assert rc == 0, err
    assert "SPY provenance warnings" not in out


def test_tolerates_partial_line(tmp_path: Path):
    # Simulate a process crash mid-write (partial final line).
    log_dir = tmp_path / "xgb_logs"
    events = _single_session([5.0])
    log_dir.mkdir(parents=True)
    path = log_dir / "2026-04-21.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")
        fh.write('{"event":"buy_filled","fill_price":1')  # truncated
    rc, out, err = _run(log_dir)
    assert rc == 0, err
    assert "mean     = +5.00" in out


def test_min_sessions_gate(tmp_path: Path):
    log_dir = tmp_path / "xgb_logs"
    _write_jsonl(log_dir / "2026-04-21.jsonl", _single_session([5.0]))
    rc, out, err = _run(log_dir, "--min-sessions", "5")
    assert rc == 0, err
    assert "insufficient data" in out


def test_numeric_slippage_strings_are_coerced(tmp_path: Path):
    log_dir = tmp_path / "xgb_logs"
    events = [
        {"event": "session_start"},
        {"event": "buy_filled", "symbol": "A",
         "slippage_bps_vs_last_close": "4.5"},
        {"event": "buy_filled", "symbol": "B",
         "slippage_bps_vs_last_close": "5.5"},
    ]
    _write_jsonl(log_dir / "2026-04-21.jsonl", events)
    rc, out, err = _run(log_dir)
    assert rc == 0, err
    assert "mean     = +5.00" in out
