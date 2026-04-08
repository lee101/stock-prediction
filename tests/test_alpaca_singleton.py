"""Tests for the alpaca_wrapper singleton + death-spiral guard.

These tests DO NOT import alpaca_wrapper (which would pull in the
network-heavy trading SDK). They import the guard module directly.

The tests:
  - paper mode: enforce_live_singleton returns None without acquiring anything
  - live mode: first call acquires lock; second call in a child process raises SystemExit(42)
  - death spiral: recording a buy and then trying to sell below floor raises
  - override env vars bypass both gates with a loud log line
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Run short python snippets in isolated subprocesses so each test gets a
# fresh state dir + fresh env.
# ---------------------------------------------------------------------------


def _run_snippet(code: str, env_extra: dict[str, str], tmp_path: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(env_extra)
    env["PYTHONPATH"] = str(REPO) + ":" + env.get("PYTHONPATH", "")
    env["UNIFIED_STATE_DIR"] = str(tmp_path / "state")
    env["XDG_STATE_HOME"] = str(tmp_path / "xdg_state")
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=env, cwd=str(REPO), text=True, capture_output=True, timeout=30,
    )


def test_paper_mode_singleton_is_noop(tmp_path):
    proc = _run_snippet(
        """
        from src.alpaca_singleton import enforce_live_singleton
        lock = enforce_live_singleton(service_name='test_paper', account_name='alpaca_test_writer')
        assert lock is None, f'expected None in paper, got {lock}'
        print('OK')
        """,
        env_extra={"ALP_PAPER": "1"},
        tmp_path=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_live_mode_singleton_acquires_and_second_fails(tmp_path):
    # First process holds the lock for the duration of a sleep; second
    # process tries to acquire the same account and must exit 42.
    holder = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent("""
            import os, time
            os.environ['ALP_PAPER'] = '0'
            from src.alpaca_singleton import enforce_live_singleton
            lock = enforce_live_singleton(service_name='holder', account_name='alpaca_test_writer')
            assert lock is not None, 'holder failed to acquire'
            print('LOCKED', flush=True)
            time.sleep(5)
            """)],
        env={
            **os.environ, "PYTHONPATH": str(REPO),
            "UNIFIED_STATE_DIR": str(tmp_path / "state"),
            "XDG_STATE_HOME": str(tmp_path / "xdg_state"),
            "ALP_PAPER": "0",
        },
        cwd=str(REPO), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        # Wait for the holder to acquire.
        for _ in range(30):
            line = holder.stdout.readline()  # type: ignore[union-attr]
            if "LOCKED" in line:
                break
        else:
            holder.kill()
            raise AssertionError("holder never announced LOCKED")

        # Now attempt a second acquire in a separate subprocess.
        proc = _run_snippet(
            """
            from src.alpaca_singleton import enforce_live_singleton
            enforce_live_singleton(service_name='second', account_name='alpaca_test_writer')
            """,
            env_extra={"ALP_PAPER": "0"},
            tmp_path=tmp_path,
        )
        assert proc.returncode == 42, (
            f"expected SystemExit(42), got {proc.returncode}\nSTDOUT={proc.stdout}\nSTDERR={proc.stderr}"
        )
        assert "REFUSING TO START" in proc.stderr
    finally:
        holder.kill()
        holder.wait(timeout=5)


def test_live_mode_override_bypasses_singleton_with_loud_log(tmp_path):
    proc = _run_snippet(
        """
        from src.alpaca_singleton import enforce_live_singleton
        lock = enforce_live_singleton(service_name='override_test', account_name='alpaca_test_writer')
        # Override returns None (no lock) but should still print.
        assert lock is None, f'expected None under override, got {lock}'
        print('BYPASSED')
        """,
        env_extra={"ALP_PAPER": "0", "ALPACA_SINGLETON_OVERRIDE": "1"},
        tmp_path=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert "BYPASSED" in proc.stdout
    assert "OVERRIDE ACTIVE" in proc.stderr  # loud audit trail


def test_death_spiral_refuses_sell_below_floor(tmp_path):
    proc = _run_snippet(
        """
        from src.alpaca_singleton import (
            enforce_live_singleton, record_buy_price,
            guard_sell_against_death_spiral, forget_all_buys,
        )
        enforce_live_singleton(service_name='ds_test', account_name='alpaca_test_writer')
        forget_all_buys()
        record_buy_price('AAPL', 200.0)
        # 0.5% tolerance → floor = 199.0. A sell at 198 must refuse.
        try:
            guard_sell_against_death_spiral('AAPL', 'sell', 198.0)
        except RuntimeError as exc:
            print('REFUSED:', exc)
        else:
            raise SystemExit('expected RuntimeError, got nothing')
        # A sell at 199.5 (inside floor) is OK.
        guard_sell_against_death_spiral('AAPL', 'sell', 199.5)
        print('ALLOWED_INSIDE')
        """,
        env_extra={"ALP_PAPER": "1"},  # paper so import is cheap but guard still works
        tmp_path=tmp_path,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "REFUSED" in proc.stdout
    assert "ALLOWED_INSIDE" in proc.stdout


def test_death_spiral_override_bypasses_with_loud_log(tmp_path):
    proc = _run_snippet(
        """
        from src.alpaca_singleton import (
            enforce_live_singleton, record_buy_price,
            guard_sell_against_death_spiral, forget_all_buys,
        )
        enforce_live_singleton(service_name='ds_override', account_name='alpaca_test_writer')
        forget_all_buys()
        record_buy_price('MSFT', 400.0)
        # Sell far below floor — override should let it through but log.
        guard_sell_against_death_spiral('MSFT', 'sell', 100.0)
        print('BYPASSED_DS')
        """,
        env_extra={
            "ALP_PAPER": "1",
            "ALPACA_DEATH_SPIRAL_OVERRIDE": "1",
        },
        tmp_path=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert "BYPASSED_DS" in proc.stdout
    assert "OVERRIDE ACTIVE" in proc.stderr


def test_death_spiral_no_record_allows_sell(tmp_path):
    proc = _run_snippet(
        """
        from src.alpaca_singleton import (
            enforce_live_singleton, guard_sell_against_death_spiral, forget_all_buys,
        )
        enforce_live_singleton(service_name='ds_no_rec', account_name='alpaca_test_writer')
        forget_all_buys()
        # No recorded buy → guard must allow (can't compare against nothing).
        guard_sell_against_death_spiral('NFLX', 'sell', 500.0)
        print('NO_RECORD_OK')
        """,
        env_extra={"ALP_PAPER": "1"},
        tmp_path=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert "NO_RECORD_OK" in proc.stdout
