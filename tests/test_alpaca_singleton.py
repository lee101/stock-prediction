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
import time
from pathlib import Path

from src import alpaca_singleton as singleton_mod


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


def test_buy_memory_path_rejects_path_like_account_name(tmp_path):
    proc = _run_snippet(
        """
        from src.alpaca_singleton import _buy_memory_path
        try:
            _buy_memory_path('../alpaca_test_writer')
        except ValueError as exc:
            print(exc)
        else:
            raise SystemExit('expected ValueError')
        """,
        env_extra={"ALP_PAPER": "1"},
        tmp_path=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Unsupported Alpaca account name" in proc.stdout


def test_record_buy_price_uses_explicit_default_account_name(tmp_path):
    proc = _run_snippet(
        """
        from src.alpaca_singleton import (
            DEFAULT_ALPACA_ACCOUNT_NAME,
            _buy_memory_path,
            record_buy_price,
        )
        record_buy_price('AAPL', 123.45)
        path = _buy_memory_path(DEFAULT_ALPACA_ACCOUNT_NAME)
        assert path.exists(), path
        print(path.name)
        """,
        env_extra={"ALP_PAPER": "1"},
        tmp_path=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert f"{'alpaca_live_writer'}_buys.json" in proc.stdout


def test_buy_memory_lock_is_reused_per_account_name() -> None:
    first = singleton_mod._buy_memory_lock("alpaca_test_writer")
    second = singleton_mod._buy_memory_lock("alpaca_test_writer")
    other = singleton_mod._buy_memory_lock("alpaca_other_writer")

    assert first is second
    assert first is not other


def test_buy_memory_file_lock_path_is_scoped_per_account_name(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("UNIFIED_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg_state"))

    path = singleton_mod._buy_memory_file_lock_path("alpaca_test_writer")

    assert path.name == "alpaca_test_writer_buys.lock"
    assert path.parent == singleton_mod._state_dir()


def test_save_buys_uses_unique_temp_file_before_replace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("UNIFIED_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg_state"))

    captured: dict[str, Path] = {}
    original_replace = singleton_mod.os.replace

    def _record_replace(src, dst):
        captured["src"] = Path(src)
        captured["dst"] = Path(dst)
        original_replace(src, dst)

    monkeypatch.setattr(singleton_mod.os, "replace", _record_replace)

    singleton_mod._save_buys(
        "alpaca_test_writer",
        {"AAPL": {"price": 123.45, "ts": 1.0, "iso": "2026-04-08T00:00:00+00:00"}},
    )

    final_path = singleton_mod._buy_memory_path("alpaca_test_writer")
    assert captured["dst"] == final_path
    assert captured["src"].parent == final_path.parent
    assert captured["src"].name.startswith("alpaca_test_writer_buys.")
    assert captured["src"].name.endswith(".tmp")
    assert captured["src"] != final_path.with_suffix(".tmp")
    assert final_path.exists()


def test_record_buy_price_prunes_using_configured_buy_memory_seconds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("UNIFIED_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg_state"))
    monkeypatch.setattr(
        singleton_mod,
        "_STATE",
        singleton_mod.SingletonState(
            account_name="alpaca_test_writer",
            buy_memory_seconds=1,
        ),
    )

    singleton_mod._save_buys(
        "alpaca_test_writer",
        {
            "STALE": {
                "price": 50.0,
                "ts": time.time() - 2.0,
                "iso": "2026-04-08T00:00:00+00:00",
            }
        },
    )

    singleton_mod.record_buy_price("AAPL", 123.45)
    data = singleton_mod._load_buys("alpaca_test_writer")

    assert "STALE" not in data
    assert data["AAPL"]["price"] == 123.45


def test_corrupt_buy_memory_is_quarantined_with_loud_log(tmp_path):
    proc = _run_snippet(
        """
        from src.alpaca_singleton import _buy_memory_path, _load_buys
        path = _buy_memory_path('alpaca_test_writer')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{not-json', encoding='utf-8')
        payload = _load_buys('alpaca_test_writer')
        assert payload == {}, payload
        quarantined = sorted(path.parent.glob('alpaca_test_writer_buys.corrupt-*.json'))
        assert len(quarantined) == 1, quarantined
        assert not path.exists(), path
        print(quarantined[0].name)
        """,
        env_extra={"ALP_PAPER": "1"},
        tmp_path=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert "alpaca_test_writer_buys.corrupt-" in proc.stdout
    assert "CORRUPT BUY MEMORY ignored" in proc.stderr


def test_record_buy_price_waits_for_cross_process_file_lock(tmp_path: Path) -> None:
    holder = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent("""
            import fcntl
            import os
            import time
            from src.alpaca_singleton import _buy_memory_file_lock_path
            path = _buy_memory_file_lock_path('alpaca_test_writer')
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a+', encoding='utf-8') as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                print('LOCKED', flush=True)
                time.sleep(1.0)
        """)],
        env={
            **os.environ,
            "PYTHONPATH": str(REPO),
            "UNIFIED_STATE_DIR": str(tmp_path / "state"),
            "XDG_STATE_HOME": str(tmp_path / "xdg_state"),
            "ALP_PAPER": "1",
        },
        cwd=str(REPO),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        for _ in range(30):
            line = holder.stdout.readline()  # type: ignore[union-attr]
            if "LOCKED" in line:
                break
        else:
            holder.kill()
            raise AssertionError("holder never announced LOCKED")

        start = time.monotonic()
        proc = _run_snippet(
            """
            from src.alpaca_singleton import _load_buys, enforce_live_singleton, record_buy_price
            enforce_live_singleton(service_name='wait_test', account_name='alpaca_test_writer')
            record_buy_price('AAPL', 123.45)
            data = _load_buys('alpaca_test_writer')
            assert data['AAPL']['price'] == 123.45
            print('RECORDED')
            """,
            env_extra={"ALP_PAPER": "1"},
            tmp_path=tmp_path,
        )
        elapsed = time.monotonic() - start
        assert proc.returncode == 0, proc.stderr
        assert "RECORDED" in proc.stdout
        assert elapsed >= 0.8, elapsed
    finally:
        holder.kill()
        holder.wait(timeout=5)


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


def test_death_spiral_blocks_short_after_buy(tmp_path):
    """`short` side must be treated identically to `sell` by the guard.

    Buying AAPL at 200 then opening a short at 198 (below the 0.5% floor of
    199) is the exact buy/short flip an RL policy can produce intra-day; the
    guard exists to crash the loop loudly rather than let it round-trip into
    a death spiral.
    """
    proc = _run_snippet(
        """
        from src.alpaca_singleton import (
            enforce_live_singleton, record_buy_price,
            guard_sell_against_death_spiral, forget_all_buys,
        )
        enforce_live_singleton(service_name='ds_short', account_name='alpaca_test_writer')
        forget_all_buys()
        record_buy_price('AAPL', 200.0)
        try:
            guard_sell_against_death_spiral('AAPL', 'short', 198.0)
        except RuntimeError as exc:
            print('REFUSED_SHORT:', exc)
        else:
            raise SystemExit('expected RuntimeError on short after buy below floor')
        # Short at 199.5 is inside floor → must be allowed.
        guard_sell_against_death_spiral('AAPL', 'short', 199.5)
        print('ALLOWED_SHORT_INSIDE')
        """,
        env_extra={"ALP_PAPER": "1"},
        tmp_path=tmp_path,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "REFUSED_SHORT" in proc.stdout
    assert "ALLOWED_SHORT_INSIDE" in proc.stdout


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


def test_death_spiral_guard_prunes_stale_buys_using_configured_buy_memory_seconds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("UNIFIED_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg_state"))
    monkeypatch.setattr(
        singleton_mod,
        "_STATE",
        singleton_mod.SingletonState(
            account_name="alpaca_test_writer",
            buy_memory_seconds=1,
        ),
    )
    singleton_mod._save_buys(
        "alpaca_test_writer",
        {
            "AAPL": {
                "price": 200.0,
                "ts": time.time() - 2.0,
                "iso": "2026-04-08T00:00:00+00:00",
            }
        },
    )

    singleton_mod.guard_sell_against_death_spiral("AAPL", "sell", 100.0)

    assert singleton_mod._load_buys("alpaca_test_writer") == {}
