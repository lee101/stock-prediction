from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

import ctrader.market_sim_ffi as market_sim_ffi
from ctrader.market_sim_ffi import (
    MAX_NATIVE_WEIGHT_SYMBOLS,
    NativeWeightEnvConfig,
    NativeWeightEnvHandle,
    load_library,
    simulate_target_weights,
)


def test_weight_sim_ffi_buy_and_hold():
    lib = load_library()
    close = np.array([[100.0], [110.0], [121.0]], dtype=np.float64)
    weights = np.array([[1.0], [1.0], [0.0]], dtype=np.float64)

    result, equity_curve = simulate_target_weights(
        close,
        weights,
        {
            "initial_cash": 10_000.0,
            "max_gross_leverage": 1.0,
            "fee_rate": 0.0,
            "borrow_rate_per_period": 0.0,
            "periods_per_year": 2.0,
            "can_short": 0,
        },
        library=lib,
    )

    assert result.total_return == pytest.approx(0.21)
    assert result.final_equity == pytest.approx(12100.0)
    assert result.annualized_return == pytest.approx(0.21)
    assert result.max_drawdown == pytest.approx(0.0)
    assert np.allclose(equity_curve, np.array([10000.0, 11000.0, 12100.0]))


def test_ensure_library_built_runs_make_for_default_library(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "market_sim.c").write_text("c", encoding="utf-8")
    (tmp_path / "market_sim.h").write_text("h", encoding="utf-8")
    calls: list[tuple[list[str], Path]] = []

    def _fake_run(cmd, *, cwd, check, stdout, stderr, text):
        calls.append((list(cmd), Path(cwd)))
        (tmp_path / "libmarket_sim.so").write_text("built", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(market_sim_ffi, "CTRADER_DIR", tmp_path)
    monkeypatch.setattr(market_sim_ffi.subprocess, "run", _fake_run)

    built_path = market_sim_ffi.ensure_library_built()

    assert built_path == tmp_path / "libmarket_sim.so"
    assert calls == [(["make", "libmarket_sim.so"], tmp_path)]


def test_build_lock_context_uses_flock_when_available(monkeypatch, tmp_path: Path) -> None:
    calls: list[object] = []

    class _FakeFcntl:
        LOCK_EX = "lock_ex"
        LOCK_UN = "lock_un"

        @staticmethod
        def flock(_fileno: int, operation: object) -> None:
            calls.append(operation)

    monkeypatch.setattr(market_sim_ffi, "_fcntl", _FakeFcntl)

    with market_sim_ffi._build_lock_context(tmp_path / ".build.lock"):
        pass

    assert calls == ["lock_ex", "lock_un"]


def test_ensure_library_built_rejects_missing_explicit_library_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing_libmarket_sim.so"

    with pytest.raises(FileNotFoundError, match="Native ctrader library does not exist"):
        market_sim_ffi.ensure_library_built(missing)


def test_ensure_library_built_rechecks_after_waiting_for_build_lock(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "market_sim.c").write_text("c", encoding="utf-8")
    (tmp_path / "market_sim.h").write_text("h", encoding="utf-8")

    @market_sim_ffi.contextmanager
    def _fake_build_lock(_lock_path: Path):
        (tmp_path / "libmarket_sim.so").write_text("built", encoding="utf-8")
        yield

    def _unexpected_run(*args, **kwargs):
        raise AssertionError("make should not run when another builder already produced the library")

    monkeypatch.setattr(market_sim_ffi, "CTRADER_DIR", tmp_path)
    monkeypatch.setattr(market_sim_ffi, "_build_lock_context", _fake_build_lock)
    monkeypatch.setattr(market_sim_ffi.subprocess, "run", _unexpected_run)

    built_path = market_sim_ffi.ensure_library_built()

    assert built_path == tmp_path / "libmarket_sim.so"


def test_load_library_reuses_cached_configured_cdll(monkeypatch, tmp_path: Path) -> None:
    lib_path = tmp_path / "libmarket_sim.so"
    lib_path.write_text("built", encoding="utf-8")
    calls: list[str] = []

    class _FakeFunction:
        argtypes = None
        restype = None

    class _FakeLib:
        def __init__(self) -> None:
            self.simulate_target_weights = _FakeFunction()
            self.compute_annualized_return = _FakeFunction()
            self.weight_env_obs_dim = _FakeFunction()
            self.weight_env_init = _FakeFunction()
            self.weight_env_free = _FakeFunction()
            self.weight_env_reset = _FakeFunction()
            self.weight_env_get_obs = _FakeFunction()
            self.weight_env_step = _FakeFunction()

    def _fake_cdll(path: str):
        calls.append(path)
        return _FakeLib()

    monkeypatch.setattr(market_sim_ffi, "_LIBRARY_CACHE", {})
    monkeypatch.setattr(market_sim_ffi, "ensure_library_built", lambda path=None: lib_path)
    monkeypatch.setattr(market_sim_ffi.ctypes, "CDLL", _fake_cdll)

    first = market_sim_ffi.load_library()
    second = market_sim_ffi.load_library()

    assert first is second
    assert calls == [str(lib_path.resolve())]


def test_load_library_refreshes_cached_cdll_after_rebuild(monkeypatch, tmp_path: Path) -> None:
    lib_path = tmp_path / "libmarket_sim.so"
    lib_path.write_text("built", encoding="utf-8")
    calls: list[str] = []

    class _FakeFunction:
        argtypes = None
        restype = None

    class _FakeLib:
        def __init__(self, tag: str) -> None:
            self.tag = tag
            self.simulate_target_weights = _FakeFunction()
            self.compute_annualized_return = _FakeFunction()
            self.weight_env_obs_dim = _FakeFunction()
            self.weight_env_init = _FakeFunction()
            self.weight_env_free = _FakeFunction()
            self.weight_env_reset = _FakeFunction()
            self.weight_env_get_obs = _FakeFunction()
            self.weight_env_step = _FakeFunction()

    def _fake_cdll(path: str):
        calls.append(path)
        return _FakeLib(f"load-{len(calls)}")

    monkeypatch.setattr(market_sim_ffi, "_LIBRARY_CACHE", {})
    monkeypatch.setattr(market_sim_ffi, "ensure_library_built", lambda path=None: lib_path)
    monkeypatch.setattr(market_sim_ffi.ctypes, "CDLL", _fake_cdll)

    first = market_sim_ffi.load_library()
    stat_result = lib_path.stat()
    updated_mtime = stat_result.st_mtime_ns + 1_000_000
    market_sim_ffi.os.utime(lib_path, ns=(updated_mtime, updated_mtime))
    second = market_sim_ffi.load_library()

    assert first is not second
    assert calls == [str(lib_path.resolve()), str(lib_path.resolve())]


def test_load_library_surfaces_cdll_load_failure(monkeypatch, tmp_path: Path) -> None:
    lib_path = tmp_path / "libmarket_sim.so"
    lib_path.write_text("built", encoding="utf-8")

    def _failing_cdll(_path: str):
        raise OSError("bad ELF header")

    monkeypatch.setattr(market_sim_ffi, "_LIBRARY_CACHE", {})
    monkeypatch.setattr(market_sim_ffi, "ensure_library_built", lambda path=None: lib_path)
    monkeypatch.setattr(market_sim_ffi.ctypes, "CDLL", _failing_cdll)

    with pytest.raises(RuntimeError, match="Failed to load native ctrader library") as exc_info:
        market_sim_ffi.load_library()

    assert str(lib_path.resolve()) in str(exc_info.value)
    assert "bad ELF header" in str(exc_info.value)


def test_ensure_library_built_surfaces_make_output_on_failure(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "market_sim.c").write_text("c", encoding="utf-8")
    (tmp_path / "market_sim.h").write_text("h", encoding="utf-8")

    def _failing_run(cmd, *, cwd, check, stdout, stderr, text):
        raise subprocess.CalledProcessError(
            returncode=2,
            cmd=cmd,
            output="compiler exploded",
        )

    monkeypatch.setattr(market_sim_ffi, "CTRADER_DIR", tmp_path)
    monkeypatch.setattr(market_sim_ffi.subprocess, "run", _failing_run)

    with pytest.raises(RuntimeError, match="Failed to build native ctrader library") as exc_info:
        market_sim_ffi.ensure_library_built()

    assert "compiler exploded" in str(exc_info.value)


def test_simulate_target_weights_rejects_symbol_count_above_native_limit() -> None:
    n_symbols = MAX_NATIVE_WEIGHT_SYMBOLS + 1
    close = np.ones((3, n_symbols), dtype=np.float64)
    weights = np.zeros((3, n_symbols), dtype=np.float64)

    with pytest.raises(ValueError, match="supports at most"):
        simulate_target_weights(
            close,
            weights,
            {"initial_cash": 10_000.0},
            library=object(),
        )


def test_simulate_target_weights_rejects_empty_bar_axis_before_touching_library() -> None:
    close = np.ones((0, 1), dtype=np.float64)
    weights = np.zeros((0, 1), dtype=np.float64)

    with pytest.raises(ValueError, match="at least one bar"):
        simulate_target_weights(
            close,
            weights,
            {"initial_cash": 10_000.0},
            library=object(),
        )


def test_simulate_target_weights_rejects_empty_symbol_axis_before_touching_library() -> None:
    close = np.ones((3, 0), dtype=np.float64)
    weights = np.zeros((3, 0), dtype=np.float64)

    with pytest.raises(ValueError, match="at least one symbol"):
        simulate_target_weights(
            close,
            weights,
            {"initial_cash": 10_000.0},
            library=object(),
        )


def test_native_weight_env_handle_rejects_symbol_count_above_native_limit() -> None:
    n_symbols = MAX_NATIVE_WEIGHT_SYMBOLS + 1
    close = np.ones((8, n_symbols), dtype=np.float64)
    env_config = NativeWeightEnvConfig(lookback=2, episode_steps=3, reward_scale=1.0)

    with pytest.raises(ValueError, match="supports at most"):
        NativeWeightEnvHandle(
            close,
            env_config,
            {"initial_cash": 10_000.0},
            library=object(),
        )


def test_native_weight_env_handle_rejects_empty_axes_before_touching_library() -> None:
    env_config = NativeWeightEnvConfig(lookback=2, episode_steps=3, reward_scale=1.0)

    with pytest.raises(ValueError, match="at least one bar"):
        NativeWeightEnvHandle(
            np.ones((0, 1), dtype=np.float64),
            env_config,
            {"initial_cash": 10_000.0},
            library=object(),
        )

    with pytest.raises(ValueError, match="at least one symbol"):
        NativeWeightEnvHandle(
            np.ones((3, 0), dtype=np.float64),
            env_config,
            {"initial_cash": 10_000.0},
            library=object(),
        )
