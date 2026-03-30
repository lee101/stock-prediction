from __future__ import annotations

import importlib
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pufferlib_market
import pytest
from pufferlib_market import binding_fallback


def _write_mktd(path: Path, *, closes: list[float]) -> None:
    num_symbols = 1
    num_timesteps = len(closes)
    features_per_sym = 16
    price_features = 5
    header = struct.pack(
        "<4sIIIII40s",
        b"MKTD",
        1,
        num_symbols,
        num_timesteps,
        features_per_sym,
        price_features,
        b"\x00" * 40,
    )
    sym_table = b"SYM0" + b"\x00" * 12
    features = np.zeros((num_timesteps, num_symbols, features_per_sym), dtype=np.float32)
    prices = np.zeros((num_timesteps, num_symbols, price_features), dtype=np.float32)
    close_array = np.asarray(closes, dtype=np.float32)
    prices[:, 0, 0] = close_array
    prices[:, 0, 1] = close_array
    prices[:, 0, 2] = close_array
    prices[:, 0, 3] = close_array
    prices[:, 0, 4] = 1_000.0

    path.write_bytes(header + sym_table + features.tobytes(order="C") + prices.tobytes(order="C"))


def test_package_binding_attribute_tracks_sys_modules_override(monkeypatch):
    fake_binding = SimpleNamespace(shared=lambda **_: None)
    monkeypatch.setitem(sys.modules, "pufferlib_market.binding", fake_binding)
    monkeypatch.setattr(pufferlib_market, "binding", fake_binding, raising=False)

    assert pufferlib_market.binding is fake_binding


def test_package_binding_falls_back_when_native_import_fails(monkeypatch):
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "pufferlib_market.binding":
            raise ImportError("native binding unavailable")
        return real_import_module(name, package)

    monkeypatch.delitem(sys.modules, "pufferlib_market.binding", raising=False)
    monkeypatch.delattr(pufferlib_market, "binding", raising=False)
    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    result = pufferlib_market._ensure_binding()

    assert result is binding_fallback
    assert sys.modules["pufferlib_market.binding"] is binding_fallback
    assert isinstance(binding_fallback.BINDING_IMPORT_ERROR, ImportError)


def test_vec_init_requires_shared_market_data(monkeypatch):
    monkeypatch.setattr(binding_fallback, "_ACTIVE_DATA", None)

    obs_bufs = np.zeros((1, 22), dtype=np.float32)
    act_bufs = np.zeros(1, dtype=np.int32)
    rew_bufs = np.zeros(1, dtype=np.float32)
    term_bufs = np.zeros(1, dtype=np.uint8)
    trunc_bufs = np.zeros(1, dtype=np.uint8)

    with pytest.raises(RuntimeError, match="MarketData not loaded"):
        binding_fallback.vec_init(
            obs_bufs,
            act_bufs,
            rew_bufs,
            term_bufs,
            trunc_bufs,
            n_envs=1,
            seed=0,
            max_steps=4,
        )


def test_fallback_vec_step_reports_trade_level_win_rate(tmp_path: Path, monkeypatch):
    path = tmp_path / "market.mktd"
    _write_mktd(path, closes=[100.0, 110.0, 121.0])

    monkeypatch.setattr(binding_fallback, "_SHARED_CACHE", {})
    monkeypatch.setattr(binding_fallback, "_ACTIVE_DATA", None)
    binding_fallback.shared(data_path=str(path))

    obs_bufs = np.zeros((1, 22), dtype=np.float32)
    act_bufs = np.zeros(1, dtype=np.int32)
    rew_bufs = np.zeros(1, dtype=np.float32)
    term_bufs = np.zeros(1, dtype=np.uint8)
    trunc_bufs = np.zeros(1, dtype=np.uint8)

    handle = binding_fallback.vec_init(
        obs_bufs,
        act_bufs,
        rew_bufs,
        term_bufs,
        trunc_bufs,
        n_envs=1,
        seed=0,
        max_steps=8,
        action_allocation_bins=1,
        action_level_bins=1,
    )
    binding_fallback.vec_reset(handle, seed=0)

    act_bufs[0] = 1
    binding_fallback.vec_step(handle)
    assert rew_bufs[0] > 0.0
    assert term_bufs[0] == 0

    binding_fallback.vec_step(handle)
    assert term_bufs[0] == 1

    log = binding_fallback.vec_log(handle)
    assert log is not None
    assert log["num_trades"] == 1
    assert log["win_rate"] == pytest.approx(1.0)
    assert log["avg_hold_hours"] == pytest.approx(2.0)
    assert log["total_return"] > 0.0

    env_log = binding_fallback.env_get(binding_fallback.vec_env_at(handle, 0))
    assert env_log == log
    assert binding_fallback.vec_log(handle) is None


def test_closed_handle_rejects_vec_step(tmp_path: Path, monkeypatch):
    path = tmp_path / "market.mktd"
    _write_mktd(path, closes=[100.0, 101.0, 102.0])

    monkeypatch.setattr(binding_fallback, "_SHARED_CACHE", {})
    monkeypatch.setattr(binding_fallback, "_ACTIVE_DATA", None)
    binding_fallback.shared(data_path=str(path))

    obs_bufs = np.zeros((1, 22), dtype=np.float32)
    act_bufs = np.zeros(1, dtype=np.int32)
    rew_bufs = np.zeros(1, dtype=np.float32)
    term_bufs = np.zeros(1, dtype=np.uint8)
    trunc_bufs = np.zeros(1, dtype=np.uint8)

    handle = binding_fallback.vec_init(
        obs_bufs,
        act_bufs,
        rew_bufs,
        term_bufs,
        trunc_bufs,
        n_envs=1,
        seed=0,
        max_steps=4,
    )
    binding_fallback.vec_close(handle)

    with pytest.raises(RuntimeError, match="Vector handle is closed"):
        binding_fallback.vec_step(handle)
