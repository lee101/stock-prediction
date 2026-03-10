from __future__ import annotations

import json
import subprocess
import sys
import struct
from pathlib import Path

import numpy as np
import pytest


def _write_mktd_v1_binary(
    path: Path,
    *,
    price: float = 100.0,
    num_timesteps: int = 2,
    close_prices: list[float] | None = None,
) -> None:
    """Write a minimal MKTD v1 binary with a single symbol and flat prices."""
    if num_timesteps < 2:
        raise ValueError("num_timesteps must be >= 2")

    magic = b"MKTD"
    version = 1
    num_symbols = 1
    features_per_sym = 16
    price_features = 5
    padding = b"\x00" * 40
    header = struct.pack(
        "<4sIIIII40s",
        magic,
        version,
        num_symbols,
        num_timesteps,
        features_per_sym,
        price_features,
        padding,
    )

    sym = b"TEST"
    sym_table = sym + b"\x00" * (16 - len(sym))

    features = np.zeros((num_timesteps, num_symbols, features_per_sym), dtype=np.float32)
    prices = np.zeros((num_timesteps, num_symbols, price_features), dtype=np.float32)
    closes = np.asarray(close_prices if close_prices is not None else [price] * num_timesteps, dtype=np.float32)
    if closes.shape != (num_timesteps,):
        raise ValueError("close_prices must match num_timesteps")
    prices[:, :, 0] = closes.reshape(num_timesteps, 1)  # open
    prices[:, :, 1] = closes.reshape(num_timesteps, 1)  # high
    prices[:, :, 2] = closes.reshape(num_timesteps, 1)  # low
    prices[:, :, 3] = closes.reshape(num_timesteps, 1)  # close
    prices[:, :, 4] = 1.0    # volume

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(sym_table)
        f.write(features.tobytes(order="C"))
        f.write(prices.tobytes(order="C"))


def test_short_does_not_create_free_money_on_flat_prices(tmp_path: Path) -> None:
    data_path = tmp_path / "flat_prices.bin"
    _write_mktd_v1_binary(data_path, price=100.0, num_timesteps=2)

    fee_rate = 0.001
    # For 1x long/short with open+close fees, flat prices implies:
    # final_cash = initial_cash * (1 - fee) / (1 + fee)
    expected_return = (1.0 - fee_rate) / (1.0 + fee_rate) - 1.0

    # Run in a fresh interpreter to avoid `binding.shared()` cache interactions.
    repo_root = Path(__file__).resolve().parents[1]
    script = f"""
import json
import numpy as np
from pathlib import Path

import pufferlib_market.binding as binding

data_path = {json.dumps(str(data_path))}
fee_rate = float({fee_rate})

binding.shared(data_path=data_path)

def run(action: int) -> float:
    num_envs = 1
    num_symbols = 1
    obs_size = num_symbols * 16 + 5 + num_symbols

    obs_buf = np.zeros((num_envs, obs_size), dtype=np.float32)
    act_buf = np.zeros((num_envs,), dtype=np.int32)
    rew_buf = np.zeros((num_envs,), dtype=np.float32)
    term_buf = np.zeros((num_envs,), dtype=np.uint8)
    trunc_buf = np.zeros((num_envs,), dtype=np.uint8)

    vec_handle = binding.vec_init(
        obs_buf,
        act_buf,
        rew_buf,
        term_buf,
        trunc_buf,
        num_envs,
        123,
        max_steps=1,
        fee_rate=fee_rate,
        max_leverage=1.0,
        periods_per_year=8760.0,
    )
    binding.vec_reset(vec_handle, 123)
    act_buf[:] = np.asarray([action], dtype=np.int32)
    binding.vec_step(vec_handle)
    log_info = binding.vec_log(vec_handle)
    binding.vec_close(vec_handle)
    return float(log_info["total_return"])

print(json.dumps({{"long": run(1), "short": run(2)}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    results = json.loads(completed.stdout.strip())
    assert results["long"] == pytest.approx(expected_return, rel=1e-4, abs=1e-7)
    assert results["short"] == pytest.approx(expected_return, rel=1e-4, abs=1e-7)
    assert results["long"] < 0.0
    assert results["short"] < 0.0


def test_short_borrow_fee_hits_flat_short_in_c_env(tmp_path: Path) -> None:
    data_path = tmp_path / "flat_prices.bin"
    _write_mktd_v1_binary(data_path, price=100.0, num_timesteps=3)

    apr = 0.10
    periods_per_year = 252.0
    expected_fee_per_step = 10_000.0 * apr / periods_per_year
    expected_short_return = -(2.0 * expected_fee_per_step) / 10_000.0

    repo_root = Path(__file__).resolve().parents[1]
    script = f"""
import json
import numpy as np

import pufferlib_market.binding as binding

binding.shared(data_path={json.dumps(str(data_path))})

def run(action: int) -> float:
    obs_buf = np.zeros((1, 22), dtype=np.float32)
    act_buf = np.zeros((1,), dtype=np.int32)
    rew_buf = np.zeros((1,), dtype=np.float32)
    term_buf = np.zeros((1,), dtype=np.uint8)
    trunc_buf = np.zeros((1,), dtype=np.uint8)
    vec_handle = binding.vec_init(
        obs_buf,
        act_buf,
        rew_buf,
        term_buf,
        trunc_buf,
        1,
        123,
        max_steps=2,
        fee_rate=0.0,
        max_leverage=1.0,
        short_borrow_apr=float({apr}),
        periods_per_year=float({periods_per_year}),
    )
    binding.vec_reset(vec_handle, 123)
    act_buf[:] = np.asarray([action], dtype=np.int32)
    binding.vec_step(vec_handle)
    act_buf[:] = np.asarray([action], dtype=np.int32)
    binding.vec_step(vec_handle)
    log_info = binding.vec_log(vec_handle)
    binding.vec_close(vec_handle)
    return float(log_info["total_return"])

print(json.dumps({{"long": run(1), "short": run(2)}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    results = json.loads(completed.stdout.strip())
    assert results["long"] == pytest.approx(0.0, abs=1e-12)
    assert results["short"] == pytest.approx(expected_short_return, rel=1e-4, abs=1e-7)


def test_long_leverage_two_x_is_supported_in_c_env(tmp_path: Path) -> None:
    data_path = tmp_path / "up_only.bin"
    _write_mktd_v1_binary(data_path, num_timesteps=2, close_prices=[100.0, 110.0])

    repo_root = Path(__file__).resolve().parents[1]
    script = f"""
import json
import numpy as np

import pufferlib_market.binding as binding

binding.shared(data_path={json.dumps(str(data_path))})

obs_buf = np.zeros((1, 22), dtype=np.float32)
act_buf = np.zeros((1,), dtype=np.int32)
rew_buf = np.zeros((1,), dtype=np.float32)
term_buf = np.zeros((1,), dtype=np.uint8)
trunc_buf = np.zeros((1,), dtype=np.uint8)
vec_handle = binding.vec_init(
    obs_buf,
    act_buf,
    rew_buf,
    term_buf,
    trunc_buf,
    1,
    123,
    max_steps=1,
    fee_rate=0.0,
    max_leverage=2.0,
    periods_per_year=252.0,
)
binding.vec_reset(vec_handle, 123)
act_buf[:] = np.asarray([1], dtype=np.int32)
binding.vec_step(vec_handle)
log_info = binding.vec_log(vec_handle)
binding.vec_close(vec_handle)
print(json.dumps(log_info))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    log_info = json.loads(completed.stdout.strip())
    assert float(log_info["total_return"]) == pytest.approx(0.2, rel=1e-6, abs=1e-9)
