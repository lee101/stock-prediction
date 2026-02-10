from __future__ import annotations

import json
import subprocess
import sys
import struct
from pathlib import Path

import numpy as np
import pytest


def _write_mktd_v1_binary(path: Path, *, price: float = 100.0, num_timesteps: int = 10) -> None:
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
    prices[:, :, 0] = price  # open
    prices[:, :, 1] = price  # high
    prices[:, :, 2] = price  # low
    prices[:, :, 3] = price  # close
    prices[:, :, 4] = 1.0  # volume

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(sym_table)
        f.write(features.tobytes(order="C"))
        f.write(prices.tobytes(order="C"))


def test_vec_step_exposes_terminal_transitions_without_clearing(tmp_path: Path) -> None:
    data_path = tmp_path / "flat_prices.bin"
    _write_mktd_v1_binary(data_path, price=100.0, num_timesteps=10)

    # Run in a fresh interpreter to avoid `binding.shared()` cache interactions.
    repo_root = Path(__file__).resolve().parents[1]
    script = f"""
import json
import numpy as np

import pufferlib_market.binding as binding

data_path = {json.dumps(str(data_path))}
binding.shared(data_path=data_path)

num_envs = 4
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
    max_steps=2,
    fee_rate=0.0,
    max_leverage=1.0,
    periods_per_year=365.0,
    reward_scale=1.0,
    reward_clip=100.0,
    cash_penalty=0.0,
    drawdown_penalty=0.0,
)
binding.vec_reset(vec_handle, 123)

act_buf[:] = np.asarray([0] * num_envs, dtype=np.int32)
binding.vec_step(vec_handle)
terms_1 = term_buf.astype(int).tolist()

binding.vec_step(vec_handle)
terms_2 = term_buf.astype(int).tolist()

binding.vec_close(vec_handle)
print(json.dumps({{"terms_1": terms_1, "terms_2": terms_2}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout.strip())
    assert payload["terms_1"] == [0, 0, 0, 0]
    assert payload["terms_2"] == [1, 1, 1, 1]

