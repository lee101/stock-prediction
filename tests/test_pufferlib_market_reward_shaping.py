from __future__ import annotations

import json
import subprocess
import sys
import struct
from pathlib import Path

import numpy as np
import pytest


def _write_mktd_v1_binary_single_symbol(path: Path, *, prices: list[float]) -> None:
    if len(prices) < 2:
        raise ValueError("prices must contain at least 2 timesteps")

    magic = b"MKTD"
    version = 1
    num_symbols = 1
    num_timesteps = len(prices)
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
    prices_arr = np.zeros((num_timesteps, num_symbols, price_features), dtype=np.float32)
    prices_arr[:, :, 0] = np.asarray(prices, dtype=np.float32).reshape(-1, 1)  # open
    prices_arr[:, :, 1] = prices_arr[:, :, 0]  # high
    prices_arr[:, :, 2] = prices_arr[:, :, 0]  # low
    prices_arr[:, :, 3] = prices_arr[:, :, 0]  # close
    prices_arr[:, :, 4] = 1.0  # volume

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(sym_table)
        f.write(features.tobytes(order="C"))
        f.write(prices_arr.tobytes(order="C"))


def test_downside_penalty_applies_on_negative_return(tmp_path: Path) -> None:
    data_path = tmp_path / "down_move.bin"
    _write_mktd_v1_binary_single_symbol(data_path, prices=[100.0, 90.0])

    repo_root = Path(__file__).resolve().parents[1]
    script = f"""
import json
import numpy as np

import pufferlib_market.binding as binding

data_path = {json.dumps(str(data_path))}
binding.shared(data_path=data_path)

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
    fee_rate=0.0,
    max_leverage=1.0,
    periods_per_year=365.0,
    reward_scale=1.0,
    reward_clip=100.0,
    cash_penalty=0.0,
    drawdown_penalty=0.0,
    downside_penalty=1.0,
    trade_penalty=0.0,
)
binding.vec_reset(vec_handle, 123)

# action=1 => open long symbol 0
act_buf[:] = np.asarray([1], dtype=np.int32)
binding.vec_step(vec_handle)

payload = {{
    "reward": float(rew_buf[0]),
    "terminal": int(term_buf[0]),
}}
binding.vec_close(vec_handle)
print(json.dumps(payload))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout.strip())
    # ret = (9000 - 10000) / 10000 = -0.1
    # reward = ret - downside_penalty * ret^2 = -0.1 - 1*0.01 = -0.11
    assert payload["terminal"] == 1
    assert payload["reward"] == pytest.approx(-0.11, rel=0, abs=1e-8)


def test_trade_penalty_counts_open_event(tmp_path: Path) -> None:
    data_path = tmp_path / "flat.bin"
    _write_mktd_v1_binary_single_symbol(data_path, prices=[100.0, 100.0])

    repo_root = Path(__file__).resolve().parents[1]
    script = f"""
import json
import numpy as np

import pufferlib_market.binding as binding

data_path = {json.dumps(str(data_path))}
binding.shared(data_path=data_path)

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
    fee_rate=0.0,
    max_leverage=1.0,
    periods_per_year=365.0,
    reward_scale=1.0,
    reward_clip=100.0,
    cash_penalty=0.0,
    drawdown_penalty=0.0,
    downside_penalty=0.0,
    trade_penalty=0.25,
)
binding.vec_reset(vec_handle, 123)

# action=1 => open long symbol 0 (ret=0 on flat prices)
act_buf[:] = np.asarray([1], dtype=np.int32)
binding.vec_step(vec_handle)

payload = {{
    "reward": float(rew_buf[0]),
    "terminal": int(term_buf[0]),
}}
binding.vec_close(vec_handle)
print(json.dumps(payload))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout.strip())
    assert payload["terminal"] == 1
    assert payload["reward"] == pytest.approx(-0.25, rel=0, abs=1e-8)

