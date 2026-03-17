from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def _write_single_symbol_price_binary(path: Path, *, prices: list[float]) -> None:
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
    price_col = np.asarray(prices, dtype=np.float32).reshape(-1, 1)
    prices_arr[:, :, 0] = price_col  # open
    prices_arr[:, :, 1] = price_col  # high
    prices_arr[:, :, 2] = price_col  # low
    prices_arr[:, :, 3] = price_col  # close
    prices_arr[:, :, 4] = 1.0        # volume

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(sym_table)
        f.write(features.tobytes(order="C"))
        f.write(prices_arr.tobytes(order="C"))


def _run_binding_episode(
    data_path: Path,
    *,
    max_steps: int,
    enable_drawdown_profit_early_exit: bool,
    drawdown_profit_early_exit_verbose: bool,
) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    kwargs = {
        "max_steps": int(max_steps),
        "fee_rate": 0.0,
        "max_leverage": 1.0,
        "periods_per_year": 365.0,
        "reward_scale": 1.0,
        "reward_clip": 100.0,
        "cash_penalty": 0.0,
        "drawdown_penalty": 0.0,
        "downside_penalty": 0.0,
        "trade_penalty": 0.0,
        "enable_drawdown_profit_early_exit": bool(enable_drawdown_profit_early_exit),
        "drawdown_profit_early_exit_verbose": bool(drawdown_profit_early_exit_verbose),
        "drawdown_profit_early_exit_min_steps": 20,
        "drawdown_profit_early_exit_progress_fraction": 0.5,
    }
    script = f"""
import json
import numpy as np

import pufferlib_market.binding as binding

data_path = {json.dumps(str(data_path))}
kwargs = json.loads({json.dumps(json.dumps(kwargs))})

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
    **kwargs,
)
binding.vec_reset(vec_handle, 123)

steps = 0
while steps < int(kwargs["max_steps"]) + 5:
    act_buf[:] = np.asarray([1], dtype=np.int32)
    binding.vec_step(vec_handle)
    steps += 1
    if int(term_buf[0]) == 1:
        break

raw_log = binding.vec_log(vec_handle) or {{}}
log_payload = {{}}
for key, value in raw_log.items():
    if isinstance(value, (int, float)):
        log_payload[key] = float(value)

payload = {{
    "steps": int(steps),
    "terminal": int(term_buf[0]),
    "reward": float(rew_buf[0]),
    "log": log_payload,
}}
binding.vec_close(vec_handle)
print(json.dumps(payload))
"""
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )


def test_c_env_early_exits_when_drawdown_exceeds_profit_after_halfway(tmp_path: Path) -> None:
    data_path = tmp_path / "drawdown_profit_exit.bin"
    _write_single_symbol_price_binary(
        data_path,
        prices=[
            100.0,
            100.0,
            105.0,
            110.0,
            108.0,
            104.0,
            100.0,
            98.0,
            95.0,
            92.0,
            90.0,
            89.0,
            88.0,
            87.0,
            87.0,
            87.0,
            87.0,
            87.0,
            87.0,
            87.0,
            87.0,
        ],
    )

    completed = _run_binding_episode(
        data_path,
        max_steps=20,
        enable_drawdown_profit_early_exit=True,
        drawdown_profit_early_exit_verbose=True,
    )
    payload = json.loads(completed.stdout.strip())

    assert payload["terminal"] == 1
    assert payload["steps"] == 9
    assert payload["log"]["total_return"] < 0.0
    assert payload["log"]["max_drawdown"] > 0.0
    assert "early stopping at 50.0%" in completed.stderr.lower()


def test_c_env_can_disable_drawdown_profit_early_exit(tmp_path: Path) -> None:
    data_path = tmp_path / "drawdown_profit_disabled.bin"
    _write_single_symbol_price_binary(
        data_path,
        prices=[
            100.0,
            100.0,
            105.0,
            110.0,
            108.0,
            104.0,
            100.0,
            98.0,
            95.0,
            92.0,
            90.0,
            89.0,
            88.0,
            87.0,
            87.0,
            87.0,
            87.0,
            87.0,
            87.0,
            87.0,
            87.0,
        ],
    )

    completed = _run_binding_episode(
        data_path,
        max_steps=20,
        enable_drawdown_profit_early_exit=False,
        drawdown_profit_early_exit_verbose=False,
    )
    payload = json.loads(completed.stdout.strip())

    assert payload["terminal"] == 1
    assert payload["steps"] == 20
    assert "early stopping" not in completed.stderr.lower()


def test_c_env_tracks_running_max_drawdown_not_just_terminal_drawdown(tmp_path: Path) -> None:
    data_path = tmp_path / "max_drawdown.bin"
    _write_single_symbol_price_binary(
        data_path,
        prices=[100.0, 110.0, 90.0, 105.0],
    )

    completed = _run_binding_episode(
        data_path,
        max_steps=3,
        enable_drawdown_profit_early_exit=False,
        drawdown_profit_early_exit_verbose=False,
    )
    payload = json.loads(completed.stdout.strip())

    assert payload["terminal"] == 1
    assert payload["steps"] == 3
    assert payload["log"]["total_return"] == pytest.approx(0.05, abs=1e-6)
    assert payload["log"]["max_drawdown"] == pytest.approx((110.0 - 90.0) / 110.0, abs=1e-6)
