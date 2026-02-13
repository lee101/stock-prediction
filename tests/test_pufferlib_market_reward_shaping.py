from __future__ import annotations

import json
import subprocess
import sys
import struct
from pathlib import Path

import numpy as np
import pytest
import math


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


def _write_mktd_v1_binary_single_symbol_ohlc(path: Path, *, ohlc: list[tuple[float, float, float, float]]) -> None:
    if len(ohlc) < 2:
        raise ValueError("ohlc must contain at least 2 timesteps")

    magic = b"MKTD"
    version = 1
    num_symbols = 1
    num_timesteps = len(ohlc)
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
    arr = np.asarray(ohlc, dtype=np.float32).reshape(num_timesteps, 4)
    prices_arr[:, :, 0] = arr[:, 0:1]  # open
    prices_arr[:, :, 1] = arr[:, 1:2]  # high
    prices_arr[:, :, 2] = arr[:, 2:3]  # low
    prices_arr[:, :, 3] = arr[:, 3:4]  # close
    prices_arr[:, :, 4] = 1.0          # volume

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(sym_table)
        f.write(features.tobytes(order="C"))
        f.write(prices_arr.tobytes(order="C"))


def _write_mktd_v1_binary_single_symbol_with_tradable_mask(
    path: Path,
    *,
    prices: list[float],
    tradable_mask: list[int],
) -> None:
    """Write a v1 file with an appended tradable mask (supported for forward-compatibility)."""
    if len(prices) != len(tradable_mask):
        raise ValueError("prices and tradable_mask must have the same length")
    _write_mktd_v1_binary_single_symbol(path, prices=prices)
    mask = np.asarray(tradable_mask, dtype=np.uint8).reshape(len(tradable_mask), 1)
    with open(path, "ab") as f:
        f.write(mask.tobytes(order="C"))


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


def test_smooth_downside_penalty_applies_softplus_shaping(tmp_path: Path) -> None:
    data_path = tmp_path / "down_move_smooth.bin"
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
    downside_penalty=0.0,
    smooth_downside_penalty=2.0,
    smooth_downside_temperature=0.02,
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
    assert payload["terminal"] == 1

    # ret = -0.1; smooth_neg = temp * softplus(-ret / temp), temp=0.02
    temp = 0.02
    ret = -0.1
    smooth_neg = temp * math.log1p(math.exp((-ret) / temp))
    expected = ret - 2.0 * (smooth_neg * smooth_neg)
    assert payload["reward"] == pytest.approx(expected, rel=0, abs=1e-6)


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


def test_cash_penalty_not_applied_when_no_symbol_tradable(tmp_path: Path) -> None:
    data_path = tmp_path / "not_tradable.bin"
    _write_mktd_v1_binary_single_symbol_with_tradable_mask(data_path, prices=[100.0, 100.0], tradable_mask=[0, 0])

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
    cash_penalty=0.5,
    drawdown_penalty=0.0,
    downside_penalty=0.0,
    trade_penalty=0.0,
)
binding.vec_reset(vec_handle, 123)

# action=0 => try to stay flat
act_buf[:] = np.asarray([0], dtype=np.int32)
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
    assert payload["reward"] == pytest.approx(0.0, rel=0, abs=1e-9)


def test_action_allocation_bins_scale_risk_exposure(tmp_path: Path) -> None:
    data_path = tmp_path / "alloc_bins.bin"
    _write_mktd_v1_binary_single_symbol(data_path, prices=[100.0, 110.0])

    repo_root = Path(__file__).resolve().parents[1]
    script = f"""
import json
import numpy as np

import pufferlib_market.binding as binding

data_path = {json.dumps(str(data_path))}
binding.shared(data_path=data_path)

num_envs = 2
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
    action_allocation_bins=2,
    action_level_bins=1,
    action_max_offset_bps=0.0,
    cash_penalty=0.0,
    drawdown_penalty=0.0,
    downside_penalty=0.0,
    trade_penalty=0.0,
)
binding.vec_reset(vec_handle, 123)

# action=1 => 50% long, action=2 => 100% long (single symbol, single level)
act_buf[:] = np.asarray([1, 2], dtype=np.int32)
binding.vec_step(vec_handle)

payload = {{
    "rewards": rew_buf.tolist(),
    "terminals": term_buf.tolist(),
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
    r_half, r_full = payload["rewards"]
    t_half, t_full = payload["terminals"]
    assert t_half == 1 and t_full == 1
    assert r_half == pytest.approx(0.05, rel=0, abs=1e-7)
    assert r_full == pytest.approx(0.10, rel=0, abs=1e-7)


def test_action_level_bins_require_bar_fill_for_limit_entry(tmp_path: Path) -> None:
    data_path = tmp_path / "limit_fill.bin"
    _write_mktd_v1_binary_single_symbol_ohlc(
        data_path,
        ohlc=[
            (100.0, 100.0, 100.0, 100.0),  # flat bar: only exact-close level can fill
            (110.0, 110.0, 110.0, 110.0),  # next bar move for mark-to-market reward
        ],
    )

    repo_root = Path(__file__).resolve().parents[1]
    script = f"""
import json
import numpy as np

import pufferlib_market.binding as binding

data_path = {json.dumps(str(data_path))}
binding.shared(data_path=data_path)

num_envs = 2
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
    action_allocation_bins=1,
    action_level_bins=3,
    action_max_offset_bps=50.0,
    cash_penalty=0.0,
    drawdown_penalty=0.0,
    downside_penalty=0.0,
    trade_penalty=0.0,
)
binding.vec_reset(vec_handle, 123)

# For 1 symbol + 1 alloc bin + 3 level bins:
# action=2 -> long level_idx=1 (0 bps, fills), action=3 -> long level_idx=2 (+50 bps, not fill on flat bar)
act_buf[:] = np.asarray([2, 3], dtype=np.int32)
binding.vec_step(vec_handle)

payload = {{
    "rewards": rew_buf.tolist(),
    "terminals": term_buf.tolist(),
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
    r_fill, r_no_fill = payload["rewards"]
    t_fill, t_no_fill = payload["terminals"]
    assert t_fill == 1 and t_no_fill == 1
    assert r_fill == pytest.approx(0.10, rel=0, abs=1e-7)
    assert r_no_fill == pytest.approx(0.0, rel=0, abs=1e-7)
