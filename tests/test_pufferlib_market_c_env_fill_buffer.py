"""Integration test: fill_buffer_bps blocks limit fills when market does not
overshoot the target by that many bps.

Mirrors Python eval `_resolve_limit_fill_price` semantics (hourly_replay.py):
- Buy trigger: low <= target_price * (1 - buffer/1e4)
- Sell trigger: high >= target_price * (1 + buffer/1e4)

Default (buffer=0) degrades to classical "target touches bar range" acceptance,
so existing training runs are unaffected.
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def _write_mktd_v1_binary(path: Path, *, close_prices: list[float], range_bps: float) -> None:
    num_timesteps = len(close_prices)
    if num_timesteps < 2:
        raise ValueError("need >= 2 timesteps")
    header = struct.pack(
        "<4sIIIII40s",
        b"MKTD",
        1,
        1,
        num_timesteps,
        16,
        5,
        b"\x00" * 40,
    )
    sym_table = b"TEST" + b"\x00" * 12
    features = np.zeros((num_timesteps, 1, 16), dtype=np.float32)
    closes = np.asarray(close_prices, dtype=np.float32).reshape(num_timesteps, 1)
    prices = np.zeros((num_timesteps, 1, 5), dtype=np.float32)
    spread = closes * (range_bps / 10000.0)
    prices[:, :, 0] = closes
    prices[:, :, 1] = closes + spread
    prices[:, :, 2] = closes - spread
    prices[:, :, 3] = closes
    prices[:, :, 4] = 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(sym_table)
        f.write(features.tobytes(order="C"))
        f.write(prices.tobytes(order="C"))


def _run_buffer_probe(data_path: Path, *, fill_buffer_bps: float) -> dict:
    """Try to open a long on a bar with range=10bps. If buffer <= 10bps: fill.
    If buffer > 10bps: market never dips below trigger → no fill, no trade."""
    repo_root = Path(__file__).resolve().parents[1]
    script = f"""
import json, warnings
import numpy as np
import pufferlib_market.binding as binding

warnings.simplefilter('ignore')
binding.shared(data_path={json.dumps(str(data_path))})

obs_buf = np.zeros((1, 22), dtype=np.float32)
act_buf = np.zeros((1,), dtype=np.int32)
rew_buf = np.zeros((1,), dtype=np.float32)
term_buf = np.zeros((1,), dtype=np.uint8)
trunc_buf = np.zeros((1,), dtype=np.uint8)
vec_handle = binding.vec_init(
    obs_buf, act_buf, rew_buf, term_buf, trunc_buf,
    1, 123,
    max_steps=3,
    fee_rate=0.0,
    max_leverage=1.0,
    periods_per_year=252.0,
    fill_slippage_bps=0.0,
    fill_buffer_bps=float({fill_buffer_bps}),
)
binding.vec_reset(vec_handle, 123)
# Try to open long three bars in a row (action=1 targets first symbol long)
act_buf[:] = np.asarray([1], dtype=np.int32); binding.vec_step(vec_handle)
act_buf[:] = np.asarray([1], dtype=np.int32); binding.vec_step(vec_handle)
act_buf[:] = np.asarray([0], dtype=np.int32); binding.vec_step(vec_handle)
log_info = binding.vec_log(vec_handle)
binding.vec_close(vec_handle)
print(json.dumps({{k: float(v) for k, v in log_info.items()}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.strip())


def test_c_env_buffer_zero_fills(tmp_path: Path) -> None:
    """buffer=0, flat closes, slippage=0 → round-trip fills cleanly at 0 PnL."""
    data = tmp_path / "narrow.bin"
    _write_mktd_v1_binary(data, close_prices=[100.0, 100.0, 100.0], range_bps=10.0)
    log = _run_buffer_probe(data, fill_buffer_bps=0.0)
    assert log["total_return"] == pytest.approx(0.0, abs=1e-6)
    assert log["num_trades"] >= 1, "buffer=0 must allow fill when target in bar range"


def test_c_env_buffer_blocks_fill_when_market_too_tight(tmp_path: Path) -> None:
    """buffer=100bps on bars with range=10bps → trigger unreachable, no fill, no PnL."""
    data = tmp_path / "narrow_block.bin"
    _write_mktd_v1_binary(data, close_prices=[100.0, 100.0, 100.0], range_bps=10.0)
    log = _run_buffer_probe(data, fill_buffer_bps=100.0)
    # No fill ever → no trade, zero PnL.
    assert log["total_return"] == pytest.approx(0.0, abs=1e-6)
    assert log["num_trades"] == 0, "buffer=100bps on 10bps-range bars must block entry"


def test_c_env_buffer_boundary_just_fits(tmp_path: Path) -> None:
    """buffer=9bps on 10bps-range bars → still fits (low=99.90 <= 100*(1-9e-4)=99.91)."""
    data = tmp_path / "narrow_just_fits.bin"
    _write_mktd_v1_binary(data, close_prices=[100.0, 100.0, 100.0], range_bps=10.0)
    log = _run_buffer_probe(data, fill_buffer_bps=9.0)
    assert log["num_trades"] >= 1, "buffer=9bps should fit inside 10bps range"


def test_c_env_buffer_negative_clamped_to_zero(tmp_path: Path) -> None:
    """Negative buffer gets clamped to 0 at binding.c (never over-permissive)."""
    data = tmp_path / "neg.bin"
    _write_mktd_v1_binary(data, close_prices=[100.0, 100.0, 100.0], range_bps=10.0)
    log = _run_buffer_probe(data, fill_buffer_bps=-5.0)
    assert log["num_trades"] >= 1
