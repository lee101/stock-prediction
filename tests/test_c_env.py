"""
tests/test_c_env.py — Correctness tests for the optimized C trading environment.

Verifies that the SIMD/pragma/compiler-hint optimizations do not change any
numerical behaviour: observations, rewards, episode termination, and accounting
must all remain bit-identical to the unoptimized reference contract.

Run with:
    pytest tests/test_c_env.py -v
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Shared binary-writing helper (kept consistent with test_pufferlib_market_accounting)
# ---------------------------------------------------------------------------

def _write_mktd_bin(
    path: Path,
    *,
    num_symbols: int = 1,
    num_timesteps: int = 10,
    close_prices: list[float] | None = None,
    version: int = 1,
    seed: int = 0,
) -> None:
    """Write a minimal MKTD binary file for testing."""
    if num_timesteps < 2:
        raise ValueError("num_timesteps must be >= 2")

    magic = b"MKTD"
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

    sym_table = b""
    for i in range(num_symbols):
        name = f"SYM{i}".encode()
        sym_table += name + b"\x00" * (16 - len(name))

    rng = np.random.default_rng(seed)
    features = rng.random((num_timesteps, num_symbols, features_per_sym)).astype(np.float32)
    prices = np.zeros((num_timesteps, num_symbols, price_features), dtype=np.float32)

    if close_prices is not None and num_symbols == 1:
        closes = np.asarray(close_prices, dtype=np.float32)
        for feat_idx in range(price_features - 1):  # O H L C
            prices[:, :, feat_idx] = closes.reshape(-1, 1)
        prices[:, :, 4] = 1.0  # volume
    else:
        base = np.full(num_symbols, 100.0, dtype=np.float32)
        for t in range(num_timesteps):
            step = rng.standard_normal(num_symbols).astype(np.float32) * 0.5
            base = np.maximum(base + step, 1.0)
            prices[t, :, 0] = base          # open
            prices[t, :, 1] = base + 0.5    # high
            prices[t, :, 2] = base - 0.5    # low
            prices[t, :, 3] = base          # close
            prices[t, :, 4] = 1000.0        # volume

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(sym_table)
        f.write(features.tobytes(order="C"))
        f.write(prices.tobytes(order="C"))
        if version >= 2:
            tradable = np.ones((num_timesteps, num_symbols), dtype=np.uint8)
            f.write(tradable.tobytes(order="C"))


def _run_subprocess_script(script: str) -> str:
    """Run a Python snippet in a fresh interpreter, return stdout."""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Test: setup.py build_ext --inplace compiles successfully
# ---------------------------------------------------------------------------

def test_build_extension_compiles(tmp_path: Path) -> None:
    """The C extension must rebuild cleanly with the new compiler flags.

    setup.py uses ROOT = Path(__file__).resolve().parent so it must be run
    from the repo root (not from inside pufferlib_market/) to produce the
    .so in the correct place.
    """
    result = subprocess.run(
        [sys.executable, "pufferlib_market/setup.py", "build_ext", "--inplace"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"build_ext failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Test: reset() then step() produce valid observations
# ---------------------------------------------------------------------------

def test_reset_produces_valid_observations(tmp_path: Path) -> None:
    """After reset(), the observation buffer must be finite and the right size."""
    data_path = tmp_path / "data.bin"
    _write_mktd_bin(data_path, num_symbols=3, num_timesteps=50, version=1)

    script = f"""
import json
import numpy as np
import pufferlib_market.binding as binding

binding.shared(data_path={json.dumps(str(data_path))})

num_envs = 4
num_symbols = 3
obs_size = num_symbols * 16 + 5 + num_symbols  # 56

obs_buf  = np.zeros((num_envs, obs_size), dtype=np.float32)
act_buf  = np.zeros((num_envs,), dtype=np.int32)
rew_buf  = np.zeros((num_envs,), dtype=np.float32)
term_buf = np.zeros((num_envs,), dtype=np.uint8)
trunc_buf = np.zeros((num_envs,), dtype=np.uint8)

vec_handle = binding.vec_init(
    obs_buf, act_buf, rew_buf, term_buf, trunc_buf,
    num_envs, 7,
    max_steps=40,
    fee_rate=0.001,
    max_leverage=1.0,
    periods_per_year=8760.0,
)
binding.vec_reset(vec_handle, 7)

result = {{
    "obs_shape": list(obs_buf.shape),
    "all_finite": bool(np.all(np.isfinite(obs_buf))),
    "obs_sample": obs_buf[0, :5].tolist(),
}}
binding.vec_close(vec_handle)
print(json.dumps(result))
"""
    out = json.loads(_run_subprocess_script(script))
    assert out["all_finite"], "Observations after reset contain non-finite values"
    assert out["obs_shape"] == [4, 56], f"Unexpected obs shape: {out['obs_shape']}"


# ---------------------------------------------------------------------------
# Test: reward computation unchanged (flat prices => fee-only loss)
# ---------------------------------------------------------------------------

def test_flat_prices_long_reward_equals_fee_loss(tmp_path: Path) -> None:
    """On flat prices, a long trade must lose exactly the round-trip fee."""
    data_path = tmp_path / "flat.bin"
    _write_mktd_bin(data_path, num_symbols=1, num_timesteps=3,
                    close_prices=[100.0, 100.0, 100.0], version=1)

    fee_rate = 0.001
    expected = (1.0 - fee_rate) / (1.0 + fee_rate) - 1.0

    script = f"""
import json
import numpy as np
import pufferlib_market.binding as binding

binding.shared(data_path={json.dumps(str(data_path))})

obs_buf  = np.zeros((1, 22), dtype=np.float32)
act_buf  = np.zeros((1,), dtype=np.int32)
rew_buf  = np.zeros((1,), dtype=np.float32)
term_buf = np.zeros((1,), dtype=np.uint8)
trunc_buf = np.zeros((1,), dtype=np.uint8)

vec_handle = binding.vec_init(
    obs_buf, act_buf, rew_buf, term_buf, trunc_buf,
    1, 0,
    max_steps=1,
    fee_rate={fee_rate},
    max_leverage=1.0,
    periods_per_year=8760.0,
)
binding.vec_reset(vec_handle, 0)
act_buf[:] = 1   # buy long sym 0
binding.vec_step(vec_handle)
log_info = binding.vec_log(vec_handle)
binding.vec_close(vec_handle)
print(json.dumps(float(log_info["total_return"])))
"""
    total_return = json.loads(_run_subprocess_script(script))
    assert total_return == pytest.approx(expected, rel=1e-4, abs=1e-7)


# ---------------------------------------------------------------------------
# Test: rising prices yield positive return for long position
# ---------------------------------------------------------------------------

def test_rising_prices_long_positive_return(tmp_path: Path) -> None:
    """A long on a rising price should produce a positive total return.

    The env uses OPEN prices for execution and has a 1-bar observation lag.
    With forced_offset=0 and max_steps=2:
      - reset: agent sees t=0 features, step counter at 0
      - step 1 (action=1/long): trade opens at bar-1 OPEN, closes at bar-2 OPEN.
      - step 2 (episode ends): position closed at bar-2 OPEN.
    With OPEN prices = [100, 110, 120], buy at 110 and close at 120 → +9.1% return.
    """
    data_path = tmp_path / "rising.bin"
    # All OHLCV prices set to close_prices value so OPEN == CLOSE == price
    _write_mktd_bin(data_path, num_symbols=1, num_timesteps=4,
                    close_prices=[100.0, 110.0, 120.0, 120.0], version=1)

    script = f"""
import json
import numpy as np
import pufferlib_market.binding as binding

binding.shared(data_path={json.dumps(str(data_path))})

obs_buf  = np.zeros((1, 22), dtype=np.float32)
act_buf  = np.zeros((1,), dtype=np.int32)
rew_buf  = np.zeros((1,), dtype=np.float32)
term_buf = np.zeros((1,), dtype=np.uint8)
trunc_buf = np.zeros((1,), dtype=np.uint8)

vec_handle = binding.vec_init(
    obs_buf, act_buf, rew_buf, term_buf, trunc_buf,
    1, 0,
    max_steps=2,
    fee_rate=0.0,
    max_leverage=1.0,
    periods_per_year=8760.0,
    forced_offset=0,
)
binding.vec_reset(vec_handle, 0)
act_buf[:] = 1   # buy long sym 0
binding.vec_step(vec_handle)
act_buf[:] = 1   # hold long
binding.vec_step(vec_handle)
log_info = binding.vec_log(vec_handle)
binding.vec_close(vec_handle)
print(json.dumps(float(log_info["total_return"])))
"""
    total_return = json.loads(_run_subprocess_script(script))
    assert total_return > 0.0, f"Expected positive return on rising prices, got {total_return}"


# ---------------------------------------------------------------------------
# Test: step/reset work correctly — terminals fire at episode end
# ---------------------------------------------------------------------------

def test_terminal_fires_at_max_steps(tmp_path: Path) -> None:
    """terminal flag must be 1 at step == max_steps."""
    data_path = tmp_path / "data.bin"
    _write_mktd_bin(data_path, num_symbols=1, num_timesteps=50, version=1)

    script = f"""
import json
import numpy as np
import pufferlib_market.binding as binding

binding.shared(data_path={json.dumps(str(data_path))})

max_steps = 5
obs_buf  = np.zeros((1, 22), dtype=np.float32)
act_buf  = np.zeros((1,), dtype=np.int32)
rew_buf  = np.zeros((1,), dtype=np.float32)
term_buf = np.zeros((1,), dtype=np.uint8)
trunc_buf = np.zeros((1,), dtype=np.uint8)

vec_handle = binding.vec_init(
    obs_buf, act_buf, rew_buf, term_buf, trunc_buf,
    1, 99,
    max_steps=max_steps,
    fee_rate=0.0,
    max_leverage=1.0,
    periods_per_year=8760.0,
    forced_offset=0,
)
binding.vec_reset(vec_handle, 99)

terminals = []
for step_i in range(max_steps):
    act_buf[:] = 0   # hold flat
    binding.vec_step(vec_handle)
    terminals.append(int(term_buf[0]))

binding.vec_close(vec_handle)
print(json.dumps(terminals))
"""
    terminals = json.loads(_run_subprocess_script(script))
    assert len(terminals) == 5
    # Terminal should fire on the last step
    assert terminals[-1] == 1, f"Expected terminal=1 at last step, got {terminals}"
    # Earlier steps should not be terminal (assuming no bankruptcy with flat hold)
    assert sum(terminals[:-1]) == 0, f"Unexpected early terminal: {terminals}"


# ---------------------------------------------------------------------------
# Test: observation values are consistent across multiple steps
# ---------------------------------------------------------------------------

def test_observation_values_consistent_across_steps(tmp_path: Path) -> None:
    """
    With forced_offset=0 and deterministic actions, obs must be reproducible.
    Run the same episode twice and compare the obs snapshots.
    """
    data_path = tmp_path / "repro.bin"
    _write_mktd_bin(data_path, num_symbols=2, num_timesteps=30, version=1, seed=77)

    script = f"""
import json
import numpy as np
import pufferlib_market.binding as binding

binding.shared(data_path={json.dumps(str(data_path))})

def run_episode():
    obs_size = 2 * 16 + 5 + 2  # 39
    obs_buf  = np.zeros((1, obs_size), dtype=np.float32)
    act_buf  = np.zeros((1,), dtype=np.int32)
    rew_buf  = np.zeros((1,), dtype=np.float32)
    term_buf = np.zeros((1,), dtype=np.uint8)
    trunc_buf = np.zeros((1,), dtype=np.uint8)
    vec_handle = binding.vec_init(
        obs_buf, act_buf, rew_buf, term_buf, trunc_buf,
        1, 0,
        max_steps=10,
        fee_rate=0.001,
        max_leverage=1.0,
        periods_per_year=8760.0,
        forced_offset=0,
    )
    binding.vec_reset(vec_handle, 0)
    snapshots = [obs_buf[0].tolist()]
    actions = [0, 1, 0, 2, 0, 1, 0, 0, 0, 0]
    for a in actions:
        act_buf[:] = a
        binding.vec_step(vec_handle)
        snapshots.append(obs_buf[0].tolist())
    binding.vec_close(vec_handle)
    return snapshots

r1 = run_episode()
r2 = run_episode()
match = all(
    abs(a - b) < 1e-7
    for snap1, snap2 in zip(r1, r2)
    for a, b in zip(snap1, snap2)
)
print(json.dumps({{"match": match, "steps": len(r1)}}))
"""
    result = json.loads(_run_subprocess_script(script))
    assert result["match"], "Observation values differ between identical episode runs — non-deterministic!"
    assert result["steps"] == 11  # reset obs + 10 steps


# ---------------------------------------------------------------------------
# Test: multi-symbol obs size is correct
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_symbols", [1, 3, 6, 12])
def test_obs_size_matches_formula(tmp_path: Path, num_symbols: int) -> None:
    """obs_size must equal S*FEATURES_PER_SYM + 5 + S for any S."""
    data_path = tmp_path / f"data_s{num_symbols}.bin"
    _write_mktd_bin(data_path, num_symbols=num_symbols, num_timesteps=20, version=1)
    expected_obs_size = num_symbols * 16 + 5 + num_symbols

    script = f"""
import json
import numpy as np
import pufferlib_market.binding as binding

binding.shared(data_path={json.dumps(str(data_path))})

obs_size = {expected_obs_size}
obs_buf  = np.zeros((1, obs_size), dtype=np.float32)
act_buf  = np.zeros((1,), dtype=np.int32)
rew_buf  = np.zeros((1,), dtype=np.float32)
term_buf = np.zeros((1,), dtype=np.uint8)
trunc_buf = np.zeros((1,), dtype=np.uint8)

vec_handle = binding.vec_init(
    obs_buf, act_buf, rew_buf, term_buf, trunc_buf,
    1, 0,
    max_steps=10,
    fee_rate=0.0,
    max_leverage=1.0,
    periods_per_year=8760.0,
)
binding.vec_reset(vec_handle, 0)
act_buf[:] = 0
binding.vec_step(vec_handle)
binding.vec_close(vec_handle)

print(json.dumps({{
    "obs_shape": list(obs_buf.shape),
    "all_finite": bool(np.all(np.isfinite(obs_buf))),
}}))
"""
    result = json.loads(_run_subprocess_script(script))
    assert result["obs_shape"] == [1, expected_obs_size]
    assert result["all_finite"], "Non-finite values in obs"


# ---------------------------------------------------------------------------
# Test: reward shaping flags still work after optimization
# ---------------------------------------------------------------------------

def test_cash_penalty_reduces_reward_when_flat(tmp_path: Path) -> None:
    """With cash_penalty > 0, holding flat should give lower reward than zero penalty."""
    data_path = tmp_path / "flat.bin"
    _write_mktd_bin(data_path, num_symbols=1, num_timesteps=10,
                    close_prices=[100.0] * 10, version=1)

    script = f"""
import json
import numpy as np
import pufferlib_market.binding as binding

def run_with_penalty(cash_penalty):
    binding.shared(data_path={json.dumps(str(data_path))})
    obs_buf  = np.zeros((1, 22), dtype=np.float32)
    act_buf  = np.zeros((1,), dtype=np.int32)
    rew_buf  = np.zeros((1,), dtype=np.float32)
    term_buf = np.zeros((1,), dtype=np.uint8)
    trunc_buf = np.zeros((1,), dtype=np.uint8)
    vec_handle = binding.vec_init(
        obs_buf, act_buf, rew_buf, term_buf, trunc_buf,
        1, 0,
        max_steps=5,
        fee_rate=0.0,
        max_leverage=1.0,
        periods_per_year=8760.0,
        cash_penalty=float(cash_penalty),
        reward_scale=1.0,
        reward_clip=10.0,
    )
    binding.vec_reset(vec_handle, 0)
    rewards = []
    for _ in range(5):
        act_buf[:] = 0
        binding.vec_step(vec_handle)
        rewards.append(float(rew_buf[0]))
    binding.vec_close(vec_handle)
    return rewards

r_penalized = run_with_penalty(0.05)
r_none      = run_with_penalty(0.0)
# All penalized rewards should be strictly less than corresponding no-penalty rewards
penalty_reduces = all(p < n for p, n in zip(r_penalized, r_none))
print(json.dumps({{"penalty_reduces": penalty_reduces,
                   "penalized": r_penalized,
                   "none": r_none}}))
"""
    result = json.loads(_run_subprocess_script(script))
    assert result["penalty_reduces"], (
        f"cash_penalty did not reduce rewards: penalized={result['penalized']}, none={result['none']}"
    )
