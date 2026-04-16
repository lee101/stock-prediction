"""Direct invariant tests for the pufferlib_market C simulator.

Covers invariants that were previously only checked indirectly (via parity with
the soft sim). These protect the C env as the ground-truth validation backend.

Follows the same subprocess + synthetic MKTD binary pattern as tests/test_c_env.py.
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


def _write_mktd_bin(
    path: Path,
    *,
    num_symbols: int = 1,
    num_timesteps: int = 10,
    close_prices: list[float] | None = None,
    version: int = 1,
    seed: int = 0,
    features_override: np.ndarray | None = None,
    hl_pad: float = 0.0,
) -> None:
    """Minimal MKTD file. If close_prices is given, all OHLC = that series,
    except HIGH = close + hl_pad and LOW = close - hl_pad (for slippage tests
    where the fill needs room inside [LOW, HIGH])."""
    if num_timesteps < 2:
        raise ValueError("num_timesteps must be >= 2")

    header = struct.pack(
        "<4sIIIII40s",
        b"MKTD",
        version,
        num_symbols,
        num_timesteps,
        16,
        5,
        b"\x00" * 40,
    )
    sym_table = b""
    for i in range(num_symbols):
        name = f"SYM{i}".encode()
        sym_table += name + b"\x00" * (16 - len(name))

    rng = np.random.default_rng(seed)
    if features_override is not None:
        features = features_override.astype(np.float32)
    else:
        features = rng.random((num_timesteps, num_symbols, 16)).astype(np.float32)
    prices = np.zeros((num_timesteps, num_symbols, 5), dtype=np.float32)
    if close_prices is not None and num_symbols == 1:
        closes = np.asarray(close_prices, dtype=np.float32)
        prices[:, :, 0] = closes.reshape(-1, 1)  # open
        prices[:, :, 1] = (closes + hl_pad).reshape(-1, 1)  # high
        prices[:, :, 2] = (closes - hl_pad).reshape(-1, 1)  # low
        prices[:, :, 3] = closes.reshape(-1, 1)  # close
        prices[:, :, 4] = 1.0
    else:
        base = np.full(num_symbols, 100.0, dtype=np.float32)
        for t in range(num_timesteps):
            base = np.maximum(base + rng.standard_normal(num_symbols).astype(np.float32) * 0.5, 1.0)
            prices[t, :, 0] = base
            prices[t, :, 1] = base + 0.5
            prices[t, :, 2] = base - 0.5
            prices[t, :, 3] = base
            prices[t, :, 4] = 1000.0

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(sym_table)
        f.write(features.tobytes(order="C"))
        f.write(prices.tobytes(order="C"))
        if version >= 2:
            f.write(np.ones((num_timesteps, num_symbols), dtype=np.uint8).tobytes(order="C"))


def _run(script: str) -> str:
    r = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    return r.stdout.strip()


# ---------------------------------------------------------------------------
# Invariant 1: zero action + zero cash_penalty + zero fee earns exactly 0
# ---------------------------------------------------------------------------

def test_zero_action_zero_penalty_earns_zero(tmp_path: Path) -> None:
    data_path = tmp_path / "flat.bin"
    _write_mktd_bin(data_path, num_symbols=1, num_timesteps=20,
                    close_prices=[100.0] * 20, version=1)

    script = f"""
import json, numpy as np
import pufferlib_market.binding as binding

binding.shared(data_path={json.dumps(str(data_path))})

obs = np.zeros((1, 22), dtype=np.float32)
act = np.zeros((1,), dtype=np.int32)
rew = np.zeros((1,), dtype=np.float32)
term = np.zeros((1,), dtype=np.uint8)
trunc = np.zeros((1,), dtype=np.uint8)

vec = binding.vec_init(
    obs, act, rew, term, trunc, 1, 0,
    max_steps=5, fee_rate=0.0, max_leverage=1.0, periods_per_year=8760.0,
    cash_penalty=0.0, forced_offset=0,
)
binding.vec_reset(vec, 0)
for _ in range(5):
    act[:] = 0
    binding.vec_step(vec)
log = binding.vec_log(vec)
binding.vec_close(vec)
print(json.dumps({{"total_return": float(log["total_return"]),
                   "num_trades": float(log["num_trades"])}}))
"""
    r = json.loads(_run(script))
    assert r["total_return"] == pytest.approx(0.0, abs=1e-6), (
        f"Flat action with zero fee/penalty must earn exactly 0, got {r['total_return']}"
    )
    assert r["num_trades"] == 0.0


# ---------------------------------------------------------------------------
# Invariant 2: decision_lag changes what the agent observes
# Write a staircase of distinct feature values; verify obs at step 0 differs
# between lag=1 and lag=3 runs.
# ---------------------------------------------------------------------------

def test_decision_lag_shifts_observation(tmp_path: Path) -> None:
    num_steps = 10
    # Distinct feature value per timestep so we can detect which bar was read
    feats = np.zeros((num_steps, 1, 16), dtype=np.float32)
    for t in range(num_steps):
        feats[t, 0, 0] = float(t + 1)  # feature 0 = 1, 2, 3, ...
    data_path = tmp_path / "stair.bin"
    _write_mktd_bin(data_path, num_symbols=1, num_timesteps=num_steps,
                    close_prices=[100.0] * num_steps, version=1,
                    features_override=feats)

    def obs_for_lag(lag: int) -> float:
        script = f"""
import json, numpy as np
import pufferlib_market.binding as binding

binding.shared(data_path={json.dumps(str(data_path))})

obs = np.zeros((1, 22), dtype=np.float32)
act = np.zeros((1,), dtype=np.int32)
rew = np.zeros((1,), dtype=np.float32)
term = np.zeros((1,), dtype=np.uint8)
trunc = np.zeros((1,), dtype=np.uint8)
vec = binding.vec_init(
    obs, act, rew, term, trunc, 1, 0,
    max_steps=6, fee_rate=0.0, max_leverage=1.0, periods_per_year=8760.0,
    forced_offset=3, decision_lag={lag}, cash_penalty=0.0,
)
binding.vec_reset(vec, 0)
# feature 0 value after reset is what agent observes before any step
v = float(obs[0, 0])
binding.vec_close(vec)
print(json.dumps(v))
"""
        return json.loads(_run(script))

    v_lag1 = obs_for_lag(1)  # should see feature at t=3-1=2 → 3.0
    v_lag3 = obs_for_lag(3)  # should see feature at t=3-3=0 → 1.0
    assert v_lag1 == pytest.approx(3.0, abs=1e-5), f"lag=1 obs: {v_lag1}"
    assert v_lag3 == pytest.approx(1.0, abs=1e-5), f"lag=3 obs: {v_lag3}"
    assert v_lag1 != v_lag3, "decision_lag must change the observation"


# ---------------------------------------------------------------------------
# Invariant 3: max_hold_hours force-closes the position
# Open a long, hold for more steps than max_hold, verify position closes.
# ---------------------------------------------------------------------------

def test_max_hold_hours_force_closes(tmp_path: Path) -> None:
    """With max_hold_hours=N and the agent repeatedly sending the same 'hold long'
    action, the env must force-close once hold_hours >= N. We detect that by the
    num_trades counter growing (each close increments it), compared to the
    max_hold=0 (disabled) baseline under the same action sequence.
    """
    data_path = tmp_path / "flat.bin"
    _write_mktd_bin(data_path, num_symbols=1, num_timesteps=30,
                    close_prices=[100.0] * 30, version=1)

    def run_with_max_hold(max_hold: int) -> float:
        script = f"""
import json, numpy as np
import pufferlib_market.binding as binding

binding.shared(data_path={json.dumps(str(data_path))})

obs = np.zeros((1, 22), dtype=np.float32)
act = np.zeros((1,), dtype=np.int32)
rew = np.zeros((1,), dtype=np.float32)
term = np.zeros((1,), dtype=np.uint8)
trunc = np.zeros((1,), dtype=np.uint8)
vec = binding.vec_init(
    obs, act, rew, term, trunc, 1, 0,
    max_steps=15, fee_rate=0.0, max_leverage=1.0, periods_per_year=8760.0,
    forced_offset=0, max_hold_hours={max_hold}, cash_penalty=0.0,
)
binding.vec_reset(vec, 0)
# Keep sending long-intent; with max_hold>0 the env will force-close + reopen repeatedly.
for _ in range(15):
    act[:] = 1
    binding.vec_step(vec)
log = binding.vec_log(vec)
binding.vec_close(vec)
print(json.dumps(float(log["num_trades"])))
"""
        return json.loads(_run(script))

    trades_disabled = run_with_max_hold(0)  # max_hold disabled
    trades_max2 = run_with_max_hold(2)
    # With max_hold disabled and a single continuous long, only the terminal
    # step closes the trade → ~1 trade logged (or 0 depending on horizon).
    # With max_hold=2 the env should force-close several times → strictly more.
    assert trades_max2 > trades_disabled + 1.5, (
        f"max_hold=2 failed to force extra closes: disabled={trades_disabled}, "
        f"max_hold=2={trades_max2}"
    )


# ---------------------------------------------------------------------------
# Invariant 4: fill_slippage_bps penalizes a round trip
# Flat prices, 0 fee, slippage 100bps: buy fills at 1.01x open, sell at open
# → round trip loses ~1%.
# ---------------------------------------------------------------------------

def test_fill_slippage_charges_buy_leg(tmp_path: Path) -> None:
    data_path = tmp_path / "flat.bin"
    # HIGH=103, LOW=97, OPEN=CLOSE=100 → 100bps buy slip still fits [97, 103].
    _write_mktd_bin(data_path, num_symbols=1, num_timesteps=10,
                    close_prices=[100.0] * 10, version=1, hl_pad=3.0)

    def run_with_slip(slip_bps: float) -> float:
        script = f"""
import json, numpy as np
import pufferlib_market.binding as binding

binding.shared(data_path={json.dumps(str(data_path))})

obs = np.zeros((1, 22), dtype=np.float32)
act = np.zeros((1,), dtype=np.int32)
rew = np.zeros((1,), dtype=np.float32)
term = np.zeros((1,), dtype=np.uint8)
trunc = np.zeros((1,), dtype=np.uint8)
vec = binding.vec_init(
    obs, act, rew, term, trunc, 1, 0,
    max_steps=3, fee_rate=0.0, max_leverage=1.0, periods_per_year=8760.0,
    forced_offset=0, fill_slippage_bps={slip_bps}, cash_penalty=0.0,
)
binding.vec_reset(vec, 0)
act[:] = 1  # long at step 1 (fills at high-side open)
binding.vec_step(vec)
act[:] = 0  # close at step 2 (OPEN price, no slippage on close)
binding.vec_step(vec)
act[:] = 0
binding.vec_step(vec)
log = binding.vec_log(vec)
binding.vec_close(vec)
print(json.dumps(float(log["total_return"])))
"""
        return json.loads(_run(script))

    r_no_slip = run_with_slip(0.0)
    r_slip = run_with_slip(100.0)  # 100 bps = 1%
    # With zero fee and flat prices, no-slip round trip = 0.
    assert r_no_slip == pytest.approx(0.0, abs=1e-5)
    # With 100 bps buy slippage, round trip loses at least ~0.9% (allow fp fuzz).
    assert r_slip < -0.005, f"fill_slippage_bps=100 should cost ~1%, got {r_slip}"


# ---------------------------------------------------------------------------
# Invariant 5: fee scales with number of round trips
# At flat prices with no slippage: fee cost ∝ number of trades.
# ---------------------------------------------------------------------------

def test_fee_scales_with_turnover(tmp_path: Path) -> None:
    data_path = tmp_path / "flat.bin"
    _write_mktd_bin(data_path, num_symbols=1, num_timesteps=30,
                    close_prices=[100.0] * 30, version=1)

    def run_with_trips(n_trips: int) -> tuple[float, float]:
        # Build action sequence: (long, flat) * n_trips, then pad with flat
        actions = []
        for _ in range(n_trips):
            actions += [1, 0]
        # Pad to 8 steps so episodes have equal horizon
        while len(actions) < 8:
            actions.append(0)
        actions_json = json.dumps(actions)
        script = f"""
import json, numpy as np
import pufferlib_market.binding as binding

binding.shared(data_path={json.dumps(str(data_path))})

obs = np.zeros((1, 22), dtype=np.float32)
act = np.zeros((1,), dtype=np.int32)
rew = np.zeros((1,), dtype=np.float32)
term = np.zeros((1,), dtype=np.uint8)
trunc = np.zeros((1,), dtype=np.uint8)
vec = binding.vec_init(
    obs, act, rew, term, trunc, 1, 0,
    max_steps=8, fee_rate=0.001, max_leverage=1.0, periods_per_year=8760.0,
    forced_offset=0, cash_penalty=0.0,
)
binding.vec_reset(vec, 0)
for a in {actions_json}:
    act[:] = a
    binding.vec_step(vec)
log = binding.vec_log(vec)
binding.vec_close(vec)
print(json.dumps({{"total_return": float(log["total_return"]),
                   "num_trades": float(log["num_trades"])}}))
"""
        r = json.loads(_run(script))
        return r["total_return"], r["num_trades"]

    ret_1, trades_1 = run_with_trips(1)
    ret_2, trades_2 = run_with_trips(2)
    ret_3, trades_3 = run_with_trips(3)

    # Each round trip closes 1 trade in the log. Expect trades proportional to n_trips.
    assert trades_2 >= trades_1 + 0.5, f"2 trips should log more trades, got {trades_1} vs {trades_2}"
    assert trades_3 >= trades_2 + 0.5, f"3 trips should log more trades, got {trades_2} vs {trades_3}"

    # Return becomes more negative with more turnover; linearity within fp tolerance.
    # per-trip loss = ret_1, so ret_3 ≈ 3*ret_1 (compounding is tiny at 1bp/trip)
    assert ret_3 < ret_2 < ret_1 < 0.0, (
        f"Fee cost must grow with turnover: ret_1={ret_1}, ret_2={ret_2}, ret_3={ret_3}"
    )
    per_trip_1 = ret_1
    per_trip_3 = ret_3 / 3.0
    assert per_trip_3 == pytest.approx(per_trip_1, rel=0.05), (
        f"Per-trip fee cost should be ~linear: ret_1={ret_1}, ret_3/3={per_trip_3}"
    )


# ---------------------------------------------------------------------------
# Invariant 6: decision_lag=2 keeps the realistic lag enforced after reset
# (the production path). This is the gate referenced in CLAUDE.md rule #1.
# ---------------------------------------------------------------------------

def test_decision_lag_2_is_explicitly_supported(tmp_path: Path) -> None:
    data_path = tmp_path / "flat.bin"
    _write_mktd_bin(data_path, num_symbols=1, num_timesteps=15,
                    close_prices=[100.0] * 15, version=1)

    script = f"""
import json, numpy as np
import pufferlib_market.binding as binding

binding.shared(data_path={json.dumps(str(data_path))})

obs = np.zeros((1, 22), dtype=np.float32)
act = np.zeros((1,), dtype=np.int32)
rew = np.zeros((1,), dtype=np.float32)
term = np.zeros((1,), dtype=np.uint8)
trunc = np.zeros((1,), dtype=np.uint8)
vec = binding.vec_init(
    obs, act, rew, term, trunc, 1, 0,
    max_steps=5, fee_rate=0.0, max_leverage=1.0, periods_per_year=8760.0,
    forced_offset=3, decision_lag=2, cash_penalty=0.0,
)
binding.vec_reset(vec, 0)
finite = bool(np.all(np.isfinite(obs)))
act[:] = 0
binding.vec_step(vec)
binding.vec_close(vec)
print(json.dumps({{"finite": finite}}))
"""
    r = json.loads(_run(script))
    assert r["finite"], "decision_lag=2 produced non-finite observations"
