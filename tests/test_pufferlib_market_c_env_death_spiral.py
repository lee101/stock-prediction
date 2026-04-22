"""Integration test: the C env death-spiral guard actually intercepts sells.

The kwargs-flow unit tests verify the config reaches the binding; this test
runs a full C env episode with a crash scenario and asserts that guard-on
vs guard-off produce divergent behavior, confirming the guard is wired
end-to-end in c_step.
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def _write_mktd_v1_binary(
    path: Path,
    *,
    close_prices: list[float],
    range_bps: float = 0.0,
) -> None:
    """Write a minimal MKTD v1 binary: 1 symbol.

    range_bps lets tests widen high/low around close so entry-side slippage
    can still fit inside the bar (resolve_limit_fill_price otherwise rejects).
    """
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
    prices[:, :, 0] = closes           # open
    prices[:, :, 1] = closes + spread  # high
    prices[:, :, 2] = closes - spread  # low
    prices[:, :, 3] = closes           # close
    prices[:, :, 4] = 1.0              # volume
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(sym_table)
        f.write(features.tobytes(order="C"))
        f.write(prices.tobytes(order="C"))


def _run_close_slippage_probe(data_path: Path, *, fill_slippage_bps: float) -> dict:
    """Open + close at flat closes, wide bars. Measure round-trip slippage.

    Entry fill and close payoff both use fill_slippage_bps (prod-parity):
    fill_price = target*(1+slip); proceeds = qty*close*(1-slip-fee).
    Flat closes + fee=0 → total_return = (1-slip)/(1+slip) - 1 ≈ -2*slip.
    """
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
    fill_slippage_bps=float({fill_slippage_bps}),
)
binding.vec_reset(vec_handle, 123)
# open, hold, close — flat prices
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


def test_c_env_close_applies_slippage(tmp_path: Path) -> None:
    """Wide flat bars, fee=0, slippage=10bps → round-trip ≈ -0.2% (2x one-sided).

    Before this fix the close path applied only fee_rate and missed slippage,
    producing a one-sided hit of ≈ -0.1% — so a regression there would show
    up as -0.1% here, not -0.2%.
    """
    data = tmp_path / "flat.bin"
    _write_mktd_v1_binary(data, close_prices=[100.0, 100.0, 100.0], range_bps=50.0)

    with_slip = _run_close_slippage_probe(data, fill_slippage_bps=10.0)
    no_slip = _run_close_slippage_probe(data, fill_slippage_bps=0.0)

    # slip=0, fee=0, flat closes → no P&L.
    assert no_slip["total_return"] == pytest.approx(0.0, abs=1e-6)
    # (1 - 0.001) / (1 + 0.001) - 1 = -0.001998...; -0.001 would mean the close
    # path is still on the legacy fee-only branch.
    assert with_slip["total_return"] == pytest.approx(-0.001998, rel=5e-3)


def _run_short_close_slippage_probe(data_path: Path, *, fill_slippage_bps: float) -> dict:
    """Open short + close at flat closes, wide bars. Mirror of long probe.

    Short round-trip at slip s: proceeds_open = qty * tp * (1-s), cover_cost =
    qty * tp * (1+s+fee) on re-entry → total_return ≈ -2s at fee=0. Same
    magnitude as the long round-trip — validates the is_short branch of
    close_position now also applies effective_fee.
    """
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
    fill_slippage_bps=float({fill_slippage_bps}),
)
binding.vec_reset(vec_handle, 123)
# action 2 = short sym 0 (1 + S*bins_block for S=1, bins=1)
act_buf[:] = np.asarray([2], dtype=np.int32); binding.vec_step(vec_handle)
act_buf[:] = np.asarray([2], dtype=np.int32); binding.vec_step(vec_handle)
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


def test_c_env_short_close_applies_slippage(tmp_path: Path) -> None:
    """Short round-trip, fee=0, slip=10bps → same ≈ -0.2% as longs."""
    data = tmp_path / "flat.bin"
    _write_mktd_v1_binary(data, close_prices=[100.0, 100.0, 100.0], range_bps=50.0)

    with_slip = _run_short_close_slippage_probe(data, fill_slippage_bps=10.0)
    no_slip = _run_short_close_slippage_probe(data, fill_slippage_bps=0.0)

    assert no_slip["total_return"] == pytest.approx(0.0, abs=1e-6)
    # Symmetric with longs at flat prices: open-side slippage on the proceeds
    # + close-side effective_fee on the cover ≈ -2*slip round-trip.
    assert with_slip["total_return"] == pytest.approx(-0.002, abs=2e-4)


def _run_episode(data_path: Path, *, guard_tolerance_bps: float, closes: list[float]) -> dict:
    """Run a 3-step episode: long at step 0, sell at step 1, hold at step 2.

    Returns binding log dict (total_return, num_trades, ...).
    """
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
    obs_buf, act_buf, rew_buf, term_buf, trunc_buf,
    1, 123,
    max_steps=3,
    fee_rate=0.0,
    max_leverage=1.0,
    periods_per_year=252.0,
    death_spiral_tolerance_bps=float({guard_tolerance_bps}),
    death_spiral_overnight_tolerance_bps=500.0,
    death_spiral_stale_after_bars=8,
)
binding.vec_reset(vec_handle, 123)
# Step 0: go long
act_buf[:] = np.asarray([1], dtype=np.int32); binding.vec_step(vec_handle)
# Step 1: attempt sell
act_buf[:] = np.asarray([0], dtype=np.int32); binding.vec_step(vec_handle)
# Step 2: hold/terminal
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


def test_c_env_guard_off_realizes_crash_sell(tmp_path: Path) -> None:
    """Baseline: with guard off, the 1% drop is realized at the sell."""
    data = tmp_path / "crash.bin"
    closes = [100.0, 99.0, 99.0]
    _write_mktd_v1_binary(data, close_prices=closes)

    result = _run_episode(data, guard_tolerance_bps=0.0, closes=closes)

    # 1 close trade; total_return -1% (no fee).
    assert result["num_trades"] == pytest.approx(1.0)
    assert result["total_return"] == pytest.approx(-0.01, rel=1e-4)


def test_c_env_guard_on_refuses_crash_sell_mark_to_market(tmp_path: Path) -> None:
    """Guard on (50bps tol): sell at 99 (1% below entry) is refused.

    Position persists to episode end; final equity is mark-to-market, not
    raw cash. Same -1% but 0 closed trades — visible divergence from
    guard-off behavior, and no spurious -100% from ignoring the position.
    """
    data = tmp_path / "crash.bin"
    closes = [100.0, 99.0, 99.0]
    _write_mktd_v1_binary(data, close_prices=closes)

    result = _run_episode(data, guard_tolerance_bps=50.0, closes=closes)

    assert result["num_trades"] == pytest.approx(0.0)
    # Mark-to-market at last-bar close: (99 - 100) / 100 = -1%.
    # If the terminal-MTM fallback is broken, this would report -100%.
    assert result["total_return"] == pytest.approx(-0.01, rel=1e-3)


def test_c_env_guard_on_allows_small_drop_within_tolerance(tmp_path: Path) -> None:
    """0.3% drop with 50bps tol: floor=99.5, sell at 99.7 > 99.5 → allowed."""
    data = tmp_path / "small.bin"
    closes = [100.0, 99.7, 99.7]
    _write_mktd_v1_binary(data, close_prices=closes)

    result = _run_episode(data, guard_tolerance_bps=50.0, closes=closes)

    # Within tolerance: sell closed, 1 trade, realized 0.3% loss
    assert result["num_trades"] == pytest.approx(1.0)
    assert result["total_return"] == pytest.approx(-0.003, rel=1e-3)
