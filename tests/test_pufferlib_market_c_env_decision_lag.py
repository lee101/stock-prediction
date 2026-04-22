"""decision_lag obs/exec separation.

Prod uses decision_lag>=2 because Chronos2 forecast + Gemini call + order
placement straddles more than one bar. The C env must show the agent
features from t-lag (never t) so training doesn't leak lookahead. This
test asserts that separation by stamping each timestep's feature vector
with its own index and reading back what the env exposes.
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def _write_signed_mktd(path: Path, *, num_timesteps: int) -> None:
    """MKTD v1 binary where features[t][0][0] = t (all other features 0).

    Flat OHLC at 100. One symbol. Gives a per-timestep signature that can be
    read straight out of obs_buf[0] after a step.
    """
    num_symbols = 1
    features_per_sym = 16
    price_feats = 5
    header = struct.pack(
        "<4sIIIII40s",
        b"MKTD",
        1,
        num_symbols,
        num_timesteps,
        features_per_sym,
        price_feats,
        b"\x00" * 40,
    )
    sym_table = b"TEST" + b"\x00" * 12
    features = np.zeros((num_timesteps, num_symbols, features_per_sym), dtype=np.float32)
    features[:, 0, 0] = np.arange(num_timesteps, dtype=np.float32)
    prices = np.zeros((num_timesteps, num_symbols, price_feats), dtype=np.float32)
    prices[:, :, :4] = 100.0
    prices[:, :, 4] = 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(sym_table)
        f.write(features.tobytes(order="C"))
        f.write(prices.tobytes(order="C"))


def _collect_obs_feature_over_steps(
    data_path: Path, *, decision_lag: int, num_steps: int
) -> list[float]:
    """Run num_steps hold-actions and return obs[0] after each (pre-action reset + every step).

    Returns a list of length num_steps + 1: element 0 is post-reset, i is post-step-i.
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
    max_steps={num_steps + 5},
    fee_rate=0.0,
    max_leverage=1.0,
    periods_per_year=252.0,
    cash_penalty=0.0,
    decision_lag={decision_lag},
)
binding.vec_reset(vec_handle, 123)
collected = [float(obs_buf[0, 0])]
for _ in range({num_steps}):
    act_buf[:] = np.asarray([0], dtype=np.int32)
    binding.vec_step(vec_handle)
    collected.append(float(obs_buf[0, 0]))
binding.vec_close(vec_handle)
print(json.dumps(collected))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.strip())


@pytest.mark.parametrize("lag", [1, 2, 3])
def test_c_env_obs_uses_features_from_t_minus_lag(tmp_path: Path, lag: int) -> None:
    """obs[0] after k steps must be features[max(0, k - lag)][0][0] = max(0, k - lag)."""
    data = tmp_path / "signed.bin"
    _write_signed_mktd(data, num_timesteps=20)

    num_steps = 10
    seen = _collect_obs_feature_over_steps(data, decision_lag=lag, num_steps=num_steps)

    expected = [float(max(0, k - lag)) for k in range(num_steps + 1)]
    assert seen == expected, f"lag={lag} obs mismatch: got {seen}, expected {expected}"
