"""Smoke tests for the ``market_sim_py`` pybind11 bindings.

These exercise both the SCALAR (legacy, action_dim=1) and DPS
(direction/size/limit_offset, action_dim=3) action modes. They run on CPU so
the tests work on dev boxes without a CUDA context.
"""

from __future__ import annotations

import os

import pytest
import torch

market_sim_py = pytest.importorskip("market_sim_py")

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
TRAINING_DATA = os.path.join(REPO_ROOT, "trainingdata")


def _have_data(symbol: str = "AAPL") -> bool:
    return os.path.isfile(os.path.join(TRAINING_DATA, f"{symbol}.csv"))


pytestmark = pytest.mark.skipif(
    not _have_data(), reason="trainingdata/AAPL.csv not present"
)


def _make_env(action_mode: str) -> "market_sim_py.MarketEnvironment":
    env = market_sim_py.MarketEnvironment(
        data_dir=TRAINING_DATA,
        log_dir=os.path.join(REPO_ROOT, "pufferlib_cpp_market_sim", "logs"),
        device="cpu",
        action_mode=action_mode,
    )
    env.set_training_mode(True)
    env.load_symbols(["AAPL"])
    return env


def test_import_works() -> None:
    assert hasattr(market_sim_py, "MarketEnvironment")


def test_scalar_mode_action_dim_and_step() -> None:
    env = _make_env("scalar")
    assert env.get_action_dim() == 1
    obs_dim = env.get_observation_dim()
    assert obs_dim > 0

    out = env.reset()
    obs = out["observations"]
    assert isinstance(obs, torch.Tensor)
    batch = obs.shape[0]
    assert obs.shape == (batch, obs_dim)

    # Zero scalar action -> no trade -> zero fees, zero realized PnL.
    action = torch.zeros(batch, dtype=torch.float32)
    step_out = env.step(action)
    assert step_out["rewards"].shape == (batch,)
    assert step_out["fees_paid"].shape == (batch,)
    assert torch.allclose(
        step_out["fees_paid"], torch.zeros(batch), atol=1e-6
    ), "zero scalar action should incur zero fees"
    assert torch.allclose(
        step_out["realized_pnl"], torch.zeros(batch), atol=1e-6
    ), "zero scalar action should yield zero realized PnL"


def test_scalar_mode_step_shapes_and_finiteness() -> None:
    env = _make_env("scalar")
    out = env.reset()
    batch = out["observations"].shape[0]

    action = torch.full((batch,), 0.5, dtype=torch.float32)
    s = env.step(action)
    for key in (
        "observations",
        "rewards",
        "fees_paid",
        "leverage_costs",
        "realized_pnl",
        "days_held",
    ):
        t = s[key]
        assert torch.isfinite(t).all(), f"{key} contains non-finite values"
    assert s["rewards"].shape == (batch,)
    assert s["observations"].shape == (batch, env.get_observation_dim())


def test_dps_mode_action_dim_and_zero_action() -> None:
    env = _make_env("dps")
    assert env.get_action_dim() == 3

    out = env.reset()
    batch = out["observations"].shape[0]

    # action_space() should reflect 3-d action shape in DPS mode.
    aspace = env.action_space()
    assert tuple(aspace) == (batch, 3)

    # Zero DPS action -> direction=0/size=0 -> no trade -> zero PnL/fees.
    action = torch.zeros((batch, 3), dtype=torch.float32)
    step_out = env.step(action)
    assert step_out["rewards"].shape == (batch,)
    assert torch.allclose(
        step_out["fees_paid"], torch.zeros(batch), atol=1e-6
    ), "zero DPS action should incur zero fees"
    assert torch.allclose(
        step_out["realized_pnl"], torch.zeros(batch), atol=1e-6
    ), "zero DPS action should yield zero realized PnL"


def test_dps_mode_leverage_cap_respected() -> None:
    """Max long DPS action (dir=+1, size=+1) should never produce more
    notional than the configured 5x leverage cap. We probe this through the
    fee channel: fee == fee_rate * notional, so fee/equity <= 5 * fee_rate."""
    env = _make_env("dps")
    out = env.reset()
    batch = out["observations"].shape[0]

    # dir=+1, size=+1, limit_offset=0 -> long at full leverage.
    action = torch.zeros((batch, 3), dtype=torch.float32)
    action[:, 0] = 1.0
    action[:, 1] = 1.0
    step_out = env.step(action)

    fees = step_out["fees_paid"]
    # Initial capital is 100k; 5x leverage notional = 500k; fee bound at
    # 10 bps = 500. Use a generous slack so we are not asserting the exact
    # fee schedule (which Unit D may tweak), only the cap.
    assert (fees <= 600.0).all(), (
        "DPS leverage cap appears violated; max fee was "
        f"{fees.max().item()} (expected <= ~500)"
    )
    assert (fees >= 0.0).all()


def test_observation_and_action_space_helpers() -> None:
    env = _make_env("scalar")
    obs_shape = env.observation_space()
    act_shape = env.action_space()
    assert len(obs_shape) == 2
    assert obs_shape[1] == env.get_observation_dim()
    # SCALAR action space is 1-d
    assert len(act_shape) == 1
