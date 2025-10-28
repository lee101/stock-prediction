from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pufferlibtraining2.config import load_plan
from pufferlibtraining2.data.loader import load_asset_frames
from pufferlibtraining2.envs.trading_env import make_vecenv


def _write_data(root: Path, symbol: str, days: int = 40) -> None:
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    base = np.linspace(100, 120, days, dtype=np.float32)
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": base,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + 0.25,
            "volume": np.full(days, 1_000_000, dtype=np.float32),
        }
    )
    frame.to_csv(root / f"{symbol}.csv", index=False)


def test_make_vecenv_serial(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for sym in ("AAPL", "MSFT"):
        _write_data(data_dir, sym)

    overrides = {
        "data": {
            "data_dir": str(data_dir),
            "symbols": ["AAPL", "MSFT"],
            "window_size": 8,
            "min_history": 32,
        },
        "env": {"device": "cpu", "reward_scale": 1.0},
        "vec": {
            "backend": "Serial",
            "num_envs": 2,
            "num_workers": 1,
            "batch_size": 2,
            "device": "cpu",
        },
        "logging": {
            "tensorboard_dir": str(tmp_path / "tb"),
            "checkpoint_dir": str(tmp_path / "ckpt"),
            "summary_path": str(tmp_path / "summary.json"),
        },
    }
    plan = load_plan(overrides=overrides)
    frames = load_asset_frames(plan.data)
    vecenv = make_vecenv(plan, frames)
    vecenv.async_reset(plan.vec.seed)
    observations, rewards, terminals, truncations, infos, env_ids, masks = vecenv.recv()

    assert observations.shape[0] == vecenv.num_agents
    assert observations.shape[1] == plan.data.window_size
    assert observations.shape[2] == len(plan.data.symbols)
    assert rewards.shape[0] == vecenv.num_agents
    assert not np.any(terminals)
