from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fastalgorithms.eth_risk_ppo.evaluate_checkpoint_windows import (
    _artifact_root_for_checkpoint,
    _evaluate_window,
    _load_training_metadata,
)


class _StubModel:
    def predict(self, obs, deterministic: bool = True):  # type: ignore[no-untyped-def]
        del obs, deterministic
        return 0, None


class _StubEnv:
    def __init__(self) -> None:
        self.start_index = 5
        self.timestamps = pd.date_range("2026-03-01", periods=16, freq="h", tz="UTC")
        self._cursor = 0
        self._steps = [
            {
                "portfolio_value": 1.01,
                "net_return": 0.01,
                "turnover": 0.20,
                "drawdown": 0.00,
                "trading_cost": 0.001,
                "weight_crypto": 0.50,
                "gross_exposure_close": 0.50,
            },
            {
                "portfolio_value": 0.9999,
                "net_return": -0.01,
                "turnover": 0.50,
                "drawdown": 0.02,
                "trading_cost": 0.002,
                "weight_crypto": 0.00,
                "gross_exposure_close": 0.00,
            },
        ]

    def reset(self, options=None):  # type: ignore[no-untyped-def]
        del options
        self._cursor = 0
        return 0, {}

    def step(self, action):  # type: ignore[no-untyped-def]
        del action
        info = dict(self._steps[self._cursor])
        terminated = self._cursor >= (len(self._steps) - 1)
        self._cursor += 1
        return 0, 0.0, terminated, False, info


def test_evaluate_window_returns_metrics_and_trajectory() -> None:
    metrics, trajectory = _evaluate_window(model=_StubModel(), env=_StubEnv(), periods_per_year=24.0 * 365.0)

    assert metrics["fills_total"] == 2
    assert metrics["fills_buy"] == 1
    assert metrics["fills_sell"] == 1
    assert len(trajectory) == 2
    assert trajectory["timestamp"].iloc[0] == "2026-03-01T05:00:00+00:00"
    assert trajectory["in_position"].tolist() == [True, False]
    assert trajectory["equity"].iloc[0] == 1.01


def test_load_training_metadata_searches_artifact_root_for_topk_checkpoint(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts" / "candidate_run"
    topk_dir = artifact_dir / "topk"
    topk_dir.mkdir(parents=True)
    checkpoint = topk_dir / "step_4096_reward_0.1234.zip"
    checkpoint.write_bytes(b"zip")
    metadata = {"args": {"data_dir": "analysis/recent"}}
    (artifact_dir / "training_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    assert _artifact_root_for_checkpoint(checkpoint) == artifact_dir
    assert _load_training_metadata(checkpoint) == metadata
