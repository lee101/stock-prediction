from __future__ import annotations

import pandas as pd

from alpacanewccrosslearning.data import (
    CovariateConfig,
    build_chronos_input,
    split_frame,
)


def _sample_frame(rows: int = 10) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100 + i for i in range(rows)],
            "high": [101 + i for i in range(rows)],
            "low": [99 + i for i in range(rows)],
            "close": [100.5 + i for i in range(rows)],
            "volume": [1000 + i * 10 for i in range(rows)],
        }
    )


def test_build_chronos_input_shapes():
    frame = _sample_frame(12)
    payload = build_chronos_input(frame)
    target = payload["target"]
    covs = payload["past_covariates"]
    assert target.shape == (4, 12)
    assert set(covs.keys()) >= {"volume", "log_volume", "return_1h"}
    for values in covs.values():
        assert len(values) == 12


def test_build_chronos_input_covariate_toggle():
    frame = _sample_frame(8)
    cfg = CovariateConfig(include_volume=False, include_log_volume=False)
    payload = build_chronos_input(frame, covariate_config=cfg)
    covs = payload["past_covariates"]
    assert "volume" not in covs
    assert "log_volume" not in covs


def test_split_frame_val_hours():
    frame = _sample_frame(20)
    train, val = split_frame(frame, val_hours=5)
    assert len(train) == 15
    assert val is not None
    assert len(val) == 5
