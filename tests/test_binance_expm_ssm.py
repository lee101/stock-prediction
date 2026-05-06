from __future__ import annotations

import pandas as pd
import torch

from experiments.binance_expm_ssm.expm_ssm import ExpmSSMConfig, _make_features, _tensorize


def _close_frame(start: str, *, curvature: float) -> pd.DataFrame:
    idx = pd.date_range(start, periods=96, freq="h", tz="UTC")
    steps = pd.Series(range(len(idx)), index=idx, dtype=float)
    return pd.DataFrame(
        {
            "BTCUSDT": 100.0 + steps + curvature * steps.pow(2),
            "ETHUSDT": 80.0 + 0.5 * steps + (curvature / 2.0) * steps.pow(2),
        },
        index=idx,
    )


def test_tensorize_can_apply_train_feature_stats_to_eval_window() -> None:
    cfg = ExpmSSMConfig(feature_lookback_hours=12, decision_lag_hours=2)
    train_close = _close_frame("2026-01-01", curvature=0.01)
    eval_close = _close_frame("2026-02-01", curvature=0.20)

    _x_train, _y_train, _symbols, train_stats = _tensorize(train_close, cfg, return_feature_stats=True)
    x_eval, _y_eval, _ = _tensorize(eval_close, cfg, feature_stats=train_stats)
    x_eval_refit, _y_eval_refit, _ = _tensorize(eval_close, cfg)

    assert not torch.allclose(x_eval, x_eval_refit)

    eval_features = _make_features(eval_close, cfg.feature_lookback_hours, feature_stats=train_stats)
    refit_features = _make_features(eval_close, cfg.feature_lookback_hours)
    start = max(cfg.feature_lookback_hours + cfg.decision_lag_hours + 1, 4)
    end = len(eval_close) - 1
    expected = torch.tensor(
        eval_features.iloc[start - cfg.decision_lag_hours : end - cfg.decision_lag_hours].to_numpy(dtype="float32"),
        dtype=torch.float32,
    )

    assert torch.allclose(x_eval, expected)
    assert abs(float(refit_features.mean(axis=0).mean())) < abs(float(eval_features.mean(axis=0).mean()))
