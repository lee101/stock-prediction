from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from differentiable_market.config import DataConfig

from differentiable_market_kronos.adapter import KronosFeatureAdapter
from differentiable_market_kronos.config import KronosFeatureConfig
from differentiable_market_kronos.kronos_embedder import KronosFeatureSpec


class StubEmbedder:
    def __init__(self, horizons=(1, 4)) -> None:
        self.feature_spec = KronosFeatureSpec(horizons=horizons, quantiles=(0.5,), include_path_stats=False)

    def features_for_context(self, x_df: pd.DataFrame, _x_ts: pd.Series) -> dict[str, float]:
        close = float(x_df["close"].iloc[-1])
        features: dict[str, float] = {}
        for horizon in self.feature_spec.horizons:
            features[f"H{horizon}_mu_end"] = close * 0.01 * horizon
            features[f"H{horizon}_sigma_end"] = float(len(x_df))
            features[f"H{horizon}_up_prob"] = 0.5
        return features


def make_frame(index: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=100.0, scale=2.0, size=len(index))
    df = pd.DataFrame(
        {
            "open": base,
            "high": base + 0.5,
            "low": base - 0.5,
            "close": base + rng.normal(0, 0.2, size=len(base)),
            "volume": rng.uniform(1e4, 2e4, size=len(base)),
        },
        index=index,
    )
    df["amount"] = df["close"] * df["volume"]
    df.index.name = "timestamp"
    return df


def test_kronos_feature_adapter_shapes(tmp_path: Path) -> None:
    index = pd.date_range("2024-01-01", periods=64, freq="h")
    frames = {
        "AAA": make_frame(index, seed=0),
        "BBB": make_frame(index, seed=1),
    }
    cfg = KronosFeatureConfig(context_length=8, horizons=(1, 4), quantiles=(0.5,), include_path_stats=False)
    data_cfg = DataConfig(root=tmp_path)
    adapter = KronosFeatureAdapter(
        cfg=cfg,
        data_cfg=data_cfg,
        symbols=tuple(frames.keys()),
        index=index,
        embedder=StubEmbedder(horizons=cfg.horizons),
        frame_override=frames,
    )

    cache = adapter.compute()
    assert cache.features.shape[0] == len(index)
    assert cache.features.shape[1] == len(frames)
    # horizons=2, metrics=3 -> feature dim 6
    assert cache.features.shape[2] == len(cfg.horizons) * 3

    torch_features = adapter.features_tensor(add_cash=True)
    assert torch_features.shape[1] == len(frames) + 1
    assert torch.allclose(torch_features[:, -1, :], torch.zeros_like(torch_features[:, -1, :]))
