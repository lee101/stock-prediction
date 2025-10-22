import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from differentiable_market_totoembedding.config import (
    DataConfig,
    EnvironmentConfig,
    EvaluationConfig,
    TotoEmbeddingConfig,
    TotoTrainingConfig,
)
from differentiable_market_totoembedding.trainer import TotoDifferentiableMarketTrainer


def _write_mock_asset(csv_path: Path, base_price: float, noise_scale: float = 0.5) -> None:
    timestamps = pd.date_range("2024-01-01", periods=200, freq="15min", tz="UTC")
    prices = base_price + np.cumsum(np.random.default_rng(0).normal(0.0, noise_scale, size=len(timestamps)))
    opens = prices + np.random.default_rng(1).normal(0.0, noise_scale, size=len(timestamps))
    highs = np.maximum(opens, prices) + np.abs(np.random.default_rng(2).normal(0.0, noise_scale * 0.5, size=len(timestamps)))
    lows = np.minimum(opens, prices) - np.abs(np.random.default_rng(3).normal(0.0, noise_scale * 0.5, size=len(timestamps)))
    data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": prices,
        }
    )
    data.to_csv(csv_path, index=False)


def test_trainer_appends_toto_embeddings(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for idx, price in enumerate((50.0, 72.5, 101.3), start=1):
        _write_mock_asset(data_dir / f"asset_{idx}.csv", base_price=price)

    data_cfg = DataConfig(root=data_dir, glob="*.csv", include_cash=True, min_timesteps=128, max_assets=3)
    env_cfg = EnvironmentConfig()
    toto_cfg = TotoEmbeddingConfig(
        context_length=32,
        embedding_dim=32,
        input_feature_dim=4,
        use_toto=False,
        freeze_backbone=True,
        batch_size=16,
        cache_dir=None,
        reuse_cache=False,
    )
    train_cfg = TotoTrainingConfig(
        lookback=32,
        rollout_groups=2,
        batch_windows=8,
        epochs=4,
        eval_interval=2,
        device="cpu",
        dtype="float32",
        save_dir=tmp_path / "runs",
        tensorboard_root=tmp_path / "tb",
        include_cash=True,
        use_muon=False,
        use_compile=False,
        toto=toto_cfg,
        best_k_checkpoints=1,
    )
    eval_cfg = EvaluationConfig(report_dir=tmp_path / "evals")

    trainer = TotoDifferentiableMarketTrainer(data_cfg, env_cfg, train_cfg, eval_cfg)

    assert trainer.train_features.shape[-1] == 4 + toto_cfg.embedding_dim
    assert trainer.eval_features.shape[-1] == 4 + toto_cfg.embedding_dim

    # Cash asset (last index) should have zeroed Toto embeddings
    cash_embeddings = trainer.train_features[:, -1, -toto_cfg.embedding_dim :]
    assert torch.allclose(cash_embeddings, torch.zeros_like(cash_embeddings))

    stats = trainer._train_step()
    assert "loss" in stats
    assert math.isfinite(stats["loss"])
