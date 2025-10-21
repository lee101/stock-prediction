from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from differentiable_market.config import DataConfig, EnvironmentConfig, EvaluationConfig, TrainingConfig
from differentiable_market.data import load_aligned_ohlc, split_train_eval
from differentiable_market.trainer import DifferentiableMarketTrainer

from differentiable_market_kronos.config import KronosFeatureConfig
from differentiable_market_kronos.trainer import DifferentiableMarketKronosTrainer


class StubAdapter:
    def __init__(self, total_len: int, asset_count: int) -> None:
        base = torch.linspace(0, total_len * asset_count - 1, total_len * asset_count)
        self.base = base.view(total_len, asset_count, 1)

    def features_tensor(self, add_cash: bool, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        tensor = self.base.to(dtype=dtype)
        if add_cash:
            zeros = torch.zeros(tensor.shape[0], 1, tensor.shape[2], dtype=dtype)
            tensor = torch.cat([tensor, zeros], dim=1)
        return tensor


@pytest.fixture(autouse=True)
def kronos_stub(monkeypatch):
    def _ensure_adapter(self):
        return StubAdapter(total_len=len(self.index), asset_count=len(self.symbols))

    monkeypatch.setattr(DifferentiableMarketKronosTrainer, "_ensure_adapter", _ensure_adapter)


def test_trainer_feature_augmentation(tmp_path: Path):
    data_cfg = DataConfig(root=Path("trainingdata"), max_assets=2)
    env_cfg = EnvironmentConfig()
    train_cfg = TrainingConfig(
        lookback=32,
        batch_windows=8,
        rollout_groups=2,
        epochs=1,
        eval_interval=10,
        use_compile=False,
        use_muon=False,
        device="cpu",
        save_dir=tmp_path / "runs",
    )
    eval_cfg = EvaluationConfig(report_dir=tmp_path / "evals")
    kronos_cfg = KronosFeatureConfig(context_length=16, horizons=(1, 4))

    trainer = DifferentiableMarketKronosTrainer(data_cfg, env_cfg, train_cfg, eval_cfg, kronos_cfg)

    ohlc_all, _, _ = load_aligned_ohlc(data_cfg)
    train_tensor, _ = split_train_eval(ohlc_all)

    base_features, _ = DifferentiableMarketTrainer._build_features(trainer, train_tensor, train_cfg.include_cash, "train")

    assert trainer.train_features.shape[-1] == base_features.shape[-1] + 1
    trainer.close()
