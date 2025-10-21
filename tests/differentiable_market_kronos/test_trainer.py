from __future__ import annotations

from pathlib import Path

import pytest
import torch

from differentiable_market.config import DataConfig, EnvironmentConfig, EvaluationConfig, TrainingConfig
from differentiable_market.data import load_aligned_ohlc, split_train_eval
from differentiable_market_kronos.config import KronosFeatureConfig
from differentiable_market_kronos.trainer import DifferentiableMarketKronosTrainer


class DummyTokenizer:
    codebook_dim = 6

    def to(self, *_args, **_kwargs):
        return self

    def eval(self) -> None:
        return None

    def encode(self, x: torch.Tensor, half: bool = True):
        tokens = torch.arange(x.shape[1], device=x.device, dtype=torch.long).unsqueeze(0)
        tokens = tokens.repeat(x.shape[0], 1)
        return [tokens, tokens]

    def indices_to_bits(self, token_pair, half: bool = True) -> torch.Tensor:
        tokens = token_pair[0].to(torch.float32)
        bits = torch.stack([tokens, torch.ones_like(tokens), -torch.ones_like(tokens)], dim=-1)
        return bits


class DummyKronosModel:
    d_model = 4

    def to(self, *_args, **_kwargs):
        return self

    def eval(self) -> None:
        return None

    def parameters(self):
        return []

    def decode_s1(self, s1_ids: torch.Tensor, s2_ids: torch.Tensor, stamp: torch.Tensor | None = None):
        batch, seq_len = s1_ids.shape
        context = torch.arange(batch * seq_len, device=s1_ids.device, dtype=torch.float32)
        context = context.view(batch, seq_len, 1).repeat(1, 1, self.d_model)
        logits = torch.zeros(batch, seq_len, 1, device=s1_ids.device)
        return logits, context


@pytest.fixture(autouse=True)
def kronos_stubs(monkeypatch):
    def _tokenizer_from_pretrained(_name: str):
        return DummyTokenizer()

    def _model_from_pretrained(_name: str):
        return DummyKronosModel()

    monkeypatch.setattr("external.kronos.model.KronosTokenizer.from_pretrained", _tokenizer_from_pretrained)
    monkeypatch.setattr("external.kronos.model.Kronos.from_pretrained", _model_from_pretrained)


def test_trainer_feature_dimension(tmp_path):
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

    kronos_cfg = KronosFeatureConfig(
        context_length=16,
        batch_size=8,
        embedding_mode="both",
        model_path="dummy",
        tokenizer_path="dummy",
    )

    trainer = DifferentiableMarketKronosTrainer(data_cfg, env_cfg, train_cfg, eval_cfg, kronos_cfg)

    ohlc_all, _, _ = load_aligned_ohlc(data_cfg)
    train_tensor, _ = split_train_eval(ohlc_all)
    base_features, _ = super(DifferentiableMarketKronosTrainer, trainer)._build_features(train_tensor, train_cfg.include_cash, "train")

    assert trainer.train_features.shape[-1] == base_features.shape[-1] + trainer.kronos_adapter.embedding_dim
    trainer.close()
