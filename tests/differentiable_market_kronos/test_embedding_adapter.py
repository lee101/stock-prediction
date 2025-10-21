from __future__ import annotations

from pathlib import Path

import torch

from differentiable_market.config import DataConfig
from differentiable_market.data import load_aligned_ohlc

from differentiable_market_kronos.config import KronosFeatureConfig
from differentiable_market_kronos.embedding import KronosEmbeddingAdapter


class DummyTokenizer:
    codebook_dim = 6

    def to(self, *_args, **_kwargs):
        return self

    def eval(self) -> None:
        return None

    def encode(self, x: torch.Tensor, half: bool = True):
        assert half is True
        batch, seq_len, _ = x.shape
        base = torch.arange(seq_len, device=x.device, dtype=torch.long)
        tokens = base.unsqueeze(0).repeat(batch, 1)
        return [tokens.clone(), tokens.clone()]

    def indices_to_bits(self, token_pair, half: bool = True) -> torch.Tensor:
        tokens = token_pair[0].to(torch.float32)
        bits = torch.stack([tokens, tokens * 0 + 1, tokens * 0 - 1], dim=-1)
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


def test_embed_slice_shapes(tmp_path):
    data_cfg = DataConfig(root=Path("trainingdata"), max_assets=2)
    ohlc, symbols, index = load_aligned_ohlc(data_cfg)
    cfg = KronosFeatureConfig(
        context_length=4,
        batch_size=8,
        embedding_mode="both",
        model_path="dummy",
        tokenizer_path="dummy",
    )
    adapter = KronosEmbeddingAdapter(
        cfg,
        data_cfg,
        symbols,
        index,
        tokenizer=DummyTokenizer(),
        model=DummyKronosModel(),
    )

    train_len = ohlc.shape[0] // 2
    embeddings = adapter.embed_slice(0, train_len, add_cash=False)

    assert embeddings.shape[0] == train_len - 1
    assert embeddings.shape[1] == len(symbols)
    assert embeddings.shape[2] == adapter.embedding_dim

    embeddings_cash = adapter.embed_slice(0, train_len, add_cash=True)
    assert embeddings_cash.shape[1] == len(symbols) + 1
    torch.testing.assert_close(embeddings_cash[:, -1, :], torch.zeros_like(embeddings_cash[:, -1, :]))
