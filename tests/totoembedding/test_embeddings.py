import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from totoembedding.embedding_model import TotoEmbeddingModel


def _make_model(seed: int = 0) -> TotoEmbeddingModel:
    torch.manual_seed(seed)
    model = TotoEmbeddingModel(use_toto=False, freeze_backbone=False)
    model.eval()
    return model


def test_similar_sequences_embed_closer_than_different():
    model = _make_model(seed=0)
    window = 6
    features = model.input_feature_dim

    base_sequence = torch.linspace(
        0.0, 1.0, steps=window * features, dtype=torch.float32
    ).reshape(window, features)
    slightly_shifted = base_sequence + 0.01
    very_different = base_sequence + 10.0

    price_data = torch.stack([base_sequence, slightly_shifted, very_different])
    symbol_ids = torch.zeros(3, dtype=torch.long)
    timestamps = torch.zeros(3, 3, dtype=torch.long)
    market_regime = torch.zeros(3, dtype=torch.long)

    with torch.no_grad():
        embeddings = model(
            price_data=price_data,
            symbol_ids=symbol_ids,
            timestamps=timestamps,
            market_regime=market_regime,
        )["embeddings"]

    dist_similar = torch.dist(embeddings[0], embeddings[1])
    dist_different = torch.dist(embeddings[0], embeddings[2])

    assert dist_similar < dist_different
    assert dist_different > 1e-4


def test_symbol_context_changes_embedding_output():
    model = _make_model(seed=1)
    window = 6
    features = model.input_feature_dim

    shared_series = torch.ones((window, features), dtype=torch.float32)
    price_data = torch.stack([shared_series, shared_series])
    symbol_ids = torch.tensor([0, 1], dtype=torch.long)
    timestamps = torch.zeros(2, 3, dtype=torch.long)
    market_regime = torch.zeros(2, dtype=torch.long)

    with torch.no_grad():
        embeddings = model(
            price_data=price_data,
            symbol_ids=symbol_ids,
            timestamps=timestamps,
            market_regime=market_regime,
        )["embeddings"]

    symbol_distance = torch.dist(embeddings[0], embeddings[1])
    assert symbol_distance > 1e-4


def test_cross_asset_attention_outputs_well_formed():
    model = _make_model(seed=2)
    batch = 2
    num_assets = 3
    window = 6
    features = model.input_feature_dim

    price_data = torch.randn(batch, window, features, dtype=torch.float32)
    symbol_ids = torch.zeros(batch, dtype=torch.long)
    timestamps = torch.zeros(batch, 3, dtype=torch.long)
    market_regime = torch.zeros(batch, dtype=torch.long)
    cross_asset_data = torch.randn(
        batch, num_assets, window, features, dtype=torch.float32
    )

    with torch.no_grad():
        outputs = model(
            price_data=price_data,
            symbol_ids=symbol_ids,
            timestamps=timestamps,
            market_regime=market_regime,
            cross_asset_data=cross_asset_data,
        )

    cross_embeddings = outputs["cross_embeddings"]
    attention_weights = outputs["attention_weights"]

    assert cross_embeddings is not None
    assert attention_weights is not None
    assert cross_embeddings.shape == (batch, model.embedding_dim)
    assert attention_weights.shape == (batch, 1, num_assets)
    attention_row_sums = attention_weights.sum(dim=-1)
    assert torch.allclose(
        attention_row_sums, torch.ones(batch, 1, dtype=attention_weights.dtype), atol=1e-4
    )
