import torch

from FastForecaster.model import FastForecasterModel


def test_fastforecaster_model_output_shape():
    model = FastForecasterModel(
        input_dim=16,
        horizon=6,
        hidden_dim=64,
        num_layers=3,
        num_heads=4,
        ff_multiplier=2,
        dropout=0.1,
        max_symbols=8,
        qk_norm=True,
        qk_norm_eps=1e-6,
    )
    x = torch.randn(5, 48, 16)
    symbol_idx = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    y = model(x, symbol_idx)

    assert y.shape == (5, 6)
    assert torch.isfinite(y).all()
