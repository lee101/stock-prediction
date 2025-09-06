#!/usr/bin/env python3
"""
Lightweight baseline training test to ensure loss decreases.

This test runs a tiny training loop on synthetic OHLC data and asserts
that the model's price-prediction loss decreases meaningfully within
dozens of steps. Kept intentionally small to run fast on CPU.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys

# Add hftraining to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../hftraining'))

from hftraining.hf_trainer import HFTrainingConfig, TransformerTradingModel


def test_baseline_training_loss_decreases():
    # Deterministic behavior
    torch.manual_seed(123)
    np.random.seed(123)

    # Tiny model and data for speed
    cfg = HFTrainingConfig(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        sequence_length=10,
        prediction_horizon=2,
        use_mixed_precision=False,
        use_gradient_checkpointing=False,
        use_data_parallel=False,
    )

    input_dim = 4  # OHLC
    model = TransformerTradingModel(cfg, input_dim)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    batch_size = 32
    seq_len = cfg.sequence_length

    # Build synthetic data that's easy to learn: targets are linear in last token
    # x_last ~ N(0,1), earlier tokens close to zero => model can map last_hidden -> targets
    x = torch.zeros(batch_size, seq_len, input_dim)
    x_last = torch.randn(batch_size, input_dim)
    x[:, -1, :] = x_last

    # Targets: simple linear mapping of last token sum; horizon=2 with different scales
    base = x_last.sum(dim=1, keepdim=True)
    targets = torch.cat([base, 2 * base], dim=1)  # shape: (B, 2)

    # Measure initial loss
    with torch.no_grad():
        out0 = model(x)
        loss0 = loss_fn(out0['price_predictions'], targets).item()

    # Train for N steps
    steps = 60
    for _ in range(steps):
        out = model(x)
        loss = loss_fn(out['price_predictions'], targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        out1 = model(x)
        loss1 = loss_fn(out1['price_predictions'], targets).item()

    # Assert loss decreased by at least 50%
    assert loss1 < loss0 * 0.5, f"Expected loss to decrease by 50%, got {loss0:.4f} -> {loss1:.4f}"

