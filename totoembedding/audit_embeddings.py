#!/usr/bin/env python3
"""
Audit Toto Embedding usage:
- Loads TotoEmbeddingModel with a specified pretrained checkpoint
- Prints backbone type and inferred d_model
- Runs a small forward pass and reports shapes and basic stats
"""

import argparse
from pathlib import Path
import torch
import numpy as np

from embedding_model import TotoEmbeddingModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pretrained', type=str, default='training/models/modern_best_sharpe.pth',
                   help='Path to pretrained model checkpoint (.pth)')
    p.add_argument('--symbols', type=int, default=21)
    p.add_argument('--window', type=int, default=30)
    p.add_argument('--batch', type=int, default=2)
    args = p.parse_args()

    ckpt = Path(args.pretrained)
    print(f"Pretrained path: {ckpt} (exists={ckpt.exists()})")

    model = TotoEmbeddingModel(
        pretrained_model_path=str(ckpt),
        num_symbols=args.symbols,
        freeze_backbone=True,
    )
    model.eval()

    print('Backbone type:', type(model.backbone).__name__)
    print('Inferred d_model:', model.backbone_dim)

    # Create a tiny synthetic batch matching expected features
    feature_dim = model.input_feature_dim
    price_data = torch.randn(args.batch, args.window, feature_dim)
    symbol_ids = torch.randint(0, args.symbols, (args.batch,))
    timestamps = torch.randint(0, 12, (args.batch, 3))  # hour/day/month will be clamped by embeddings
    market_regime = torch.randint(0, 4, (args.batch,))

    with torch.no_grad():
        out = model(
            price_data=price_data,
            symbol_ids=symbol_ids,
            timestamps=timestamps,
            market_regime=market_regime,
        )

    emb = out['embeddings']
    print('Embeddings shape:', tuple(emb.shape))
    print('Embeddings stats: mean={:.4f}, std={:.4f}'.format(emb.mean().item(), emb.std().item()))

    # Check trainable vs frozen params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: total={total:,} trainable={trainable:,} (frozen backbone expected)")

    # Quick signal check
    zero_like = torch.zeros_like(emb)
    diff = (emb - zero_like).abs().mean().item()
    print('Non-zero embedding check (mean abs):', diff)


if __name__ == '__main__':
    main()

