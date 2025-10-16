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

from totoembedding.embedding_model import TotoEmbeddingModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pretrained', type=str, default='',
                   help='Optional: Path to fallback checkpoint (.pth) when not using Toto')
    p.add_argument('--use_toto', action='store_true', help='Use real Toto backbone')
    p.add_argument('--toto_model_id', type=str, default='Datadog/Toto-Open-Base-1.0')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--symbols', type=int, default=21)
    p.add_argument('--window', type=int, default=30)
    p.add_argument('--batch', type=int, default=2)
    args = p.parse_args()

    ckpt = Path(args.pretrained) if args.pretrained else None
    if ckpt is not None:
        print(f"Pretrained path: {ckpt} (exists={ckpt.exists()})")

    model = TotoEmbeddingModel(
        pretrained_model_path=str(ckpt) if ckpt is not None else None,
        num_symbols=args.symbols,
        freeze_backbone=True,
        use_toto=args.use_toto,
        toto_model_id=args.toto_model_id,
        toto_device=args.device,
    )
    model.eval()

    backbone_type = type(getattr(model, 'backbone', None)).__name__ if getattr(model, 'backbone', None) is not None else 'Toto'
    mode = getattr(model, '_backbone_mode', 'unknown')
    print('Backbone type:', backbone_type)
    print('Backbone mode:', mode)
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
