#!/usr/bin/env python3
"""
Single-batch supervised fit using Shampoo optimizer and Muon scheduler.

Fits y = 3x + 2 on a single batch, showing loss decreasing over steps.

Usage examples:
  python training/single_batch_shampoo_muon.py --optimizer shampoo --scheduler muon
  python training/single_batch_shampoo_muon.py --optimizer adamw --scheduler muon --lr 0.01
  python training/single_batch_shampoo_muon.py --optimizer shampoo --no-scheduler
"""

import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from hftraining.modern_optimizers import get_optimizer
from hftraining.improved_schedulers import get_improved_scheduler


def make_line_data(n=256, noise=0.02, seed=123):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand((n, 1), generator=g) * 2 - 1  # [-1,1]
    y = 3.0 * x + 2.0
    if noise > 0:
        y = y + noise * torch.randn_like(y, generator=g)
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', type=str, default='shampoo', help='Optimizer name (shampoo, adamw, lion, etc.)')
    parser.add_argument('--scheduler', type=str, default='muon', help='Scheduler name (muon, cosine, etc.)')
    parser.add_argument('--no-scheduler', action='store_true', help='Disable scheduler')
    parser.add_argument('--steps', type=int, default=200, help='Number of optimization steps over the single batch')
    parser.add_argument('--lr', type=float, default=5e-2, help='Learning rate')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Create single batch
    x, y = make_line_data(n=256, noise=0.02, seed=args.seed)

    # Simple linear model y = ax + b
    model = nn.Linear(1, 1)

    # Optimizer and optional scheduler
    opt = get_optimizer(args.optimizer, model.parameters(), lr=args.lr, weight_decay=0.0)
    if not args.no_scheduler and args.scheduler:
        sched = get_improved_scheduler(
            opt,
            args.scheduler,
            warmup_steps=max(5, args.steps // 20),
            hold_steps=max(10, args.steps // 10),
            total_steps=args.steps,
            min_lr_ratio=0.1,
        )
    else:
        sched = None

    print('=' * 72)
    print('Single-batch line fit')
    print(f'- Optimizer: {args.optimizer}')
    print(f'- Scheduler: {args.scheduler if sched is not None else "none"}')
    print(f'- Steps: {args.steps}, LR: {args.lr}')
    print('=' * 72)

    # Train on the same batch repeatedly
    for t in range(1, args.steps + 1):
        pred = model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        opt.zero_grad()

        if t % max(1, args.steps // 10) == 0 or t == 1:
            a = model.weight.detach().item()
            b = model.bias.detach().item()
            lr_now = sched.get_last_lr()[0] if sched is not None else args.lr
            print(f'Step {t:4d} | loss={loss.item():.6f} | a={a:+.3f} b={b:+.3f} | lr={lr_now:.5g}')

    # Final summary
    final_pred = model(x)
    final_loss = F.mse_loss(final_pred, y).item()
    a = model.weight.detach().item()
    b = model.bias.detach().item()
    print('-' * 72)
    print(f'Final   | loss={final_loss:.6f} | a={a:+.3f} b={b:+.3f}')
    print('Target  | a=+3.000 b=+2.000')
    print('=' * 72)


if __name__ == '__main__':
    main()

