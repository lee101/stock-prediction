#!/usr/bin/env python3
"""
HFTraining demo: compare Shampoo+Muon vs AdamW+Cosine on a toy task.

Runs a supervised single-batch line-fit (y = 3x + 2) for a small number
of steps and prints the loss curves and final parameters, so you can see
if Shampoo+Muon adds performance for your environment.

Usage:
  python hftraining/single_batch_shampoo_muon.py --steps 200
  python hftraining/single_batch_shampoo_muon.py --steps 300 --lr 0.03
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from hftraining.modern_optimizers import get_optimizer
from hftraining.improved_schedulers import get_improved_scheduler


def make_line_data(n=256, noise=0.02, seed=123):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand((n, 1), generator=g) * 2 - 1
    y = 3.0 * x + 2.0
    if noise > 0:
        y = y + noise * torch.randn_like(y, generator=g)
    return x, y


def run_once(optimizer_name: str, scheduler_name: str, steps: int, lr: float):
    x, y = make_line_data(n=256, noise=0.02, seed=123)
    model = nn.Linear(1, 1)
    opt = get_optimizer(optimizer_name, model.parameters(), lr=lr, weight_decay=0.0)
    sched = None
    if scheduler_name:
        sched = get_improved_scheduler(
            opt,
            scheduler_name,
            warmup_steps=max(5, steps // 20),
            hold_steps=max(10, steps // 10),
            total_steps=steps,
            min_lr_ratio=0.1,
        )

    losses = []
    for t in range(steps):
        pred = model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        opt.zero_grad()
        losses.append(float(loss.item()))
    a = model.weight.detach().item()
    b = model.bias.detach().item()
    return losses, a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--lr', type=float, default=5e-2)
    args = ap.parse_args()

    # Baseline: AdamW + cosine restart
    base_losses, base_a, base_b = run_once('adamw', 'cosine_restart', args.steps, max(1e-3, args.lr / 5))
    # Candidate: Shampoo + muon (warmup-hold-cosine)
    shp_losses, shp_a, shp_b = run_once('shampoo', 'muon', args.steps, args.lr)

    def summarise(name, losses, a, b):
        return {
            'name': name,
            'loss_start': losses[0],
            'loss_mid': losses[len(losses)//2],
            'loss_end': losses[-1],
            'a': a,
            'b': b,
        }

    s_base = summarise('adamw+cos_restart', base_losses, base_a, base_b)
    s_shp = summarise('shampoo+muon', shp_losses, shp_a, shp_b)

    print('=' * 72)
    print('Single-batch line fit comparison (y = 3x + 2)')
    for s in (s_base, s_shp):
        print(f"{s['name']:<20} | loss: start={s['loss_start']:.5f} mid={s['loss_mid']:.5f} end={s['loss_end']:.5f} | a={s['a']:+.3f} b={s['b']:+.3f}")
    # Simple verdict
    better = 'shampoo+muon' if s_shp['loss_end'] < s_base['loss_end'] else 'adamw+cos_restart'
    print('-' * 72)
    print(f"Lower final loss: {better}")
    print('=' * 72)


if __name__ == '__main__':
    main()

