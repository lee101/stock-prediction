#!/usr/bin/env python3
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from hftraining.modern_optimizers import get_optimizer
from hftraining.improved_schedulers import get_improved_scheduler


def make_line_data(n=512, noise=0.01, seed=123):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand((n, 1), generator=g) * 2 - 1  # [-1,1]
    y = 3.0 * x + 2.0
    if noise > 0:
        y = y + noise * torch.randn_like(y, generator=g)
    return x, y


def train_model(optimizer_name: str, scheduler_type: str = None, steps: int = 300, lr: float = 3e-2):
    x, y = make_line_data(n=256, noise=0.02)
    model = nn.Linear(1, 1)

    opt = get_optimizer(optimizer_name, model.parameters(), lr=lr, weight_decay=0.0)
    if scheduler_type is not None:
        sched = get_improved_scheduler(opt, scheduler_type, warmup_steps=25, hold_steps=50, total_steps=steps, min_lr_ratio=0.1)
    else:
        sched = None

    loss_hist = []
    for t in range(steps):
        pred = model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        opt.zero_grad()
        loss_hist.append(float(loss.item()))
    # Return final loss and learned params
    a = model.weight.detach().item()
    b = model.bias.detach().item()
    return loss_hist[-1], (a, b), loss_hist


def test_shampoo_linefit_converges():
    final_loss, (a, b), _ = train_model('shampoo', scheduler_type=None, steps=250, lr=0.05)
    # Should fit y ~ 3x+2 fairly well
    assert final_loss < 1e-2
    assert abs(a - 3.0) < 0.2
    assert abs(b - 2.0) < 0.2


def test_muon_scheduler_progression():
    # Verify the Muon-style scheduler produces warmup->hold->decay shape
    x, y = make_line_data(n=128, noise=0.02)
    model = nn.Linear(1, 1)
    opt = get_optimizer('adamw', model.parameters(), lr=1e-2, weight_decay=0.0)
    sched = get_improved_scheduler(opt, 'muon', warmup_steps=5, hold_steps=10, total_steps=40, min_lr_ratio=0.2)

    lrs = []
    for t in range(40):
        pred = model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad()
        lrs.append(sched.get_last_lr()[0])

    # LR should start small, rise during warmup, hold, then decay
    assert lrs[0] < lrs[4]  # warmup increasing
    assert abs(lrs[5] - lrs[10]) < 1e-10  # flat hold section
    assert lrs[-1] < lrs[15]  # decayed by the end

