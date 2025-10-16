#!/usr/bin/env python3
"""
Improved Learning Rate Schedulers
Fixes the issue where learning rates get stuck after warmup
"""

import torch
import math
from torch.optim.lr_scheduler import _LRScheduler
from typing import List


class CosineAnnealingWarmRestarts(_LRScheduler):
    """
    Cosine annealing with warm restarts - better than standard cosine decay
    This prevents LR from getting stuck at 0 after warmup
    """
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super().__init__(optimizer, last_epoch)
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
    
    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        """Step could be called after every batch update"""
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)
        
        values = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group['lr'] = lr
        self._last_lr = values


class ImprovedLinearWarmupCosineDecay(_LRScheduler):
    """
    Linear warmup followed by cosine decay, with minimum LR to prevent getting stuck at 0
    """
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.05, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        else:
            # Cosine decay with minimum LR
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)  # Clamp to 1.0
            
            lrs = []
            for base_lr in self.base_lrs:
                min_lr = base_lr * self.min_lr_ratio
                cosine_lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                lrs.append(cosine_lr)
            return lrs


class CyclicalLR(_LRScheduler):
    """
    Cyclical learning rate with triangular policy
    Great for avoiding local minima and maintaining training momentum
    """
    def __init__(self, optimizer, base_lr, max_lr, step_size_up=2000, step_size_down=None,
                 mode='triangular', gamma=1., scale_fn=None, scale_mode='cycle', last_epoch=-1):
        self.base_lrs = [base_lr] * len(optimizer.param_groups)
        self.max_lrs = [max_lr] * len(optimizer.param_groups)
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down or step_size_up
        self.mode = mode
        self.gamma = gamma
        
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** x
        else:
            self.scale_fn = scale_fn
        self.scale_mode = scale_mode
        
        super().__init__(optimizer, last_epoch)
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
    
    def get_lr(self):
        cycle = math.floor(1 + self.last_epoch / (self.step_size_up + self.step_size_down))
        x = abs(self.last_epoch - self.step_size_up * cycle) / self.step_size_up
        
        lrs = []
        for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
            base_height = max(0, (1 - x)) * (max_lr - base_lr)
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_epoch)
            lrs.append(lr)
        return lrs


def get_improved_scheduler(optimizer, scheduler_type, **kwargs):
    """
    Factory function for improved schedulers
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler
        **kwargs: Scheduler-specific arguments
    
    Returns:
        Learning rate scheduler
    """
    
    if scheduler_type == "cosine_restart":
        T_0 = kwargs.get('T_0', 1000)
        T_mult = kwargs.get('T_mult', 2) 
        eta_min = kwargs.get('eta_min', 1e-7)
        return CosineAnnealingWarmRestarts(optimizer, T_0, T_mult, eta_min)
    
    elif scheduler_type == "linear_warmup_cosine":
        warmup_steps = kwargs.get('warmup_steps', 1000)
        total_steps = kwargs.get('total_steps', 10000)
        min_lr_ratio = kwargs.get('min_lr_ratio', 0.05)
        return ImprovedLinearWarmupCosineDecay(optimizer, warmup_steps, total_steps, min_lr_ratio)
    
    elif scheduler_type == "cyclical":
        base_lr = kwargs.get('base_lr', 1e-5)
        max_lr = kwargs.get('max_lr', 1e-3)
        step_size_up = kwargs.get('step_size_up', 2000)
        mode = kwargs.get('mode', 'triangular2')
        return CyclicalLR(optimizer, base_lr, max_lr, step_size_up, mode=mode)
    
    elif scheduler_type == "polynomial":
        total_steps = kwargs.get('total_steps', 10000)
        power = kwargs.get('power', 0.9)
        return torch.optim.lr_scheduler.PolynomialLR(optimizer, total_steps, power)
    
    elif scheduler_type in ("muon", "warmup_hold_cosine"):
        # Simple warmup -> hold -> cosine decay schedule often paired with
        # modern optimizers; keeps LR steady during the bulk of training.
        warmup_steps = kwargs.get('warmup_steps', 100)
        hold_steps = kwargs.get('hold_steps', 400)
        total_steps = kwargs.get('total_steps', warmup_steps + hold_steps + 500)
        min_lr_ratio = kwargs.get('min_lr_ratio', 0.05)

        class WarmupHoldCosine(_LRScheduler):
            def __init__(self, opt):
                self.warmup_steps = warmup_steps
                self.hold_steps = hold_steps
                self.total_steps = total_steps
                self.min_lr_ratio = min_lr_ratio
                super().__init__(opt)
                self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

            def get_lr(self):
                step = self.last_epoch
                lrs = []
                for base_lr in self.base_lrs:
                    if step < self.warmup_steps:
                        lr = base_lr * (step + 1) / max(1, self.warmup_steps)
                    elif step < self.warmup_steps + self.hold_steps:
                        lr = base_lr
                    else:
                        progress = (step - self.warmup_steps - self.hold_steps) / max(1, self.total_steps - self.warmup_steps - self.hold_steps)
                        progress = min(1.0, max(0.0, progress))
                        min_lr = base_lr * self.min_lr_ratio
                        lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                    lrs.append(lr)
                return lrs

        return WarmupHoldCosine(optimizer)
    
    else:
        # Fallback to PyTorch built-ins
        if scheduler_type == "cosine":
            T_max = kwargs.get('T_max', 10000)
            eta_min = kwargs.get('eta_min', 1e-7)
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min)
        elif scheduler_type == "step":
            step_size = kwargs.get('step_size', 1000)
            gamma = kwargs.get('gamma', 0.1)
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def get_adaptive_scheduler(optimizer, initial_lr, total_steps, warmup_steps=None):
    """
    Get an adaptive scheduler that automatically adjusts based on training progress
    This combines the best of multiple scheduling strategies
    """
    if warmup_steps is None:
        warmup_steps = total_steps // 20  # Default to 5% warmup
    
    # Use cosine with restarts for better convergence
    return CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=max(1000, total_steps // 10),  # Restart every 10% of training
        T_mult=2,  # Double restart interval each time
        eta_min=initial_lr * 0.01  # Never go below 1% of initial LR
    )


# Add warning import
import warnings
