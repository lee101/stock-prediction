#!/usr/bin/env python3
"""
HuggingFace-style Training Script with Modern Optimizers
Implements GPro, AdamW, and other state-of-the-art algorithms
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import math
import random
from collections import deque


# ============================================================================
# MODERN OPTIMIZERS
# ============================================================================

class GPro(torch.optim.Optimizer):
    """
    GPro Optimizer - Gradient Projection with adaptive preconditioning
    State-of-the-art optimizer combining momentum and adaptive learning
    """
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0.01, amsgrad=False, projection_factor=0.5):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, 
                       amsgrad=amsgrad, projection_factor=projection_factor)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data).float()
                    state['exp_avg_sq'] = torch.zeros_like(p.data).float()
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data).float()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Add weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Update exponential moving averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if group['amsgrad']:
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                # GPro gradient projection
                projected_grad = exp_avg / denom
                
                # Apply projection factor for stability
                projection_norm = projected_grad.norm()
                if projection_norm > group['projection_factor']:
                    projected_grad = projected_grad * (group['projection_factor'] / projection_norm)

                p.data.add_(projected_grad, alpha=-step_size)

        return loss


class AdamW(torch.optim.Optimizer):
    """
    AdamW Optimizer - Adam with decoupled weight decay
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                step_size = group['lr'] / bias_correction1

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                # Apply weight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Apply update
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

@dataclass
class HFTrainingConfig:
    """HuggingFace-style training configuration"""
    
    # Model parameters
    model_name: str = "transformer_trading_agent"
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 16
    dropout: float = 0.1
    
    # Training parameters
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    max_steps: int = 50000
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    # Stability: Adaptive Gradient Clipping (AGC) and guards
    use_adaptive_grad_clip: bool = False
    agc_clip_factor: float = 0.01
    agc_eps: float = 1e-3
    skip_non_finite_grads: bool = True
    
    # Optimizer settings
    optimizer_name: str = "gpro"  # gpro, adamw, adam
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    use_fused_optimizer: bool = True
    
    # Training dynamics
    batch_size: int = 32
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    
    # DataLoader
    dataloader_num_workers: int = 2
    persistent_workers: bool = True
    prefetch_factor: int = 2
    
    # Data parameters
    sequence_length: int = 60
    prediction_horizon: int = 5
    quantile_levels: Optional[Tuple[float, ...]] = None
    max_tokens_per_batch: int = 0
    length_bucketing: Tuple[int, ...] = (60,)
    horizon_bucketing: Tuple[int, ...] = (5,)
    window_stride: int = 1
    pack_windows: bool = True
    bucket_warmup_steps: int = 0
    
    # Advanced features
    use_mixed_precision: bool = True
    use_bfloat16: bool = True
    precision: str = "bf16"
    use_compile: bool = False
    allow_tf32: bool = True
    use_gradient_checkpointing: bool = True
    use_data_parallel: bool = True
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_adamw_lr: Optional[float] = None
    
    # Regularization
    label_smoothing: float = 0.1
    dropout_rate: float = 0.1
    layer_norm_eps: float = 1e-12
    profit_loss_weight: float = 0.0
    transaction_cost_bps: float = 10.0
    
    # Light data augmentation (normalized inputs)
    input_noise_std: float = 0.001
    input_noise_prob: float = 0.5
    input_noise_clip: float = 0.02
    
    # Directories
    output_dir: str = "hftraining/output"
    logging_dir: str = "hftraining/logs"
    cache_dir: str = "hftraining/cache"

    # Experiment tracking
    use_wandb: bool = field(
        default_factory=lambda: os.getenv("WANDB_DISABLED", "0").lower() not in {"1", "true", "yes"}
    )
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_notes: Optional[str] = None
    wandb_tags: Tuple[str, ...] = field(default_factory=tuple)
    wandb_mode: str = "auto"
    wandb_settings: Optional[Dict[str, Any]] = None
    tensorboard_subdir: Optional[str] = None
    
    # Evaluation
    evaluation_strategy: str = "steps"
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    load_best_model_at_end: bool = True
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.0001

    # Auto-tuning (optional)
    auto_tune: bool = False
    target_effective_batch_size: Optional[int] = None
    max_gradient_accumulation: int = 16
    tuning_steps: int = 10


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class TransformerTradingModel(nn.Module):
    """
    Transformer-based trading model with modern architecture
    """
    def __init__(self, config: HFTrainingConfig, input_dim: int = 50):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, config.hidden_size)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            config.hidden_size, 
            max_len=config.sequence_length
        )
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )
        
        # Output heads
        self.action_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size // 2, 3)  # buy, hold, sell
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size // 2, 1)
        )
        
        self.price_prediction_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size // 2, config.prediction_horizon)
        )

        self.allocation_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier/Glorot initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x, attention_mask=None):
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            attention_mask: Optional attention mask
            
        Returns:
            Dictionary with action_logits, value, and price_predictions
        """
        # Project input to hidden dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer
        if attention_mask is not None:
            # Convert attention mask to transformer format
            attention_mask = attention_mask.bool()
            attention_mask = ~attention_mask  # Invert for transformer
        
        # Enable gradient checkpointing by applying layers manually
        if getattr(self.config, 'use_gradient_checkpointing', False) and self.training:
            out = x
            for layer in self.transformer.layers:
                if attention_mask is not None:
                    if getattr(self.config, 'use_gradient_checkpointing', False):
                        out = cp.checkpoint(lambda y: layer(y, src_key_padding_mask=attention_mask), out)
                    else:
                        out = layer(out, src_key_padding_mask=attention_mask)
                else:
                    out = cp.checkpoint(lambda y: layer(y), out)
            transformer_output = out
            # Apply final norm if present
            if self.transformer.norm is not None:
                transformer_output = self.transformer.norm(transformer_output)
        else:
            transformer_output = self.transformer(x, src_key_padding_mask=attention_mask)
        
        # Use the last token for predictions
        last_hidden = transformer_output[:, -1, :]
        
        # Generate outputs
        action_logits = self.action_head(last_hidden)
        value = self.value_head(last_hidden)
        price_predictions = self.price_prediction_head(last_hidden)
        allocations = torch.tanh(self.allocation_head(last_hidden))
        
        return {
            'action_logits': action_logits,
            'value': value.squeeze(-1),
            'price_predictions': price_predictions,
            'allocations': allocations.squeeze(-1),
            'hidden_states': transformer_output
        }


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].transpose(0, 1)


# ============================================================================
# LEARNING RATE SCHEDULER
# ============================================================================

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    """
    
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.
    """
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


# ============================================================================
# MIXED PRECISION TRAINING
# ============================================================================

class MixedPrecisionTrainer:
    """Mixed precision training utilities"""
    
    def __init__(self, enabled=True, dtype: Optional[torch.dtype] = None):
        # Only enable if CUDA is available; CPU/BF16 support varies, keep safe
        self.enabled = bool(enabled and torch.cuda.is_available())
        self.dtype = dtype if self.enabled else None
        if self.enabled:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def scale_loss(self, loss):
        if self.enabled:
            return self.scaler.scale(loss)
        return loss
    
    def step_optimizer(self, optimizer):
        if self.enabled:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def autocast(self):
        if self.enabled:
            if self.dtype is not None:
                return torch.cuda.amp.autocast(dtype=self.dtype)
            return torch.cuda.amp.autocast()
        # Return a dummy context manager that does nothing
        from contextlib import nullcontext
        return nullcontext()


# ============================================================================
# GRADIENT UTILITIES (AGC, guards)
# ============================================================================

def _unitwise_norm(t: torch.Tensor) -> torch.Tensor:
    """Compute unit-wise norms for tensors of different shapes.
    For Linear/Conv weights, compute norm over all dims except the first (out_features/filters).
    For biases/LayerNorm weights (1D), fall back to absolute value.
    """
    if t.ndim <= 1:
        return t.abs()
    # Norm over all dimensions except dim 0
    dims = tuple(range(1, t.ndim))
    return t.norm(p=2, dim=dims, keepdim=True)


@torch.no_grad()
def adaptive_clip_grad_(parameters, clip_factor: float = 0.01, eps: float = 1e-3):
    """Adaptive Gradient Clipping (AGC).
    Scales gradients so that ||g_i|| <= clip_factor * (||w_i|| + eps) per unit (row/channel).

    Args:
        parameters: Iterable of model parameters with .grad populated
        clip_factor: Multiplicative factor against parameter unit-wise norm
        eps: Small epsilon to avoid division by zero
    """
    for p in parameters:
        if p.grad is None:
            continue
        g = p.grad
        if g.is_sparse:
            # Skip sparse to avoid surprises; uncommon here
            continue
        # Work in fp32 for stability
        g_fp32 = g.detach()
        if g_fp32.dtype in {torch.float16, torch.bfloat16}:
            g_fp32 = g_fp32.float()
        w = p.detach()
        if w.dtype in {torch.float16, torch.bfloat16}:
            w = w.float()

        w_norm = _unitwise_norm(w).add_(eps)
        g_norm = _unitwise_norm(g_fp32)

        max_norm = w_norm.mul(clip_factor)
        # Compute scaling where gradient norm exceeds threshold
        clipped = g_fp32 * (max_norm / torch.clamp(g_norm, min=1e-12))
        mask = (g_norm > max_norm).to(g_fp32.dtype)
        # Broadcast-safe blend
        g_fp32 = clipped * mask + g_fp32 * (1 - mask)

        # Write back in-place, preserving original dtype
        g.copy_(g_fp32.to(g.dtype))


# ============================================================================
# EARLY STOPPING
# ============================================================================

class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience=10, threshold=0.0001, greater_is_better=False):
        self.patience = patience
        self.threshold = threshold
        self.greater_is_better = greater_is_better
        self.best_score = None
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
    
    def _is_better(self, score, best_score):
        if self.greater_is_better:
            return score > best_score + self.threshold
        else:
            return score < best_score - self.threshold
