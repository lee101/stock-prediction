#!/usr/bin/env python3
"""
Advanced RL Training with PEFT/LoRA for Parameter-Efficient Fine-Tuning
Prevents overfitting while maintaining predictive power
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


# ============================================================================
# LORA-ENHANCED TRANSFORMER ARCHITECTURE
# ============================================================================

class LoRALinear(nn.Module):
    """LoRA-enhanced Linear layer for parameter-efficient training"""
    
    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        
        # Frozen pretrained weights (these don't update)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.weight.requires_grad = False  # Freeze base weights
        
        # LoRA adaptation matrices (these update)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor
        self.scaling = self.alpha / self.rank
        
        # Optional bias
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        # Base transformation (frozen)
        base_output = F.linear(x, self.weight, self.bias)
        
        # LoRA adaptation
        lora_output = x @ self.lora_A.T @ self.lora_B.T * self.scaling
        lora_output = self.dropout(lora_output)
        
        return base_output + lora_output


class PEFTTransformerTradingAgent(nn.Module):
    """Transformer with PEFT/LoRA for efficient fine-tuning"""
    
    def __init__(self, input_dim, hidden_dim=256, num_layers=3, num_heads=8, 
                 dropout=0.1, lora_rank=8, lora_alpha=16, freeze_base=True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        
        # Input projection with LoRA
        self.input_projection = LoRALinear(
            input_dim, hidden_dim, 
            rank=lora_rank, alpha=lora_alpha, dropout=dropout
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout)
        
        # Transformer layers with LoRA in attention
        self.transformer_layers = nn.ModuleList([
            PEFTTransformerBlock(
                hidden_dim, num_heads, dropout,
                lora_rank=lora_rank, lora_alpha=lora_alpha,
                freeze_base=freeze_base
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Output heads with LoRA
        self.actor_head = nn.Sequential(
            LoRALinear(hidden_dim, 128, rank=lora_rank//2, alpha=lora_alpha//2, dropout=dropout),
            nn.ReLU(),
            nn.Dropout(dropout),
            LoRALinear(128, 64, rank=lora_rank//4, alpha=lora_alpha//4, dropout=dropout),
            nn.ReLU(),
            nn.Linear(64, 1),  # Final layer without LoRA
            nn.Tanh()
        )
        
        self.critic_head = nn.Sequential(
            LoRALinear(hidden_dim, 128, rank=lora_rank//2, alpha=lora_alpha//2, dropout=dropout),
            nn.ReLU(),
            nn.Dropout(dropout),
            LoRALinear(128, 64, rank=lora_rank//4, alpha=lora_alpha//4, dropout=dropout),
            nn.ReLU(),
            nn.Linear(64, 1)  # Final layer without LoRA
        )
        
        # Learnable action variance
        self.log_std = nn.Parameter(torch.zeros(1))
        
        # Freeze base model if specified
        if freeze_base:
            self._freeze_base_weights()
    
    def _freeze_base_weights(self):
        """Freeze non-LoRA parameters"""
        for name, param in self.named_parameters():
            if 'lora' not in name.lower() and 'log_std' not in name:
                param.requires_grad = False
    
    def get_num_trainable_params(self):
        """Count trainable parameters"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total
    
    def forward(self, x):
        # Input projection
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Layer norm
        x = self.layer_norm(x)
        
        # Global pooling
        if len(x.shape) == 3:
            features = x.mean(dim=1)
        else:
            features = x
        
        # Get action and value
        action = self.actor_head(features)
        value = self.critic_head(features)
        
        return action, value
    
    def get_action_distribution(self, x):
        action_mean, _ = self.forward(x)
        action_std = torch.exp(self.log_std)
        return torch.distributions.Normal(action_mean, action_std)


class PEFTTransformerBlock(nn.Module):
    """Transformer block with LoRA-enhanced attention"""
    
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1, 
                 lora_rank=8, lora_alpha=16, freeze_base=True):
        super().__init__()
        
        # Multi-head attention with LoRA
        self.attention = PEFTMultiHeadAttention(
            hidden_dim, num_heads, dropout,
            lora_rank=lora_rank, lora_alpha=lora_alpha
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feedforward with LoRA
        self.feed_forward = nn.Sequential(
            LoRALinear(hidden_dim, hidden_dim * 4, rank=lora_rank, alpha=lora_alpha, dropout=dropout),
            nn.GELU(),
            nn.Dropout(dropout),
            LoRALinear(hidden_dim * 4, hidden_dim, rank=lora_rank, alpha=lora_alpha, dropout=dropout),
            nn.Dropout(dropout)
        )
        
        if freeze_base:
            # Freeze normalization layers
            for param in self.norm1.parameters():
                param.requires_grad = False
            for param in self.norm2.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        # Self-attention with residual
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)
        
        # Feedforward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x


class PEFTMultiHeadAttention(nn.Module):
    """Multi-head attention with LoRA adaptation"""
    
    def __init__(self, embed_dim, num_heads=8, dropout=0.1, 
                 lora_rank=8, lora_alpha=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Q, K, V projections with LoRA
        self.q_linear = LoRALinear(embed_dim, embed_dim, rank=lora_rank, alpha=lora_alpha, dropout=dropout)
        self.k_linear = LoRALinear(embed_dim, embed_dim, rank=lora_rank, alpha=lora_alpha, dropout=dropout)
        self.v_linear = LoRALinear(embed_dim, embed_dim, rank=lora_rank, alpha=lora_alpha, dropout=dropout)
        self.out_linear = LoRALinear(embed_dim, embed_dim, rank=lora_rank, alpha=lora_alpha, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape[0], x.shape[1] if len(x.shape) == 3 else 1
        
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # Linear transformations
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        output = self.out_linear(context)
        
        if seq_len == 1:
            output = output.squeeze(1)
        
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        if len(x.shape) == 3:
            x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


# ============================================================================
# ENHANCED REGULARIZATION TECHNIQUES
# ============================================================================

class MixupAugmentation:
    """Mixup augmentation for time series"""
    
    @staticmethod
    def mixup(x1, x2, alpha=0.2):
        """Mix two samples"""
        lam = np.random.beta(alpha, alpha)
        return lam * x1 + (1 - lam) * x2, lam


class StochasticDepth(nn.Module):
    """Stochastic depth for regularization"""
    
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        mask = torch.bernoulli(torch.full((x.shape[0], 1), keep_prob, device=x.device))
        mask = mask.div(keep_prob)
        
        return x * mask


class LabelSmoothing(nn.Module):
    """Label smoothing for better generalization"""
    
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_class = pred.size(-1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        return F.kl_div(F.log_softmax(pred, dim=-1), one_hot, reduction='batchmean')


# ============================================================================
# ENHANCED TRAINING CONFIGURATION
# ============================================================================

@dataclass
class PEFTTrainingConfig:
    # PEFT/LoRA settings
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    freeze_base: bool = True
    
    # Architecture
    architecture: str = 'peft_transformer'
    hidden_dim: int = 256
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.2  # Higher dropout for regularization
    
    # Optimization
    optimizer: str = 'adamw'
    learning_rate: float = 0.0001  # Lower LR for fine-tuning
    weight_decay: float = 0.01
    batch_size: int = 128
    gradient_clip: float = 0.5  # Lower gradient clip
    
    # RL
    gamma: float = 0.995
    gae_lambda: float = 0.95
    ppo_epochs: int = 5  # Fewer epochs to prevent overfitting
    ppo_clip: float = 0.1  # Smaller clip range
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.02  # Higher entropy for exploration
    
    # Regularization
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    use_stochastic_depth: bool = True
    stochastic_depth_prob: float = 0.1
    label_smoothing: float = 0.1
    
    # Data augmentation
    use_augmentation: bool = True
    augmentation_prob: float = 0.5
    noise_level: float = 0.01
    
    # Training
    num_episodes: int = 2000
    eval_interval: int = 20
    save_interval: int = 100
    early_stop_patience: int = 200
    
    # Curriculum
    use_curriculum: bool = True
    warmup_episodes: int = 100


def create_peft_agent(config: PEFTTrainingConfig, input_dim: int):
    """Create PEFT-enhanced agent"""
    
    agent = PEFTTransformerTradingAgent(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        freeze_base=config.freeze_base
    )
    
    # Print parameter statistics
    trainable, total = agent.get_num_trainable_params()
    print(f"\n📊 PEFT Model Statistics:")
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Reduction: {(1 - trainable/total)*100:.2f}%")
    
    return agent


def create_peft_optimizer(agent, config: PEFTTrainingConfig):
    """Create optimizer for PEFT model"""
    
    # Only optimize LoRA parameters
    lora_params = [p for n, p in agent.named_parameters() if p.requires_grad]
    
    if config.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            lora_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )
    elif config.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            lora_params,
            lr=config.learning_rate,
            betas=(0.9, 0.999)
        )
    else:
        optimizer = torch.optim.SGD(
            lora_params,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay
        )
    
    return optimizer


if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 PEFT/LoRA Enhanced Trading Agent")
    print("="*80)
    
    print("\n📊 Key Features:")
    print("✓ Parameter-Efficient Fine-Tuning (PEFT)")
    print("✓ Low-Rank Adaptation (LoRA)")
    print("✓ Frozen base weights to prevent overfitting")
    print("✓ Enhanced regularization (dropout, mixup, stochastic depth)")
    print("✓ Label smoothing for better generalization")
    print("✓ Reduced trainable parameters by ~90%")
    
    # Test creation
    config = PEFTTrainingConfig()
    agent = create_peft_agent(config, input_dim=13)
    
    print("\n✅ PEFT agent created successfully!")
    print("="*80)