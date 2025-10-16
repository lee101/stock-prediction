#!/usr/bin/env python3
"""
Modern Transformer-based Trading Agent with HuggingFace Best Practices
Addresses overfitting through proper scaling, regularization, and modern techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import math
from collections import deque
import random


# ============================================================================
# MODERN TRANSFORMER ARCHITECTURE WITH PROPER SCALING
# ============================================================================

class ModernTransformerConfig:
    """Configuration for modern transformer with appropriate scaling"""
    def __init__(
        self,
        # Model architecture - MUCH smaller to prevent overfitting
        d_model: int = 128,  # Reduced from 256
        n_heads: int = 4,    # Reduced from 8
        n_layers: int = 2,   # Reduced from 3
        d_ff: int = 256,     # 2x d_model instead of 4x
        
        # Regularization - MUCH stronger
        dropout: float = 0.4,         # Increased from 0.1-0.2
        attention_dropout: float = 0.3,
        path_dropout: float = 0.2,    # Stochastic depth
        layer_drop: float = 0.1,      # Layer dropout
        
        # Input/output
        input_dim: int = 13,
        action_dim: int = 1,
        
        # Training hyperparameters
        max_position_embeddings: int = 100,
        layer_norm_eps: float = 1e-6,
        
        # Advanced regularization
        weight_decay: float = 0.01,
        label_smoothing: float = 0.1,
        gradient_checkpointing: bool = True,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.path_dropout = path_dropout
        self.layer_drop = layer_drop
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.gradient_checkpointing = gradient_checkpointing


class RMSNorm(nn.Module):
    """RMS Normalization (modern alternative to LayerNorm)"""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) - modern positional encoding"""
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
        
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Applies Rotary Position Embedding to the query and key tensors."""
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class ModernMultiHeadAttention(nn.Module):
    """Modern multi-head attention with RoPE, flash attention patterns, and proper scaling"""
    
    def __init__(self, config: ModernTransformerConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = self.d_model // self.n_heads
        
        assert self.d_model % self.n_heads == 0
        
        # Use grouped query attention pattern (more efficient)
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        
        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.max_position_embeddings)
        
        # Attention dropout
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        
        # Scale factor
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        cos, sin = self.rotary_emb(v, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.o_proj(out)
        
        return out, attn_weights


class ModernFeedForward(nn.Module):
    """Modern feed-forward with SwiGLU activation (used in modern LLMs)"""
    
    def __init__(self, config: ModernTransformerConfig):
        super().__init__()
        self.config = config
        
        # SwiGLU requires 3 linear layers instead of 2
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        # SwiGLU: silu(gate) * up
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        intermediate = gate * up
        intermediate = self.dropout(intermediate)
        return self.down_proj(intermediate)


class StochasticDepth(nn.Module):
    """Stochastic Depth for regularization (drops entire layers randomly)"""
    
    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x, residual):
        if not self.training:
            return x + residual
        
        keep_prob = 1 - self.drop_prob
        if torch.rand(1).item() > keep_prob:
            return residual  # Skip the layer completely
        else:
            return x + residual


class ModernTransformerLayer(nn.Module):
    """Modern transformer layer with RMSNorm, SwiGLU, and stochastic depth"""
    
    def __init__(self, config: ModernTransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Pre-normalization (modern approach)
        self.input_layernorm = RMSNorm(config.d_model, config.layer_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.d_model, config.layer_norm_eps)
        
        # Attention and feed-forward
        self.self_attn = ModernMultiHeadAttention(config)
        self.mlp = ModernFeedForward(config)
        
        # Stochastic depth (layer dropout)
        # Increase drop probability linearly with depth
        layer_drop_prob = config.layer_drop * (layer_idx / config.n_layers)
        self.stochastic_depth = StochasticDepth(layer_drop_prob)
        
        # Path dropout (different from regular dropout)
        self.path_dropout = nn.Dropout(config.path_dropout)
        
    def forward(self, x, attention_mask=None):
        # Pre-norm attention
        residual = x
        x = self.input_layernorm(x)
        attn_out, attn_weights = self.self_attn(x, attention_mask)
        attn_out = self.path_dropout(attn_out)
        x = self.stochastic_depth(attn_out, residual)
        
        # Pre-norm feed-forward
        residual = x
        x = self.post_attention_layernorm(x)
        ff_out = self.mlp(x)
        ff_out = self.path_dropout(ff_out)
        x = self.stochastic_depth(ff_out, residual)
        
        return x, attn_weights


class ModernTransformerTradingAgent(nn.Module):
    """Modern transformer trading agent with proper scaling and regularization"""
    
    def __init__(self, config: ModernTransformerConfig):
        super().__init__()
        self.config = config
        
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(config.input_dim, config.d_model),
            nn.Dropout(config.dropout)
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            ModernTransformerLayer(config, i) for i in range(config.n_layers)
        ])
        
        # Final norm
        self.norm = RMSNorm(config.d_model, config.layer_norm_eps)
        
        # Output heads with proper initialization
        self.actor_head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.action_dim),
            nn.Tanh()
        )
        
        self.critic_head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 1)
        )
        
        # Learnable action variance
        self.log_std = nn.Parameter(torch.zeros(config.action_dim))
        
        # Initialize weights properly
        self.apply(self._init_weights)
        
        # Gradient checkpointing for memory efficiency
        if config.gradient_checkpointing:
            self.gradient_checkpointing_enable()
    
    def _init_weights(self, module):
        """Proper weight initialization following modern practices"""
        if isinstance(module, nn.Linear):
            # Xavier/Glorot initialization for linear layers
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency"""
        for layer in self.layers:
            layer._use_gradient_checkpointing = True
    
    def forward(self, x, attention_mask=None):
        """Forward pass through the transformer"""
        # Handle different input shapes
        if len(x.shape) == 2:
            # (batch_size, seq_len * features) -> (batch_size, seq_len, features)
            batch_size = x.shape[0]
            seq_len = x.shape[1] // self.config.input_dim
            x = x.view(batch_size, seq_len, self.config.input_dim)
        
        # Input embedding
        x = self.input_embedding(x)
        
        # Through transformer layers
        all_attentions = []
        for layer in self.layers:
            if hasattr(layer, '_use_gradient_checkpointing') and self.training:
                try:
                    from torch.utils.checkpoint import checkpoint
                    x, attn_weights = checkpoint(layer, x, attention_mask, use_reentrant=False)
                except (ImportError, AttributeError):
                    # Fallback to regular forward pass if checkpointing is not available
                    x, attn_weights = layer(x, attention_mask)
            else:
                x, attn_weights = layer(x, attention_mask)
            all_attentions.append(attn_weights)
        
        # Final normalization
        x = self.norm(x)
        
        # Global pooling (mean over sequence dimension)
        pooled = x.mean(dim=1)
        
        # Get action and value
        action_mean = self.actor_head(pooled)
        value = self.critic_head(pooled)
        
        return action_mean, value, all_attentions
    
    def get_action_distribution(self, x, attention_mask=None):
        """Get action distribution for sampling"""
        action_mean, _, _ = self.forward(x, attention_mask)
        action_std = torch.exp(self.log_std)
        return torch.distributions.Normal(action_mean, action_std)
    
    def get_num_parameters(self):
        """Get number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# MODERN TRAINING CONFIGURATION
# ============================================================================

@dataclass
class ModernTrainingConfig:
    """Modern training configuration with proper scaling"""
    
    # Model architecture
    model_config: ModernTransformerConfig = None
    
    # Training hyperparameters - MUCH LOWER learning rates
    learning_rate: float = 5e-5        # Much lower, following modern practices
    min_learning_rate: float = 1e-6    # Minimum LR for scheduler
    weight_decay: float = 0.01         # Proper weight decay
    beta1: float = 0.9
    beta2: float = 0.95                # Higher beta2 for stability
    eps: float = 1e-8
    
    # Batch sizes - larger with gradient accumulation
    batch_size: int = 32               # Smaller physical batch
    gradient_accumulation_steps: int = 8  # Effective batch = 32 * 8 = 256
    max_grad_norm: float = 1.0         # Gradient clipping
    
    # Scheduler
    scheduler_type: str = "cosine_with_restarts"  # or "linear_warmup"
    warmup_ratio: float = 0.1          # 10% warmup
    num_training_steps: int = 10000    # Total training steps
    num_cycles: float = 1.0            # For cosine with restarts
    
    # RL specific
    gamma: float = 0.995
    gae_lambda: float = 0.95
    ppo_epochs: int = 4                # Fewer epochs to prevent overfitting
    ppo_clip: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # Training control
    num_episodes: int = 5000           # More episodes for better training
    eval_interval: int = 50            # More frequent evaluation
    save_interval: int = 200
    
    # Early stopping
    patience: int = 300                # Early stopping patience
    min_improvement: float = 0.001     # Minimum improvement threshold
    
    # Data scaling
    train_data_size: int = 10000       # 10x more data
    synthetic_noise: float = 0.02      # More varied synthetic data
    
    # Regularization
    use_mixup: bool = True
    mixup_alpha: float = 0.4
    label_smoothing: float = 0.1
    
    def __post_init__(self):
        if self.model_config is None:
            self.model_config = ModernTransformerConfig()


# ============================================================================
# MODERN PPO TRAINER WITH SCALED TRAINING
# ============================================================================

class ModernPPOTrainer:
    """Modern PPO trainer with proper scaling and regularization"""
    
    def __init__(self, config: ModernTrainingConfig, device='cuda'):
        self.config = config
        self.device = device
        
        # Create model
        self.model = ModernTransformerTradingAgent(config.model_config).to(device)
        
        print(f"\n🤖 Model created with {self.model.get_num_parameters():,} parameters")
        
        # Optimizer with proper settings
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        if config.scheduler_type == "cosine_with_restarts":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=int(config.num_training_steps * config.warmup_ratio),
                num_training_steps=config.num_training_steps,
                num_cycles=config.num_cycles
            )
        else:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=int(config.num_training_steps * config.warmup_ratio),
                num_training_steps=config.num_training_steps
            )
        
        # TensorBoard logging
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(f'traininglogs/modern_{timestamp}')
        self.global_step = 0
        self.episode_num = 0
        
        # Training state
        self.best_performance = -float('inf')
        self.patience_counter = 0
        self.training_metrics = {
            'episode_rewards': [],
            'episode_profits': [],
            'episode_sharpes': [],
            'actor_losses': [],
            'critic_losses': [],
            'learning_rates': []
        }
        
        # Gradient accumulation
        self.accumulation_counter = 0
    
    def select_action(self, state, deterministic=False):
        """Select action using the model"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            dist = self.model.get_action_distribution(state_tensor)
            if deterministic:
                action = dist.mean
            else:
                action = dist.sample()
            
            action_mean, value, _ = self.model(state_tensor)
            
            return action.cpu().numpy()[0], value.cpu().item()
    
    def compute_gae(self, rewards, values, dones, next_value):
        """Generalized Advantage Estimation with proper scaling"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def mixup_batch(self, states, actions, advantages, returns):
        """Apply mixup augmentation"""
        if not self.config.use_mixup or len(states) < 2:
            return states, actions, advantages, returns
        
        batch_size = len(states)
        indices = torch.randperm(batch_size)
        
        lam = np.random.beta(self.config.mixup_alpha, self.config.mixup_alpha)
        
        mixed_states = lam * states + (1 - lam) * states[indices]
        mixed_actions = lam * actions + (1 - lam) * actions[indices]
        mixed_advantages = lam * advantages + (1 - lam) * advantages[indices]
        mixed_returns = lam * returns + (1 - lam) * returns[indices]
        
        return mixed_states, mixed_actions, mixed_advantages, mixed_returns
    
    def update_policy(self, states, actions, old_log_probs, advantages, returns):
        """PPO policy update with gradient accumulation"""
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Apply mixup augmentation
        if self.config.use_mixup:
            states, actions, advantages, returns = self.mixup_batch(
                states, actions, advantages, returns
            )
        
        total_loss = 0
        total_actor_loss = 0
        total_critic_loss = 0
        
        for epoch in range(self.config.ppo_epochs):
            # Get current predictions
            dist = self.model.get_action_distribution(states)
            action_mean, values, _ = self.model(states)
            values = values.squeeze()
            
            # Compute log probabilities
            log_probs = dist.log_prob(actions).sum(dim=-1)
            
            # PPO loss
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.ppo_clip, 1 + self.config.ppo_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss with clipping
            value_loss_unclipped = F.mse_loss(values, returns)
            value_loss = value_loss_unclipped  # Can add value clipping here if needed
            
            # Entropy bonus
            entropy = dist.entropy().mean()
            
            # Total loss
            loss = (
                actor_loss + 
                self.config.value_loss_coef * value_loss - 
                self.config.entropy_coef * entropy
            )
            
            # Scale loss by gradient accumulation steps
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            self.accumulation_counter += 1
            
            # Update only after accumulating enough gradients
            if self.accumulation_counter % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Log learning rate
                current_lr = self.scheduler.get_last_lr()[0]
                self.writer.add_scalar('Training/LearningRate', current_lr, self.global_step)
                self.training_metrics['learning_rates'].append(current_lr)
                
                self.global_step += 1
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            total_actor_loss += actor_loss.item()
            total_critic_loss += value_loss.item()
        
        # Average losses
        avg_loss = total_loss / self.config.ppo_epochs
        avg_actor_loss = total_actor_loss / self.config.ppo_epochs
        avg_critic_loss = total_critic_loss / self.config.ppo_epochs
        
        # Log metrics
        self.training_metrics['actor_losses'].append(avg_actor_loss)
        self.training_metrics['critic_losses'].append(avg_critic_loss)
        
        self.writer.add_scalar('Loss/Actor', avg_actor_loss, self.global_step)
        self.writer.add_scalar('Loss/Critic', avg_critic_loss, self.global_step)
        self.writer.add_scalar('Loss/Total', avg_loss, self.global_step)
        self.writer.add_scalar('Loss/Entropy', entropy.item(), self.global_step)
        
        return avg_loss
    
    def train_episode(self, env, max_steps=1000):
        """Train one episode with modern techniques"""
        state = env.reset()
        
        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
        
        episode_reward = 0
        episode_steps = 0
        
        for step in range(max_steps):
            action, value = self.select_action(state)
            
            next_state, reward, done, info = env.step([action])
            
            # Store experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            dones.append(done)
            
            # Compute log prob for PPO
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                dist = self.model.get_action_distribution(state_tensor)
                log_prob = dist.log_prob(torch.FloatTensor([action]).to(self.device)).cpu().item()
            log_probs.append(log_prob)
            
            episode_reward += reward
            episode_steps += 1
            state = next_state
            
            if done:
                break
        
        # Compute advantages and returns
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            _, next_value, _ = self.model(next_state_tensor)
            next_value = next_value.cpu().item()
        
        advantages = self.compute_gae(rewards, values, dones, next_value)
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        # Update policy
        if len(states) > 0:
            loss = self.update_policy(states, actions, log_probs, advantages, returns)
        
        # Track metrics
        self.training_metrics['episode_rewards'].append(episode_reward)
        
        if hasattr(env, 'get_metrics'):
            metrics = env.get_metrics()
            self.training_metrics['episode_profits'].append(metrics.get('total_return', 0))
            self.training_metrics['episode_sharpes'].append(metrics.get('sharpe_ratio', 0))
            
            # Log episode metrics
            self.writer.add_scalar('Episode/Reward', episode_reward, self.episode_num)
            self.writer.add_scalar('Episode/TotalReturn', metrics.get('total_return', 0), self.episode_num)
            self.writer.add_scalar('Episode/SharpeRatio', metrics.get('sharpe_ratio', 0), self.episode_num)
            self.writer.add_scalar('Episode/MaxDrawdown', metrics.get('max_drawdown', 0), self.episode_num)
            self.writer.add_scalar('Episode/NumTrades', metrics.get('num_trades', 0), self.episode_num)
            self.writer.add_scalar('Episode/WinRate', metrics.get('win_rate', 0), self.episode_num)
            self.writer.add_scalar('Episode/Steps', episode_steps, self.episode_num)
        
        self.episode_num += 1
        
        return episode_reward, episode_steps
    
    def evaluate(self, env, num_episodes=5):
        """Evaluate the model"""
        total_reward = 0
        total_return = 0
        
        for _ in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = self.select_action(state, deterministic=True)
                state, reward, done, _ = env.step([action])
                episode_reward += reward
            
            total_reward += episode_reward
            
            if hasattr(env, 'get_metrics'):
                metrics = env.get_metrics()
                total_return += metrics.get('total_return', 0)
        
        avg_reward = total_reward / num_episodes
        avg_return = total_return / num_episodes
        
        return avg_reward, avg_return
    
    def should_stop_early(self, current_performance):
        """Check if training should stop early"""
        if current_performance > self.best_performance + self.config.min_improvement:
            self.best_performance = current_performance
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.patience
    
    def train(self, env, val_env=None, num_episodes=None):
        """Main training loop with enhanced logging"""
        if num_episodes is None:
            num_episodes = self.config.num_episodes
        
        best_reward = -float('inf')
        best_sharpe = -float('inf')
        best_profit = -float('inf')
        
        # Track recent metrics for moving averages
        recent_losses = deque(maxlen=10)
        recent_rewards = deque(maxlen=10)
        
        for episode in range(num_episodes):
            # Train episode
            reward, steps = self.train_episode(env)
            recent_rewards.append(reward)
            
            # Get current loss (average of recent losses)
            if self.training_metrics['actor_losses']:
                current_loss = self.training_metrics['actor_losses'][-1]
                recent_losses.append(current_loss)
                avg_loss = np.mean(recent_losses)
            else:
                avg_loss = 0.0
            
            # Get current learning rate
            current_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.config.learning_rate
            
            # Validation evaluation
            val_reward = 0.0
            val_profit = 0.0
            val_sharpe = 0.0
            val_drawdown = 0.0
            status = "Training"
            
            if (episode + 1) % self.config.eval_interval == 0:
                # Validate on training env first for quick metrics
                env.reset()
                state = env.reset()
                done = False
                while not done:
                    action, _ = self.select_action(state, deterministic=True)
                    state, _, done, _ = env.step([action])
                
                train_metrics = env.get_metrics()
                
                # Validate on validation env if provided
                if val_env is not None:
                    val_reward, val_return = self.evaluate(val_env, num_episodes=3)
                    
                    # Get detailed validation metrics
                    val_env.reset()
                    state = val_env.reset()
                    done = False
                    while not done:
                        action, _ = self.select_action(state, deterministic=True)
                        state, _, done, _ = val_env.step([action])
                    
                    val_metrics = val_env.get_metrics()
                    val_profit = val_return
                    val_sharpe = val_metrics.get('sharpe_ratio', 0)
                    val_drawdown = val_metrics.get('max_drawdown', 0)
                else:
                    # Use training metrics if no validation env
                    val_reward = reward
                    val_profit = train_metrics.get('total_return', 0)
                    val_sharpe = train_metrics.get('sharpe_ratio', 0)
                    val_drawdown = train_metrics.get('max_drawdown', 0)
                
                # Combined performance metric
                performance = val_sharpe + val_profit * 10
                
                # Check for improvements
                improved = False
                if val_reward > best_reward:
                    best_reward = val_reward
                    self.save_checkpoint('models/modern_best_reward.pth', episode, val_reward)
                    improved = True
                
                if val_sharpe > best_sharpe:
                    best_sharpe = val_sharpe
                    self.save_checkpoint('models/modern_best_sharpe.pth', episode, val_sharpe)
                    improved = True
                
                if val_profit > best_profit:
                    best_profit = val_profit
                    self.save_checkpoint('models/modern_best_profit.pth', episode, val_profit)
                    improved = True
                
                status = "🔥BEST" if improved else "Eval"
                
                # Log evaluation metrics
                self.writer.add_scalar('Evaluation/Reward', val_reward, episode)
                self.writer.add_scalar('Evaluation/Return', val_profit, episode)
                self.writer.add_scalar('Evaluation/Sharpe', val_sharpe, episode)
                self.writer.add_scalar('Evaluation/Performance', performance, episode)
                
                # Early stopping check
                if self.should_stop_early(performance):
                    print(f"\n⏹️  Early stopping at episode {episode + 1} - No improvement for {self.patience_counter} evaluations")
                    break
            
            # Print progress every episode with nice formatting
            if episode == 0 or (episode + 1) % max(1, num_episodes // 200) == 0 or (episode + 1) % self.config.eval_interval == 0:
                print(f"{episode+1:7d} "
                      f"{np.mean(recent_rewards):8.3f} "
                      f"{steps:6d} "
                      f"{avg_loss:8.4f} "
                      f"{current_lr:10.6f} "
                      f"{val_reward:8.3f} "
                      f"{val_profit:8.2%} "
                      f"{val_sharpe:7.3f} "
                      f"{val_drawdown:7.2%} "
                      f"{status}")
            
            # Save checkpoints
            if (episode + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f'models/modern_checkpoint_ep{episode + 1}.pth', episode)
        
        print("="*100)
        print(f"🏁 Training complete! Best metrics:")
        print(f"   Best Reward: {best_reward:.4f}")
        print(f"   Best Sharpe: {best_sharpe:.4f}")
        print(f"   Best Profit: {best_profit:.2%}")
        
        return self.training_metrics
    
    def save_checkpoint(self, filepath, episode=None, metric=None):
        """Save model checkpoint"""
        Path(filepath).parent.mkdir(exist_ok=True, parents=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'metrics': self.training_metrics,
            'episode': episode,
            'metric': metric,
            'global_step': self.global_step
        }
        
        torch.save(checkpoint, filepath)
        if metric is not None:
            tqdm.write(f"🔥 Best model saved: {filepath} (metric: {metric:.4f})")
    
    def close(self):
        """Clean up resources"""
        self.writer.close()


if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 MODERN TRANSFORMER TRADING SYSTEM")
    print("="*80)
    print("\n📊 Key Improvements:")
    print("✓ Much smaller model (128 dim, 2 layers, 4 heads)")
    print("✓ Strong regularization (dropout 0.4, weight decay)")
    print("✓ Modern architecture (RoPE, RMSNorm, SwiGLU)")
    print("✓ Low learning rates (5e-5) with cosine scheduling")
    print("✓ Gradient accumulation for large effective batches")
    print("✓ Proper early stopping and plateau detection")
    print("✓ 10x more training data")
    print("✓ Modern optimizer (AdamW) and scheduling")
    print("="*80)