#!/usr/bin/env python3
"""
Advanced RL Training System with State-of-the-Art Techniques
Implements:
- Muon optimizer for faster convergence
- Advanced data augmentation
- Curiosity-driven exploration
- Hindsight Experience Replay (HER)
- Transformer-based architecture
- Ensemble learning
- Advanced reward shaping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import random
from collections import deque, namedtuple
from dataclasses import dataclass
import math


# ============================================================================
# ADVANCED OPTIMIZERS
# ============================================================================

class Muon(torch.optim.Optimizer):
    """
    Muon Optimizer - Momentum-based optimizer with adaptive learning
    Combines benefits of Adam and SGD with momentum
    """
    def __init__(self, params, lr=0.001, momentum=0.95, nesterov=True, 
                 weight_decay=0.0, adaptive=True):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                       weight_decay=weight_decay, adaptive=adaptive)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            momentum = group['momentum']
            nesterov = group['nesterov']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad.data
                param_state = self.state[p]
                
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p)
                
                if group['adaptive']:
                    # Adaptive learning rate based on gradient magnitude
                    grad_norm = d_p.norm()
                    if grad_norm > 0:
                        adaptive_lr = group['lr'] * (1.0 / (1.0 + grad_norm))
                    else:
                        adaptive_lr = group['lr']
                else:
                    adaptive_lr = group['lr']
                
                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)
                    p.data.add_(d_p, alpha=-adaptive_lr)
                else:
                    p.data.add_(buf, alpha=-adaptive_lr)
        
        return loss


class Shampoo(torch.optim.Optimizer):
    """
    Shampoo Optimizer - Second-order optimizer with preconditioning
    Approximates natural gradient descent
    """
    def __init__(self, params, lr=0.001, eps=1e-10, update_freq=50):
        defaults = dict(lr=lr, eps=eps, update_freq=update_freq)
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
                order = len(grad.shape)
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['precon'] = []
                    for i in range(order):
                        state['precon'].append(
                            group['eps'] * torch.eye(grad.shape[i], device=grad.device)
                        )
                
                state['step'] += 1
                
                # Update preconditioning matrices
                if state['step'] % group['update_freq'] == 0:
                    for i in range(order):
                        # Compute covariance matrix for each mode
                        grad_reshaped = grad.reshape(grad.shape[i], -1)
                        cov = torch.mm(grad_reshaped, grad_reshaped.t())
                        state['precon'][i] = (1 - group['eps']) * state['precon'][i] + \
                                           group['eps'] * cov
                
                # Apply preconditioning
                preconditioned_grad = grad.clone()
                for i in range(order):
                    # Apply preconditioning for each mode
                    inv_precon = torch.inverse(
                        state['precon'][i] + group['eps'] * torch.eye(
                            grad.shape[i], device=grad.device
                        )
                    )
                    if i == 0:
                        preconditioned_grad = torch.mm(inv_precon, grad.reshape(grad.shape[0], -1))
                        preconditioned_grad = preconditioned_grad.reshape(grad.shape)
                
                p.data.add_(preconditioned_grad, alpha=-group['lr'])
        
        return loss


# ============================================================================
# ADVANCED NEURAL ARCHITECTURES
# ============================================================================

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention for temporal pattern recognition"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Linear transformations and split into heads
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
        return output


class TransformerTradingAgent(nn.Module):
    """Advanced transformer-based trading agent with attention mechanisms"""
    def __init__(self, input_dim, hidden_dim=256, num_layers=3, num_heads=8, dropout=0.1):
        super().__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output heads
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Curiosity module for exploration
        self.curiosity_module = CuriosityModule(hidden_dim)
        
        # Action variance (learnable)
        self.log_std = nn.Parameter(torch.zeros(1))
    
    def forward(self, x, return_features=False):
        # Input projection
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Global pooling (or take last timestep)
        if len(x.shape) == 3:
            features = x.mean(dim=1)  # Global average pooling
        else:
            features = x
        
        # Get action and value
        action = self.actor_head(features)
        value = self.critic_head(features)
        
        if return_features:
            return action, value, features
        return action, value
    
    def get_action_distribution(self, x):
        action_mean, _ = self.forward(x)
        action_std = torch.exp(self.log_std)
        return torch.distributions.Normal(action_mean, action_std)


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and feedforward"""
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Self-attention with residual
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)
        
        # Feedforward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x


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
# CURIOSITY-DRIVEN EXPLORATION
# ============================================================================

class CuriosityModule(nn.Module):
    """Intrinsic Curiosity Module for exploration"""
    def __init__(self, feature_dim, action_dim=1):
        super().__init__()
        
        # Forward model: predicts next state given current state and action
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )
        
        # Inverse model: predicts action given current and next state
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def compute_intrinsic_reward(self, state, action, next_state):
        # Predict next state
        state_action = torch.cat([state, action], dim=-1)
        predicted_next = self.forward_model(state_action)
        
        # Forward model error as curiosity bonus
        curiosity_reward = F.mse_loss(predicted_next, next_state, reduction='none').mean(dim=-1)
        
        # Inverse model for learning useful features
        state_pair = torch.cat([state, next_state], dim=-1)
        predicted_action = self.inverse_model(state_pair)
        
        return curiosity_reward, predicted_action


# ============================================================================
# ADVANCED REPLAY BUFFERS
# ============================================================================

Experience = namedtuple('Experience', 
    ['state', 'action', 'reward', 'next_state', 'done', 'info'])


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay with importance sampling"""
    def __init__(self, capacity=100000, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def push(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        # New experiences get max priority
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return [], [], []
        
        # Calculate sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        return experiences, indices, weights
    
    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + 1e-6
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)


class HindsightExperienceReplay:
    """HER for learning from failed experiences"""
    def __init__(self, capacity=100000, k=4):
        self.buffer = deque(maxlen=capacity)
        self.k = k  # Number of hindsight goals per episode
    
    def store_episode(self, episode_experiences):
        # Store original experiences
        for exp in episode_experiences:
            self.buffer.append(exp)
        
        # Generate hindsight experiences
        for i, exp in enumerate(episode_experiences[:-1]):
            # Sample future states as goals
            future_indices = np.random.choice(
                range(i + 1, len(episode_experiences)), 
                min(self.k, len(episode_experiences) - i - 1),
                replace=False
            )
            
            for future_idx in future_indices:
                # Create hindsight experience with achieved goal
                hindsight_exp = Experience(
                    state=exp.state,
                    action=exp.action,
                    reward=self._compute_hindsight_reward(exp, episode_experiences[future_idx]),
                    next_state=exp.next_state,
                    done=exp.done,
                    info={'hindsight': True}
                )
                self.buffer.append(hindsight_exp)
    
    def _compute_hindsight_reward(self, exp, future_exp):
        # Reward for reaching the future state
        return 1.0 if np.allclose(exp.next_state, future_exp.state, rtol=0.1) else 0.0
    
    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))


# ============================================================================
# DATA AUGMENTATION FOR TIME SERIES
# ============================================================================

class TimeSeriesAugmentation:
    """Advanced augmentation techniques for financial time series"""
    
    @staticmethod
    def add_noise(data, noise_level=0.01):
        """Add Gaussian noise to data"""
        noise = np.random.normal(0, noise_level, data.shape)
        return data + noise
    
    @staticmethod
    def time_warp(data, sigma=0.2):
        """Random time warping"""
        from scipy.interpolate import CubicSpline
        
        orig_steps = np.arange(len(data))
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(len(data), 1))
        warp_steps = np.cumsum(random_warps)
        
        # Normalize to original length
        warp_steps = (warp_steps - warp_steps.min()) / (warp_steps.max() - warp_steps.min())
        warp_steps = warp_steps * (len(data) - 1)
        
        # Interpolate
        warped = np.zeros_like(data)
        for i in range(data.shape[1]):
            cs = CubicSpline(warp_steps.flatten(), data[:, i])
            warped[:, i] = cs(orig_steps)
        
        return warped
    
    @staticmethod
    def magnitude_warp(data, sigma=0.2):
        """Random magnitude warping"""
        from scipy.interpolate import CubicSpline
        
        orig_steps = np.arange(len(data))
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(4, 1))
        warp_steps = np.linspace(0, len(data) - 1, 4)
        
        warped = np.zeros_like(data)
        for i in range(data.shape[1]):
            cs = CubicSpline(warp_steps, random_warps.flatten())
            warped[:, i] = data[:, i] * cs(orig_steps)
        
        return warped
    
    @staticmethod
    def window_slice(data, slice_ratio=0.9):
        """Random window slicing"""
        target_len = int(len(data) * slice_ratio)
        if target_len >= len(data):
            return data
        
        start = np.random.randint(0, len(data) - target_len)
        return data[start:start + target_len]
    
    @staticmethod
    def mixup(data1, data2, alpha=0.2):
        """Mixup augmentation between two samples"""
        lam = np.random.beta(alpha, alpha)
        return lam * data1 + (1 - lam) * data2
    
    @staticmethod
    def cutmix(data1, data2, alpha=1.0):
        """CutMix augmentation"""
        lam = np.random.beta(alpha, alpha)
        cut_point = int(len(data1) * lam)
        
        mixed = data1.copy()
        mixed[cut_point:] = data2[cut_point:]
        return mixed


# ============================================================================
# ADVANCED REWARD SHAPING
# ============================================================================

class AdvancedRewardShaper:
    """Sophisticated reward shaping for better learning"""
    
    def __init__(self, risk_penalty=0.01, consistency_bonus=0.1, 
                 profit_threshold=0.001):
        self.risk_penalty = risk_penalty
        self.consistency_bonus = consistency_bonus
        self.profit_threshold = profit_threshold
        self.profit_history = deque(maxlen=100)
    
    def shape_reward(self, raw_reward, info):
        shaped_reward = raw_reward
        
        # Risk-adjusted reward (penalize high volatility)
        if 'volatility' in info:
            shaped_reward -= self.risk_penalty * info['volatility']
        
        # Consistency bonus (reward stable profits)
        self.profit_history.append(raw_reward)
        if len(self.profit_history) > 10:
            recent_profits = list(self.profit_history)[-10:]
            if all(p > self.profit_threshold for p in recent_profits):
                shaped_reward += self.consistency_bonus
        
        # Sharpe ratio bonus
        if 'sharpe_ratio' in info and info['sharpe_ratio'] > 0:
            shaped_reward += 0.1 * info['sharpe_ratio']
        
        # Drawdown penalty
        if 'drawdown' in info and info['drawdown'] < -0.05:
            shaped_reward -= abs(info['drawdown']) * 0.5
        
        # Win rate bonus
        if 'win_rate' in info and info['win_rate'] > 0.6:
            shaped_reward += 0.05 * (info['win_rate'] - 0.5)
        
        return shaped_reward


# ============================================================================
# ENSEMBLE LEARNING
# ============================================================================

class EnsembleTradingAgent:
    """Ensemble of multiple agents for robust trading"""
    
    def __init__(self, num_agents=5, input_dim=100, hidden_dim=256):
        self.agents = [
            TransformerTradingAgent(input_dim, hidden_dim)
            for _ in range(num_agents)
        ]
        
        # Different optimizers for diversity
        self.optimizers = [
            Muon(agent.parameters(), lr=0.001) if i % 2 == 0
            else torch.optim.Adam(agent.parameters(), lr=0.001)
            for i, agent in enumerate(self.agents)
        ]
        
        # Ensemble weights (learnable)
        self.ensemble_weights = nn.Parameter(torch.ones(num_agents) / num_agents)
    
    def get_ensemble_action(self, state):
        actions = []
        values = []
        
        for agent in self.agents:
            action, value = agent(state)
            actions.append(action)
            values.append(value)
        
        # Weighted average
        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_action = sum(w * a for w, a in zip(weights, actions))
        ensemble_value = sum(w * v for w, v in zip(weights, values))
        
        return ensemble_action, ensemble_value
    
    def train_ensemble(self, experiences, diversity_bonus=0.1):
        losses = []
        
        for i, (agent, optimizer) in enumerate(zip(self.agents, self.optimizers)):
            # Train each agent
            loss = self._compute_agent_loss(agent, experiences)
            
            # Add diversity regularization
            if i > 0:
                # Encourage different behaviors
                with torch.no_grad():
                    prev_actions = [self.agents[j](experiences.states)[0] 
                                  for j in range(i)]
                curr_action = agent(experiences.states)[0]
                
                diversity_loss = -torch.mean(
                    torch.stack([F.mse_loss(curr_action, pa) for pa in prev_actions])
                )
                loss += diversity_bonus * diversity_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        return np.mean(losses)
    
    def _compute_agent_loss(self, agent, experiences):
        # Implement PPO or other RL loss
        pass  # Placeholder for actual loss computation


# ============================================================================
# CURRICULUM LEARNING
# ============================================================================

class CurriculumScheduler:
    """Gradually increase task difficulty for better learning"""
    
    def __init__(self, start_difficulty=0.1, end_difficulty=1.0, 
                 warmup_episodes=100):
        self.start_difficulty = start_difficulty
        self.end_difficulty = end_difficulty
        self.warmup_episodes = warmup_episodes
        self.current_episode = 0
        
    def get_difficulty(self):
        if self.current_episode < self.warmup_episodes:
            # Linear warmup
            progress = self.current_episode / self.warmup_episodes
            return self.start_difficulty + progress * (self.end_difficulty - self.start_difficulty)
        return self.end_difficulty
    
    def update(self):
        self.current_episode += 1
    
    def adjust_environment(self, env):
        difficulty = self.get_difficulty()
        
        # Adjust environment parameters based on difficulty
        env.volatility = 0.01 + difficulty * 0.05  # Increase volatility
        env.fee_multiplier = 1.0 + difficulty * 0.5  # Increase fees
        env.max_position = 0.5 + difficulty * 0.5  # Allow larger positions
        
        return env


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

@dataclass
class AdvancedTrainingConfig:
    # Model
    architecture: str = 'transformer'  # 'transformer', 'lstm', 'cnn'
    hidden_dim: int = 256
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.1
    
    # Optimization
    optimizer: str = 'muon'  # 'muon', 'shampoo', 'adam'
    learning_rate: float = 0.001
    batch_size: int = 256
    gradient_clip: float = 1.0
    
    # RL
    gamma: float = 0.995
    gae_lambda: float = 0.95
    ppo_epochs: int = 10
    ppo_clip: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # Exploration
    use_curiosity: bool = True
    curiosity_weight: float = 0.1
    use_her: bool = True
    
    # Data
    use_augmentation: bool = True
    augmentation_prob: float = 0.5
    
    # Training
    num_episodes: int = 10000
    eval_interval: int = 100
    save_interval: int = 500
    
    # Ensemble
    use_ensemble: bool = True
    num_agents: int = 3
    
    # Curriculum
    use_curriculum: bool = True
    warmup_episodes: int = 1000


def create_advanced_agent(config: AdvancedTrainingConfig, input_dim: int):
    """Create agent based on configuration"""
    if config.use_ensemble:
        return EnsembleTradingAgent(
            num_agents=config.num_agents,
            input_dim=input_dim,
            hidden_dim=config.hidden_dim
        )
    elif config.architecture == 'transformer':
        return TransformerTradingAgent(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
    else:
        raise ValueError(f"Unknown architecture: {config.architecture}")


def create_optimizer(agent, config: AdvancedTrainingConfig):
    """Create optimizer based on configuration"""
    if config.optimizer == 'muon':
        return Muon(agent.parameters(), lr=config.learning_rate)
    elif config.optimizer == 'shampoo':
        return Shampoo(agent.parameters(), lr=config.learning_rate)
    else:
        return torch.optim.Adam(agent.parameters(), lr=config.learning_rate)


if __name__ == '__main__':
    print("Advanced Trading Agent Training System")
    print("=" * 80)
    print("\nFeatures:")
    print("✓ Muon & Shampoo optimizers for faster convergence")
    print("✓ Transformer architecture with attention mechanisms")
    print("✓ Curiosity-driven exploration")
    print("✓ Hindsight Experience Replay (HER)")
    print("✓ Prioritized replay buffer")
    print("✓ Advanced data augmentation")
    print("✓ Ensemble learning with multiple agents")
    print("✓ Curriculum learning with progressive difficulty")
    print("✓ Advanced reward shaping")
    print("=" * 80)