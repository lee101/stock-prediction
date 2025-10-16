#!/usr/bin/env python3
"""
HuggingFace-style RL Trainer with Toto Embeddings
Incorporates modern optimizers, mixed precision, and advanced training techniques
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
from torch.cuda.amp import autocast, GradScaler
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import math
from collections import deque, namedtuple
import random
import sys

# Import modern optimizers
from modern_optimizers import GPro, Lion, AdaFactor

# Import toto embedding system
sys.path.append('../totoembedding')
from embedding_model import TotoEmbeddingModel
from pretrained_loader import PretrainedWeightLoader

from multi_asset_env import MultiAssetTradingEnv


@dataclass
class HFRLConfig:
    """Configuration for HuggingFace-style RL training"""
    
    # Model architecture
    hidden_size: int = 512
    num_heads: int = 8
    num_layers: int = 6
    intermediate_size: int = 2048
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Toto embedding configuration
    embedding_dim: int = 128
    freeze_toto_embeddings: bool = True
    toto_pretrained_path: str = "../training/models/modern_best_sharpe.pth"
    
    # Training parameters
    learning_rate: float = 5e-5
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Optimizer selection
    optimizer_type: str = "gpro"  # "gpro", "adamw", "lion", "adafactor"
    use_8bit_adam: bool = False
    
    # Mixed precision and efficiency
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = True
    gradient_accumulation_steps: int = 4
    
    # RL specific
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # Training schedule
    num_train_epochs: int = 100
    batch_size: int = 32
    mini_batch_size: int = 8
    buffer_size: int = 100000
    
    # Evaluation
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 50
    
    # Directories
    output_dir: str = "models/hf_rl"
    logging_dir: str = "logs/hf_rl"
    
    # Advanced features
    use_layer_norm_bias: bool = False
    layer_norm_eps: float = 1e-12
    rope_scaling: Optional[Dict] = None
    use_flash_attention: bool = False
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.0001


class TotoTransformerRL(nn.Module):
    """
    Transformer-based RL model with frozen Toto embeddings
    Follows HuggingFace architecture patterns
    """
    
    def __init__(self, config: HFRLConfig, observation_dim: int, action_dim: int):
        super().__init__()
        self.config = config
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        
        # Load and freeze Toto embeddings
        self.toto_embeddings = self._load_toto_embeddings()
        if config.freeze_toto_embeddings:
            for param in self.toto_embeddings.parameters():
                param.requires_grad = False
        
        # Project non-embedding observations to hidden size
        non_embedding_dim = observation_dim - config.embedding_dim
        self.obs_projection = nn.Linear(non_embedding_dim, config.hidden_size)
        
        # Combine embeddings with observations
        self.embedding_projection = nn.Linear(config.embedding_dim, config.hidden_size)
        
        # Layer normalization
        self.pre_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN architecture for stability
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )
        
        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, action_dim)
        )
        
        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1)
        )
        
        # Auxiliary heads for multi-task learning
        self.return_prediction_head = nn.Linear(config.hidden_size, 1)
        self.market_regime_head = nn.Linear(config.hidden_size, 4)  # 4 market regimes
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Special initialization for policy head (smaller values for stable training)
        with torch.no_grad():
            self.policy_head[-1].weight.data *= 0.01
            self.value_head[-1].weight.data *= 0.01
    
    def _load_toto_embeddings(self) -> TotoEmbeddingModel:
        """Load pre-trained Toto embeddings"""
        try:
            model = TotoEmbeddingModel(
                pretrained_model_path=self.config.toto_pretrained_path,
                embedding_dim=self.config.embedding_dim,
                freeze_backbone=True
            )
            model.eval()
            print("Loaded Toto embeddings successfully")
            return model
        except Exception as e:
            print(f"Warning: Could not load Toto embeddings: {e}")
            # Return identity module as fallback
            return nn.Identity()
    
    def _init_weights(self, module):
        """Initialize weights following HuggingFace conventions"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
    
    def forward(
        self,
        observations: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with gradient checkpointing support
        """
        batch_size = observations.shape[0]
        
        # Split observations into embeddings and other features
        toto_features = observations[:, :self.config.embedding_dim]
        other_features = observations[:, self.config.embedding_dim:]
        
        # Process Toto embeddings (frozen or trainable)
        with torch.no_grad() if self.config.freeze_toto_embeddings else torch.enable_grad():
            # Toto embeddings are already computed, just project them
            embedded_features = self.embedding_projection(toto_features)
        
        # Project other observations
        projected_obs = self.obs_projection(other_features)
        
        # Combine features
        combined_features = embedded_features + projected_obs
        combined_features = self.pre_ln(combined_features)
        
        # Add sequence dimension if needed
        if len(combined_features.shape) == 2:
            combined_features = combined_features.unsqueeze(1)
        
        # Apply transformer with optional gradient checkpointing
        if self.config.gradient_checkpointing and self.training:
            transformer_output = torch.utils.checkpoint.checkpoint(
                self.transformer,
                combined_features,
                attention_mask
            )
        else:
            transformer_output = self.transformer(combined_features, attention_mask)
        
        # Pool transformer output (use last token or mean pooling)
        if len(transformer_output.shape) == 3:
            pooled_output = transformer_output.mean(dim=1)
        else:
            pooled_output = transformer_output
        
        # Generate outputs
        action_logits = self.policy_head(pooled_output)
        state_values = self.value_head(pooled_output).squeeze(-1)
        
        # Auxiliary predictions
        predicted_returns = self.return_prediction_head(pooled_output).squeeze(-1)
        market_regime_logits = self.market_regime_head(pooled_output)
        
        # Apply tanh to actions for bounded continuous control
        actions = torch.tanh(action_logits)
        
        if return_dict:
            return {
                'actions': actions,
                'action_logits': action_logits,
                'state_values': state_values,
                'predicted_returns': predicted_returns,
                'market_regime_logits': market_regime_logits,
                'hidden_states': pooled_output
            }
        else:
            return actions, state_values


class PPOTrainer:
    """
    Proximal Policy Optimization trainer with HuggingFace-style training loop
    """
    
    def __init__(
        self,
        config: HFRLConfig,
        model: TotoTransformerRL,
        env: MultiAssetTradingEnv,
        eval_env: Optional[MultiAssetTradingEnv] = None
    ):
        self.config = config
        self.model = model
        self.env = env
        self.eval_env = eval_env or env
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Experience buffer
        self.rollout_buffer = RolloutBuffer(
            buffer_size=config.buffer_size,
            observation_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            device=self.device
        )
        
        # Logging
        self.writer = SummaryWriter(config.logging_dir)
        self.global_step = 0
        self.episode = 0
        
        # Metrics tracking
        self.train_metrics = defaultdict(list)
        self.eval_metrics = defaultdict(list)
        
        print(f"PPOTrainer initialized on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration"""
        # Separate parameters for weight decay
        no_decay = ["bias", "LayerNorm.weight", "ln", "embeddings"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        
        if self.config.optimizer_type == "gpro":
            return GPro(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                eps=self.config.adam_epsilon
            )
        elif self.config.optimizer_type == "lion":
            return Lion(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == "adafactor":
            return AdaFactor(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                scale_parameter=True,
                relative_step=False,
                warmup_init=False
            )
        else:  # Default to AdamW
            return torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                eps=self.config.adam_epsilon
            )
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup"""
        try:
            from transformers import get_linear_schedule_with_warmup
            return get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=self.config.num_train_epochs * 1000  # Approximate
            )
        except ImportError:
            # Fallback to a simple linear scheduler
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=self.config.warmup_steps
            )
    
    def collect_rollouts(self, n_rollout_steps: int = 2048) -> bool:
        """
        Collect experience by interacting with the environment
        """
        self.model.eval()
        obs = self.env.reset()
        
        for step in range(n_rollout_steps):
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Get action from policy
                outputs = self.model(obs_tensor)
                actions = outputs['actions'].cpu().numpy()[0]
                values = outputs['state_values'].cpu().numpy()[0]
                
                # Add exploration noise during training
                if self.model.training:
                    noise = np.random.normal(0, 0.1, actions.shape)
                    actions = np.clip(actions + noise, -1, 1)
            
            # Step environment
            next_obs, reward, done, info = self.env.step(actions)
            
            # Store experience
            self.rollout_buffer.add(
                obs=obs,
                action=actions,
                reward=reward,
                value=values,
                done=done
            )
            
            obs = next_obs
            
            if done:
                obs = self.env.reset()
                self.episode += 1
                
                # Log episode metrics
                if 'portfolio_value' in info:
                    self.writer.add_scalar('Episode/Portfolio_Value', info['portfolio_value'], self.episode)
                if 'total_return' in info:
                    self.writer.add_scalar('Episode/Total_Return', info['total_return'], self.episode)
        
        # Compute returns and advantages
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            last_values = self.model(obs_tensor)['state_values'].cpu().numpy()[0]
        
        self.rollout_buffer.compute_returns_and_advantages(
            last_values=last_values,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda
        )
        
        return True
    
    def train_epoch(self):
        """
        Train for one epoch using collected rollouts
        """
        self.model.train()
        
        # Get data from rollout buffer
        batch_size = self.config.mini_batch_size
        
        for epoch in range(10):  # PPO typically uses multiple epochs per rollout
            for batch in self.rollout_buffer.get_batches(batch_size):
                # Move batch to device
                observations = batch['observations'].to(self.device)
                actions = batch['actions'].to(self.device)
                old_values = batch['values'].to(self.device)
                old_log_probs = batch['log_probs'].to(self.device)
                advantages = batch['advantages'].to(self.device)
                returns = batch['returns'].to(self.device)
                
                # Forward pass with mixed precision
                with autocast(enabled=self.config.use_mixed_precision):
                    outputs = self.model(observations)
                    
                    # Calculate action probabilities
                    action_logits = outputs['action_logits']
                    dist = torch.distributions.Normal(action_logits, 0.1)
                    log_probs = dist.log_prob(actions).sum(dim=-1)
                    
                    # Calculate losses
                    # Policy loss (PPO clip)
                    ratio = torch.exp(log_probs - old_log_probs)
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss
                    values = outputs['state_values']
                    value_loss = F.mse_loss(values, returns)
                    
                    # Entropy bonus for exploration
                    entropy = dist.entropy().mean()
                    
                    # Auxiliary losses
                    return_loss = F.mse_loss(outputs['predicted_returns'], returns)
                    
                    # Total loss
                    loss = (
                        policy_loss + 
                        self.config.value_loss_coef * value_loss - 
                        self.config.entropy_coef * entropy +
                        0.1 * return_loss  # Auxiliary task weight
                    )
                
                # Backward pass with gradient accumulation
                if self.config.gradient_accumulation_steps > 1:
                    loss = loss / self.config.gradient_accumulation_steps
                
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Gradient clipping
                if self.config.max_grad_norm:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                # Optimizer step
                if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    self.writer.add_scalar('Loss/Policy', policy_loss.item(), self.global_step)
                    self.writer.add_scalar('Loss/Value', value_loss.item(), self.global_step)
                    self.writer.add_scalar('Loss/Total', loss.item(), self.global_step)
                    self.writer.add_scalar('Metrics/Entropy', entropy.item(), self.global_step)
                    self.writer.add_scalar('LR', self.scheduler.get_last_lr()[0], self.global_step)
                
                self.global_step += 1
        
        # Clear rollout buffer
        self.rollout_buffer.reset()
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the current policy
        """
        self.model.eval()
        eval_rewards = []
        eval_returns = []
        eval_sharpes = []
        
        for _ in range(num_episodes):
            obs = self.eval_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                    actions = self.model(obs_tensor)['actions'].cpu().numpy()[0]
                
                obs, reward, done, info = self.eval_env.step(actions)
                episode_reward += reward
            
            eval_rewards.append(episode_reward)
            
            # Get portfolio metrics
            metrics = self.eval_env.get_portfolio_metrics()
            if metrics:
                eval_returns.append(metrics.get('total_return', 0))
                eval_sharpes.append(metrics.get('sharpe_ratio', 0))
        
        results = {
            'eval_reward': np.mean(eval_rewards),
            'eval_return': np.mean(eval_returns) if eval_returns else 0,
            'eval_sharpe': np.mean(eval_sharpes) if eval_sharpes else 0,
            'eval_reward_std': np.std(eval_rewards)
        }
        
        # Log evaluation results
        for key, value in results.items():
            self.writer.add_scalar(f'Eval/{key}', value, self.global_step)
        
        return results
    
    def train(self):
        """
        Main training loop following HuggingFace conventions
        """
        print("Starting training...")
        best_eval_reward = -np.inf
        patience_counter = 0
        
        for epoch in tqdm(range(self.config.num_train_epochs), desc="Training"):
            # Collect rollouts
            self.collect_rollouts()
            
            # Train on collected data
            self.train_epoch()
            
            # Evaluate periodically
            if (epoch + 1) % 10 == 0:
                eval_results = self.evaluate()
                
                print(f"\nEpoch {epoch + 1}:")
                print(f"  Eval Reward: {eval_results['eval_reward']:.4f}")
                print(f"  Eval Return: {eval_results['eval_return']:.2%}")
                print(f"  Eval Sharpe: {eval_results['eval_sharpe']:.2f}")
                
                # Save best model
                if eval_results['eval_reward'] > best_eval_reward:
                    best_eval_reward = eval_results['eval_reward']
                    patience_counter = 0
                    self.save_model(f"{self.config.output_dir}/best_model.pth")
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            
            # Regular checkpointing
            if (epoch + 1) % 50 == 0:
                self.save_model(f"{self.config.output_dir}/checkpoint_epoch_{epoch + 1}.pth")
        
        # Save final model
        self.save_model(f"{self.config.output_dir}/final_model.pth")
        print("Training completed!")
        
        return self.eval_metrics
    
    def save_model(self, path: str):
        """Save model checkpoint"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'global_step': self.global_step,
            'episode': self.episode,
            'eval_metrics': self.eval_metrics,
            'train_metrics': self.train_metrics
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.global_step = checkpoint.get('global_step', 0)
        self.episode = checkpoint.get('episode', 0)
        self.eval_metrics = checkpoint.get('eval_metrics', defaultdict(list))
        self.train_metrics = checkpoint.get('train_metrics', defaultdict(list))
        
        print(f"Model loaded from {path}")


class RolloutBuffer:
    """
    Rollout buffer for PPO with GAE
    """
    
    def __init__(self, buffer_size: int, observation_dim: int, action_dim: int, device: torch.device):
        self.buffer_size = buffer_size
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.device = device
        
        self.reset()
    
    def reset(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.log_probs = []
        self.advantages = None
        self.returns = None
        self.ptr = 0
    
    def add(self, obs, action, reward, value, done):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_returns_and_advantages(self, last_values: float, gamma: float, gae_lambda: float):
        """
        Compute returns and GAE advantages
        """
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        
        # Add last value
        values = np.append(values, last_values)
        
        # Compute GAE
        advantages = np.zeros_like(rewards)
        last_gae_lam = 0
        
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[-1]
                next_values = last_values
            else:
                next_non_terminal = 1.0 - dones[step + 1]
                next_values = values[step + 1]
            
            delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
            advantages[step] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        
        self.advantages = advantages
        self.returns = advantages + values[:-1]
    
    def get_batches(self, batch_size: int):
        """
        Generate batches for training
        """
        n_samples = len(self.observations)
        indices = np.random.permutation(n_samples)
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            yield {
                'observations': torch.tensor(np.array(self.observations)[batch_indices], dtype=torch.float32),
                'actions': torch.tensor(np.array(self.actions)[batch_indices], dtype=torch.float32),
                'values': torch.tensor(np.array(self.values)[batch_indices], dtype=torch.float32),
                'log_probs': torch.zeros(len(batch_indices)),  # Will be recomputed
                'advantages': torch.tensor(self.advantages[batch_indices], dtype=torch.float32),
                'returns': torch.tensor(self.returns[batch_indices], dtype=torch.float32)
            }


from collections import defaultdict

if __name__ == "__main__":
    # Example usage
    config = HFRLConfig(
        optimizer_type="gpro",
        use_mixed_precision=True,
        gradient_checkpointing=True,
        freeze_toto_embeddings=True
    )
    
    # Create environment
    env = MultiAssetTradingEnv(
        data_dir="../trainingdata/train",
        initial_balance=100000
    )
    
    # Create model
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    model = TotoTransformerRL(config, obs_dim, action_dim)
    
    # Create trainer
    trainer = PPOTrainer(config, model, env)
    
    # Train
    trainer.train()