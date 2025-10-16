#!/usr/bin/env python3
"""
RL Trainer for Multi-Asset Trading with Toto Embeddings
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
from collections import deque, namedtuple
import random
from typing import Dict, List, Tuple, Optional, Any
import gymnasium as gym

from multi_asset_env import MultiAssetTradingEnv

# Import toto embedding system
import sys
sys.path.append('../totoembedding')
from embedding_model import TotoEmbeddingModel
from pretrained_loader import PretrainedWeightLoader


class TotoRLAgent(nn.Module):
    """RL Agent that uses Toto embeddings for multi-asset trading"""
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        embedding_dim: int = 128,
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.2,
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        
        # Separate embedding features from other observations
        self.embedding_processor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dims[0] // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Process remaining observation features
        other_obs_dim = observation_dim - embedding_dim
        self.obs_processor = nn.Sequential(
            nn.Linear(other_obs_dim, hidden_dims[0] // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Main network layers
        layers = []
        input_dim = hidden_dims[0]
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # Separate value and advantage heads for dueling architecture
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, 1)
        )
        
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, action_dim)
        )
        
        # Action scaling layer (tanh output)
        self.action_scale = nn.Tanh()
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize network weights"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        batch_size = observation.shape[0]
        
        # Split observation into embedding and other features
        embedding_features = observation[:, :self.embedding_dim]
        other_features = observation[:, self.embedding_dim:]
        
        # Process embedding features
        emb_processed = self.embedding_processor(embedding_features)
        
        # Process other observation features
        obs_processed = self.obs_processor(other_features)
        
        # Combine processed features
        combined = torch.cat([emb_processed, obs_processed], dim=-1)
        
        # Main backbone
        features = self.backbone(combined)
        
        # Dueling network: V(s) + A(s,a) - mean(A(s,a))
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        
        # Dueling combination
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        
        # Scale to [-1, 1] for continuous actions
        actions = self.action_scale(q_values)
        
        return actions
    
    def get_q_values(self, observation: torch.Tensor) -> torch.Tensor:
        """Get Q-values for critic evaluation"""
        batch_size = observation.shape[0]
        
        # Split observation into embedding and other features
        embedding_features = observation[:, :self.embedding_dim]
        other_features = observation[:, self.embedding_dim:]
        
        # Process features
        emb_processed = self.embedding_processor(embedding_features)
        obs_processed = self.obs_processor(other_features)
        combined = torch.cat([emb_processed, obs_processed], dim=-1)
        
        # Get features
        features = self.backbone(combined)
        
        # Get value and advantage
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        
        # Return raw Q-values (before tanh scaling)
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        
        return q_values


class TotoRLTrainer:
    """RL Trainer for multi-asset trading with Toto embeddings"""
    
    def __init__(
        self,
        env_config: Dict[str, Any] = None,
        agent_config: Dict[str, Any] = None,
        training_config: Dict[str, Any] = None,
        pretrained_model_path: str = None
    ):
        # Default configurations
        self.env_config = env_config or {}
        self.agent_config = agent_config or {
            'hidden_dims': [512, 256, 128],
            'dropout': 0.2,
            'use_layer_norm': True
        }
        self.training_config = training_config or {
            'batch_size': 128,
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'tau': 0.005,
            'buffer_size': 100000,
            'warmup_steps': 1000,
            'update_freq': 4,
            'target_update_freq': 100,
            'episodes': 1000,
            'max_steps': 2000,
            'epsilon_start': 1.0,
            'epsilon_end': 0.05,
            'epsilon_decay': 0.995
        }
        
        # Setup environment
        self.env = MultiAssetTradingEnv(**self.env_config)
        self.test_env = MultiAssetTradingEnv(**self.env_config)  # For evaluation
        
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        # Create agent networks
        self.agent = TotoRLAgent(
            observation_dim=obs_dim,
            action_dim=action_dim,
            **self.agent_config
        )
        
        self.target_agent = TotoRLAgent(
            observation_dim=obs_dim,
            action_dim=action_dim,
            **self.agent_config
        )
        
        # Copy weights to target network
        self.target_agent.load_state_dict(self.agent.state_dict())
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.agent.parameters(),
            lr=self.training_config['learning_rate'],
            weight_decay=1e-5
        )
        
        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=self.training_config['buffer_size'],
            obs_dim=obs_dim,
            action_dim=action_dim
        )
        
        # Training state
        self.step_count = 0
        self.episode_count = 0
        self.epsilon = self.training_config['epsilon_start']
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_metrics = []
        self.losses = []
        
        # Setup tensorboard
        log_dir = f"runs/toto_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir)
        
        # Load pretrained weights if available
        if pretrained_model_path:
            self.load_pretrained_weights(pretrained_model_path)
        
        print(f"TotoRLTrainer initialized:")
        print(f"  Observation space: {obs_dim}")
        print(f"  Action space: {action_dim}")
        print(f"  Agent parameters: {sum(p.numel() for p in self.agent.parameters()):,}")
        print(f"  Tensorboard: {log_dir}")
    
    def load_pretrained_weights(self, model_path: str):
        """Load pretrained weights into the agent"""
        try:
            loader = PretrainedWeightLoader()
            result = loader.load_compatible_weights(
                self.agent,
                model_path,
                exclude_patterns=[
                    r'.*action.*',
                    r'.*output.*',
                    r'.*head.*',
                    r'.*classifier.*'
                ]
            )
            print(f"Loaded pretrained weights: {result['load_ratio']:.2%} parameters")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
    
    def select_action(
        self, 
        observation: np.ndarray, 
        epsilon: float = None,
        eval_mode: bool = False
    ) -> np.ndarray:
        """Select action using epsilon-greedy policy"""
        if epsilon is None:
            epsilon = self.epsilon
        
        if not eval_mode and random.random() < epsilon:
            # Random action
            return self.env.action_space.sample()
        else:
            # Greedy action
            with torch.no_grad():
                obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                action = self.agent(obs_tensor).squeeze(0).cpu().numpy()
                
                # Add small amount of noise for exploration during training
                if not eval_mode:
                    noise = np.random.normal(0, 0.1, size=action.shape)
                    action = np.clip(action + noise, -1.0, 1.0)
                
                return action
    
    def train_step(self):
        """Perform one training step"""
        if len(self.replay_buffer) < self.training_config['batch_size']:
            return
        
        batch = self.replay_buffer.sample(self.training_config['batch_size'])
        
        # Convert to tensors
        obs = torch.tensor(batch['obs'], dtype=torch.float32)
        actions = torch.tensor(batch['actions'], dtype=torch.float32)
        rewards = torch.tensor(batch['rewards'], dtype=torch.float32)
        next_obs = torch.tensor(batch['next_obs'], dtype=torch.float32)
        dones = torch.tensor(batch['dones'], dtype=torch.bool)
        
        # Current Q-values
        current_q = self.agent.get_q_values(obs)
        
        # Target Q-values
        with torch.no_grad():
            next_actions = self.agent(next_obs)  # Double DQN: use main network for action selection
            next_q = self.target_agent.get_q_values(next_obs)
            
            # For continuous actions, we need to compute Q(s', a') where a' is the predicted action
            # This is a simplified approach - could be enhanced with proper continuous Q-learning
            target_q = rewards.unsqueeze(-1) + (1 - dones.unsqueeze(-1).float()) * self.training_config['gamma'] * next_q
        
        # Compute loss (MSE between predicted and target Q-values)
        # For continuous control, we use the Q-values directly
        loss = F.mse_loss(current_q, target_q.detach())
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        # Update target network
        if self.step_count % self.training_config['target_update_freq'] == 0:
            self.update_target_network()
        
        # Track loss
        self.losses.append(loss.item())
        
        # Log to tensorboard
        if self.step_count % 100 == 0:
            self.writer.add_scalar('Loss/Training', loss.item(), self.step_count)
            self.writer.add_scalar('Epsilon', self.epsilon, self.step_count)
    
    def update_target_network(self):
        """Update target network using soft updates"""
        tau = self.training_config['tau']
        
        for target_param, param in zip(self.target_agent.parameters(), self.agent.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        
        best_reward = -np.inf
        patience_counter = 0
        max_patience = 50
        
        for episode in tqdm(range(self.training_config['episodes']), desc="Training"):
            self.episode_count = episode
            
            # Reset environment
            obs = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(self.training_config['max_steps']):
                # Select action
                action = self.select_action(obs)
                
                # Take step
                next_obs, reward, done, info = self.env.step(action)
                
                # Store in replay buffer
                self.replay_buffer.push(obs, action, reward, next_obs, done)
                
                # Train agent
                if self.step_count % self.training_config['update_freq'] == 0:
                    self.train_step()
                
                # Update state
                obs = next_obs
                episode_reward += reward
                episode_length += 1
                self.step_count += 1
                
                if done:
                    break
            
            # Decay epsilon
            self.epsilon = max(
                self.training_config['epsilon_end'],
                self.epsilon * self.training_config['epsilon_decay']
            )
            
            # Track episode metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Get portfolio metrics
            portfolio_metrics = self.env.get_portfolio_metrics()
            self.episode_metrics.append(portfolio_metrics)
            
            # Log to tensorboard
            self.writer.add_scalar('Reward/Episode', episode_reward, episode)
            self.writer.add_scalar('Length/Episode', episode_length, episode)
            
            if portfolio_metrics:
                for key, value in portfolio_metrics.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(f'Portfolio/{key}', value, episode)
            
            # Evaluation and checkpointing
            if episode % 50 == 0 and episode > 0:
                eval_metrics = self.evaluate()
                
                avg_reward = np.mean(self.episode_rewards[-50:])
                print(f"\nEpisode {episode}:")
                print(f"  Average Reward (last 50): {avg_reward:.4f}")
                print(f"  Epsilon: {self.epsilon:.3f}")
                print(f"  Buffer Size: {len(self.replay_buffer)}")
                
                if portfolio_metrics:
                    print(f"  Portfolio Return: {portfolio_metrics.get('total_return', 0):.2%}")
                    print(f"  Sharpe Ratio: {portfolio_metrics.get('sharpe_ratio', 0):.2f}")
                
                # Save best model
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    patience_counter = 0
                    self.save_model(f"models/toto_rl_best.pth")
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= max_patience:
                    print(f"Early stopping after {patience_counter} episodes without improvement")
                    break
                
                # Regular checkpoint
                if episode % 200 == 0:
                    self.save_model(f"models/toto_rl_checkpoint_{episode}.pth")
        
        print("Training completed!")
        
        # Final evaluation and save
        final_metrics = self.evaluate(num_episodes=10)
        self.save_model("models/toto_rl_final.pth")
        
        return final_metrics
    
    def evaluate(self, num_episodes: int = 5) -> Dict[str, float]:
        """Evaluate the current policy"""
        eval_rewards = []
        eval_metrics = []
        
        for _ in range(num_episodes):
            obs = self.test_env.reset()
            episode_reward = 0
            
            for _ in range(self.training_config['max_steps']):
                action = self.select_action(obs, epsilon=0.0, eval_mode=True)
                obs, reward, done, info = self.test_env.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_metrics.append(self.test_env.get_portfolio_metrics())
        
        # Aggregate metrics
        avg_reward = np.mean(eval_rewards)
        
        aggregated_metrics = {
            'eval_reward': avg_reward,
            'eval_std': np.std(eval_rewards)
        }
        
        # Aggregate portfolio metrics
        if eval_metrics and eval_metrics[0]:
            for key in eval_metrics[0].keys():
                values = [m.get(key, 0) for m in eval_metrics if m]
                if values and all(isinstance(v, (int, float)) for v in values):
                    aggregated_metrics[f'eval_{key}'] = np.mean(values)
        
        # Log to tensorboard
        for key, value in aggregated_metrics.items():
            self.writer.add_scalar(f'Eval/{key}', value, self.episode_count)
        
        return aggregated_metrics
    
    def save_model(self, filepath: str):
        """Save model checkpoint"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'agent_state_dict': self.agent.state_dict(),
            'target_agent_state_dict': self.target_agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_metrics': self.episode_metrics,
            'env_config': self.env_config,
            'agent_config': self.agent_config,
            'training_config': self.training_config
        }
        
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.target_agent.load_state_dict(checkpoint['target_agent_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
        self.epsilon = checkpoint['epsilon']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_metrics = checkpoint['episode_metrics']
        
        print(f"Model loaded from {filepath}")


# Experience Replay Buffer
Experience = namedtuple('Experience', ['obs', 'action', 'reward', 'next_obs', 'done'])


class ReplayBuffer:
    """Experience replay buffer for RL training"""
    
    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
    
    def push(self, obs, action, reward, next_obs, done):
        """Add experience to buffer"""
        experience = Experience(obs, action, reward, next_obs, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample batch from buffer"""
        experiences = random.sample(self.buffer, batch_size)
        
        batch = {
            'obs': np.array([e.obs for e in experiences]),
            'actions': np.array([e.action for e in experiences]),
            'rewards': np.array([e.reward for e in experiences]),
            'next_obs': np.array([e.next_obs for e in experiences]),
            'dones': np.array([e.done for e in experiences])
        }
        
        return batch
    
    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":
    # Example usage
    trainer = TotoRLTrainer(
        env_config={
            'data_dir': '../trainingdata/train',
            'initial_balance': 100000.0,
            'max_positions': 10
        },
        training_config={
            'episodes': 2000,
            'batch_size': 128,
            'learning_rate': 1e-4
        }
    )
    
    trainer.train()