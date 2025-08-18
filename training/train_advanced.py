#!/usr/bin/env python3
"""
Advanced Training Script with State-of-the-Art Techniques
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

from advanced_trainer import (
    AdvancedTrainingConfig,
    TransformerTradingAgent,
    EnsembleTradingAgent,
    Muon, Shampoo,
    PrioritizedReplayBuffer,
    HindsightExperienceReplay,
    TimeSeriesAugmentation,
    AdvancedRewardShaper,
    CurriculumScheduler,
    Experience
)
from trading_env import DailyTradingEnv
from trading_config import get_trading_costs
from train_full_model import load_and_prepare_data, generate_synthetic_data


class AdvancedPPOTrainer:
    """Advanced PPO trainer with all modern techniques"""
    
    def __init__(self, agent, config: AdvancedTrainingConfig, device='cuda'):
        self.agent = agent
        self.config = config
        self.device = device
        
        # Optimizer
        if config.optimizer == 'muon':
            self.optimizer = Muon(agent.parameters(), lr=config.learning_rate)
        elif config.optimizer == 'shampoo':
            self.optimizer = Shampoo(agent.parameters(), lr=config.learning_rate)
        else:
            self.optimizer = torch.optim.AdamW(
                agent.parameters(), 
                lr=config.learning_rate,
                weight_decay=0.01
            )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=100, T_mult=2
        )
        
        # Replay buffers
        self.replay_buffer = PrioritizedReplayBuffer(capacity=100000)
        self.her_buffer = HindsightExperienceReplay() if config.use_her else None
        
        # Reward shaper
        self.reward_shaper = AdvancedRewardShaper()
        
        # Curriculum scheduler
        self.curriculum = CurriculumScheduler() if config.use_curriculum else None
        
        # Data augmentation
        self.augmenter = TimeSeriesAugmentation() if config.use_augmentation else None
        
        # Metrics tracking
        self.metrics = {
            'episode_rewards': [],
            'episode_profits': [],
            'episode_sharpes': [],
            'actor_losses': [],
            'critic_losses': [],
            'curiosity_rewards': [],
            'learning_rates': []
        }
        
        # Move agent to device
        if hasattr(agent, 'to'):
            agent.to(device)
        elif hasattr(agent, 'agents'):  # Ensemble
            for a in agent.agents:
                a.to(device)
    
    def select_action(self, state, deterministic=False):
        """Select action using the agent"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Apply augmentation during training
            if not deterministic and self.augmenter and np.random.random() < self.config.augmentation_prob:
                state_np = state_tensor.cpu().numpy()[0]
                augmented = self.augmenter.add_noise(state_np, noise_level=0.005)
                state_tensor = torch.FloatTensor(augmented).unsqueeze(0).to(self.device)
            
            if isinstance(self.agent, EnsembleTradingAgent):
                action, value = self.agent.get_ensemble_action(state_tensor)
            else:
                dist = self.agent.get_action_distribution(state_tensor)
                if deterministic:
                    action = dist.mean
                else:
                    action = dist.sample()
                _, value = self.agent(state_tensor)
            
            return action.cpu().numpy()[0], value.cpu().item()
    
    def compute_gae(self, rewards, values, dones, next_value):
        """Generalized Advantage Estimation"""
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
    
    def update_policy(self, states, actions, old_log_probs, advantages, returns):
        """PPO policy update with advanced techniques"""
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0
        for _ in range(self.config.ppo_epochs):
            # Get current predictions
            if isinstance(self.agent, EnsembleTradingAgent):
                actions_pred, values = self.agent.get_ensemble_action(states)
                # Compute log probs for ensemble
                log_probs = -0.5 * ((actions - actions_pred) ** 2).sum(dim=-1)
            else:
                dist = self.agent.get_action_distribution(states)
                log_probs = dist.log_prob(actions).sum(dim=-1)
                _, values = self.agent(states)
            
            values = values.squeeze()
            
            # PPO loss
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.ppo_clip, 1 + self.config.ppo_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, returns)
            
            # Entropy bonus
            if not isinstance(self.agent, EnsembleTradingAgent):
                entropy = dist.entropy().mean()
            else:
                entropy = torch.tensor(0.0)  # No entropy for ensemble
            
            # Total loss
            loss = actor_loss + self.config.value_loss_coef * value_loss - self.config.entropy_coef * entropy
            
            # Curiosity loss if applicable
            if self.config.use_curiosity and hasattr(self.agent, 'curiosity_module'):
                # Compute curiosity loss here
                pass  # Implement based on state transitions
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.agent.parameters() if hasattr(self.agent, 'parameters') 
                else [p for a in self.agent.agents for p in a.parameters()],
                self.config.gradient_clip
            )
            
            self.optimizer.step()
            total_loss += loss.item()
        
        # Update learning rate
        self.scheduler.step()
        
        # Track metrics
        self.metrics['actor_losses'].append(actor_loss.item())
        self.metrics['critic_losses'].append(value_loss.item())
        self.metrics['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
        
        return total_loss / self.config.ppo_epochs
    
    def train_episode(self, env, max_steps=1000):
        """Train one episode with advanced techniques"""
        state = env.reset()
        
        # Adjust difficulty if using curriculum
        if self.curriculum:
            env = self.curriculum.adjust_environment(env)
            self.curriculum.update()
        
        episode_experiences = []
        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
        
        episode_reward = 0
        episode_steps = 0
        
        for step in range(max_steps):
            # Select action
            action, value = self.select_action(state)
            
            # Environment step
            next_state, reward, done, info = env.step([action])
            
            # Shape reward
            shaped_reward = self.reward_shaper.shape_reward(reward, info)
            
            # Store experience
            exp = Experience(state, action, shaped_reward, next_state, done, info)
            episode_experiences.append(exp)
            
            # For PPO update
            states.append(state)
            actions.append(action)
            rewards.append(shaped_reward)
            values.append(value)
            dones.append(done)
            
            # Compute log prob for PPO
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                if isinstance(self.agent, EnsembleTradingAgent):
                    log_prob = 0  # Simplified for ensemble
                else:
                    dist = self.agent.get_action_distribution(state_tensor)
                    log_prob = dist.log_prob(torch.FloatTensor([action]).to(self.device)).cpu().item()
            log_probs.append(log_prob)
            
            episode_reward += reward
            episode_steps += 1
            state = next_state
            
            if done:
                break
        
        # Store in replay buffers
        for exp in episode_experiences:
            self.replay_buffer.push(exp)
        
        if self.her_buffer:
            self.her_buffer.store_episode(episode_experiences)
        
        # Compute advantages and returns
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            if isinstance(self.agent, EnsembleTradingAgent):
                _, next_value = self.agent.get_ensemble_action(next_state_tensor)
            else:
                _, next_value = self.agent(next_state_tensor)
            next_value = next_value.cpu().item()
        
        advantages = self.compute_gae(rewards, values, dones, next_value)
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        # Update policy
        if len(states) > 0:
            loss = self.update_policy(states, actions, log_probs, advantages, returns)
        
        # Track metrics
        self.metrics['episode_rewards'].append(episode_reward)
        if hasattr(env, 'get_metrics'):
            metrics = env.get_metrics()
            self.metrics['episode_profits'].append(metrics.get('total_return', 0))
            self.metrics['episode_sharpes'].append(metrics.get('sharpe_ratio', 0))
        
        return episode_reward, episode_steps
    
    def train(self, env, num_episodes=None):
        """Main training loop"""
        if num_episodes is None:
            num_episodes = self.config.num_episodes
        
        best_reward = -float('inf')
        
        with tqdm(total=num_episodes, desc="Training") as pbar:
            for episode in range(num_episodes):
                # Train episode
                reward, steps = self.train_episode(env)
                
                # Update progress bar
                pbar.set_postfix({
                    'reward': f'{reward:.3f}',
                    'steps': steps,
                    'lr': f'{self.metrics["learning_rates"][-1]:.6f}' if self.metrics["learning_rates"] else 0
                })
                pbar.update(1)
                
                # Evaluation
                if (episode + 1) % self.config.eval_interval == 0:
                    eval_reward = self.evaluate(env)
                    
                    if eval_reward > best_reward:
                        best_reward = eval_reward
                        self.save_checkpoint(f'models/best_advanced_model.pth')
                    
                    tqdm.write(f"\nEpisode {episode + 1} Evaluation: {eval_reward:.3f}")
                
                # Save checkpoint
                if (episode + 1) % self.config.save_interval == 0:
                    self.save_checkpoint(f'models/checkpoint_ep{episode + 1}.pth')
        
        return self.metrics
    
    def evaluate(self, env, num_episodes=5):
        """Evaluate the agent"""
        total_reward = 0
        
        for _ in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = self.select_action(state, deterministic=True)
                state, reward, done, _ = env.step([action])
                episode_reward += reward
            
            total_reward += episode_reward
        
        return total_reward / num_episodes
    
    def save_checkpoint(self, filepath):
        """Save model checkpoint"""
        Path(filepath).parent.mkdir(exist_ok=True, parents=True)
        
        checkpoint = {
            'config': self.config.__dict__,
            'metrics': self.metrics,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict()
        }
        
        if isinstance(self.agent, EnsembleTradingAgent):
            checkpoint['ensemble_states'] = [
                agent.state_dict() for agent in self.agent.agents
            ]
            checkpoint['ensemble_weights'] = self.agent.ensemble_weights
        else:
            checkpoint['agent_state'] = self.agent.state_dict()
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")


def main():
    """Main training function"""
    print("\n" + "="*80)
    print("🚀 ADVANCED RL TRADING SYSTEM")
    print("="*80)
    
    # Configuration
    config = AdvancedTrainingConfig(
        architecture='transformer',
        optimizer='muon',  # Advanced optimizer for better convergence
        learning_rate=0.001,
        num_episodes=5000,  # Production training
        eval_interval=50,
        save_interval=200,
        use_curiosity=True,
        use_her=True,
        use_augmentation=True,
        use_ensemble=False,  # Set to True for ensemble
        use_curriculum=True,
        batch_size=256,
        ppo_epochs=10,
        hidden_dim=256,
        num_layers=3
    )
    
    print("\n📋 Configuration:")
    print(f"  Architecture: {config.architecture}")
    print(f"  Optimizer: {config.optimizer}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Use Curiosity: {config.use_curiosity}")
    print(f"  Use HER: {config.use_her}")
    print(f"  Use Augmentation: {config.use_augmentation}")
    print(f"  Use Ensemble: {config.use_ensemble}")
    print(f"  Use Curriculum: {config.use_curriculum}")
    
    # Load data
    print("\n📊 Loading data...")
    df = generate_synthetic_data(1000)  # Or load real data
    
    # Split data
    train_size = int(len(df) * 0.8)
    train_df = df[:train_size]
    test_df = df[train_size:]
    
    # Get realistic trading costs
    costs = get_trading_costs('stock', 'alpaca')  # Near-zero fees for stocks
    
    # Create environment
    print("\n🌍 Creating environment...")
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 
                'Rsi', 'Macd', 'Bb_Position', 'Volume_Ratio']
    available_features = [f for f in features if f in train_df.columns]
    
    train_env = DailyTradingEnv(
        train_df,
        window_size=30,
        initial_balance=100000,
        transaction_cost=costs.commission,
        spread_pct=costs.spread_pct,
        slippage_pct=costs.slippage_pct,
        features=available_features
    )
    
    test_env = DailyTradingEnv(
        test_df,
        window_size=30,
        initial_balance=100000,
        transaction_cost=costs.commission,
        spread_pct=costs.spread_pct,
        slippage_pct=costs.slippage_pct,
        features=available_features
    )
    
    # Create agent
    print("\n🤖 Creating advanced agent...")
    input_dim = 30 * (len(available_features) + 3)
    
    if config.use_ensemble:
        agent = EnsembleTradingAgent(
            num_agents=config.num_agents,
            input_dim=input_dim,
            hidden_dim=config.hidden_dim
        )
    else:
        # Reshape input for transformer (batch, seq_len, features)
        class ReshapeWrapper(nn.Module):
            def __init__(self, agent, window_size=30):
                super().__init__()
                self.agent = agent
                self.window_size = window_size
            
            def forward(self, x):
                # Reshape from (batch, flat_features) to (batch, seq_len, features)
                if len(x.shape) == 2:
                    batch_size = x.shape[0]
                    features_per_step = x.shape[1] // self.window_size
                    x = x.view(batch_size, self.window_size, features_per_step)
                return self.agent(x)
            
            def get_action_distribution(self, x):
                if len(x.shape) == 2:
                    batch_size = x.shape[0]
                    features_per_step = x.shape[1] // self.window_size
                    x = x.view(batch_size, self.window_size, features_per_step)
                return self.agent.get_action_distribution(x)
        
        features_per_step = input_dim // 30  # 30 is window_size
        base_agent = TransformerTradingAgent(
            input_dim=features_per_step,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        agent = ReshapeWrapper(base_agent, window_size=30)
    
    # Create trainer
    print("\n🎓 Creating advanced trainer...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    trainer = AdvancedPPOTrainer(agent, config, device)
    
    # Train
    print("\n🏋️ Starting advanced training...")
    print("="*80)
    
    start_time = datetime.now()
    metrics = trainer.train(train_env, num_episodes=config.num_episodes)
    training_time = (datetime.now() - start_time).total_seconds()
    
    print(f"\n✅ Training complete in {training_time:.1f} seconds")
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    test_reward = trainer.evaluate(test_env, num_episodes=10)
    
    # Get final metrics
    test_env.reset()
    state = test_env.reset()
    done = False
    
    while not done:
        action, _ = trainer.select_action(state, deterministic=True)
        state, _, done, _ = test_env.step([action])
    
    final_metrics = test_env.get_metrics()
    
    print("\n💰 FINAL RESULTS:")
    print("="*80)
    print(f"  Test Reward:     {test_reward:.4f}")
    print(f"  Total Return:    {final_metrics.get('total_return', 0):.2%}")
    print(f"  Sharpe Ratio:    {final_metrics.get('sharpe_ratio', 0):.3f}")
    print(f"  Max Drawdown:    {final_metrics.get('max_drawdown', 0):.2%}")
    print(f"  Number of Trades: {final_metrics.get('num_trades', 0)}")
    print(f"  Win Rate:        {final_metrics.get('win_rate', 0):.2%}")
    print("="*80)
    
    # Plot training curves
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(metrics['episode_rewards'])
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    
    # Episode profits
    if metrics['episode_profits']:
        axes[0, 1].plot(metrics['episode_profits'])
        axes[0, 1].set_title('Episode Returns')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Return (%)')
    
    # Sharpe ratios
    if metrics['episode_sharpes']:
        axes[0, 2].plot(metrics['episode_sharpes'])
        axes[0, 2].set_title('Sharpe Ratios')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Sharpe')
    
    # Losses
    axes[1, 0].plot(metrics['actor_losses'], label='Actor', alpha=0.7)
    axes[1, 0].plot(metrics['critic_losses'], label='Critic', alpha=0.7)
    axes[1, 0].set_title('Training Losses')
    axes[1, 0].set_xlabel('Update')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    
    # Learning rate
    axes[1, 1].plot(metrics['learning_rates'])
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_xlabel('Update')
    axes[1, 1].set_ylabel('LR')
    
    # Final performance
    axes[1, 2].bar(['Return', 'Sharpe', 'Win Rate'], 
                   [final_metrics.get('total_return', 0) * 100,
                    final_metrics.get('sharpe_ratio', 0),
                    final_metrics.get('win_rate', 0) * 100])
    axes[1, 2].set_title('Final Performance')
    axes[1, 2].set_ylabel('Value')
    
    plt.suptitle('Advanced RL Trading System Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save results
    Path('results').mkdir(exist_ok=True)
    plt.savefig(f'results/advanced_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    
    # Save metrics
    with open(f'results/advanced_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
        json.dump({
            'config': config.__dict__,
            'final_metrics': final_metrics,
            'training_time': training_time,
            'test_reward': test_reward
        }, f, indent=2, default=float)
    
    print("\n📊 Results saved to results/")
    print("\n🎉 Advanced training complete!")


if __name__ == '__main__':
    main()