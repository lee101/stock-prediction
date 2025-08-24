import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Any, Optional
from collections import deque
import pandas as pd
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
    
    def add(self, state, action, logprob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)


class PPOTrainer:
    def __init__(
        self,
        agent,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        k_epochs: int = 4,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        log_dir: str = './traininglogs'
    ):
        self.agent = agent.to(device)
        self.device = device
        
        self.optimizer = optim.Adam([
            {'params': agent.actor_mean.parameters(), 'lr': lr_actor},
            {'params': agent.critic.parameters(), 'lr': lr_critic},
            {'params': [agent.action_var], 'lr': lr_actor}
        ])
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        
        self.memory = Memory()
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'actor_losses': [],
            'critic_losses': [],
            'total_losses': []
        }
        
        # Initialize TensorBoard writer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(f'{log_dir}/ppo_{timestamp}')
        self.global_step = 0
        self.episode_count = 0
    
    def select_action(self, state: np.ndarray, deterministic: bool = False):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, action_logprob, value = self.agent.act(state_tensor, deterministic)
            
        return (
            action.cpu().numpy().flatten(),
            action_logprob.cpu().numpy().flatten(),
            value.cpu().numpy().flatten()
        )
    
    def store_transition(self, state, action, logprob, reward, value, done):
        self.memory.add(state, action, logprob, reward, value, done)
    
    def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool]) -> tuple:
        n = len(rewards)
        advantages = np.zeros(n)
        returns = np.zeros(n)
        
        gae = 0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return returns, advantages
    
    def update(self):
        if len(self.memory.states) == 0:
            return
        
        states = torch.FloatTensor(np.array(self.memory.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.memory.actions)).to(self.device)
        old_logprobs = torch.FloatTensor(np.array(self.memory.logprobs)).to(self.device)
        
        returns, advantages = self.compute_gae(
            self.memory.rewards,
            self.memory.values,
            self.memory.dones
        )
        
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_actor_loss = 0
        total_critic_loss = 0
        total_loss = 0
        
        for _ in range(self.k_epochs):
            logprobs, values, dist_entropy = self.agent.evaluate(states, actions)
            values = values.squeeze()
            
            ratio = torch.exp(logprobs - old_logprobs)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            critic_loss = nn.MSELoss()(values, returns)
            
            entropy_loss = -dist_entropy.mean()
            
            loss = actor_loss + self.value_loss_coef * critic_loss + self.entropy_coef * entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_loss += loss.item()
        
        avg_actor_loss = total_actor_loss / self.k_epochs
        avg_critic_loss = total_critic_loss / self.k_epochs
        avg_total_loss = total_loss / self.k_epochs
        
        self.training_history['actor_losses'].append(avg_actor_loss)
        self.training_history['critic_losses'].append(avg_critic_loss)
        self.training_history['total_losses'].append(avg_total_loss)
        
        # Log to TensorBoard
        self.writer.add_scalar('Loss/Actor', avg_actor_loss, self.global_step)
        self.writer.add_scalar('Loss/Critic', avg_critic_loss, self.global_step)
        self.writer.add_scalar('Loss/Total', avg_total_loss, self.global_step)
        self.writer.add_scalar('Loss/Entropy', entropy_loss.item(), self.global_step)
        
        # Log advantages and returns statistics
        self.writer.add_scalar('Stats/Advantages_Mean', advantages.mean().item(), self.global_step)
        self.writer.add_scalar('Stats/Advantages_Std', advantages.std().item(), self.global_step)
        self.writer.add_scalar('Stats/Returns_Mean', returns.mean().item(), self.global_step)
        self.writer.add_scalar('Stats/Returns_Std', returns.std().item(), self.global_step)
        
        # Log ratio statistics
        with torch.no_grad():
            final_ratio = torch.exp(logprobs - old_logprobs)
            self.writer.add_scalar('Stats/Ratio_Mean', final_ratio.mean().item(), self.global_step)
            self.writer.add_scalar('Stats/Ratio_Max', final_ratio.max().item(), self.global_step)
            self.writer.add_scalar('Stats/Ratio_Min', final_ratio.min().item(), self.global_step)
        
        self.global_step += 1
        self.memory.clear()
        
        return {
            'actor_loss': avg_actor_loss,
            'critic_loss': avg_critic_loss,
            'total_loss': avg_total_loss
        }
    
    def train_episode(self, env, max_steps: int = 1000, deterministic: bool = False):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            action, logprob, value = self.select_action(state, deterministic)
            
            next_state, reward, done, info = env.step(action)
            
            if not deterministic:
                self.store_transition(
                    state, action, logprob, reward,
                    value[0], done
                )
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        if not deterministic:
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_lengths'].append(episode_length)
            
            # Log episode metrics to TensorBoard
            self.writer.add_scalar('Episode/Reward', episode_reward, self.episode_count)
            self.writer.add_scalar('Episode/Length', episode_length, self.episode_count)
            self.writer.add_scalar('Episode/Final_Balance', info['balance'], self.episode_count)
            
            # Get environment metrics if available
            if hasattr(env, 'get_metrics'):
                metrics = env.get_metrics()
                self.writer.add_scalar('Metrics/Total_Return', metrics.get('total_return', 0), self.episode_count)
                self.writer.add_scalar('Metrics/Sharpe_Ratio', metrics.get('sharpe_ratio', 0), self.episode_count)
                self.writer.add_scalar('Metrics/Max_Drawdown', metrics.get('max_drawdown', 0), self.episode_count)
                self.writer.add_scalar('Metrics/Num_Trades', metrics.get('num_trades', 0), self.episode_count)
                self.writer.add_scalar('Metrics/Win_Rate', metrics.get('win_rate', 0), self.episode_count)
            
            self.episode_count += 1
        
        return episode_reward, episode_length, info
    
    def train(self, env, num_episodes: int = 1000, update_interval: int = 10,
              eval_interval: int = 50, save_interval: int = 100,
              save_dir: str = './models', top_k: int = 5):
        
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        best_reward = -np.inf
        
        # Track top-k models by profitability (total return)
        top_k_models = []  # List of (episode, total_return, model_path)
        
        for episode in range(num_episodes):
            episode_reward, episode_length, info = self.train_episode(env)
            
            if (episode + 1) % update_interval == 0:
                update_info = self.update()
                print(f"Episode {episode + 1}: Updated policy - "
                      f"Actor Loss: {update_info['actor_loss']:.4f}, "
                      f"Critic Loss: {update_info['critic_loss']:.4f}")
            
            if (episode + 1) % eval_interval == 0:
                eval_reward, _, eval_info = self.train_episode(env, deterministic=True)
                metrics = env.get_metrics()
                
                total_return = metrics.get('total_return', 0)
                
                print(f"\nEpisode {episode + 1} Evaluation:")
                print(f"  Reward: {eval_reward:.4f}")
                print(f"  Total Return: {total_return:.2%}")
                print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
                print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
                print(f"  Num Trades: {metrics.get('num_trades', 0)}")
                print(f"  Win Rate: {metrics.get('win_rate', 0):.2%}\n")
                
                # Save best model by reward
                if eval_reward > best_reward:
                    best_reward = eval_reward
                    self.save_checkpoint(save_path / 'best_model.pth')
                    print(f"  New best model saved (reward: {eval_reward:.4f})")
                
                # Track top-k models by profitability
                model_info = (episode + 1, total_return, f'top_{episode + 1}_profit_{total_return:.4f}.pth')
                top_k_models.append(model_info)
                
                # Sort by total return (descending) and keep only top-k
                top_k_models.sort(key=lambda x: x[1], reverse=True)
                
                # Save current model if it's in top-k
                if len(top_k_models) <= top_k or model_info in top_k_models[:top_k]:
                    top_k_path = save_path / f'top_profit_{episode + 1}_return_{total_return:.4f}.pth'
                    self.save_checkpoint(top_k_path)
                    print(f"  Model saved to top-{top_k} profitable models")
                
                # Remove models outside top-k
                if len(top_k_models) > top_k:
                    for _, _, old_path in top_k_models[top_k:]:
                        old_file = save_path / old_path
                        if old_file.exists() and 'top_profit_' in str(old_file):
                            old_file.unlink()
                            print(f"  Removed model outside top-{top_k}: {old_path}")
                    top_k_models = top_k_models[:top_k]
            
            if (episode + 1) % save_interval == 0:
                checkpoint_path = save_path / f'checkpoint_ep{episode + 1}.pth'
                self.save_checkpoint(checkpoint_path)
        
        # Save summary of top-k models
        if top_k_models:
            summary = {
                'top_k_models': [
                    {
                        'episode': ep,
                        'total_return': ret,
                        'filename': path
                    }
                    for ep, ret, path in top_k_models
                ]
            }
            import json
            with open(save_path / 'top_k_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\nTop-{top_k} models summary saved to top_k_summary.json")
        
        return self.training_history
    
    def save_checkpoint(self, filepath: str):
        torch.save({
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history
        }, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', self.training_history)
        print(f"Checkpoint loaded from {filepath}")
    
    def close(self):
        """Close the TensorBoard writer"""
        self.writer.close()