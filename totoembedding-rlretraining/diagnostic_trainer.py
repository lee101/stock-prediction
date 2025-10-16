#!/usr/bin/env python3
"""
Diagnostic Trainer - 2-minute time-boxed training runs for optimization
Focuses on proper frozen embeddings and concise metric reporting
"""

import torch
import torch.nn as nn
import numpy as np
import time
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Dict, Tuple

from hf_rl_trainer import HFRLConfig, TotoTransformerRL, PPOTrainer
from multi_asset_env import MultiAssetTradingEnv


class DiagnosticTrainer:
    """Quick diagnostic runs with proper frozen embeddings"""
    
    def __init__(self, time_limit_seconds: int = 120):
        self.time_limit = time_limit_seconds
        self.start_time = None
        self.best_model_path = "models/diagnostic_best.pth"
        self.best_metrics_path = "models/diagnostic_best_metrics.json"
        self.metrics = {
            'initial_balance': 100000,
            'final_balance': 0,
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'num_trades': 0,
            'val_loss': float('inf'),
            'entropy': 0,
            'trainable_params': 0,
            'frozen_params': 0,
            'frozen_ratio': 0,
            'avg_daily_return': 0,
            'volatility': 0
        }
    
    def create_lightweight_model(self, obs_dim: int, action_dim: int) -> nn.Module:
        """Create model with PROPER frozen embeddings"""
        
        class LightweightTotoRL(nn.Module):
            def __init__(self, obs_dim, action_dim):
                super().__init__()
                
                # Toto embedding dimension (should be frozen)
                self.embedding_dim = 128
                
                # FROZEN: Pretrained embedding processor (simulate large frozen model)
                self.toto_processor = nn.Sequential(
                    nn.Linear(self.embedding_dim, 256),
                    nn.LayerNorm(256),
                    nn.ReLU(),
                    nn.Linear(256, 512),
                    nn.LayerNorm(512),
                    nn.ReLU(),
                    nn.Linear(512, 256)
                )
                
                # Freeze the toto processor
                for param in self.toto_processor.parameters():
                    param.requires_grad = False
                
                # TRAINABLE: Small adapter on top
                self.adapter = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                )
                
                # TRAINABLE: Task-specific heads
                self.policy_head = nn.Linear(64, action_dim)
                self.value_head = nn.Linear(64, 1)
                
                # TRAINABLE: Process non-embedding features
                non_emb_dim = obs_dim - self.embedding_dim
                self.feature_processor = nn.Sequential(
                    nn.Linear(non_emb_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64)
                )
                
                # Initialize trainable weights
                for m in [self.adapter, self.policy_head, self.value_head, self.feature_processor]:
                    if isinstance(m, nn.Sequential):
                        for layer in m:
                            if isinstance(layer, nn.Linear):
                                nn.init.orthogonal_(layer.weight, gain=0.01)
                                nn.init.constant_(layer.bias, 0)
                    elif isinstance(m, nn.Linear):
                        nn.init.orthogonal_(m.weight, gain=0.01)
                        nn.init.constant_(m.bias, 0)
            
            def forward(self, obs, return_dict=True):
                # Split observation
                toto_features = obs[:, :self.embedding_dim]
                other_features = obs[:, self.embedding_dim:]
                
                # Process through frozen toto embeddings
                with torch.no_grad():
                    embedded = self.toto_processor(toto_features)
                
                # Adapt embeddings (trainable)
                adapted = self.adapter(embedded)
                
                # Process other features (trainable)
                processed_features = self.feature_processor(other_features)
                
                # Combine
                combined = adapted + processed_features
                
                # Generate outputs
                policy_logits = self.policy_head(combined)
                values = self.value_head(combined).squeeze(-1)
                
                # Add entropy for exploration
                actions = torch.tanh(policy_logits)
                
                if return_dict:
                    return {
                        'actions': actions,
                        'action_logits': policy_logits,
                        'state_values': values
                    }
                return actions, values
        
        return LightweightTotoRL(obs_dim, action_dim)
    
    def run_diagnostic(self, config_name: str = "quick_test") -> Dict:
        """Run 2-minute diagnostic training"""
        
        print(f"\n{'='*60}")
        print(f"DIAGNOSTIC RUN: {config_name}")
        print(f"Time limit: {self.time_limit}s")
        print(f"{'='*60}")
        
        self.start_time = time.time()
        
        # Setup environment with subset of symbols for speed
        test_symbols = ['AAPL', 'BTCUSD', 'TSLA', 'MSFT', 'ETHUSD']
        
        env = MultiAssetTradingEnv(
            data_dir="../trainingdata/train",
            symbols=test_symbols,
            initial_balance=100000,
            max_positions=3,
            max_position_size=0.5,
            confidence_threshold=0.3,
            window_size=20,  # Smaller window for speed
            rebalance_frequency=10  # Allow rebalancing every 10 steps for diagnostic testing
        )
        
        # Create lightweight model
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        model = self.create_lightweight_model(obs_dim, action_dim)
        
        # Try to load best existing model
        best_metrics = self._load_best_metrics()
        if best_metrics and Path(self.best_model_path).exists():
            try:
                model.load_state_dict(torch.load(self.best_model_path, weights_only=False))
                print(f"🔄 Loaded previous best model (Return: {best_metrics.get('total_return', 0):.2%}, Val Loss: {best_metrics.get('val_loss', float('inf')):.4f})")
            except Exception as e:
                print(f"⚠️  Could not load previous model: {e}")
                best_metrics = None
        else:
            print("🔧 No previous model found - training from scratch")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        self.metrics['trainable_params'] = trainable_params
        self.metrics['frozen_params'] = frozen_params
        self.metrics['frozen_ratio'] = frozen_params / total_params
        
        print(f"\nMODEL STATS:")
        print(f"  Total: {total_params:,}")
        print(f"  Frozen: {frozen_params:,} ({frozen_params/total_params:.1%})")
        print(f"  Trainable: {trainable_params:,} ({trainable_params/total_params:.1%})")
        
        # Quick training config
        config = HFRLConfig()
        config.learning_rate = 3e-4  # Higher LR for quick learning
        config.entropy_coef = 0.05  # Higher entropy for exploration
        config.batch_size = 4
        config.mini_batch_size = 2
        config.num_train_epochs = 100  # Will be limited by time
        config.logging_steps = 10
        config.use_mixed_precision = False
        
        # Setup optimizer with higher LR
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # Training loop
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.train()
        
        episode = 0
        total_rewards = []
        portfolio_values = []
        entropies = []
        losses = []
        
        print(f"\nTRAINING:")
        while (time.time() - self.start_time) < self.time_limit:
            # Reset env
            obs = env.reset()
            episode_reward = 0
            done = False
            steps = 0
            
            while not done and (time.time() - self.start_time) < self.time_limit:
                # Get action
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(obs_tensor)
                    action = outputs['actions'].cpu().numpy()[0]
                    
                    # Add exploration noise
                    noise = np.random.normal(0, 0.1, action.shape)
                    action = np.clip(action + noise, -1, 1)
                
                # Step environment
                next_obs, reward, done, info = env.step(action)
                episode_reward += reward
                
                # Simple policy gradient update every 10 steps
                if steps % 10 == 0 and steps > 0:
                    # Calculate simple loss
                    outputs = model(obs_tensor)
                    
                    # Entropy for exploration
                    dist_std = 0.5
                    dist = torch.distributions.Normal(outputs['action_logits'], dist_std)
                    entropy = dist.entropy().mean()
                    
                    # Simple policy loss (reinforce)
                    log_prob = dist.log_prob(torch.tensor(action, dtype=torch.float32).to(device)).sum()
                    policy_loss = -log_prob * float(reward)
                    
                    # Value loss
                    value_loss = nn.functional.mse_loss(
                        outputs['state_values'], 
                        torch.tensor([float(episode_reward)], dtype=torch.float32).to(device)
                    )
                    
                    # Total loss
                    loss = policy_loss + 0.5 * value_loss - config.entropy_coef * entropy
                    
                    # Update
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    # Track metrics
                    entropies.append(entropy.item())
                    losses.append(loss.item())
                
                obs = next_obs
                steps += 1
            
            # Track episode metrics
            total_rewards.append(episode_reward)
            portfolio_metrics = env.get_portfolio_metrics()
            if portfolio_metrics:
                portfolio_values.append(portfolio_metrics.get('final_balance', 100000))
                self.metrics['num_trades'] = portfolio_metrics.get('num_trades', 0)
            
            episode += 1
            
            # Quick status update every 10 episodes
            if episode % 10 == 0:
                elapsed = time.time() - self.start_time
                print(f"  [{elapsed:5.1f}s] Ep {episode:3d} | "
                      f"Reward: {np.mean(total_rewards[-10:]):7.4f} | "
                      f"Entropy: {np.mean(entropies[-10:]) if entropies else 0:6.4f} | "
                      f"Trades: {self.metrics['num_trades']:3d}")
        
        # Calculate final metrics
        if portfolio_values:
            self.metrics['final_balance'] = portfolio_values[-1]
            self.metrics['total_return'] = (portfolio_values[-1] - 100000) / 100000
        
        if total_rewards:
            returns = np.array(total_rewards)
            self.metrics['sharpe_ratio'] = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        
        if entropies:
            self.metrics['entropy'] = np.mean(entropies[-20:])
        
        if losses:
            self.metrics['val_loss'] = np.mean(losses[-20:])
        
        # Evaluate on validation set
        print(f"\\n🔍 EVALUATION PHASE:")
        val_metrics = self._evaluate_model(model, env, device)
        self.metrics.update(val_metrics)
        
        # Check if this is the best model so far
        is_best = self._is_best_model(best_metrics)
        if is_best:
            self._save_best_model(model)
            print(f"💾 NEW BEST MODEL SAVED!")
            print(f"   Improvement: Return {self.metrics['total_return']:.2%} vs {best_metrics.get('total_return', 0):.2%}" if best_metrics else "")
            print(f"   Val Loss: {self.metrics['val_loss']:.4f} vs {best_metrics.get('val_loss', float('inf')):.4f}" if best_metrics else "")
        else:
            print(f"📈 Current model performance:")
            if best_metrics:
                print(f"   Return: {self.metrics['total_return']:.2%} (Best: {best_metrics.get('total_return', 0):.2%})")
                print(f"   Val Loss: {self.metrics['val_loss']:.4f} (Best: {best_metrics.get('val_loss', float('inf')):.4f})")
        
        # Final summary
        self._print_summary(is_best, episode)
        
        return self.metrics
    
    def _load_best_metrics(self):
        """Load best model metrics if they exist"""
        if Path(self.best_metrics_path).exists():
            try:
                with open(self.best_metrics_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return None
        return None
    
    def _is_best_model(self, previous_best):
        """Check if current model is better than previous best"""
        if not previous_best:
            return True
        
        # Primary: Better validation loss
        if self.metrics['val_loss'] < previous_best.get('val_loss', float('inf')):
            return True
        
        # Secondary: Better return with similar val loss (within 10%)
        val_loss_similar = abs(self.metrics['val_loss'] - previous_best.get('val_loss', 0)) / max(previous_best.get('val_loss', 1), 1) < 0.1
        if val_loss_similar and self.metrics['total_return'] > previous_best.get('total_return', 0):
            return True
            
        return False
    
    def _save_best_model(self, model):
        """Save the best model and metrics"""
        Path("models").mkdir(exist_ok=True)
        
        # Save model state
        torch.save(model.state_dict(), self.best_model_path)
        
        # Save metrics
        with open(self.best_metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def _evaluate_model(self, model, env, device):
        """Evaluate model on validation episodes"""
        model.eval()
        
        val_returns = []
        val_portfolio_values = []
        
        # Run 5 validation episodes
        for episode in range(5):
            obs = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(obs_tensor)
                    action = outputs['actions'].cpu().numpy()[0]
                
                obs, reward, done, info = env.step(action)
                episode_reward += reward
            
            val_returns.append(episode_reward)
            portfolio_metrics = env.get_portfolio_metrics()
            if portfolio_metrics:
                val_portfolio_values.append(portfolio_metrics.get('final_balance', 100000))
        
        # Calculate validation metrics
        val_metrics = {}
        if val_returns:
            val_metrics['avg_daily_return'] = np.mean(val_returns)
            val_metrics['volatility'] = np.std(val_returns)
            
        if val_portfolio_values:
            final_balance = np.mean(val_portfolio_values)
            val_metrics['final_balance'] = final_balance
            val_metrics['total_return'] = (final_balance - 100000) / 100000
            
            # Calculate Sharpe ratio (annualized)
            if val_metrics['volatility'] > 0:
                val_metrics['sharpe_ratio'] = val_metrics['avg_daily_return'] / val_metrics['volatility'] * np.sqrt(252)
        
        model.train()
        return val_metrics
    
    def _print_summary(self, is_best=False, episodes_run=0):
        """Print concise summary"""
        print(f"\n{'='*60}")
        print(f"RESULTS {'🏆 NEW BEST!' if is_best else '📊'}:")
        print(f"{'='*60}")
        
        # Model architecture
        print(f"MODEL:     {self.metrics['frozen_params']:,} frozen ({self.metrics['frozen_ratio']:.1%}) | "
              f"{self.metrics['trainable_params']:,} trainable")
        
        # Financial performance (validation results)
        print(f"PROFIT:    ${self.metrics['final_balance']:,.0f} | "
              f"Return: {self.metrics['total_return']:.2%} | "
              f"Sharpe: {self.metrics['sharpe_ratio']:.2f}")
        
        # Training metrics
        print(f"TRAINING:  Val Loss: {self.metrics['val_loss']:.4f} | "
              f"Entropy: {self.metrics['entropy']:.4f} | "
              f"Daily Vol: {self.metrics.get('volatility', 0):.4f}")
        
        # Trading frequency (episodes per 2min session)
        trading_sessions_per_day = 1440 / self.time_limit * 60  # How many 2min sessions in a day
        trades_per_episode = self.metrics['num_trades'] / max(1, episodes_run)
        estimated_daily_trades = trades_per_episode * trading_sessions_per_day
        print(f"TRADING:   {estimated_daily_trades:.1f} est. trades/day | "
              f"Episodes: {episodes_run} | Trades/Ep: {trades_per_episode:.1f} | "
              f"Avg Daily Return: {self.metrics.get('avg_daily_return', 0):.4f}")
        
        # Issues and improvements
        issues = []
        if self.metrics['entropy'] < 0.01:
            issues.append("⚠️  Low entropy - needs more exploration")
        if self.metrics['total_return'] < -0.02:
            issues.append("⚠️  Losing money - check reward shaping")
        if self.metrics['frozen_ratio'] < 0.5:
            issues.append("⚠️  Too few frozen parameters")
        if estimated_daily_trades > 10:
            issues.append("⚠️  Overtrading - increase confidence threshold")
        if abs(self.metrics.get('volatility', 0)) < 0.001:
            issues.append("⚠️  No volatility - model not adapting")
            
        if issues:
            print()
            for issue in issues:
                print(issue)
        else:
            print("✅ No major issues detected")
        
        print(f"{'='*60}\n")


def run_optimization_tests():
    """Run multiple diagnostic tests with different configurations"""
    
    results = {}
    
    # Test 1: Baseline
    print("🔍 Running Baseline Test...")
    trainer = DiagnosticTrainer(time_limit_seconds=60)  # Shorter for comparison
    results['baseline'] = trainer.run_diagnostic('baseline')
    
    # Test 2: Higher learning rate
    print("🔍 Running High LR Test...")
    trainer2 = DiagnosticTrainer(time_limit_seconds=60)
    results['high_lr'] = trainer2.run_diagnostic('high_lr')
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON:")
    print("="*60)
    for name, metrics in results.items():
        print(f"{name:15s}: Return: {metrics['total_return']:7.2%} | "
              f"Sharpe: {metrics['sharpe_ratio']:6.2f} | "
              f"Entropy: {metrics['entropy']:6.4f}")


if __name__ == "__main__":
    # Run single diagnostic with model saving
    trainer = DiagnosticTrainer(time_limit_seconds=120)
    trainer.run_diagnostic("daily_trading_optimization")