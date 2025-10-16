#!/usr/bin/env python3
"""
Main Training Script for Toto RL System
Integrates embedding model with RL training for multi-asset trading
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

from rl_trainer import TotoRLTrainer
from multi_asset_env import MultiAssetTradingEnv

# Import embedding system
import sys
sys.path.append('../totoembedding')
from embedding_model import TotoEmbeddingModel, TotoEmbeddingDataset
from pretrained_loader import PretrainedWeightLoader


class TotoRLPipeline:
    """Complete pipeline for training Toto RL system"""
    
    def __init__(self, config_path: str = None):
        # Load configuration
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self.get_default_config()
        
        # Setup paths
        self.setup_paths()
        
        # Initialize components
        self.pretrained_loader = PretrainedWeightLoader()
        self.embedding_model = None
        self.rl_trainer = None
        
        print("TotoRLPipeline initialized")
        print(f"Data directory: {self.config['data']['train_dir']}")
        print(f"Output directory: {self.config['output']['model_dir']}")
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'data': {
                'train_dir': '../trainingdata/train',
                'test_dir': '../trainingdata/test',
                'symbols': [
                    'AAPL', 'ADBE', 'ADSK', 'BTCUSD', 'COIN', 'COUR', 'CRWD',
                    'ETHUSD', 'GOOG', 'LTCUSD', 'MSFT', 'NET', 'NFLX', 'NVDA',
                    'PAXGUSD', 'PYPL', 'SAP', 'SONY', 'TSLA', 'U', 'UNIUSD'
                ]
            },
            'embedding': {
                'pretrained_model': '../training/models/modern_best_sharpe.pth',
                'embedding_dim': 128,
                'freeze_backbone': True,
                'train_embeddings': False,  # Whether to train embedding model first
                'embedding_epochs': 50
            },
            'environment': {
                'initial_balance': 100000.0,
                'max_positions': 10,
                'transaction_cost': 0.001,
                'window_size': 30
            },
            'agent': {
                'hidden_dims': [512, 256, 128],
                'dropout': 0.2,
                'use_layer_norm': True
            },
            'training': {
                'episodes': 2000,
                'batch_size': 128,
                'learning_rate': 1e-4,
                'gamma': 0.99,
                'epsilon_start': 1.0,
                'epsilon_end': 0.05,
                'epsilon_decay': 0.995,
                'buffer_size': 100000,
                'update_freq': 4,
                'target_update_freq': 100
            },
            'output': {
                'model_dir': 'models',
                'results_dir': 'results',
                'plots_dir': 'plots'
            }
        }
    
    def setup_paths(self):
        """Setup output directories"""
        for path_key in ['model_dir', 'results_dir', 'plots_dir']:
            Path(self.config['output'][path_key]).mkdir(parents=True, exist_ok=True)
    
    def train_embedding_model(self) -> str:
        """Train or load embedding model"""
        embedding_model_path = f"{self.config['output']['model_dir']}/toto_embeddings.pth"
        
        if Path(embedding_model_path).exists() and not self.config['embedding']['train_embeddings']:
            print("Loading existing embedding model...")
            return embedding_model_path
        
        if not self.config['embedding']['train_embeddings']:
            print("Skipping embedding training - using pretrained backbone only")
            return None
        
        print("Training embedding model...")
        
        # Create embedding model
        embedding_model = TotoEmbeddingModel(
            pretrained_model_path=self.config['embedding']['pretrained_model'],
            embedding_dim=self.config['embedding']['embedding_dim'],
            num_symbols=len(self.config['data']['symbols']),
            freeze_backbone=self.config['embedding']['freeze_backbone']
        )
        
        # Create dataset
        dataset = TotoEmbeddingDataset(
            data_dir=self.config['data']['train_dir'],
            symbols=self.config['data']['symbols']
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=64, 
            shuffle=True
        )
        
        # Train embedding model (simplified training loop)
        optimizer = torch.optim.AdamW(embedding_model.parameters(), lr=1e-4)
        criterion = torch.nn.MSELoss()
        
        embedding_model.train()
        for epoch in range(self.config['embedding']['embedding_epochs']):
            total_loss = 0
            
            for batch in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = embedding_model(
                    price_data=batch['price_data'],
                    symbol_ids=batch['symbol_id'],
                    timestamps=batch['timestamp'],
                    market_regime=batch['regime']
                )
                
                # Simple prediction task - predict next return
                embeddings = outputs['embeddings']
                predicted_return = torch.mean(embeddings, dim=-1)  # Simplified
                actual_return = batch['target_return']
                
                loss = criterion(predicted_return, actual_return)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            if epoch % 10 == 0:
                print(f"Embedding Epoch {epoch}: Loss = {avg_loss:.6f}")
        
        # Save embedding model
        torch.save({
            'state_dict': embedding_model.state_dict(),
            'config': self.config['embedding']
        }, embedding_model_path)
        
        print(f"Embedding model saved to {embedding_model_path}")
        return embedding_model_path
    
    def create_rl_trainer(self, embedding_model_path: str = None) -> TotoRLTrainer:
        """Create and configure RL trainer"""
        env_config = {
            'data_dir': self.config['data']['train_dir'],
            'symbols': self.config['data']['symbols'],
            'embedding_model_path': embedding_model_path,
            **self.config['environment']
        }
        
        trainer = TotoRLTrainer(
            env_config=env_config,
            agent_config=self.config['agent'],
            training_config=self.config['training'],
            pretrained_model_path=self.config['embedding']['pretrained_model']
        )
        
        return trainer
    
    def train_rl_agent(self, trainer: TotoRLTrainer) -> Dict[str, Any]:
        """Train the RL agent"""
        print("Training RL agent...")
        
        # Train the agent
        final_metrics = trainer.train()
        
        # Save training results
        results = {
            'final_metrics': final_metrics,
            'episode_rewards': trainer.episode_rewards,
            'episode_metrics': trainer.episode_metrics,
            'config': self.config
        }
        
        results_path = f"{self.config['output']['results_dir']}/training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Training results saved to {results_path}")
        return results
    
    def evaluate_performance(self, trainer: TotoRLTrainer) -> Dict[str, Any]:
        """Evaluate trained model performance"""
        print("Evaluating model performance...")
        
        # Test on held-out data
        test_env_config = self.config['environment'].copy()
        test_env_config['data_dir'] = self.config['data']['test_dir']
        
        test_env = MultiAssetTradingEnv(**test_env_config)
        
        # Run evaluation episodes
        eval_results = []
        num_eval_episodes = 10
        
        for episode in range(num_eval_episodes):
            obs = test_env.reset()
            episode_reward = 0
            
            while True:
                action = trainer.select_action(obs, epsilon=0.0, eval_mode=True)
                obs, reward, done, info = test_env.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            metrics = test_env.get_portfolio_metrics()
            metrics['episode_reward'] = episode_reward
            eval_results.append(metrics)
        
        # Aggregate results
        eval_summary = {}
        for key in eval_results[0].keys():
            values = [r[key] for r in eval_results if isinstance(r[key], (int, float))]
            if values:
                eval_summary[f'{key}_mean'] = np.mean(values)
                eval_summary[f'{key}_std'] = np.std(values)
        
        # Save evaluation results
        eval_path = f"{self.config['output']['results_dir']}/evaluation_results.json"
        with open(eval_path, 'w') as f:
            json.dump({
                'summary': eval_summary,
                'episodes': eval_results
            }, f, indent=2, default=str)
        
        print(f"Evaluation results saved to {eval_path}")
        return eval_summary
    
    def create_visualizations(self, trainer: TotoRLTrainer, eval_results: Dict[str, Any]):
        """Create training and evaluation visualizations"""
        print("Creating visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Toto RL Training Results', fontsize=16, fontweight='bold')
        
        # 1. Episode Rewards
        ax1 = axes[0, 0]
        rewards = trainer.episode_rewards
        if rewards:
            episodes = range(len(rewards))
            ax1.plot(episodes, rewards, alpha=0.3, color='blue')
            
            # Moving average
            window = 50
            if len(rewards) > window:
                moving_avg = pd.Series(rewards).rolling(window).mean()
                ax1.plot(episodes, moving_avg, color='red', linewidth=2, label=f'MA({window})')
                ax1.legend()
            
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.set_title('Training Rewards')
            ax1.grid(True, alpha=0.3)
        
        # 2. Portfolio Performance
        ax2 = axes[0, 1]
        if trainer.episode_metrics and trainer.episode_metrics[0]:
            returns = [m.get('total_return', 0) for m in trainer.episode_metrics if m]
            if returns:
                episodes = range(len(returns))
                ax2.plot(episodes, np.array(returns) * 100, color='green', linewidth=2)
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Total Return (%)')
                ax2.set_title('Portfolio Returns')
                ax2.grid(True, alpha=0.3)
        
        # 3. Sharpe Ratio Evolution
        ax3 = axes[0, 2]
        if trainer.episode_metrics and trainer.episode_metrics[0]:
            sharpe_ratios = [m.get('sharpe_ratio', 0) for m in trainer.episode_metrics if m]
            if sharpe_ratios:
                episodes = range(len(sharpe_ratios))
                ax3.plot(episodes, sharpe_ratios, color='orange', linewidth=2)
                ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Sharpe=1.0')
                ax3.set_xlabel('Episode')
                ax3.set_ylabel('Sharpe Ratio')
                ax3.set_title('Risk-Adjusted Returns')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
        
        # 4. Drawdown Analysis
        ax4 = axes[1, 0]
        if trainer.episode_metrics and trainer.episode_metrics[0]:
            drawdowns = [abs(m.get('max_drawdown', 0)) * 100 for m in trainer.episode_metrics if m]
            if drawdowns:
                episodes = range(len(drawdowns))
                ax4.plot(episodes, drawdowns, color='red', linewidth=2)
                ax4.fill_between(episodes, drawdowns, alpha=0.3, color='red')
                ax4.set_xlabel('Episode')
                ax4.set_ylabel('Max Drawdown (%)')
                ax4.set_title('Maximum Drawdown')
                ax4.grid(True, alpha=0.3)
        
        # 5. Trading Activity
        ax5 = axes[1, 1]
        if trainer.episode_metrics and trainer.episode_metrics[0]:
            num_trades = [m.get('num_trades', 0) for m in trainer.episode_metrics if m]
            if num_trades:
                episodes = range(len(num_trades))
                ax5.plot(episodes, num_trades, color='purple', linewidth=2)
                ax5.set_xlabel('Episode')
                ax5.set_ylabel('Number of Trades')
                ax5.set_title('Trading Activity')
                ax5.grid(True, alpha=0.3)
        
        # 6. Evaluation Summary
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        if eval_results:
            # Create summary text
            summary_text = "Final Evaluation Results:\n\n"
            key_metrics = [
                'total_return_mean', 'sharpe_ratio_mean', 'max_drawdown_mean',
                'num_trades_mean', 'total_fees_mean'
            ]
            
            for key in key_metrics:
                if key in eval_results:
                    value = eval_results[key]
                    if 'return' in key or 'drawdown' in key:
                        summary_text += f"{key.replace('_mean', '').replace('_', ' ').title()}: {value:.2%}\n"
                    elif 'ratio' in key:
                        summary_text += f"{key.replace('_mean', '').replace('_', ' ').title()}: {value:.2f}\n"
                    else:
                        summary_text += f"{key.replace('_mean', '').replace('_', ' ').title()}: {value:.2f}\n"
            
            ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
                    fontsize=12, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f"{self.config['output']['plots_dir']}/training_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training visualization saved to {plot_path}")
        
        plt.show()
    
    def run_full_pipeline(self):
        """Run the complete Toto RL training pipeline"""
        print("\n" + "="*60)
        print("STARTING TOTO RL TRAINING PIPELINE")
        print("="*60)
        
        start_time = datetime.now()
        
        try:
            # Step 1: Train/load embedding model
            embedding_model_path = self.train_embedding_model()
            
            # Step 2: Create RL trainer
            rl_trainer = self.create_rl_trainer(embedding_model_path)
            
            # Step 3: Train RL agent
            training_results = self.train_rl_agent(rl_trainer)
            
            # Step 4: Evaluate performance
            eval_results = self.evaluate_performance(rl_trainer)
            
            # Step 5: Create visualizations
            self.create_visualizations(rl_trainer, eval_results)
            
            # Summary
            end_time = datetime.now()
            duration = end_time - start_time
            
            print("\n" + "="*60)
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print("="*60)
            print(f"Total Duration: {duration}")
            print(f"Final Portfolio Return: {eval_results.get('total_return_mean', 0):.2%}")
            print(f"Final Sharpe Ratio: {eval_results.get('sharpe_ratio_mean', 0):.2f}")
            print(f"Max Drawdown: {eval_results.get('max_drawdown_mean', 0):.2%}")
            print(f"Models saved to: {self.config['output']['model_dir']}")
            print(f"Results saved to: {self.config['output']['results_dir']}")
            
        except Exception as e:
            print(f"Pipeline failed with error: {e}")
            raise
    
    def save_config(self, filepath: str = None):
        """Save current configuration"""
        if filepath is None:
            filepath = f"{self.config['output']['results_dir']}/config.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Configuration saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Train Toto RL System')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--episodes', type=int, default=2000, help='Number of training episodes')
    parser.add_argument('--symbols', type=str, nargs='+', help='Symbols to trade')
    parser.add_argument('--balance', type=float, default=100000, help='Initial balance')
    parser.add_argument('--train-embeddings', action='store_true', help='Train embedding model')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = TotoRLPipeline(args.config)
    
    # Override config with command line arguments
    if args.episodes:
        pipeline.config['training']['episodes'] = args.episodes
    if args.symbols:
        pipeline.config['data']['symbols'] = args.symbols
    if args.balance:
        pipeline.config['environment']['initial_balance'] = args.balance
    if args.train_embeddings:
        pipeline.config['embedding']['train_embeddings'] = True
    
    # Save updated config
    pipeline.save_config()
    
    # Run pipeline
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()