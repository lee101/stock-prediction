#!/usr/bin/env python3
"""
Automated Training Improvement Cycle
Trains models iteratively, analyzes results, and automatically improves hyperparameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training/improvement_cycle.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class StableStockDataset(Dataset):
    """Stable dataset with proper normalization"""
    
    def __init__(self, n_samples=10000, sequence_length=60):
        self.sequence_length = sequence_length
        
        # Generate synthetic data
        np.random.seed(42)  # For reproducibility
        
        # Generate price data
        returns = np.random.normal(0.0001, 0.01, n_samples)
        price = 100 * np.exp(np.cumsum(returns))
        
        # Create features
        features = []
        for i in range(len(price) - 1):
            feature = [
                price[i],
                price[i] * (1 + np.random.normal(0, 0.001)),  # Open
                price[i] * (1 + abs(np.random.normal(0, 0.002))),  # High  
                price[i] * (1 - abs(np.random.normal(0, 0.002))),  # Low
                np.random.lognormal(10, 0.5)  # Volume
            ]
            features.append(feature)
        
        features = np.array(features)
        
        # Proper normalization
        self.mean = features.mean(axis=0, keepdims=True)
        self.std = features.std(axis=0, keepdims=True) + 1e-8
        self.features = (features - self.mean) / self.std
        
        # Create targets
        price_changes = np.diff(price) / price[:-1]
        self.targets = np.zeros(len(price_changes), dtype=np.int64)
        self.targets[price_changes < -0.001] = 0
        self.targets[price_changes > 0.001] = 2  
        self.targets[(price_changes >= -0.001) & (price_changes <= 0.001)] = 1
        
        # Convert to tensors
        self.features = torch.FloatTensor(self.features)
        self.targets = torch.LongTensor(self.targets)
        
        logger.info(f"Dataset created: {len(self.features)} samples, {self.features.shape[1]} features")
        logger.info(f"Target distribution: {np.bincount(self.targets.numpy())}")
    
    def __len__(self):
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        x = self.features[idx:idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length]
        return x, y


class StableTransformer(nn.Module):
    """Stable Transformer with proper initialization"""
    
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        
        # Smaller model for stability
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 3)
        
        # Careful initialization
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)
    
    def forward(self, x):
        # Add checks for NaN
        if torch.isnan(x).any():
            logger.warning("NaN in input!")
            x = torch.nan_to_num(x, nan=0.0)
        
        x = self.input_projection(x)
        x = self.input_norm(x)
        x = self.transformer(x)
        x = self.output_norm(x[:, -1, :])
        x = self.classifier(x)
        
        return x


class ImprovementCycleTrainer:
    """Automated training with improvement cycles"""
    
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cycle_results = []
        self.best_config = None
        self.best_loss = float('inf')
        
        # Create main results directory
        self.results_dir = Path('training/improvement_cycles')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Improvement Cycle Trainer initialized on {self.device}")
    
    def train_single_cycle(self, config: Dict[str, Any], cycle_num: int) -> Dict[str, Any]:
        """Train a single cycle with given config"""
        
        logger.info(f"\n{'='*50}")
        logger.info(f"CYCLE {cycle_num}: Starting training")
        logger.info(f"Config: {json.dumps(config, indent=2)}")
        logger.info(f"{'='*50}\n")
        
        # Create cycle directory
        cycle_dir = self.results_dir / f'cycle_{cycle_num}'
        cycle_dir.mkdir(exist_ok=True)
        
        # Save config
        with open(cycle_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # Dataset
        dataset = StableStockDataset(n_samples=5000, sequence_length=config['sequence_length'])
        train_loader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0  # Avoid multiprocessing issues
        )
        
        # Model
        model = StableTransformer(
            input_dim=5,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            dropout=config['dropout']
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Training metrics
        train_losses = []
        train_accs = []
        best_cycle_loss = float('inf')
        
        # Training loop
        for epoch in range(config['num_epochs']):
            model.train()
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0
            nan_batches = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                output = model(data)
                loss = criterion(output, target)
                
                # Check for NaN
                if torch.isnan(loss):
                    nan_batches += 1
                    continue
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                
                # Metrics
                epoch_loss += loss.item()
                pred = output.argmax(dim=1)
                epoch_correct += (pred == target).sum().item()
                epoch_total += target.size(0)
            
            # Calculate epoch metrics
            if epoch_total > 0:
                avg_loss = epoch_loss / (len(train_loader) - nan_batches) if (len(train_loader) - nan_batches) > 0 else float('inf')
                accuracy = epoch_correct / epoch_total
            else:
                avg_loss = float('inf')
                accuracy = 0.0
            
            train_losses.append(avg_loss)
            train_accs.append(accuracy)
            
            if avg_loss < best_cycle_loss:
                best_cycle_loss = avg_loss
                torch.save(model.state_dict(), cycle_dir / 'best_model.pth')
            
            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch}/{config['num_epochs']}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}, NaN batches={nan_batches}")
        
        # Save training history
        history = {
            'losses': train_losses,
            'accuracies': train_accs,
            'config': config,
            'best_loss': best_cycle_loss,
            'final_loss': train_losses[-1] if train_losses else float('inf'),
            'final_accuracy': train_accs[-1] if train_accs else 0.0,
            'improvement': (train_losses[0] - train_losses[-1]) / train_losses[0] * 100 if len(train_losses) > 1 and train_losses[0] != 0 else 0
        }
        
        with open(cycle_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # Plot training curves
        self.plot_cycle_results(train_losses, train_accs, cycle_dir)
        
        return history
    
    def analyze_cycle(self, history: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cycle results and suggest improvements"""
        
        improvements = {
            'learning_rate': None,
            'batch_size': None,
            'hidden_dim': None,
            'num_layers': None,
            'dropout': None,
            'weight_decay': None
        }
        
        config = history['config']
        
        # Analyze loss behavior
        if history['improvement'] < 5:  # Less than 5% improvement
            # Try increasing learning rate
            improvements['learning_rate'] = min(config['learning_rate'] * 2, 1e-2)
            logger.info("Low improvement - increasing learning rate")
        
        elif history['improvement'] > 50:  # Very high improvement, might be unstable
            # Reduce learning rate for stability
            improvements['learning_rate'] = config['learning_rate'] * 0.5
            logger.info("High improvement - reducing learning rate for stability")
        
        # Check final loss
        if history['final_loss'] > 0.9:  # High loss
            # Increase model capacity
            improvements['hidden_dim'] = min(config['hidden_dim'] * 2, 256)
            improvements['num_layers'] = min(config['num_layers'] + 1, 6)
            logger.info("High final loss - increasing model capacity")
        
        # Check accuracy
        if history['final_accuracy'] < 0.4:  # Poor accuracy
            # Adjust regularization
            improvements['dropout'] = max(config['dropout'] * 0.5, 0.05)
            improvements['weight_decay'] = config.get('weight_decay', 0.01) * 0.5
            logger.info("Poor accuracy - reducing regularization")
        
        elif history['final_accuracy'] > 0.6:  # Good accuracy, might overfit
            # Increase regularization
            improvements['dropout'] = min(config['dropout'] * 1.5, 0.3)
            improvements['weight_decay'] = config.get('weight_decay', 0.01) * 1.5
            logger.info("Good accuracy - increasing regularization")
        
        # Remove None values
        improvements = {k: v for k, v in improvements.items() if v is not None}
        
        return improvements
    
    def create_improved_config(self, base_config: Dict[str, Any], improvements: Dict[str, Any]) -> Dict[str, Any]:
        """Create improved configuration"""
        
        new_config = base_config.copy()
        new_config.update(improvements)
        
        # Ensure valid values
        new_config['num_heads'] = min(new_config['num_heads'], new_config['hidden_dim'] // 8)
        new_config['num_heads'] = max(new_config['num_heads'], 1)
        
        return new_config
    
    def plot_cycle_results(self, losses: List[float], accs: List[float], save_dir: Path):
        """Plot training curves for a cycle"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(losses, 'b-', label='Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Accuracy plot
        ax2.plot(accs, 'g-', label='Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Accuracy')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / 'training_curves.png', dpi=100)
        plt.close()
    
    def plot_improvement_summary(self):
        """Plot summary of all cycles"""
        
        if not self.cycle_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract metrics
        cycles = list(range(1, len(self.cycle_results) + 1))
        final_losses = [r['final_loss'] for r in self.cycle_results]
        final_accs = [r['final_accuracy'] for r in self.cycle_results]
        improvements = [r['improvement'] for r in self.cycle_results]
        learning_rates = [r['config']['learning_rate'] for r in self.cycle_results]
        
        # Loss progression
        axes[0, 0].plot(cycles, final_losses, 'b-o', label='Final Loss')
        axes[0, 0].set_xlabel('Cycle')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Progression')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Accuracy progression
        axes[0, 1].plot(cycles, final_accs, 'g-o', label='Final Accuracy')
        axes[0, 1].set_xlabel('Cycle')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy Progression')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Improvement per cycle
        axes[1, 0].bar(cycles, improvements, color='orange', alpha=0.7)
        axes[1, 0].set_xlabel('Cycle')
        axes[1, 0].set_ylabel('Improvement (%)')
        axes[1, 0].set_title('Training Improvement per Cycle')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate evolution
        axes[1, 1].semilogy(cycles, learning_rates, 'r-o', label='Learning Rate')
        axes[1, 1].set_xlabel('Cycle')
        axes[1, 1].set_ylabel('Learning Rate (log scale)')
        axes[1, 1].set_title('Learning Rate Evolution')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.suptitle('Training Improvement Cycle Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'improvement_summary.png', dpi=150)
        plt.close()
    
    def run_improvement_cycles(self, num_cycles: int = 5):
        """Run multiple improvement cycles"""
        
        logger.info(f"\nStarting {num_cycles} improvement cycles")
        logger.info("="*60)
        
        current_config = self.base_config.copy()
        
        for cycle in range(1, num_cycles + 1):
            # Train cycle
            history = self.train_single_cycle(current_config, cycle)
            self.cycle_results.append(history)
            
            # Update best configuration
            if history['final_loss'] < self.best_loss:
                self.best_loss = history['final_loss']
                self.best_config = current_config.copy()
                logger.info(f"New best configuration found! Loss: {self.best_loss:.4f}")
            
            # Analyze and improve
            if cycle < num_cycles:  # Don't improve on last cycle
                improvements = self.analyze_cycle(history)
                current_config = self.create_improved_config(current_config, improvements)
                
                logger.info(f"\nCycle {cycle} Results:")
                logger.info(f"  Final Loss: {history['final_loss']:.4f}")
                logger.info(f"  Final Accuracy: {history['final_accuracy']:.4f}")
                logger.info(f"  Improvement: {history['improvement']:.2f}%")
                logger.info(f"  Suggested improvements: {improvements}")
        
        # Generate final report
        self.generate_final_report()
        
        return self.best_config, self.cycle_results
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'num_cycles': len(self.cycle_results),
            'best_loss': self.best_loss,
            'best_config': self.best_config,
            'cycle_summaries': []
        }
        
        for i, result in enumerate(self.cycle_results, 1):
            summary = {
                'cycle': i,
                'final_loss': result['final_loss'],
                'final_accuracy': result['final_accuracy'],
                'improvement': result['improvement'],
                'config': result['config']
            }
            report['cycle_summaries'].append(summary)
        
        # Calculate overall statistics
        all_losses = [r['final_loss'] for r in self.cycle_results]
        all_accs = [r['final_accuracy'] for r in self.cycle_results]
        
        report['overall_stats'] = {
            'best_loss': min(all_losses),
            'worst_loss': max(all_losses),
            'avg_loss': np.mean(all_losses),
            'best_accuracy': max(all_accs),
            'worst_accuracy': min(all_accs),
            'avg_accuracy': np.mean(all_accs),
            'total_improvement': (all_losses[0] - all_losses[-1]) / all_losses[0] * 100 if all_losses[0] != 0 else 0
        }
        
        # Save report
        with open(self.results_dir / 'final_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Plot summary
        self.plot_improvement_summary()
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("IMPROVEMENT CYCLE COMPLETE!")
        logger.info("="*60)
        logger.info(f"Total cycles run: {len(self.cycle_results)}")
        logger.info(f"Best loss achieved: {report['overall_stats']['best_loss']:.4f}")
        logger.info(f"Best accuracy achieved: {report['overall_stats']['best_accuracy']:.4f}")
        logger.info(f"Total improvement: {report['overall_stats']['total_improvement']:.2f}%")
        logger.info(f"\nBest configuration:")
        for key, value in self.best_config.items():
            logger.info(f"  {key}: {value}")
        logger.info(f"\nFull report saved to: {self.results_dir / 'final_report.json'}")
        logger.info(f"Visualization saved to: {self.results_dir / 'improvement_summary.png'}")


def main():
    """Main function to run improvement cycles"""
    
    # Base configuration
    base_config = {
        'sequence_length': 30,
        'batch_size': 32,
        'hidden_dim': 64,
        'num_layers': 2,
        'num_heads': 4,
        'dropout': 0.1,
        'learning_rate': 5e-4,
        'weight_decay': 0.01,
        'num_epochs': 20
    }
    
    # Create trainer
    trainer = ImprovementCycleTrainer(base_config)
    
    # Run improvement cycles
    best_config, results = trainer.run_improvement_cycles(num_cycles=5)
    
    return best_config, results


if __name__ == "__main__":
    best_config, results = main()