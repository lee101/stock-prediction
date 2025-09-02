#!/usr/bin/env python3
"""
Quick 2-minute RL training for realistic trading
Saves best model based on profit metrics
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
import json
import logging

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.append(os.path.dirname(current_dir))

from config import create_config
from train_hf import HFTrainer, StockDataset
from hf_trainer import TransformerTradingModel, HFTrainingConfig
from data_utils import StockDataProcessor, download_stock_data, split_data
from profit_tracker import ProfitTracker, ProfitAwareLoss

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RLTradingTrainer:
    """RL-based trading trainer with profit maximization"""
    
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Profit tracking
        self.profit_tracker = ProfitTracker(
            initial_capital=10000,
            commission=0.001,
            max_position_size=0.3
        )
        
        # Best model tracking
        self.best_profit = -float('inf')
        self.best_sharpe = -float('inf')
        self.best_model_state = None
        
        # Optimizer with adaptive learning rate
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=5e-4,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2
        )
        
        # Loss with profit awareness
        self.loss_fn = ProfitAwareLoss(
            price_loss_weight=0.5,
            action_loss_weight=0.3,
            profit_loss_weight=0.2,
            profit_tracker=self.profit_tracker
        )
        
    def train_step(self, batch):
        """Single training step with profit tracking"""
        self.model.train()
        
        # Extract inputs and targets from batch dict
        inputs = batch['input_ids'].to(self.device)
        targets = batch['labels'].to(self.device)
        
        # Forward pass - model returns a dict
        model_outputs = self.model(inputs)
        predictions = model_outputs['price_predictions']
        
        # Calculate trading signals from price predictions
        signals = self.generate_signals(predictions, targets)
        
        # Simulate trades and calculate profit
        profit_metrics = self.profit_tracker.calculate_metrics_from_predictions(
            predictions=predictions,
            actual_prices=targets[:, :predictions.shape[1], 3] if len(targets.shape) > 2 else targets,
            action_logits=model_outputs.get('action_logits')
        )
        
        # Combined loss
        # Adjust target shape to match predictions
        target_prices = targets[:, :predictions.shape[1], 3] if len(targets.shape) > 2 else targets[:, :predictions.shape[1]]
        mse_loss = nn.functional.mse_loss(predictions, target_prices)
        profit_loss = -profit_metrics.total_return / 100  # Normalize
        sharpe_loss = -profit_metrics.sharpe_ratio / 10
        
        total_loss = (
            0.5 * mse_loss +
            0.3 * profit_loss +
            0.2 * sharpe_loss
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'loss': total_loss.item(),
            'mse': mse_loss.item(),
            'profit': profit_metrics.total_return,
            'sharpe': profit_metrics.sharpe_ratio,
            'trades': profit_metrics.total_trades
        }
    
    def generate_signals(self, predictions, targets):
        """Generate trading signals from predictions"""
        # predictions is shape (batch, horizon) - predicted prices
        # Use first prediction for immediate signal
        if len(predictions.shape) == 2:
            pred_prices = predictions[:, 0]  # First prediction
        else:
            pred_prices = predictions.squeeze()
        
        # Get current prices from targets
        if len(targets.shape) == 3:
            current_prices = targets[:, 0, 3]  # Close price
        else:
            current_prices = targets[:, 0] if len(targets.shape) == 2 else targets
        
        # Calculate predicted returns
        pred_returns = (pred_prices - current_prices) / (current_prices + 1e-8)
        
        # Buy signal if predicted return > 0.5%
        buy_signals = (pred_returns > 0.005).float()
        
        # Sell signal if predicted return < -0.5%
        sell_signals = (pred_returns < -0.005).float()
        
        # Hold otherwise (0 = hold, 1 = buy, -1 = sell)
        signals = buy_signals - sell_signals
        
        return signals
    
    def validate(self):
        """Validation with profit metrics"""
        self.model.eval()
        
        total_profit = 0
        total_sharpe = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Extract inputs and targets from batch dict
                inputs = batch['input_ids'].to(self.device)
                targets = batch['labels'].to(self.device)
                
                # Forward pass - model returns a dict
                model_outputs = self.model(inputs)
                predictions = model_outputs['price_predictions']
                signals = self.generate_signals(predictions, targets)
                
                profit_metrics = self.profit_tracker.calculate_metrics_from_predictions(
                    predictions=predictions,
                    actual_prices=targets[:, :predictions.shape[1], 3] if len(targets.shape) > 2 else targets,
                    action_logits=model_outputs.get('action_logits')
                )
                
                total_profit += profit_metrics.total_return
                total_sharpe += profit_metrics.sharpe_ratio
                num_batches += 1
        
        avg_profit = total_profit / max(num_batches, 1)
        avg_sharpe = total_sharpe / max(num_batches, 1)
        
        # Save best model
        if avg_profit > self.best_profit:
            self.best_profit = avg_profit
            self.best_sharpe = avg_sharpe
            self.best_model_state = self.model.state_dict().copy()
            logger.info(f"💰 New best model! Profit: {avg_profit:.2f}%, Sharpe: {avg_sharpe:.3f}")
        
        return {
            'profit': avg_profit,
            'sharpe': avg_sharpe
        }
    
    def train(self, max_minutes=2):
        """Train for specified time with early stopping"""
        start_time = time.time()
        epoch = 0
        
        logger.info(f"Starting {max_minutes}-minute training session...")
        
        while (time.time() - start_time) / 60 < max_minutes:
            epoch += 1
            epoch_metrics = {
                'loss': [], 'mse': [], 'profit': [], 'sharpe': [], 'trades': []
            }
            
            # Training
            for batch in self.train_loader:
                if (time.time() - start_time) / 60 >= max_minutes:
                    break
                    
                metrics = self.train_step(batch)
                for k, v in metrics.items():
                    epoch_metrics[k].append(v)
            
            # Validation
            val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Log progress
            avg_loss = np.mean(epoch_metrics['loss'])
            avg_profit = np.mean(epoch_metrics['profit'])
            avg_sharpe = np.mean(epoch_metrics['sharpe'])
            
            logger.info(
                f"Epoch {epoch} | "
                f"Loss: {avg_loss:.4f} | "
                f"Train Profit: {avg_profit:.2f}% | "
                f"Val Profit: {val_metrics['profit']:.2f}% | "
                f"Val Sharpe: {val_metrics['sharpe']:.3f}"
            )
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"✅ Loaded best model with profit: {self.best_profit:.2f}%")
        
        elapsed_time = (time.time() - start_time) / 60
        logger.info(f"Training completed in {elapsed_time:.2f} minutes")
        
        return self.model

def main():
    """Run quick RL training session"""
    
    # Setup paths
    checkpoint_dir = Path("hftraining/checkpoints/rl_quick")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Download fresh data
    logger.info("Downloading stock data...")
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    all_data = []
    for symbol in stocks:
        stock_data = download_stock_data(symbol, start_date='2020-01-01')
        if symbol in stock_data:
            df = stock_data[symbol]
            logger.info(f"Downloaded {len(df)} records for {symbol}")
            all_data.append(df)
    
    # Combine data
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total data points: {len(combined_df)}")
    
    # Process data
    processor = StockDataProcessor()
    features = processor.prepare_features(combined_df)
    processor.fit_scalers(features)
    normalized_data = processor.transform(features)
    
    # Create datasets
    train_data, val_data, _ = split_data(normalized_data, 0.7, 0.15, 0.15)
    
    train_dataset = StockDataset(
        train_data,
        sequence_length=30,
        prediction_horizon=5
    )
    
    val_dataset = StockDataset(
        val_data,
        sequence_length=30,
        prediction_horizon=5
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )
    
    # Create model
    input_dim = normalized_data.shape[1]
    hf_config = HFTrainingConfig(
        hidden_size=512,
        num_layers=8,
        num_heads=8,
        sequence_length=30,
        prediction_horizon=5
    )
    
    model = TransformerTradingModel(hf_config, input_dim=input_dim)
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Check for existing best model
    best_model_path = checkpoint_dir / "best_model.pth"
    if best_model_path.exists():
        logger.info("Loading existing best model...")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model with previous best profit: {checkpoint.get('best_profit', 'N/A'):.2f}%")
    
    # Create trainer
    trainer = RLTradingTrainer(model, train_loader, val_loader)
    
    # Train
    trained_model = trainer.train(max_minutes=2)
    
    # Save best model
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'best_profit': trainer.best_profit,
        'best_sharpe': trainer.best_sharpe,
        'timestamp': datetime.now().isoformat(),
        'config': hf_config
    }, best_model_path)
    
    logger.info(f"💾 Best model saved to {best_model_path}")
    
    # Save training report
    report = {
        'best_profit': float(trainer.best_profit),
        'best_sharpe': float(trainer.best_sharpe),
        'training_time': datetime.now().isoformat(),
        'stocks_trained': stocks,
        'data_points': len(combined_df),
        'model_params': sum(p.numel() for p in model.parameters())
    }
    
    report_path = checkpoint_dir / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"📊 Training report saved to {report_path}")
    
    # Display final results
    logger.info("\n" + "="*50)
    logger.info("TRAINING COMPLETE - RESULTS:")
    logger.info(f"Best Profit: {trainer.best_profit:.2f}%")
    logger.info(f"Best Sharpe Ratio: {trainer.best_sharpe:.3f}")
    logger.info(f"Model saved: {best_model_path}")
    logger.info("="*50)

if __name__ == "__main__":
    main()