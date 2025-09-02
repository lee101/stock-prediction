#!/usr/bin/env python3
"""
Profit Tracking and Trading Simulation for Training
Calculates actual profit metrics during training and logs to TensorBoard
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd


@dataclass
class TradingMetrics:
    """Trading performance metrics"""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_profit: float
    avg_loss: float
    total_trades: int
    profitable_trades: int
    losing_trades: int
    cumulative_returns: List[float]
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging"""
        return {
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'avg_profit': self.avg_profit,
            'avg_loss': self.avg_loss,
            'total_trades': self.total_trades,
            'profitable_trades': self.profitable_trades,
            'losing_trades': self.losing_trades,
            'final_cumulative_return': self.cumulative_returns[-1] if self.cumulative_returns else 0
        }


class ProfitTracker:
    """
    Tracks profit metrics during training
    Simulates trading based on model predictions
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: float = 0.001,  # 0.1% commission
        slippage: float = 0.0005,   # 0.05% slippage
        max_position_size: float = 0.3,  # Max 30% of capital per trade
        stop_loss: float = 0.02,     # 2% stop loss
        take_profit: float = 0.05,   # 5% take profit
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.max_position_size = max_position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        # Trading state
        self.reset()
    
    def reset(self):
        """Reset trading state"""
        self.capital = self.initial_capital
        self.position = 0  # Current position size
        self.entry_price = 0
        self.trades = []
        self.returns = []
        self.cumulative_returns = []
        self.prices_history = []
        self.actions_history = []
        self.positions_history = []
    
    def calculate_metrics_from_predictions(
        self,
        predictions: torch.Tensor,
        actual_prices: torch.Tensor,
        action_logits: Optional[torch.Tensor] = None
    ) -> TradingMetrics:
        """
        Calculate trading metrics from model predictions
        
        Args:
            predictions: Predicted prices (batch_size, horizon)
            actual_prices: Actual future prices (batch_size, horizon)
            action_logits: Action predictions (batch_size, 3) [buy, hold, sell]
        """
        
        batch_size = predictions.shape[0]
        horizon = predictions.shape[1] if len(predictions.shape) > 1 else 1
        
        # Convert to numpy for easier manipulation
        pred_prices = predictions.detach().cpu().numpy()
        actual_prices = actual_prices.detach().cpu().numpy()
        
        if action_logits is not None:
            actions = torch.argmax(action_logits, dim=-1).detach().cpu().numpy()
        else:
            # Generate actions based on predicted price movements
            actions = self._generate_actions_from_predictions(pred_prices)
        
        # Simulate trading for each sample in batch
        batch_returns = []
        batch_trades = []
        
        for i in range(batch_size):
            if len(pred_prices.shape) > 1:
                pred = pred_prices[i]
                actual = actual_prices[i]
            else:
                pred = pred_prices[i:i+1]
                actual = actual_prices[i:i+1]
            
            action = actions[i] if i < len(actions) else 1  # Default to hold
            
            # Calculate return based on action and actual price movement
            trade_return = self._simulate_trade(
                action=action,
                predicted_price=pred[0] if len(pred) > 0 else 0,
                actual_price=actual[0] if len(actual) > 0 else 0,
                current_price=actual[0] if len(actual) > 0 else 100  # Default price
            )
            
            batch_returns.append(trade_return)
            if trade_return != 0:
                batch_trades.append(trade_return)
        
        # Calculate aggregate metrics
        metrics = self._calculate_aggregate_metrics(batch_returns, batch_trades)
        
        return metrics
    
    def _generate_actions_from_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Generate trading actions from price predictions"""
        actions = []
        
        for pred in predictions:
            if len(pred.shape) == 0:
                # Scalar prediction
                price_change = 0
            elif len(pred) == 1:
                price_change = 0
            else:
                # Calculate expected price change
                price_change = (pred[-1] - pred[0]) / (pred[0] + 1e-8)
            
            # Generate action based on threshold
            if price_change > 0.01:  # 1% up
                actions.append(0)  # Buy
            elif price_change < -0.01:  # 1% down
                actions.append(2)  # Sell
            else:
                actions.append(1)  # Hold
        
        return np.array(actions)
    
    def _simulate_trade(
        self,
        action: int,
        predicted_price: float,
        actual_price: float,
        current_price: float
    ) -> float:
        """
        Simulate a single trade and return the profit/loss
        
        Args:
            action: 0=buy, 1=hold, 2=sell
            predicted_price: Model's predicted price
            actual_price: Actual future price
            current_price: Current market price
        """
        
        if action == 0:  # Buy
            if self.position <= 0:  # Not already long
                # Calculate position size based on confidence
                confidence = min(abs(predicted_price - current_price) / current_price, 1.0)
                position_size = self.max_position_size * confidence
                
                # Account for commission and slippage
                entry_cost = current_price * (1 + self.commission + self.slippage)
                exit_price = actual_price * (1 - self.commission - self.slippage)
                
                # Calculate return
                trade_return = (exit_price - entry_cost) / entry_cost
                
                # Apply stop loss and take profit
                if trade_return < -self.stop_loss:
                    trade_return = -self.stop_loss
                elif trade_return > self.take_profit:
                    trade_return = self.take_profit
                
                return trade_return * position_size
        
        elif action == 2:  # Sell/Short
            if self.position >= 0:  # Not already short
                # Calculate position size
                confidence = min(abs(current_price - predicted_price) / current_price, 1.0)
                position_size = self.max_position_size * confidence
                
                # Account for commission and slippage (reversed for short)
                entry_credit = current_price * (1 - self.commission - self.slippage)
                exit_cost = actual_price * (1 + self.commission + self.slippage)
                
                # Calculate return (reversed for short)
                trade_return = (entry_credit - exit_cost) / entry_credit
                
                # Apply stop loss and take profit
                if trade_return < -self.stop_loss:
                    trade_return = -self.stop_loss
                elif trade_return > self.take_profit:
                    trade_return = self.take_profit
                
                return trade_return * position_size
        
        # Hold - no return
        return 0.0
    
    def _calculate_aggregate_metrics(
        self,
        returns: List[float],
        trades: List[float]
    ) -> TradingMetrics:
        """Calculate aggregate trading metrics"""
        
        returns = np.array(returns)
        trades = np.array(trades) if trades else np.array([0])
        
        # Filter out zero returns for trade statistics
        non_zero_returns = returns[returns != 0]
        
        if len(non_zero_returns) == 0:
            # No trades made
            return TradingMetrics(
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                avg_profit=0.0,
                avg_loss=0.0,
                total_trades=0,
                profitable_trades=0,
                losing_trades=0,
                cumulative_returns=[0.0]
            )
        
        # Calculate metrics
        total_return = np.sum(non_zero_returns)
        cumulative_returns = np.cumsum(returns)
        
        # Sharpe ratio (annualized, assuming daily returns)
        if len(non_zero_returns) > 1:
            sharpe_ratio = np.mean(non_zero_returns) / (np.std(non_zero_returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Max drawdown
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / (running_max + 1e-8)
        max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0
        
        # Win rate and average P&L
        profitable_trades = non_zero_returns[non_zero_returns > 0]
        losing_trades = non_zero_returns[non_zero_returns < 0]
        
        win_rate = len(profitable_trades) / len(non_zero_returns) if len(non_zero_returns) > 0 else 0.0
        avg_profit = np.mean(profitable_trades) if len(profitable_trades) > 0 else 0.0
        avg_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0.0
        
        return TradingMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            avg_profit=avg_profit,
            avg_loss=avg_loss,
            total_trades=len(non_zero_returns),
            profitable_trades=len(profitable_trades),
            losing_trades=len(losing_trades),
            cumulative_returns=cumulative_returns.tolist()
        )
    
    def update_with_batch(
        self,
        model_outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        data_processor=None
    ) -> TradingMetrics:
        """
        Update profit tracking with a batch of predictions
        
        Args:
            model_outputs: Model predictions including price_predictions and action_logits
            batch: Input batch with actual prices
            data_processor: Optional data processor for inverse scaling
        """
        
        # Extract predictions and actuals
        price_predictions = model_outputs.get('price_predictions')
        action_logits = model_outputs.get('action_logits')
        
        # Get actual prices from batch
        if 'labels' in batch:
            actual_prices = batch['labels']
            if len(actual_prices.shape) > 2:
                # Extract close prices (usually index 3)
                actual_prices = actual_prices[:, :, 3]
        else:
            # Generate dummy prices for testing
            actual_prices = price_predictions * (1 + torch.randn_like(price_predictions) * 0.01)
        
        # Inverse transform if processor is provided
        if data_processor is not None and hasattr(data_processor, 'inverse_transform'):
            # This would need proper implementation based on your processor
            pass
        
        # Calculate metrics
        metrics = self.calculate_metrics_from_predictions(
            predictions=price_predictions,
            actual_prices=actual_prices,
            action_logits=action_logits
        )
        
        return metrics


class ProfitAwareLoss(torch.nn.Module):
    """
    Custom loss function that incorporates profit considerations
    """
    
    def __init__(
        self,
        price_loss_weight: float = 0.5,
        action_loss_weight: float = 0.3,
        profit_loss_weight: float = 0.2,
        profit_tracker: Optional[ProfitTracker] = None
    ):
        super().__init__()
        self.price_loss_weight = price_loss_weight
        self.action_loss_weight = action_loss_weight
        self.profit_loss_weight = profit_loss_weight
        self.profit_tracker = profit_tracker or ProfitTracker()
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate combined loss with profit awareness
        
        Returns:
            Total loss and individual loss components
        """
        
        total_loss = 0
        loss_components = {}
        
        # Price prediction loss (MSE)
        if 'price_predictions' in predictions and 'price_targets' in targets:
            price_loss = F.mse_loss(
                predictions['price_predictions'],
                targets['price_targets']
            )
            total_loss += self.price_loss_weight * price_loss
            loss_components['price_loss'] = price_loss.item()
        
        # Action prediction loss (Cross Entropy)
        if 'action_logits' in predictions and 'action_labels' in targets:
            action_loss = F.cross_entropy(
                predictions['action_logits'],
                targets['action_labels']
            )
            total_loss += self.action_loss_weight * action_loss
            loss_components['action_loss'] = action_loss.item()
        
        # Profit-aware loss component
        if self.profit_loss_weight > 0 and 'price_predictions' in predictions:
            # Calculate simulated profit metrics
            metrics = self.profit_tracker.calculate_metrics_from_predictions(
                predictions=predictions['price_predictions'],
                actual_prices=targets.get('price_targets', predictions['price_predictions']),
                action_logits=predictions.get('action_logits')
            )
            
            # Create profit loss (negative Sharpe ratio as loss)
            profit_loss = -metrics.sharpe_ratio
            
            # Add risk penalty for high drawdown
            risk_penalty = metrics.max_drawdown * 0.5
            
            profit_loss_tensor = torch.tensor(
                profit_loss + risk_penalty,
                device=predictions['price_predictions'].device
            )
            
            total_loss += self.profit_loss_weight * profit_loss_tensor
            loss_components['profit_loss'] = profit_loss
            loss_components['sharpe_ratio'] = metrics.sharpe_ratio
            loss_components['max_drawdown'] = metrics.max_drawdown
        
        return total_loss, loss_components


def integrate_profit_tracking(trainer, profit_tracker: Optional[ProfitTracker] = None):
    """
    Integrate profit tracking into existing trainer
    
    Args:
        trainer: HFTrainer instance
        profit_tracker: Optional custom profit tracker
    """
    
    if profit_tracker is None:
        profit_tracker = ProfitTracker()
    
    # Store original training_step
    original_training_step = trainer.training_step
    
    def training_step_with_profit(batch):
        """Enhanced training step with profit tracking"""
        
        # Run original training step
        loss = original_training_step(batch)
        
        # Calculate profit metrics periodically
        if trainer.global_step % 100 == 0:  # Every 100 steps
            with torch.no_grad():
                # Get model predictions
                outputs = trainer.model(
                    batch['input_ids'],
                    attention_mask=batch.get('attention_mask')
                )
                
                # Calculate profit metrics
                metrics = profit_tracker.update_with_batch(
                    model_outputs=outputs,
                    batch=batch
                )
                
                # Log to TensorBoard
                profit_metrics = {
                    f'profit/total_return': metrics.total_return,
                    f'profit/sharpe_ratio': metrics.sharpe_ratio,
                    f'profit/max_drawdown': metrics.max_drawdown,
                    f'profit/win_rate': metrics.win_rate,
                    f'profit/total_trades': metrics.total_trades,
                }
                
                trainer.log_metrics(profit_metrics)
                
                # Log to console periodically
                if trainer.global_step % 500 == 0:
                    trainer.training_logger.info(
                        f"📊 Profit Metrics - Return: {metrics.total_return:.2%}, "
                        f"Sharpe: {metrics.sharpe_ratio:.2f}, "
                        f"Win Rate: {metrics.win_rate:.2%}"
                    )
        
        return loss
    
    # Replace training step
    trainer.training_step = training_step_with_profit
    trainer.profit_tracker = profit_tracker
    
    return trainer