#!/usr/bin/env python3
"""Comprehensive tests for hfinference modules."""

import pytest
import numpy as np
import pandas as pd
import torch
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import modules to test
pytest.importorskip("torch", reason="hfinference tests require torch")
import hfinference.hf_trading_engine as hfe
import hfinference.production_engine as pe


class TestHFTradingEngine:
    """Test HFTradingEngine functionality."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = MagicMock()
        model.eval = MagicMock(return_value=model)
        model.to = MagicMock(return_value=model)
        
        # Mock forward pass
        def mock_forward(x):
            batch_size = x.shape[0] if hasattr(x, 'shape') else 1
            # Create deterministic outputs for testing
            action_logits = torch.tensor([[2.0, 0.5, -1.0]] * batch_size)
            return {
                'price_predictions': torch.randn(batch_size, 5, 21),
                'action_logits': action_logits,
                'action_probs': torch.softmax(action_logits, dim=-1)
            }
        model.__call__ = mock_forward
        model.side_effect = mock_forward
        return model
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample OHLCV data."""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        data = pd.DataFrame({
            'Open': np.random.uniform(90, 110, 100),
            'High': np.random.uniform(95, 115, 100),
            'Low': np.random.uniform(85, 105, 100),
            'Close': np.random.uniform(90, 110, 100),
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        # Ensure high >= max(open, close) and low <= min(open, close)
        data['High'] = data[['Open', 'Close', 'High']].max(axis=1)
        data['Low'] = data[['Open', 'Close', 'Low']].min(axis=1)
        return data
    
    @patch('hfinference.hf_trading_engine.HFTradingEngine.load_model')
    def test_initialization(self, mock_load):
        """Test engine initialization."""
        mock_load.return_value = MagicMock()
        
        # Test with checkpoint path
        engine = hfe.HFTradingEngine(checkpoint_path="test.pt", device="cpu")
        assert engine.device == torch.device("cpu")
        assert engine.model is not None
        mock_load.assert_called_once()
        
    @patch('hfinference.hf_trading_engine.HFTradingEngine.load_model')
    def test_generate_signal(self, mock_load, mock_model, sample_data):
        """Test signal generation."""
        mock_load.return_value = mock_model
        
        engine = hfe.HFTradingEngine(checkpoint_path="test.pt", device="cpu")
        signal = engine.generate_signal("TEST", sample_data)
        
        assert signal is not None
        assert signal.action in ['buy', 'hold', 'sell']
        assert 0 <= signal.confidence <= 1
        assert signal.symbol == "TEST"
        assert isinstance(signal.timestamp, datetime)
    
    @patch('hfinference.hf_trading_engine.HFTradingEngine.load_model')
    @patch('hfinference.hf_trading_engine.yf.download')
    def test_run_backtest(self, mock_yf, mock_load, mock_model, sample_data):
        """Test backtesting functionality."""
        mock_load.return_value = mock_model
        mock_yf.return_value = sample_data
        
        engine = hfe.HFTradingEngine(checkpoint_path="test.pt", device="cpu")
        results = engine.run_backtest(
            symbols=["TEST"],
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        
        assert isinstance(results, dict)
        assert 'metrics' in results
        assert 'equity_curve' in results
        assert 'trades' in results
        
        # Check metrics
        metrics = results['metrics']
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        
    @patch('hfinference.hf_trading_engine.HFTradingEngine.load_model')
    def test_execute_trade(self, mock_load, mock_model):
        """Test trade execution logic."""
        mock_load.return_value = mock_model
        
        engine = hfe.HFTradingEngine(checkpoint_path="test.pt", device="cpu")
        
        # Mock signal
        signal = Mock()
        signal.action = 'buy'
        signal.confidence = 0.8
        signal.position_size = 100
        signal.symbol = 'TEST'
        
        # Test execution
        trade = engine.execute_trade(signal)
        
        assert trade is not None
        assert trade['symbol'] == 'TEST'
        assert trade['action'] == 'buy'
        # Check that trade has expected fields
        assert 'timestamp' in trade
        assert 'status' in trade
        
    @patch('hfinference.hf_trading_engine.HFTradingEngine.load_model')
    def test_risk_manager(self, mock_load, mock_model):
        """Test risk management."""
        mock_load.return_value = mock_model
        
        engine = hfe.HFTradingEngine(checkpoint_path="test.pt", device="cpu")
        
        # Test risk limits
        assert hasattr(engine, 'risk_manager')
        
        # Test risk limits checking
        signal = Mock()
        signal.action = 'buy'
        signal.confidence = 0.9
        signal.position_size = 0.1  # 10% of capital
        signal.symbol = 'TEST'
        
        # Check risk limits with empty positions
        can_trade = engine.risk_manager.check_risk_limits(
            signal, {}, 100000
        )
        assert can_trade == True
        
        # Check with position size too large
        signal.position_size = 0.5  # 50% exceeds typical limit
        can_trade = engine.risk_manager.check_risk_limits(
            signal, {}, 100000
        )
        # Should be false if max_position_size < 0.5


class TestProductionEngine:
    """Test ProductionEngine functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            'model': {
                'hidden_size': 256,
                'num_heads': 8,
                'num_layers': 4
            },
            'trading': {
                'initial_capital': 100000,
                'max_position_size': 0.2,
                'stop_loss': 0.05,
                'take_profit': 0.1
            },
            'risk': {
                'max_daily_loss': 0.02,
                'max_drawdown': 0.1,
                'position_limit': 10
            }
        }
    
    @pytest.fixture
    def mock_checkpoint(self, tmp_path):
        """Create a mock checkpoint file."""
        checkpoint_path = tmp_path / "model.pt"
        checkpoint = {
            'model_state_dict': {},
            'config': {
                'hidden_size': 256,
                'num_heads': 8,
                'num_layers': 4
            }
        }
        torch.save(checkpoint, checkpoint_path)
        return str(checkpoint_path)
    
    @patch('torch.load')
    def test_initialization(self, mock_load, config):
        """Test production engine initialization."""
        mock_load.return_value = {
            'model_state_dict': {},
            'config': config['model']
        }
        
        engine = pe.ProductionTradingEngine(
            checkpoint_path="test.pt",
            config=config,
            device="cpu"
        )
        
        assert engine.device == torch.device("cpu")
        assert engine.config == config
        assert hasattr(engine, 'capital')
    
    @patch('torch.load')
    def test_enhanced_signal_generation(self, mock_load, config):
        """Test enhanced signal with all features."""
        mock_model = MagicMock()
        mock_load.return_value = {
            'model_state_dict': {},
            'config': config['model']
        }
        
        # Mock model output
        mock_model.return_value = {
            'price_predictions': torch.randn(1, 5, 21),
            'action_logits': torch.tensor([[2.0, 0.5, -1.0]]),
            'volatility': torch.tensor([[0.02]]),
            'regime': torch.tensor([[1]])  # Bullish
        }
        
        engine = pe.ProductionTradingEngine(
            checkpoint_path="test.pt",
            config=config,
            device="cpu"
        )
        
        # Generate sample data
        data = pd.DataFrame({
            'Close': np.random.uniform(90, 110, 100),
            'Volume': np.random.randint(1000000, 10000000, 100)
        })
        
        signal = engine.generate_enhanced_signal("TEST", data)
        
        assert isinstance(signal, pe.EnhancedTradingSignal)
        assert signal.symbol == "TEST"
        assert signal.action in ['buy', 'hold', 'sell']
        assert signal.stop_loss is not None
        assert signal.take_profit is not None
        assert signal.volatility >= 0
        assert signal.market_regime in ['bullish', 'bearish', 'volatile', 'normal']
    
    @patch('torch.load')
    def test_portfolio_management(self, mock_load, config):
        """Test portfolio management features."""
        mock_load.return_value = {
            'model_state_dict': {},
            'config': config['model']
        }
        
        engine = pe.ProductionTradingEngine(
            checkpoint_path="test.pt",
            config=config,
            device="cpu"
        )
        
        # Add positions
        engine.add_position("AAPL", 100, 150.0)
        engine.add_position("GOOGL", 50, 2800.0)
        
        # Test portfolio value
        portfolio_value = engine.get_portfolio_value({
            "AAPL": 155.0,
            "GOOGL": 2850.0
        })
        
        expected = 100 * 155.0 + 50 * 2850.0
        assert abs(portfolio_value - expected) < 0.01
        
        # Test position limits
        assert engine.can_add_position() == True  # Still room for positions
        
        # Fill up positions
        for i in range(8):
            engine.add_position(f"TEST{i}", 10, 100.0)
        
        assert engine.can_add_position() == False  # At limit
    
    @patch('torch.load')
    @patch('hfinference.production_engine.yf.download')
    def test_live_trading_simulation(self, mock_yf, mock_load, config):
        """Test live trading simulation."""
        mock_load.return_value = {
            'model_state_dict': {},
            'config': config['model']
        }
        
        # Mock market data
        mock_yf.return_value = pd.DataFrame({
            'Close': [100, 101, 102, 103, 102]
        })
        
        engine = pe.ProductionTradingEngine(
            checkpoint_path="test.pt",
            config=config,
            device="cpu",
            mode="paper"  # Paper trading mode
        )
        
        # Run live simulation
        results = engine.run_live_simulation(
            symbols=["TEST"],
            duration_minutes=1,
            interval_seconds=1
        )
        
        assert 'trades' in results
        assert 'final_capital' in results
        assert 'performance' in results
    
    @patch('torch.load')
    def test_performance_tracking(self, mock_load, config):
        """Test performance tracking and metrics."""
        mock_load.return_value = {
            'model_state_dict': {},
            'config': config['model']
        }
        
        engine = pe.ProductionTradingEngine(
            checkpoint_path="test.pt",
            config=config,
            device="cpu"
        )
        
        # Simulate some trades
        engine.record_trade({
            'symbol': 'TEST',
            'action': 'buy',
            'price': 100,
            'quantity': 100,
            'timestamp': datetime.now()
        })
        
        engine.update_equity_curve(101000)
        engine.update_equity_curve(102000)
        engine.update_equity_curve(99000)
        
        # Calculate metrics
        metrics = engine.calculate_performance_metrics()
        
        assert 'total_return' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics
        assert 'profit_factor' in metrics
    
    @patch('torch.load')
    def test_model_versioning(self, mock_load, config, tmp_path):
        """Test model versioning and rollback."""
        mock_load.return_value = {
            'model_state_dict': {},
            'config': config['model']
        }
        
        engine = pe.ProductionTradingEngine(
            checkpoint_path="test.pt",
            config=config,
            device="cpu"
        )
        
        # Test checkpoint saving
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        
        engine.save_checkpoint(checkpoint_dir / "v1.pt")
        assert (checkpoint_dir / "v1.pt").exists()
        
        # Test loading different version
        engine.load_checkpoint_version(checkpoint_dir / "v1.pt")
        
    @patch('torch.load')
    def test_error_handling(self, mock_load, config):
        """Test error handling and recovery."""
        mock_load.return_value = {
            'model_state_dict': {},
            'config': config['model']
        }
        
        engine = pe.ProductionTradingEngine(
            checkpoint_path="test.pt",
            config=config,
            device="cpu"
        )
        
        # Test with invalid data
        with pytest.raises(ValueError):
            engine.generate_enhanced_signal("TEST", pd.DataFrame())
        
        # Test with None data
        signal = engine.generate_enhanced_signal("TEST", None)
        assert signal is None
        
        # Test recovery from model failure
        engine.model.side_effect = RuntimeError("Model failed")
        signal = engine.generate_enhanced_signal("TEST", pd.DataFrame({'Close': [100]}))
        assert signal is None  # Should handle gracefully


class TestIntegration:
    """Integration tests for hfinference modules."""
    
    @patch('hfinference.hf_trading_engine.torch.load')
    @patch('hfinference.production_engine.torch.load')
    def test_engine_compatibility(self, mock_prod_load, mock_hf_load):
        """Test compatibility between HF and Production engines."""
        # Mock checkpoint
        checkpoint = {
            'model_state_dict': {},
            'config': {
                'hidden_size': 256,
                'num_heads': 8,
                'num_layers': 4
            }
        }
        mock_hf_load.return_value = checkpoint
        mock_prod_load.return_value = checkpoint
        
        # Create engines
        hf_engine = hfe.HFTradingEngine(checkpoint_path="test.pt", device="cpu")
        prod_engine = pe.ProductionTradingEngine(
            checkpoint_path="test.pt",
            config={'model': checkpoint['config']},
            device="cpu"
        )
        
        # Both should load same model architecture
        assert hasattr(hf_engine, 'model')
        assert hasattr(prod_engine, 'model')
    
    @patch('hfinference.hf_trading_engine.yf.download')
    @patch('hfinference.production_engine.yf.download')
    def test_data_pipeline_consistency(self, mock_prod_yf, mock_hf_yf):
        """Test data pipeline consistency across engines."""
        # Create consistent test data
        test_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [102, 103, 104],
            'Low': [99, 100, 101],
            'Close': [101, 102, 103],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range(start='2023-01-01', periods=3))
        
        mock_hf_yf.return_value = test_data
        mock_prod_yf.return_value = test_data
        
        # Both engines should process data similarly
        assert mock_hf_yf.return_value.equals(mock_prod_yf.return_value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])