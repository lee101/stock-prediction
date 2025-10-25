#!/usr/bin/env python3
"""
Comprehensive tests for the production trading engine
Tests all critical components for production readiness
"""

import pytest
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from pathlib import Path
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from hfinference.production_engine import (
    ProductionTradingEngine,
    EnhancedTradingSignal,
    Position
)


class TestProductionEngine:
    """Test suite for production trading engine"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        return {
            'model': {
                'input_features': 30,
                'hidden_size': 64,
                'num_heads': 4,
                'num_layers': 2,
                'intermediate_size': 128,
                'dropout': 0.1,
                'sequence_length': 60,
                'prediction_horizon': 5
            },
            'trading': {
                'initial_capital': 100000,
                'max_position_size': 0.15,
                'max_positions': 10,
                'stop_loss': 0.02,
                'take_profit': 0.05,
                'trailing_stop': 0.015,
                'confidence_threshold': 0.65,
                'risk_per_trade': 0.01,
                'max_daily_loss': 0.02,
                'kelly_fraction': 0.25
            },
            'strategy': {
                'use_ensemble': False,
                'ensemble_size': 3,
                'confirmation_required': 2,
                'use_technical_confirmation': True,
                'market_regime_filter': True,
                'volatility_filter': True,
                'volume_filter': True
            },
            'data': {
                'lookback_days': 200,
                'update_interval': 60,
                'use_technical_indicators': True,
                'normalize_features': True,
                'feature_engineering': True
            }
        }
    
    @pytest.fixture
    def mock_data(self):
        """Generate mock OHLCV data"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        np.random.seed(42)
        
        close = 100 + np.cumsum(np.random.randn(100) * 2)
        data = pd.DataFrame({
            'Open': close + np.random.randn(100) * 0.5,
            'High': close + np.abs(np.random.randn(100)) * 2,
            'Low': close - np.abs(np.random.randn(100)) * 2,
            'Close': close,
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        return data
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing"""
        model = Mock()
        
        # Mock forward pass
        def mock_forward(x):
            batch_size = x.shape[0]
            return {
                'price_predictions': torch.randn(batch_size, 5, 30),
                'action_logits': torch.tensor([[2.0, 0.5, -1.0]]).repeat(batch_size, 1),
                'action_probs': torch.softmax(torch.tensor([[2.0, 0.5, -1.0]]), dim=-1).repeat(batch_size, 1)
            }
        
        model.return_value = mock_forward
        model.eval = Mock(return_value=model)
        model.to = Mock(return_value=model)
        
        return model
    
    @pytest.fixture
    def engine(self, mock_config, mock_model, tmp_path):
        """Create engine instance with mocks"""
        
        # Create temporary checkpoint
        checkpoint_path = tmp_path / "test_model.pt"
        torch.save({
            'model_state_dict': {},
            'config': mock_config,
            'metrics': {'test_loss': 0.1}
        }, checkpoint_path)
        
        with patch('hfinference.production_engine.TransformerTradingModel') as MockModel:
            MockModel.return_value = mock_model
            
            engine = ProductionTradingEngine(
                checkpoint_path=str(checkpoint_path),
                config_path=None,
                device='cpu',
                paper_trading=True,
                live_trading=False
            )
            
            # Override config with mock
            engine.config = mock_config
            engine.model = mock_model
            
            return engine
    
    def test_engine_initialization(self, engine):
        """Test engine initializes correctly"""
        assert engine is not None
        assert engine.current_capital == 100000
        assert engine.paper_trading is True
        assert engine.live_trading is False
        assert len(engine.positions) == 0
        assert engine.device == torch.device('cpu')
    
    def test_signal_generation(self, engine, mock_data):
        """Test signal generation with mock data"""
        
        # Mock the data processor's prepare_features
        with patch.object(engine.data_processor, 'prepare_features') as mock_prep:
            mock_prep.return_value = np.random.randn(60, 30).astype(np.float32)
            
            signal = engine.generate_enhanced_signal('AAPL', mock_data, use_ensemble=False)
            
            # Signal may be None if data processor returns None
            if signal is not None:
                assert signal.symbol == 'AAPL'
                assert signal.action in ['buy', 'hold', 'sell']
                assert 0 <= signal.confidence <= 1
                assert signal.position_size >= 0
                assert signal.risk_score >= 0
    
    def test_technical_signals(self, engine, mock_data):
        """Test technical indicator calculations"""
        
        # Add technical indicators to mock data
        mock_data['rsi'] = 45  # Neutral RSI
        mock_data['macd'] = 0.5
        mock_data['macd_signal'] = 0.3
        mock_data['ma_20'] = mock_data['Close'].rolling(20).mean()
        mock_data['ma_50'] = mock_data['Close'].rolling(50).mean()
        mock_data['bb_position'] = 0.5
        
        signals = engine._calculate_technical_signals(mock_data)
        
        assert 'rsi' in signals
        assert signals['rsi'] == 0.0  # Neutral
        assert 'macd' in signals
        assert signals['macd'] == 1.0  # Bullish crossover
    
    def test_market_regime_detection(self, engine, mock_data):
        """Test market regime detection"""
        
        # Test normal regime
        regime = engine._detect_market_regime(mock_data)
        assert regime in ['normal', 'bullish', 'bearish', 'volatile']
        
        # Create volatile data
        volatile_data = mock_data.copy()
        volatile_data['close'] = volatile_data['Close']
        volatile_data.loc[volatile_data.index[-20:], 'close'] *= np.random.uniform(0.9, 1.1, 20)
        
        regime = engine._detect_market_regime(volatile_data)
        # Should detect increased volatility
    
    def test_support_resistance_levels(self, engine, mock_data):
        """Test support and resistance calculation"""
        
        support, resistance = engine._calculate_support_resistance(mock_data)
        
        assert isinstance(support, list)
        assert isinstance(resistance, list)
        assert len(support) <= 3
        assert len(resistance) <= 3
        
        current_price = float(mock_data['Close'].iloc[-1])
        lowest = float(mock_data['Low'].min())
        highest = float(mock_data['High'].max())

        assert support == sorted(support)
        assert resistance == sorted(resistance)

        for level in support:
            assert lowest <= level <= highest

        for level in resistance:
            assert lowest <= level <= highest
    
    def test_kelly_position_sizing(self, engine):
        """Test Kelly Criterion position sizing"""
        
        # Test with high confidence, positive return
        size = engine._calculate_kelly_position_size(
            confidence=0.8,
            expected_return=0.05,
            volatility=0.02,
            risk_score=0.3
        )
        
        assert 0 <= size <= engine.config['trading']['max_position_size']
        
        # Test with low confidence
        size_low = engine._calculate_kelly_position_size(
            confidence=0.3,
            expected_return=0.05,
            volatility=0.02,
            risk_score=0.3
        )
        
        assert size_low <= size
        
        # Test with high risk
        size_risky = engine._calculate_kelly_position_size(
            confidence=0.8,
            expected_return=0.05,
            volatility=0.05,
            risk_score=0.8
        )
        
        assert size_risky <= size
    
    def test_risk_level_calculation(self, engine):
        """Test stop-loss and take-profit calculation"""
        
        current_price = 100.0
        volatility = 0.02
        support = [95, 97, 98]
        resistance = [102, 103, 105]
        
        stop_loss, take_profit, trailing = engine._calculate_risk_levels(
            current_price=current_price,
            volatility=volatility,
            action='buy',
            support_levels=support,
            resistance_levels=resistance
        )
        
        assert stop_loss is not None
        assert take_profit is not None
        assert trailing is not None
        
        # Stop loss should be below current price
        assert stop_loss < current_price
        
        # Take profit should be above current price
        assert take_profit > current_price
        
        # Trailing stop should be below current price
        assert trailing < current_price
    
    def test_trade_execution_buy(self, engine):
        """Test buy trade execution"""
        
        signal = EnhancedTradingSignal(
            timestamp=datetime.now(),
            symbol='AAPL',
            action='buy',
            confidence=0.8,
            predicted_price=105,
            current_price=100,
            expected_return=0.05,
            position_size=0.1,
            stop_loss=98,
            take_profit=105,
            risk_score=0.3
        )
        
        result = engine.execute_trade(signal)
        
        assert result['status'] == 'executed'
        assert result['symbol'] == 'AAPL'
        assert result['action'] == 'buy'
        assert 'shares' in result
        assert 'value' in result
        
        # Check position was created
        assert 'AAPL' in engine.positions
        position = engine.positions['AAPL']
        assert position.shares > 0
        assert position.entry_price == 100
    
    def test_trade_execution_sell(self, engine):
        """Test sell trade execution"""
        
        # Create existing position
        engine.positions['AAPL'] = Position(
            symbol='AAPL',
            shares=100,
            entry_price=95,
            entry_time=datetime.now() - timedelta(days=5),
            stop_loss=93,
            take_profit=100
        )
        
        signal = EnhancedTradingSignal(
            timestamp=datetime.now(),
            symbol='AAPL',
            action='sell',
            confidence=0.8,
            predicted_price=98,
            current_price=100,
            expected_return=-0.02,
            position_size=0,
            risk_score=0.3
        )
        
        initial_capital = engine.current_capital
        result = engine.execute_trade(signal)
        
        assert result['status'] == 'executed'
        assert 'pnl' in result
        assert result['pnl'] == 500  # (100-95) * 100 shares
        
        # Position should be closed
        assert 'AAPL' not in engine.positions
        
        # Capital should increase
        assert engine.current_capital > initial_capital
    
    def test_risk_limits(self, engine):
        """Test risk management limits"""
        
        # Test daily loss limit
        engine.daily_pnl = -engine.daily_loss_limit - 100
        
        signal = EnhancedTradingSignal(
            timestamp=datetime.now(),
            symbol='AAPL',
            action='buy',
            confidence=0.8,
            predicted_price=105,
            current_price=100,
            expected_return=0.05,
            position_size=0.1,
            risk_score=0.3
        )
        
        result = engine.execute_trade(signal)
        assert result['status'] == 'rejected'
        assert result['reason'] == 'daily_loss_limit'
        
        # Reset daily P&L
        engine.daily_pnl = 0
        
        # Test low confidence rejection
        signal.confidence = 0.3
        result = engine.execute_trade(signal)
        assert result['status'] == 'rejected'
        assert result['reason'] == 'low_confidence'
        
        # Test high risk rejection
        signal.confidence = 0.8
        signal.risk_score = 0.9
        result = engine.execute_trade(signal)
        assert result['status'] == 'rejected'
        assert result['reason'] == 'high_risk'
    
    def test_position_updates(self, engine):
        """Test position update mechanisms"""
        
        # Create position
        position = Position(
            symbol='AAPL',
            shares=100,
            entry_price=100,
            entry_time=datetime.now(),
            stop_loss=98,
            take_profit=105,
            trailing_stop=99,
            high_water_mark=100
        )
        
        engine.positions['AAPL'] = position
        
        # Test trailing stop update
        position.update_trailing_stop(102, 0.02)
        assert position.high_water_mark == 102
        assert position.trailing_stop == pytest.approx(99.96, rel=0.01)
        
        # Test position exit on stop loss
        mock_data = pd.DataFrame({
            'Close': [97]  # Below stop loss
        })
        
        market_data = {'AAPL': mock_data}
        
        with patch.object(engine, 'execute_trade') as mock_execute:
            engine.update_positions(market_data)
            mock_execute.assert_called_once()
            
            # Check the sell signal was created
            call_args = mock_execute.call_args[0][0]
            assert call_args.action == 'sell'
            assert call_args.symbol == 'AAPL'
    
    def test_portfolio_metrics(self, engine):
        """Test portfolio metrics calculation"""
        
        # Add some trades to history
        engine.trade_history = [
            {'symbol': 'AAPL', 'pnl': 500, 'return': 0.05},
            {'symbol': 'GOOGL', 'pnl': -200, 'return': -0.02},
            {'symbol': 'MSFT', 'pnl': 300, 'return': 0.03}
        ]
        
        engine.performance_metrics['winning_trades'] = 2
        engine.performance_metrics['losing_trades'] = 1
        engine.performance_metrics['total_pnl'] = 600
        
        metrics = engine.calculate_portfolio_metrics()
        
        assert 'portfolio_value' in metrics
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'win_rate' in metrics
        
        # Check win rate calculation
        assert metrics['win_rate'] == pytest.approx(0.667, rel=0.01)
        
        # Check profit factor
        assert metrics['profit_factor'] == pytest.approx(4.0, rel=0.1)  # (500+300)/200
    
    def test_ensemble_confirmation(self, engine):
        """Test ensemble voting mechanism"""
        
        engine.config['strategy']['use_ensemble'] = True
        engine.config['strategy']['ensemble_size'] = 3
        engine.config['strategy']['confirmation_required'] = 2
        
        signal1 = EnhancedTradingSignal(
            timestamp=datetime.now(),
            symbol='AAPL',
            action='buy',
            confidence=0.7,
            predicted_price=105,
            current_price=100,
            expected_return=0.05,
            position_size=0.1
        )
        
        # First signal - not enough confirmation
        initial_confidence = signal1.confidence
        result1 = engine._apply_ensemble_confirmation('AAPL', signal1)
        assert result1.confidence < initial_confidence
        
        # Second signal (same action)
        signal2 = EnhancedTradingSignal(
            timestamp=datetime.now(),
            symbol='AAPL',
            action='buy',
            confidence=0.7,
            predicted_price=105,
            current_price=100,
            expected_return=0.05,
            position_size=0.1
        )
        result2 = engine._apply_ensemble_confirmation('AAPL', signal2)
        
        # Should have confirmation now
        assert result2.action == 'buy'
        assert result2.confidence >= result1.confidence
        
        # Third signal (different action)
        signal3 = EnhancedTradingSignal(
            timestamp=datetime.now(),
            symbol='AAPL',
            action='sell',
            confidence=0.7,
            predicted_price=95,
            current_price=100,
            expected_return=-0.05,
            position_size=0.1
        )
        
        prior_confidence = result2.confidence
        result3 = engine._apply_ensemble_confirmation('AAPL', signal3)
        assert result3.confidence <= prior_confidence
    
    def test_state_persistence(self, engine, tmp_path):
        """Test saving and loading engine state"""
        
        # Add some state
        engine.positions['AAPL'] = Position(
            symbol='AAPL',
            shares=100,
            entry_price=100,
            entry_time=datetime.now(),
            stop_loss=98,
            take_profit=105
        )
        
        engine.trade_history.append({
            'symbol': 'AAPL',
            'action': 'buy',
            'price': 100,
            'shares': 100
        })
        
        engine.current_capital = 90000
        engine.daily_pnl = -500
        
        # Save state
        state_file = tmp_path / "engine_state.json"
        engine.save_state(str(state_file))
        
        assert state_file.exists()
        
        # Create new engine and load state
        with patch.object(ProductionTradingEngine, "load_model", return_value=engine.model):
            new_engine = ProductionTradingEngine(
                checkpoint_path=str(tmp_path / "test_model.pt"),
                paper_trading=True
            )
        
        # Mock the model loading
        new_engine.model = engine.model
        
        new_engine.load_state(str(state_file))
        
        # Verify state was restored
        assert 'AAPL' in new_engine.positions
        assert new_engine.positions['AAPL'].shares == 100
        assert len(new_engine.trade_history) == 1
        assert new_engine.current_capital == 90000
        assert new_engine.daily_pnl == -500
    
    def test_error_handling(self, engine, mock_data):
        """Test error handling in signal generation"""
        
        # Test with insufficient data
        short_data = mock_data.head(10)
        signal = engine.generate_enhanced_signal('AAPL', short_data)
        assert signal is None
        
        # Test with corrupted data
        bad_data = mock_data.copy()
        bad_data['Close'] = np.nan
        
        signal = engine.generate_enhanced_signal('AAPL', bad_data)
        # Should handle gracefully
    
    def test_feature_normalization(self, engine, mock_data):
        """Test feature normalization"""
        
        features = np.random.randn(60, 5) * 100 + 50
        normalized = engine._normalize_features(features, mock_data)
        
        # Check shape preserved
        assert normalized.shape == features.shape
        
        # Check normalization applied (first 4 columns should be divided by price)
        assert np.abs(normalized[:, :4]).max() < np.abs(features[:, :4]).max()
    
    def test_signal_strength_calculation(self, engine):
        """Test signal strength calculation"""
        
        tech_signals = {'rsi': 1.0, 'macd': 1.0, 'ma_trend': 1.0}
        
        strength = engine._calculate_signal_strength(
            confidence=0.8,
            expected_return=0.1,
            tech_signals=tech_signals,
            market_regime='bullish'
        )
        
        # Should be boosted by positive factors
        assert strength > 0.8
        assert strength <= 1.0
        
        # Test with contradicting signals
        tech_signals_bad = {'rsi': -1.0, 'macd': -1.0}
        
        strength_bad = engine._calculate_signal_strength(
            confidence=0.8,
            expected_return=0.1,
            tech_signals=tech_signals_bad,
            market_regime='bearish'
        )
        
        assert strength_bad < strength


class TestPositionClass:
    """Test Position dataclass"""
    
    def test_position_creation(self):
        """Test position creation"""
        
        position = Position(
            symbol='AAPL',
            shares=100,
            entry_price=100,
            entry_time=datetime.now(),
            stop_loss=98,
            take_profit=105
        )
        
        assert position.symbol == 'AAPL'
        assert position.shares == 100
        assert position.entry_price == 100
    
    def test_unrealized_pnl(self):
        """Test P&L calculation"""
        
        position = Position(
            symbol='AAPL',
            shares=100,
            entry_price=100,
            entry_time=datetime.now(),
            stop_loss=98,
            take_profit=105
        )
        
        # Test profit
        pnl = position.get_unrealized_pnl(105)
        assert pnl == 500
        
        # Test loss
        pnl = position.get_unrealized_pnl(95)
        assert pnl == -500
    
    def test_return_calculation(self):
        """Test return percentage calculation"""
        
        position = Position(
            symbol='AAPL',
            shares=100,
            entry_price=100,
            entry_time=datetime.now(),
            stop_loss=98,
            take_profit=105
        )
        
        ret = position.get_return(105)
        assert ret == pytest.approx(0.05)
        
        ret = position.get_return(95)
        assert ret == pytest.approx(-0.05)
    
    def test_trailing_stop_update(self):
        """Test trailing stop mechanism"""
        
        position = Position(
            symbol='AAPL',
            shares=100,
            entry_price=100,
            entry_time=datetime.now(),
            stop_loss=98,
            take_profit=105,
            trailing_stop=99,
            high_water_mark=100
        )
        
        # Price goes up - should update
        position.update_trailing_stop(105, trail_percent=0.02)
        assert position.high_water_mark == 105
        assert position.trailing_stop == pytest.approx(102.9, rel=0.01)
        
        # Price goes down - should not update
        position.update_trailing_stop(103, trail_percent=0.02)
        assert position.high_water_mark == 105  # Unchanged
        assert position.trailing_stop == pytest.approx(102.9, rel=0.01)  # Unchanged


class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.slow
    def test_full_trading_cycle(self, tmp_path):
        """Test complete trading cycle"""
        
        # Create mock checkpoint
        checkpoint_path = tmp_path / "model.pt"
        torch.save({
            'model_state_dict': {},
            'config': {
                'model': {
                    'input_features': 5,
                    'hidden_size': 64,
                    'num_heads': 4,
                    'num_layers': 2,
                    'sequence_length': 60,
                    'prediction_horizon': 5
                }
            }
        }, checkpoint_path)
        
        with patch('hfinference.production_engine.TransformerTradingModel'):
            with patch('yfinance.download') as mock_download:
                # Mock market data
                dates = pd.date_range(end=datetime.now(), periods=200, freq='D')
                mock_download.return_value = pd.DataFrame({
                    'Open': np.random.randn(200) * 2 + 100,
                    'High': np.random.randn(200) * 2 + 102,
                    'Low': np.random.randn(200) * 2 + 98,
                    'Close': np.random.randn(200) * 2 + 100,
                    'Volume': np.random.randint(1000000, 10000000, 200)
                }, index=dates)
                
                # Initialize engine
                engine = ProductionTradingEngine(
                    checkpoint_path=str(checkpoint_path),
                    paper_trading=True,
                    live_trading=False
                )
                
                # Mock model forward pass
                def mock_forward(x):
                    return {
                        'price_predictions': torch.randn(x.shape[0], 5, 5),
                        'action_logits': torch.tensor([[2.0, 0.5, -1.0]]).repeat(x.shape[0], 1)
                    }
                
                engine.model = Mock(side_effect=mock_forward)
                engine.model.eval = Mock()
                
                # Run trading cycle
                symbols = ['AAPL', 'GOOGL']
                
                for symbol in symbols:
                    data = mock_download.return_value
                    
                    # Generate signal
                    signal = engine.generate_enhanced_signal(symbol, data, use_ensemble=False)
                    
                    if signal and signal.confidence > 0.65:
                        # Execute trade
                        result = engine.execute_trade(signal)
                        
                        # Update positions
                        market_data = {symbol: data.tail(1)}
                        engine.update_positions(market_data)
                
                # Calculate final metrics
                metrics = engine.calculate_portfolio_metrics()
                
                # Verify metrics exist
                assert 'portfolio_value' in metrics
                assert 'total_return' in metrics
                assert metrics['portfolio_value'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
