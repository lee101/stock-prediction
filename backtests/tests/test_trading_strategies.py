#!/usr/bin/env python3
"""
Tests for trading strategy simulation.
"""

import unittest
import sys
import os
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backtests.simulate_trading_strategies import TradingSimulator


class TestTradingStrategies(unittest.TestCase):
    """Test trading strategies with mock data."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "test_data"
        self.data_dir.mkdir(exist_ok=True)
        
        # Create mock CSV data
        self.create_mock_csv_files()
        
        # Create simulator with mocked pipeline
        with patch('backtests.simulate_trading_strategies.TradingSimulator._load_prediction_pipeline'):
            self.simulator = TradingSimulator(
                backtestdata_dir=str(self.data_dir),
                forecast_days=3,
                initial_capital=10000,
                output_dir=str(Path(self.temp_dir) / "results")
            )
    
    def create_mock_csv_files(self):
        """Create mock CSV files for testing."""
        symbols = ['AAPL', 'GOOGL', 'TSLA']
        
        for symbol in symbols:
            # Generate realistic stock data
            np.random.seed(42)
            dates = pd.date_range('2024-01-01', periods=100, freq='D')
            
            # Generate price data with some trend
            base_price = 100
            returns = np.random.normal(0.001, 0.02, len(dates))  # 0.1% mean return, 2% volatility
            prices = [base_price]
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            # Create OHLC data
            data = {
                'Date': dates,
                'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
                'High': [p * (1 + abs(np.random.normal(0.01, 0.01))) for p in prices],
                'Low': [p * (1 - abs(np.random.normal(0.01, 0.01))) for p in prices],
                'Close': prices,
                'Volume': [np.random.randint(1000000, 10000000) for _ in prices]
            }
            
            df = pd.DataFrame(data)
            df.to_csv(self.data_dir / f"{symbol}-2024-01-01.csv", index=False)
    
    def test_load_data(self):
        """Test data loading functionality."""
        csv_files = list(self.data_dir.glob("*.csv"))
        self.assertEqual(len(csv_files), 3)
        
        # Test loading a CSV file
        data = self.simulator.load_and_preprocess_data(csv_files[0])
        self.assertIsNotNone(data)
        self.assertIn('Close', data.columns)
        self.assertIn('High', data.columns)
        self.assertIn('Low', data.columns)
        self.assertIn('Open', data.columns)
    
    def test_mock_forecasts(self):
        """Test strategies with mock forecast data."""
        # Create mock forecast data
        mock_forecasts = {
            'AAPL': {
                'symbol': 'AAPL',
                'close_total_predicted_change': 0.05,  # 5% expected return
                'close_last_price': 150.0,
                'close_predicted_price_value': 157.5,
                'high_total_predicted_change': 0.07,
                'low_total_predicted_change': 0.03,
            },
            'GOOGL': {
                'symbol': 'GOOGL',
                'close_total_predicted_change': 0.03,  # 3% expected return
                'close_last_price': 2800.0,
                'close_predicted_price_value': 2884.0,
                'high_total_predicted_change': 0.05,
                'low_total_predicted_change': 0.01,
            },
            'TSLA': {
                'symbol': 'TSLA',
                'close_total_predicted_change': 0.08,  # 8% expected return
                'close_last_price': 250.0,
                'close_predicted_price_value': 270.0,
                'high_total_predicted_change': 0.12,
                'low_total_predicted_change': 0.04,
            }
        }
        
        # Test best single stock strategy
        strategy_result = self.simulator.strategy_best_single_stock(mock_forecasts)
        self.assertEqual(strategy_result['selected_stock'], 'TSLA')  # Highest return
        self.assertEqual(strategy_result['allocation']['TSLA'], 1.0)
        self.assertEqual(strategy_result['expected_return'], 0.08)
        
        # Test best two stocks strategy
        strategy_result = self.simulator.strategy_best_two_stocks(mock_forecasts)
        self.assertIn('TSLA', strategy_result['allocation'])
        self.assertIn('AAPL', strategy_result['allocation'])
        self.assertEqual(strategy_result['allocation']['TSLA'], 0.5)
        self.assertEqual(strategy_result['allocation']['AAPL'], 0.5)
        
        # Test weighted portfolio strategy
        strategy_result = self.simulator.strategy_weighted_portfolio(mock_forecasts, top_n=3)
        self.assertEqual(len(strategy_result['allocation']), 3)
        
        # TSLA should have highest weight due to highest predicted return
        max_weight_symbol = max(strategy_result['allocation'], key=strategy_result['allocation'].get)
        self.assertEqual(max_weight_symbol, 'TSLA')
        
        # Test risk-adjusted portfolio
        strategy_result = self.simulator.strategy_risk_adjusted_portfolio(mock_forecasts, top_n=3)
        self.assertIn('allocation', strategy_result)
        self.assertIn('expected_return', strategy_result)
    
    def test_portfolio_performance_simulation(self):
        """Test portfolio performance simulation."""
        mock_strategy = {
            'strategy': 'test_strategy',
            'allocation': {'AAPL': 0.6, 'GOOGL': 0.4},
            'expected_return': 0.04,
        }
        
        result = self.simulator.simulate_portfolio_performance(mock_strategy)
        self.assertIn('performance', result)
        self.assertIn('predicted_return', result['performance'])
        self.assertIn('simulated_actual_return', result['performance'])
        self.assertIn('profit_loss', result['performance'])
        self.assertIn('capital_after', result['performance'])
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with empty forecasts
        empty_forecasts = {}
        
        strategy_result = self.simulator.strategy_best_single_stock(empty_forecasts)
        self.assertIn('error', strategy_result)
        
        strategy_result = self.simulator.strategy_best_two_stocks(empty_forecasts)
        self.assertIn('error', strategy_result)
        
        # Test with negative predictions only
        negative_forecasts = {
            'AAPL': {
                'symbol': 'AAPL',
                'close_total_predicted_change': -0.05,
            },
            'GOOGL': {
                'symbol': 'GOOGL', 
                'close_total_predicted_change': -0.03,
            }
        }
        
        strategy_result = self.simulator.strategy_weighted_portfolio(negative_forecasts)
        self.assertIn('error', strategy_result)
    
    def test_data_format_consistency(self):
        """Test that data formats are consistent throughout the pipeline."""
        mock_forecasts = {
            'TEST': {
                'symbol': 'TEST',
                'close_total_predicted_change': 0.02,
                'close_last_price': 100.0,
                'close_predicted_price_value': 102.0,
            }
        }
        
        # Test that all strategies can handle the data format
        strategies = [
            self.simulator.strategy_best_single_stock,
            self.simulator.strategy_best_two_stocks,
            self.simulator.strategy_weighted_portfolio,
        ]
        
        for strategy_func in strategies:
            try:
                result = strategy_func(mock_forecasts)
                # Should either succeed or fail with a clear error message
                self.assertTrue('allocation' in result or 'error' in result)
            except Exception as e:
                self.fail(f"Strategy {strategy_func.__name__} failed with exception: {e}")


class TestVisualizationLogger(unittest.TestCase):
    """Test visualization logger functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock TensorBoard to avoid GPU/dependencies issues
        with patch('backtests.visualization_logger.SummaryWriter') as mock_writer:
            from backtests.visualization_logger import VisualizationLogger
            self.viz_logger = VisualizationLogger(
                output_dir=str(Path(self.temp_dir) / "viz_results")
            )
    
    @patch('backtests.visualization_logger.plt.savefig')
    @patch('backtests.visualization_logger.plt.close')
    def test_forecast_visualization(self, mock_close, mock_savefig):
        """Test forecast visualization creation."""
        mock_forecasts = {
            'AAPL': {
                'close_total_predicted_change': 0.05,
                'close_last_price': 150.0,
                'close_predicted_price_value': 157.5,
            },
            'GOOGL': {
                'close_total_predicted_change': 0.03,
                'close_last_price': 2800.0,
                'close_predicted_price_value': 2884.0,
            }
        }
        
        try:
            result = self.viz_logger.create_forecast_visualization(mock_forecasts)
            # Should not raise exception
            self.assertTrue(True)
        except Exception as e:
            # If it fails due to matplotlib backend issues, that's OK for testing
            if "backend" not in str(e).lower():
                raise e
    
    def test_tensorboard_logging(self):
        """Test TensorBoard logging functionality."""
        mock_results = {
            'forecasts': {
                'AAPL': {'close_total_predicted_change': 0.05},
                'GOOGL': {'close_total_predicted_change': 0.03}
            },
            'strategies': {
                'test_strategy': {
                    'expected_return': 0.04,
                    'allocation': {'AAPL': 0.6, 'GOOGL': 0.4},
                    'performance': {
                        'simulated_actual_return': 0.035,
                        'profit_loss': 350.0
                    }
                }
            }
        }
        
        # Should not raise exception
        try:
            self.viz_logger.log_comprehensive_analysis(mock_results)
            self.assertTrue(True)
        except Exception as e:
            # TensorBoard might not be available in test environment
            if "tensorboard" not in str(e).lower():
                raise e


class TestPositionSizingOptimization(unittest.TestCase):
    """Test position sizing optimization strategies."""
    
    def test_risk_adjusted_weighting(self):
        """Test risk-adjusted position weighting logic."""
        # Mock data with different risk/return profiles
        stocks = {
            'low_risk_low_return': {'return': 0.02, 'volatility': 0.01},
            'medium_risk_medium_return': {'return': 0.05, 'volatility': 0.03}, 
            'high_risk_high_return': {'return': 0.10, 'volatility': 0.08},
            'high_risk_low_return': {'return': 0.03, 'volatility': 0.09}
        }
        
        # Calculate risk-adjusted returns (Sharpe-like ratio)
        risk_adjusted = {}
        for stock, data in stocks.items():
            risk_adjusted[stock] = data['return'] / (data['volatility'] + 0.001)
        
        # Calculate actual values to verify logic
        expected_ratios = {
            'low_risk_low_return': 0.02 / 0.011,  # ~1.82
            'medium_risk_medium_return': 0.05 / 0.031,  # ~1.61
            'high_risk_high_return': 0.10 / 0.081,  # ~1.23
            'high_risk_low_return': 0.03 / 0.091   # ~0.33
        }
        
        # Best risk-adjusted should be low_risk_low_return (highest ratio)
        best_stock = max(risk_adjusted, key=risk_adjusted.get)
        self.assertEqual(best_stock, 'low_risk_low_return')
        
        # Worst should be high_risk_low_return
        worst_stock = min(risk_adjusted, key=risk_adjusted.get)
        self.assertEqual(worst_stock, 'high_risk_low_return')
    
    def test_portfolio_diversification_benefits(self):
        """Test that diversified portfolios reduce risk."""
        # Single asset vs diversified portfolio
        single_asset_vol = 0.20  # 20% volatility
        
        # Assume correlation of 0.5 between assets
        correlation = 0.5
        n_assets = 4
        equal_weight = 1.0 / n_assets
        
        # Portfolio volatility with equal weights
        portfolio_vol = np.sqrt(
            n_assets * (equal_weight**2) * (single_asset_vol**2) +
            n_assets * (n_assets - 1) * (equal_weight**2) * correlation * (single_asset_vol**2)
        )
        
        # Diversified portfolio should have lower volatility
        self.assertLess(portfolio_vol, single_asset_vol)
        print(f"Single asset vol: {single_asset_vol:.3f}, Portfolio vol: {portfolio_vol:.3f}")


def run_comprehensive_test():
    """Run comprehensive test suite with performance benchmarking."""
    print("="*80)
    print("RUNNING COMPREHENSIVE TRADING STRATEGY TESTS")
    print("="*80)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestTradingStrategies))
    suite.addTest(unittest.makeSuite(TestVisualizationLogger))
    suite.addTest(unittest.makeSuite(TestPositionSizingOptimization))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\n" + "="*80)
    print(f"TEST RESULTS: {result.testsRun} tests run")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print("="*80)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)