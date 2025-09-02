#!/usr/bin/env python3
"""
Production Trading Engine
Uses base model + specialists for optimal predictions
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import yfinance as yf
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

# Import our production model
import sys
sys.path.append('hftraining')
from train_production_v2 import ProductionTransformerModel, TrainingConfig


@dataclass
class PredictionResult:
    """Result from production prediction"""
    symbol: str
    timestamp: datetime
    
    # Multi-horizon predictions
    predictions_1d: Dict[str, float]
    predictions_5d: Dict[str, float] 
    predictions_10d: Dict[str, float]
    
    # Trading signals
    action_1d: str  # 'buy', 'hold', 'sell'
    action_5d: str
    action_10d: str
    
    # Confidence scores
    confidence_1d: float
    confidence_5d: float
    confidence_10d: float
    
    # Ensemble metrics
    base_weight: float
    specialist_weight: float
    
    # Risk metrics
    volatility_forecast: float
    market_regime: str  # 'bull', 'bear', 'sideways'
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'predictions': {
                '1d': self.predictions_1d,
                '5d': self.predictions_5d,
                '10d': self.predictions_10d
            },
            'actions': {
                '1d': self.action_1d,
                '5d': self.action_5d,
                '10d': self.action_10d
            },
            'confidence': {
                '1d': self.confidence_1d,
                '5d': self.confidence_5d,
                '10d': self.confidence_10d
            },
            'ensemble_weights': {
                'base': self.base_weight,
                'specialist': self.specialist_weight
            },
            'risk': {
                'volatility_forecast': self.volatility_forecast,
                'market_regime': self.market_regime
            }
        }


class ProductionTradingEngine:
    """Production trading engine with ensemble predictions"""
    
    def __init__(self, models_dir: str = 'hftraining/models'):
        self.models_dir = Path(models_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.setup_logging()
        
        # Load ensemble configuration
        self.load_ensemble_config()
        
        # Load models
        self.base_model = self.load_base_model()
        self.specialists = self.load_specialists()
        
        # Data processor
        from robust_data_pipeline import AdvancedDataProcessor
        self.data_processor = AdvancedDataProcessor()
        
        self.logger.info(f"Production engine ready. Base model + {len(self.specialists)} specialists")
    
    def setup_logging(self):
        """Setup production logging"""
        log_dir = Path('hfinference/logs/production')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'engine_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_ensemble_config(self):
        """Load ensemble configuration"""
        config_path = self.models_dir / 'ensemble' / 'metadata.json'
        
        if not config_path.exists():
            raise FileNotFoundError(f"Ensemble config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.ensemble_config = json.load(f)
        
        # Recreate training config
        self.config = TrainingConfig(**self.ensemble_config['config'])
    
    def load_base_model(self) -> ProductionTransformerModel:
        """Load base model"""
        model_path = Path(self.ensemble_config['base_model_path'])
        
        if not model_path.exists():
            raise FileNotFoundError(f"Base model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model (need input features - assume from config)
        model = ProductionTransformerModel(self.config, 21)  # Default feature count
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.to(self.device)
        model.eval()
        
        self.logger.info(f"Base model loaded from {model_path}")
        return model
    
    def load_specialists(self) -> Dict[str, ProductionTransformerModel]:
        """Load all specialist models"""
        specialists = {}
        
        for symbol, model_path in self.ensemble_config['specialist_paths'].items():
            path = Path(model_path)
            
            if not path.exists():
                self.logger.warning(f"Specialist not found: {path}")
                continue
            
            try:
                checkpoint = torch.load(path, map_location=self.device)
                
                model = ProductionTransformerModel(self.config, 21)
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                model.to(self.device)
                model.eval()
                
                specialists[symbol] = model
                
            except Exception as e:
                self.logger.error(f"Failed to load specialist {symbol}: {e}")
        
        self.logger.info(f"Loaded {len(specialists)} specialist models")
        return specialists
    
    def get_market_data(self, symbol: str, days: int = 100) -> pd.DataFrame:
        """Get recent market data for symbol"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if len(df) < self.config.sequence_length:
                raise ValueError(f"Insufficient data for {symbol}")
            
            # Clean data
            df.columns = df.columns.str.lower()
            df = df.reset_index()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get data for {symbol}: {e}")
            raise
    
    def prepare_sequence(self, df: pd.DataFrame) -> torch.Tensor:
        """Prepare sequence for model input"""
        # Process with technical indicators
        processed_data = self.data_processor.process_dataframe(df)
        
        # Get last sequence_length samples
        sequence_data = processed_data[-self.config.sequence_length:]
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence_data).unsqueeze(0)
        
        return sequence_tensor.to(self.device)
    
    def calculate_ensemble_weights(self, symbol: str, recent_performance: Dict = None) -> Tuple[float, float]:
        """Calculate ensemble weights for base vs specialist"""
        
        # Default weights
        base_weight = 0.3
        specialist_weight = 0.7
        
        # Adjust based on recent performance if available
        if recent_performance:
            specialist_accuracy = recent_performance.get(f'{symbol}_accuracy', 0.5)
            base_accuracy = recent_performance.get('base_accuracy', 0.5)
            
            # Weight towards better performing model
            total_accuracy = specialist_accuracy + base_accuracy
            if total_accuracy > 0:
                specialist_weight = specialist_accuracy / total_accuracy
                base_weight = base_accuracy / total_accuracy
        
        # Ensure weights sum to 1
        total_weight = base_weight + specialist_weight
        base_weight /= total_weight
        specialist_weight /= total_weight
        
        return base_weight, specialist_weight
    
    @torch.no_grad()
    def predict(self, symbol: str, days_history: int = 100) -> PredictionResult:
        """Generate production prediction for symbol"""
        
        try:
            # Get market data
            df = self.get_market_data(symbol, days_history)
            sequence = self.prepare_sequence(df)
            
            # Get base model prediction
            base_outputs = self.base_model(sequence)
            
            # Get specialist prediction if available
            specialist_outputs = None
            if symbol in self.specialists:
                specialist_outputs = self.specialists[symbol](sequence)
            
            # Calculate ensemble weights
            base_weight, specialist_weight = self.calculate_ensemble_weights(symbol)
            
            # If no specialist, use base model only
            if specialist_outputs is None:
                specialist_weight = 0
                base_weight = 1.0
                self.logger.warning(f"No specialist available for {symbol}, using base model only")
            
            # Ensemble predictions for each horizon
            predictions = {}
            actions = {}
            confidences = {}
            
            for horizon in [1, 5, 10]:
                horizon_key = f'horizon_{horizon}'
                
                if self.config.multi_horizon and horizon_key in base_outputs:
                    # Multi-horizon predictions
                    base_pred = base_outputs[horizon_key]['action_probs']
                    base_confidence = torch.max(base_pred).item()
                    
                    if specialist_outputs and horizon_key in specialist_outputs:
                        specialist_pred = specialist_outputs[horizon_key]['action_probs']
                        specialist_confidence = torch.max(specialist_pred).item()
                        
                        # Ensemble probabilities
                        ensemble_probs = (base_weight * base_pred + 
                                        specialist_weight * specialist_pred)
                        ensemble_confidence = torch.max(ensemble_probs).item()
                    else:
                        ensemble_probs = base_pred
                        ensemble_confidence = base_confidence
                    
                    # Convert to action
                    action_idx = torch.argmax(ensemble_probs).item()
                    action_map = {0: 'buy', 1: 'hold', 2: 'sell'}
                    
                    predictions[f'{horizon}d'] = {
                        'buy_prob': float(ensemble_probs[0, 0]),
                        'hold_prob': float(ensemble_probs[0, 1]),
                        'sell_prob': float(ensemble_probs[0, 2])
                    }
                    
                    actions[f'{horizon}d'] = action_map[action_idx]
                    confidences[f'{horizon}d'] = ensemble_confidence
                
                else:
                    # Single horizon fallback
                    if 'action_probs' in base_outputs:
                        base_pred = base_outputs['action_probs']
                        ensemble_probs = base_pred
                        
                        if specialist_outputs and 'action_probs' in specialist_outputs:
                            specialist_pred = specialist_outputs['action_probs']
                            ensemble_probs = (base_weight * base_pred + 
                                            specialist_weight * specialist_pred)
                        
                        action_idx = torch.argmax(ensemble_probs).item()
                        action_map = {0: 'buy', 1: 'hold', 2: 'sell'}
                        
                        predictions[f'{horizon}d'] = {
                            'buy_prob': float(ensemble_probs[0, 0]),
                            'hold_prob': float(ensemble_probs[0, 1]), 
                            'sell_prob': float(ensemble_probs[0, 2])
                        }
                        
                        actions[f'{horizon}d'] = action_map[action_idx]
                        confidences[f'{horizon}d'] = torch.max(ensemble_probs).item()
            
            # Market regime detection
            regime_probs = base_outputs.get('market_regime', torch.zeros(1, 3))
            regime_idx = torch.argmax(regime_probs).item()
            regime_map = {0: 'bull', 1: 'bear', 2: 'sideways'}
            market_regime = regime_map[regime_idx]
            
            # Volatility forecast (simple heuristic from recent data)
            recent_prices = df['close'].tail(20).values
            returns = np.diff(np.log(recent_prices))
            volatility_forecast = float(np.std(returns) * np.sqrt(252))  # Annualized
            
            # Create result
            result = PredictionResult(
                symbol=symbol,
                timestamp=datetime.now(),
                
                predictions_1d=predictions.get('1d', {}),
                predictions_5d=predictions.get('5d', {}),
                predictions_10d=predictions.get('10d', {}),
                
                action_1d=actions.get('1d', 'hold'),
                action_5d=actions.get('5d', 'hold'),
                action_10d=actions.get('10d', 'hold'),
                
                confidence_1d=confidences.get('1d', 0.33),
                confidence_5d=confidences.get('5d', 0.33),
                confidence_10d=confidences.get('10d', 0.33),
                
                base_weight=base_weight,
                specialist_weight=specialist_weight,
                
                volatility_forecast=volatility_forecast,
                market_regime=market_regime
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {symbol}: {e}")
            raise
    
    def predict_portfolio(self, symbols: List[str]) -> Dict[str, PredictionResult]:
        """Generate predictions for multiple symbols"""
        self.logger.info(f"Generating predictions for {len(symbols)} symbols")
        
        results = {}
        
        for symbol in symbols:
            try:
                result = self.predict(symbol)
                results[symbol] = result
                self.logger.info(f"{symbol}: {result.action_1d} (conf: {result.confidence_1d:.3f})")
                
            except Exception as e:
                self.logger.error(f"Failed to predict {symbol}: {e}")
                continue
        
        return results
    
    def save_predictions(self, results: Dict[str, PredictionResult], filename: Optional[str] = None):
        """Save predictions to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"predictions_{timestamp}.json"
        
        output_dir = Path('hfinference/predictions')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        serializable_results = {
            symbol: result.to_dict() for symbol, result in results.items()
        }
        
        output_path = output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Predictions saved to {output_path}")
        return output_path


def main():
    """Test production engine"""
    print("Testing Production Trading Engine")
    print("="*50)
    
    # Create engine
    try:
        engine = ProductionTradingEngine()
    except FileNotFoundError as e:
        print(f"Models not found: {e}")
        print("Please run train_production_v2.py first")
        return
    
    # Test symbols
    test_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    
    # Generate predictions
    results = engine.predict_portfolio(test_symbols)
    
    # Display results
    print(f"\nGenerated predictions for {len(results)} symbols:")
    print("-" * 80)
    
    for symbol, result in results.items():
        print(f"{symbol}:")
        print(f"  1-day: {result.action_1d} (confidence: {result.confidence_1d:.3f})")
        print(f"  5-day: {result.action_5d} (confidence: {result.confidence_5d:.3f})")
        print(f"  Market regime: {result.market_regime}")
        print(f"  Volatility: {result.volatility_forecast:.2f}")
        print(f"  Ensemble: {result.specialist_weight:.2f} specialist, {result.base_weight:.2f} base")
        print()
    
    # Save results
    output_path = engine.save_predictions(results)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()