#!/usr/bin/env python3
"""
Production Training System V2
Multi-stage training: Base model -> Stock specialization -> Ensemble
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import numpy as np
import pandas as pd
# yfinance removed; use local CSVs from trainingdata/
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging
import warnings
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import concurrent.futures
import pickle

warnings.filterwarnings('ignore')

from robust_data_pipeline import create_robust_dataloader, AdvancedDataProcessor, DataValidator


@dataclass
class TrainingConfig:
    """Configuration for production training"""
    # Base model training
    base_epochs: int = 50
    base_lr: float = 1e-4
    base_batch_size: int = 32
    
    # Specialization training  
    specialist_epochs: int = 30
    specialist_lr: float = 5e-5
    specialist_batch_size: int = 16
    
    # Architecture
    hidden_size: int = 1024
    num_heads: int = 16
    num_layers: int = 12
    sequence_length: int = 60
    prediction_horizon: int = 5
    
    # Advanced features
    use_moe: bool = True  # Mixture of Experts
    num_experts: int = 8
    multi_horizon: bool = True  # Predict multiple time horizons
    cross_stock_attention: bool = True
    
    # Data
    major_stocks: List[str] = None
    start_date: str = '2020-01-01'
    validation_split: float = 0.15
    test_split: float = 0.15
    
    # Paths
    base_model_dir: str = 'hftraining/models/base'
    specialist_dir: str = 'hftraining/models/specialists'
    ensemble_dir: str = 'hftraining/models/ensemble'


class MixtureOfExpertsLayer(nn.Module):
    """Mixture of Experts for different market conditions"""
    
    def __init__(self, hidden_size: int, num_experts: int, expert_size: int):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        
        # Gating network
        self.gate = nn.Linear(hidden_size, num_experts)
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, expert_size),
                nn.GELU(),
                nn.Linear(expert_size, hidden_size)
            ) for _ in range(num_experts)
        ])
        
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)
        
        # Gate weights
        gate_weights = F.softmax(self.gate(x_flat), dim=-1)
        
        # Expert outputs
        expert_outputs = torch.stack([
            expert(x_flat) for expert in self.experts
        ], dim=1)  # [batch*seq, num_experts, hidden_size]
        
        # Weighted combination
        gate_weights = gate_weights.unsqueeze(-1)  # [batch*seq, num_experts, 1]
        output = (gate_weights * expert_outputs).sum(dim=1)
        
        return output.view(batch_size, seq_len, hidden_size)


class ProductionTransformerModel(nn.Module):
    """Production-ready transformer with advanced features"""
    
    def __init__(self, config: TrainingConfig, input_features: int):
        super().__init__()
        
        self.config = config
        self.input_features = input_features
        self.hidden_size = config.hidden_size
        
        # Input processing
        self.input_projection = nn.Sequential(
            nn.Linear(input_features, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, config.sequence_length, config.hidden_size) * 0.02
        )
        
        # Market condition embedding (bull/bear/sideways)
        self.market_condition_embed = nn.Embedding(3, config.hidden_size // 4)
        
        # Transformer layers with MoE
        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            # Standard transformer layer
            layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_size * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.layers.append(layer)
            
            # Add MoE every few layers
            if config.use_moe and i % 3 == 2:
                moe_layer = MixtureOfExpertsLayer(
                    config.hidden_size, 
                    config.num_experts,
                    config.hidden_size * 2
                )
                self.layers.append(moe_layer)
        
        self.final_norm = nn.LayerNorm(config.hidden_size)
        
        # Multi-head attention for cross-stock relationships
        if config.cross_stock_attention:
            self.cross_stock_attention = nn.MultiheadAttention(
                embed_dim=config.hidden_size,
                num_heads=config.num_heads // 2,
                dropout=0.1,
                batch_first=True
            )
        
        # Output heads for different horizons
        if config.multi_horizon:
            self.horizons = [1, 5, 10]  # 1-day, 5-day, 10-day predictions
            self.price_heads = nn.ModuleDict({
                f'horizon_{h}': nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size // 2),
                    nn.LayerNorm(config.hidden_size // 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(config.hidden_size // 2, h * input_features)
                ) for h in self.horizons
            })
            
            self.action_heads = nn.ModuleDict({
                f'horizon_{h}': nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size // 4),
                    nn.LayerNorm(config.hidden_size // 4),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(config.hidden_size // 4, 3)  # Buy, Hold, Sell
                ) for h in self.horizons
            })
        else:
            # Single horizon
            self.price_head = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 2),
                nn.LayerNorm(config.hidden_size // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(config.hidden_size // 2, config.prediction_horizon * input_features)
            )
            
            self.action_head = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 4),
                nn.LayerNorm(config.hidden_size // 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(config.hidden_size // 4, 3)
            )
        
        # Market regime classifier (auxiliary task)
        self.regime_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, 3)  # Bull, Bear, Sideways
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Proper weight initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)
    
    def detect_market_condition(self, x):
        """Detect market condition from price movements"""
        # Simple heuristic based on recent price changes
        price_changes = x[:, :, 3]  # Close prices
        recent_trend = price_changes[:, -10:].mean(dim=1) - price_changes[:, -20:-10].mean(dim=1)
        
        conditions = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        conditions[recent_trend > 0.01] = 0  # Bull
        conditions[recent_trend < -0.01] = 1  # Bear  
        # Rest remain 2 (Sideways)
        
        return conditions
    
    def forward(self, x, cross_stock_context=None):
        batch_size, seq_len, features = x.shape
        
        # Project input
        hidden = self.input_projection(x)
        
        # Add positional encoding
        hidden = hidden + self.positional_encoding[:, :seq_len, :]
        
        # Detect and embed market condition
        market_conditions = self.detect_market_condition(x)
        market_embed = self.market_condition_embed(market_conditions)
        hidden[:, 0, :self.hidden_size//4] += market_embed
        
        # Apply transformer layers
        for layer in self.layers:
            if isinstance(layer, MixtureOfExpertsLayer):
                hidden = layer(hidden)
            else:
                hidden = layer(hidden)
        
        hidden = self.final_norm(hidden)
        
        # Cross-stock attention if available
        if hasattr(self, 'cross_stock_attention') and cross_stock_context is not None:
            attended, _ = self.cross_stock_attention(
                hidden, cross_stock_context, cross_stock_context
            )
            hidden = hidden + attended * 0.1  # Small residual connection
        
        # Pool sequence (attention-based)
        attention_weights = F.softmax(
            torch.sum(hidden * hidden[:, -1:, :], dim=-1), dim=1
        )
        pooled = torch.sum(hidden * attention_weights.unsqueeze(-1), dim=1)
        
        outputs = {}
        
        # Generate predictions for different horizons
        if self.config.multi_horizon:
            for horizon in self.horizons:
                price_pred = self.price_heads[f'horizon_{horizon}'](pooled)
                price_pred = price_pred.view(batch_size, horizon, self.input_features)
                
                action_logits = self.action_heads[f'horizon_{horizon}'](pooled)
                
                outputs[f'horizon_{horizon}'] = {
                    'price_predictions': price_pred,
                    'action_logits': action_logits,
                    'action_probs': torch.softmax(action_logits, dim=-1)
                }
        else:
            price_pred = self.price_head(pooled)
            price_pred = price_pred.view(batch_size, self.config.prediction_horizon, self.input_features)
            
            action_logits = self.action_head(pooled)
            
            outputs = {
                'price_predictions': price_pred,
                'action_logits': action_logits,
                'action_probs': torch.softmax(action_logits, dim=-1)
            }
        
        # Market regime prediction (auxiliary task)
        outputs['market_regime'] = self.regime_classifier(pooled)
        
        return outputs


class ProductionTrainer:
    """Production trainer with multi-stage pipeline"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_logging()
        
        # Create directories
        for directory in [config.base_model_dir, config.specialist_dir, config.ensemble_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self):
        """Setup production logging"""
        log_dir = Path('hftraining/logs/production')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_enhanced_data(self) -> Dict[str, np.ndarray]:
        """Load enhanced data for all stocks"""
        
        if self.config.major_stocks is None:
            # Major stocks across different sectors
            self.config.major_stocks = [
                # Tech
                'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'TSLA',
                # Finance
                'JPM', 'BAC', 'WFC', 'GS', 'MS',
                # Healthcare
                'JNJ', 'PFE', 'UNH', 'ABBV',
                # Consumer
                'KO', 'PG', 'WMT', 'HD', 'MCD',
                # Energy
                'XOM', 'CVX', 'COP'
            ]
        
        self.logger.info(f"Loading enhanced data for {len(self.config.major_stocks)} stocks")
        
        processor = AdvancedDataProcessor()
        validator = DataValidator()
        stock_data = {}
        
        def process_stock(symbol):
            try:
                self.logger.info(f"Processing {symbol}")
                # Load from trainingdata CSVs
                base = Path('trainingdata')
                candidates = list(base.glob(f"{symbol}.csv"))
                if not candidates:
                    candidates = [p for p in base.glob("*.csv") if symbol.lower() in p.stem.lower()]
                if not candidates:
                    self.logger.warning(f"No local CSV for {symbol}")
                    return symbol, None
                df = pd.read_csv(candidates[0])
                df.columns = df.columns.str.lower()
                if 'date' in df.columns:
                    try:
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.sort_values('date')
                    except Exception:
                        pass

                if len(df) < 500:  # Need substantial history
                    self.logger.warning(f"Insufficient data for {symbol}")
                    return symbol, None
                
                # Clean and validate
                df.columns = df.columns.str.lower()
                df = df.reset_index()
                df = validator.validate_dataframe(df)
                
                # Enhanced processing
                processed_data = processor.process_dataframe(df)
                
                self.logger.info(f"{symbol}: {processed_data.shape}")
                return symbol, processed_data
                
            except Exception as e:
                self.logger.error(f"Failed to process {symbol}: {e}")
                return symbol, None
        
        # Process stocks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(process_stock, self.config.major_stocks))
        
        # Collect results
        for symbol, data in results:
            if data is not None:
                stock_data[symbol] = data
        
        self.logger.info(f"Successfully loaded {len(stock_data)} stocks")
        
        # Save processed data
        data_path = Path('hftraining/data/processed_stocks.pkl')
        data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(data_path, 'wb') as f:
            pickle.dump(stock_data, f)
        
        return stock_data
    
    def create_base_dataset(self, stock_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Create combined dataset for base model training"""
        self.logger.info("Creating base model dataset")
        
        all_data = []
        for symbol, data in stock_data.items():
            all_data.append(data)
        
        combined_data = np.vstack(all_data)
        np.random.shuffle(combined_data)  # Mix different stocks
        
        self.logger.info(f"Base dataset shape: {combined_data.shape}")
        return combined_data
    
    def train_base_model(self, stock_data: Dict[str, np.ndarray]) -> ProductionTransformerModel:
        """Stage 1: Train base model on all stocks"""
        self.logger.info("Stage 1: Training base model")
        
        # Create base dataset
        base_data = self.create_base_dataset(stock_data)
        input_features = base_data.shape[1]
        
        # Split data
        train_size = int((1 - self.config.validation_split - self.config.test_split) * len(base_data))
        val_size = int(self.config.validation_split * len(base_data))
        
        train_data = base_data[:train_size]
        val_data = base_data[train_size:train_size + val_size]
        
        # Create data loaders
        train_loader = create_robust_dataloader(
            train_data,
            batch_size=self.config.base_batch_size,
            sequence_length=self.config.sequence_length,
            prediction_horizon=self.config.prediction_horizon,
            shuffle=True,
            num_workers=4,
            augment=True
        )
        
        val_loader = create_robust_dataloader(
            val_data,
            batch_size=self.config.base_batch_size,
            sequence_length=self.config.sequence_length,
            prediction_horizon=self.config.prediction_horizon,
            shuffle=False,
            num_workers=2,
            augment=False
        )
        
        # Create model
        model = ProductionTransformerModel(self.config, input_features)
        model.to(self.device)
        
        # Training setup - use Shampoo optimizer
        try:
            from modern_optimizers import Shampoo
            optimizer = Shampoo(
                model.parameters(),
                lr=self.config.base_lr,
                betas=(0.9, 0.999),
                eps=1e-10,
                weight_decay=0.01
            )
            self.logger.info("Using Shampoo optimizer")
        except ImportError:
            self.logger.warning("Shampoo not available, falling back to AdamW")
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config.base_lr,
                weight_decay=0.01
            )
        
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=500, T_mult=2, eta_min=1e-6
        )
        
        scaler = GradScaler()
        best_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        model.train()
        step = 0
        
        for epoch in range(self.config.base_epochs):
            self.logger.info(f"Base model epoch {epoch + 1}/{self.config.base_epochs}")
            
            epoch_losses = []
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                with autocast():
                    outputs = model(batch['input_ids'].to(self.device))
                    
                    # Multi-task loss
                    total_loss = 0
                    
                    if self.config.multi_horizon:
                        for horizon in [1, 5, 10]:
                            horizon_outputs = outputs[f'horizon_{horizon}']
                            
                            # Price prediction loss - use available horizon data
                            batch_horizon = min(horizon, batch['labels'].size(1))
                            target_labels = batch['labels'][:, :batch_horizon, :].to(self.device)
                            pred_labels = horizon_outputs['price_predictions'][:, :batch_horizon, :]
                            
                            price_loss = F.mse_loss(pred_labels, target_labels)
                            
                            # Action prediction loss
                            action_loss = F.cross_entropy(
                                horizon_outputs['action_logits'],
                                batch['action_labels'].squeeze(-1).to(self.device)
                            )
                            
                            # Weight shorter horizons more heavily
                            horizon_weight = 1.0 / horizon
                            total_loss += horizon_weight * (price_loss + 0.5 * action_loss)
                    else:
                        price_loss = F.mse_loss(
                            outputs['price_predictions'],
                            batch['labels'].to(self.device)
                        )
                        action_loss = F.cross_entropy(
                            outputs['action_logits'],
                            batch['action_labels'].squeeze(-1).to(self.device)
                        )
                        total_loss = price_loss + 0.5 * action_loss
                    
                    # Market regime loss (auxiliary)
                    if 'market_regime' in outputs:
                        regime_targets = torch.randint(0, 3, (batch['input_ids'].size(0),)).to(self.device)
                        regime_loss = F.cross_entropy(outputs['market_regime'], regime_targets)
                        total_loss += 0.1 * regime_loss
                
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                epoch_losses.append(total_loss.item())
                step += 1
                
                if step % 100 == 0:
                    self.logger.info(f"Step {step} | Loss: {total_loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
            
            # Validation
            if epoch % 5 == 0:
                val_loss = self.validate_model(model, val_loader)
                self.logger.info(f"Epoch {epoch} | Val Loss: {val_loss:.4f}")
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'config': self.config.__dict__,
                        'epoch': epoch,
                        'loss': val_loss
                    }, Path(self.config.base_model_dir) / 'best.pt')
                else:
                    patience_counter += 1
                    if patience_counter >= 5:
                        self.logger.info("Early stopping for base model")
                        break
        
        self.logger.info(f"Base model training complete. Best loss: {best_loss:.4f}")
        return model
    
    def validate_model(self, model, val_loader):
        """Validate model performance"""
        model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch['input_ids'].to(self.device))
                
                if self.config.multi_horizon:
                    loss = 0
                    for horizon in [1, 5, 10]:
                        horizon_outputs = outputs[f'horizon_{horizon}']
                        batch_horizon = min(horizon, batch['labels'].size(1))
                        target_labels = batch['labels'][:, :batch_horizon, :].to(self.device)
                        pred_labels = horizon_outputs['price_predictions'][:, :batch_horizon, :]
                        price_loss = F.mse_loss(pred_labels, target_labels)
                        loss += price_loss / horizon  # Weight by horizon
                else:
                    loss = F.mse_loss(
                        outputs['price_predictions'],
                        batch['labels'].to(self.device)
                    )
                
                total_loss += loss.item()
                num_batches += 1
        
        model.train()
        return total_loss / num_batches
    
    def train_specialist_models(self, base_model: ProductionTransformerModel, stock_data: Dict[str, np.ndarray]):
        """Stage 2: Train stock-specific specialists"""
        self.logger.info("Stage 2: Training specialist models")
        
        specialists = {}
        
        for symbol, data in stock_data.items():
            self.logger.info(f"Training specialist for {symbol}")
            
            # Split stock-specific data
            train_size = int(0.8 * len(data))
            train_data = data[:train_size]
            val_data = data[train_size:]
            
            # Create specialist model (copy of base model)
            specialist = ProductionTransformerModel(self.config, data.shape[1])
            specialist.load_state_dict(base_model.state_dict())
            specialist.to(self.device)
            
            # Create data loaders
            train_loader = create_robust_dataloader(
                train_data,
                batch_size=self.config.specialist_batch_size,
                sequence_length=self.config.sequence_length,
                prediction_horizon=self.config.prediction_horizon,
                shuffle=True,
                num_workers=2,
                augment=True
            )
            
            val_loader = create_robust_dataloader(
                val_data,
                batch_size=self.config.specialist_batch_size,
                sequence_length=self.config.sequence_length,
                prediction_horizon=self.config.prediction_horizon,
                shuffle=False,
                num_workers=1,
                augment=False
            )
            
            # Specialist training with lower learning rate - use Shampoo
            try:
                from modern_optimizers import Shampoo
                optimizer = Shampoo(
                    specialist.parameters(),
                    lr=self.config.specialist_lr,
                    betas=(0.9, 0.999),
                    eps=1e-10,
                    weight_decay=0.005
                )
                self.logger.info(f"Using Shampoo optimizer for specialist {name}")
            except ImportError:
                self.logger.warning(f"Shampoo not available for specialist {name}, falling back to AdamW")
                optimizer = optim.AdamW(
                    specialist.parameters(),
                    lr=self.config.specialist_lr,
                    weight_decay=0.005
                )
            
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
            )
            
            scaler = GradScaler()
            best_loss = float('inf')
            
            # Fine-tuning loop
            specialist.train()
            
            for epoch in range(self.config.specialist_epochs):
                epoch_losses = []
                
                for batch in train_loader:
                    optimizer.zero_grad()
                    
                    with autocast():
                        outputs = specialist(batch['input_ids'].to(self.device))
                        
                        if self.config.multi_horizon:
                            total_loss = 0
                            for horizon in [1, 5, 10]:
                                horizon_outputs = outputs[f'horizon_{horizon}']
                                batch_horizon = min(horizon, batch['labels'].size(1))
                                target_labels = batch['labels'][:, :batch_horizon, :].to(self.device)
                                pred_labels = horizon_outputs['price_predictions'][:, :batch_horizon, :]
                                price_loss = F.mse_loss(pred_labels, target_labels)
                                action_loss = F.cross_entropy(
                                    horizon_outputs['action_logits'],
                                    batch['action_labels'].squeeze(-1).to(self.device)
                                )
                                horizon_weight = 1.0 / horizon
                                total_loss += horizon_weight * (price_loss + 0.3 * action_loss)
                        else:
                            price_loss = F.mse_loss(
                                outputs['price_predictions'],
                                batch['labels'].to(self.device)
                            )
                            action_loss = F.cross_entropy(
                                outputs['action_logits'],
                                batch['action_labels'].squeeze(-1).to(self.device)
                            )
                            total_loss = price_loss + 0.3 * action_loss
                    
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(specialist.parameters(), 0.5)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    epoch_losses.append(total_loss.item())
                
                # Validation
                val_loss = self.validate_model(specialist, val_loader)
                scheduler.step(val_loss)
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    # Save specialist
                    torch.save({
                        'model_state_dict': specialist.state_dict(),
                        'config': self.config.__dict__,
                        'symbol': symbol,
                        'epoch': epoch,
                        'loss': val_loss
                    }, Path(self.config.specialist_dir) / f'{symbol}.pt')
                
                if epoch % 10 == 0:
                    self.logger.info(f"{symbol} epoch {epoch} | Val Loss: {val_loss:.4f}")
            
            specialists[symbol] = specialist
            self.logger.info(f"{symbol} specialist complete. Best loss: {best_loss:.4f}")
        
        return specialists
    
    def run_production_pipeline(self):
        """Run complete production training pipeline"""
        self.logger.info("Starting production training pipeline")
        
        # Load data
        stock_data = self.load_enhanced_data()
        
        # Stage 1: Train base model
        base_model = self.train_base_model(stock_data)
        
        # Stage 2: Train specialists
        specialists = self.train_specialist_models(base_model, stock_data)
        
        # Stage 3: Save metadata for ensemble
        ensemble_metadata = {
            'base_model_path': str(Path(self.config.base_model_dir) / 'best.pt'),
            'specialist_paths': {
                symbol: str(Path(self.config.specialist_dir) / f'{symbol}.pt')
                for symbol in specialists.keys()
            },
            'config': self.config.__dict__,
            'training_date': datetime.now().isoformat()
        }
        
        with open(Path(self.config.ensemble_dir) / 'metadata.json', 'w') as f:
            json.dump(ensemble_metadata, f, indent=2)
        
        self.logger.info("Production pipeline complete!")
        return base_model, specialists


def main():
    """Run production training"""
    
    config = TrainingConfig(
        # Base training
        base_epochs=60,
        base_lr=8e-5,
        base_batch_size=24,
        
        # Specialization
        specialist_epochs=25,
        specialist_lr=3e-5,
        specialist_batch_size=12,
        
        # Advanced architecture
        hidden_size=1024,
        num_heads=16,
        num_layers=12,
        use_moe=True,
        num_experts=6,
        multi_horizon=True,
        cross_stock_attention=True,
        
        # Data
        start_date='2020-01-01'
    )
    
    print("Starting production training pipeline")
    print("="*80)
    print(f"Base model: {config.base_epochs} epochs, {config.hidden_size}d, {config.num_layers} layers")
    print(f"Specialists: {config.specialist_epochs} epochs for each stock")
    print(f"Advanced features: MoE={config.use_moe}, Multi-horizon={config.multi_horizon}")
    print("="*80)
    
    trainer = ProductionTrainer(config)
    base_model, specialists = trainer.run_production_pipeline()
    
    print(f"\nProduction training complete!")
    print(f"Base model saved to: {config.base_model_dir}")
    print(f"Specialists saved to: {config.specialist_dir}")
    print(f"Models ready for {len(specialists)} stocks")


if __name__ == "__main__":
    main()
