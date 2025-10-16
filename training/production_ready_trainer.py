#!/usr/bin/env python3
"""
Production-Ready HuggingFace Training Pipeline
Fully scaled and ready for deployment
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
import logging
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionStockDataset(Dataset):
    """Production dataset with all features and optimizations"""
    
    def __init__(
        self,
        data_dir: str,
        symbols: list = None,
        seq_len: int = 60,
        pred_horizon: int = 5,
        max_samples: int = 100000,
        augment: bool = True
    ):
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon
        self.augment = augment
        self.samples = []
        
        data_path = Path(data_dir)
        
        # Auto-detect all symbols if not specified
        if symbols is None:
            symbols = [f.stem for f in data_path.glob('*.csv')]
            symbols = [s for s in symbols if not any(x in s for x in ['metadata', 'combined'])]
            logger.info(f"Auto-detected {len(symbols)} symbols")
        
        total_samples = 0
        for symbol in symbols:
            if total_samples >= max_samples:
                break
                
            file_path = data_path / f"{symbol}.csv"
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, index_col=0)
                    
                    # Extract features
                    features = self.extract_features(df)
                    
                    if features is not None and len(features) > self.seq_len + self.pred_horizon:
                        # Create sequences
                        for i in range(min(500, len(features) - self.seq_len - self.pred_horizon)):
                            if total_samples >= max_samples:
                                break
                                
                            seq = features[i:i+self.seq_len]
                            target = features[i+self.seq_len:i+self.seq_len+self.pred_horizon]
                            
                            # Action label
                            price_change = (target[0, 3] - seq[-1, 3]) / (abs(seq[-1, 3]) + 1e-8)
                            
                            if price_change > 0.01:
                                action = 0  # Buy
                            elif price_change < -0.01:
                                action = 2  # Sell
                            else:
                                action = 1  # Hold
                            
                            self.samples.append((seq, target, action))
                            total_samples += 1
                            
                except Exception as e:
                    logger.warning(f"Failed to process {symbol}: {e}")
        
        logger.info(f"Created {len(self.samples)} total samples")
    
    def extract_features(self, df):
        """Extract normalized OHLCV + technical indicators"""
        try:
            # Get price columns
            price_cols = []
            for col_set in [['open', 'high', 'low', 'close'], ['Open', 'High', 'Low', 'Close']]:
                if all(c in df.columns for c in col_set):
                    price_cols = col_set
                    break
            
            if len(price_cols) < 4:
                return None
            
            ohlc = df[price_cols].values
            
            # Normalize
            ohlc_norm = (ohlc - ohlc.mean(axis=0)) / (ohlc.std(axis=0) + 1e-8)
            
            # Add volume if available
            volume = np.ones(len(ohlc))  # Default
            for vol_col in ['volume', 'Volume']:
                if vol_col in df.columns:
                    volume = df[vol_col].values
                    break
            
            volume_norm = (volume - volume.mean()) / (volume.std() + 1e-8)
            
            # Add technical indicators
            close = ohlc[:, 3]
            
            # Returns
            returns = np.zeros_like(close)
            returns[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-8)
            
            # SMA ratios
            sma_20 = pd.Series(close).rolling(20, min_periods=1).mean().values
            sma_ratio = close / (sma_20 + 1e-8)
            
            # RSI
            rsi = self.calculate_rsi(close)
            
            # Volatility
            volatility = pd.Series(returns).rolling(20, min_periods=1).std().values
            
            # Combine all features
            features = np.column_stack([
                ohlc_norm,
                volume_norm,
                returns,
                sma_ratio,
                rsi,
                volatility
            ])
            
            return features
            
        except Exception as e:
            logger.debug(f"Feature extraction error: {e}")
            return None
    
    def calculate_rsi(self, prices, period=14):
        """RSI calculation"""
        deltas = np.diff(prices, prepend=prices[0])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = pd.Series(gains).rolling(period, min_periods=1).mean().values
        avg_losses = pd.Series(losses).rolling(period, min_periods=1).mean().values
        
        rs = avg_gains / (avg_losses + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi / 100.0
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        seq, target, action = self.samples[idx]
        
        seq_tensor = torch.FloatTensor(seq)
        target_tensor = torch.FloatTensor(target)
        
        # Augmentation
        if self.augment and np.random.random() < 0.3:
            noise = torch.randn_like(seq_tensor) * 0.01
            seq_tensor = seq_tensor + noise
        
        return {
            'input_ids': seq_tensor,
            'labels': target_tensor,
            'action_labels': torch.tensor(action, dtype=torch.long),
            'attention_mask': torch.ones(self.seq_len)
        }


class ProductionTransformer(nn.Module):
    """Production-ready transformer model"""
    
    def __init__(
        self,
        input_dim=9,
        hidden_dim=256,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        seq_len=60,
        pred_horizon=5,
        num_features=9
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.pred_horizon = pred_horizon
        self.num_features = num_features
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = self.create_positional_encoding(seq_len, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output heads
        self.norm = nn.LayerNorm(hidden_dim)
        
        self.price_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, pred_horizon * num_features)
        )
        
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)
        )
    
    def create_positional_encoding(self, seq_len, hidden_dim):
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(seq_len, hidden_dim)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() *
            -(np.log(10000.0) / hidden_dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def forward(self, input_ids=None, labels=None, action_labels=None, attention_mask=None, **kwargs):
        batch_size, seq_len, input_dim = input_ids.shape
        
        # Project input
        x = self.input_proj(input_ids)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Transformer
        x = self.transformer(x)
        
        # Normalize
        x = self.norm(x)
        
        # Pool (mean)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(x.size())
            sum_embeddings = torch.sum(x * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = x.mean(dim=1)
        
        # Predictions
        price_pred = self.price_head(pooled)
        action_logits = self.action_head(pooled)
        
        # Calculate loss
        loss = None
        if labels is not None or action_labels is not None:
            loss = 0.0
            
            if labels is not None:
                price_pred_reshaped = price_pred.view(
                    batch_size, self.pred_horizon, self.num_features
                )
                price_loss = F.mse_loss(price_pred_reshaped, labels)
                loss += price_loss
            
            if action_labels is not None:
                action_loss = F.cross_entropy(action_logits, action_labels)
                loss += action_loss * 0.5
        
        return {
            'loss': loss,
            'logits': action_logits,
            'price_predictions': price_pred
        }


def create_production_trainer(model, train_dataset, eval_dataset, output_dir="./production_model"):
    """Create production-ready trainer"""
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        
        # Training parameters
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=4,
        
        # Learning rate
        learning_rate=5e-5,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        
        # Optimization
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        
        # Logging
        logging_steps=20,
        report_to=[],
        
        # Performance
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        
        # Other
        seed=42,
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3)
        ],
    )
    
    return trainer


def deploy_for_inference(model_path="./production_model"):
    """Load trained model for inference"""
    
    # Load model
    model = ProductionTransformer()
    checkpoint = torch.load(f"{model_path}/pytorch_model.bin", map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    
    logger.info(f"Model loaded from {model_path}")
    
    def predict(data):
        """Make predictions on new data"""
        with torch.no_grad():
            input_tensor = torch.FloatTensor(data).unsqueeze(0)
            output = model(input_ids=input_tensor)
            
            # Get action prediction
            action_probs = F.softmax(output['logits'], dim=-1)
            action = action_probs.argmax(dim=-1).item()
            
            # Get price prediction
            price_pred = output['price_predictions']
            
            return {
                'action': ['Buy', 'Hold', 'Sell'][action],
                'action_probs': action_probs.squeeze().tolist(),
                'price_prediction': price_pred.squeeze().tolist()
            }
    
    return predict


def main():
    """Main training and deployment pipeline"""
    logger.info("="*80)
    logger.info("PRODUCTION-READY TRAINING PIPELINE")
    logger.info("="*80)
    
    # Create datasets
    logger.info("Loading datasets...")
    
    train_dataset = ProductionStockDataset(
        data_dir="../trainingdata/train",
        symbols=None,  # Use all
        seq_len=60,
        pred_horizon=5,
        max_samples=50000,  # Limit for reasonable training time
        augment=True
    )
    
    eval_dataset = ProductionStockDataset(
        data_dir="../trainingdata/train",
        symbols=['SPY', 'QQQ', 'AAPL', 'GOOGL'],
        seq_len=60,
        pred_horizon=5,
        max_samples=5000,
        augment=False
    )
    
    logger.info(f"Dataset sizes - Train: {len(train_dataset):,}, Eval: {len(eval_dataset):,}")
    
    # Create model
    model = ProductionTransformer(
        input_dim=9,
        hidden_dim=256,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        seq_len=60,
        pred_horizon=5,
        num_features=9
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    # Create trainer
    trainer = create_production_trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir="./production_model"
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    trainer.save_model()
    logger.info("Model saved!")
    
    # Evaluate
    eval_results = trainer.evaluate()
    logger.info(f"Final evaluation: {eval_results}")
    
    # Save results
    results = {
        'eval_results': eval_results,
        'model_params': total_params,
        'train_size': len(train_dataset),
        'eval_size': len(eval_dataset),
        'timestamp': datetime.now().isoformat()
    }
    
    with open("./production_model/training_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Test deployment
    logger.info("\n" + "="*80)
    logger.info("TESTING DEPLOYMENT")
    logger.info("="*80)
    
    # Create a simple inference function
    torch.save(model.state_dict(), "./production_model/pytorch_model.bin")
    
    # Test inference
    predict_fn = deploy_for_inference("./production_model")
    
    # Get a sample
    sample = train_dataset[0]['input_ids'].numpy()
    prediction = predict_fn(sample)
    
    logger.info(f"Sample prediction: {prediction['action']}")
    logger.info(f"Action probabilities: Buy={prediction['action_probs'][0]:.2%}, "
              f"Hold={prediction['action_probs'][1]:.2%}, "
              f"Sell={prediction['action_probs'][2]:.2%}")
    
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE! Model ready for deployment.")
    logger.info("="*80)


if __name__ == "__main__":
    main()