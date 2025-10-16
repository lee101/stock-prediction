#!/usr/bin/env python3
"""
Quick test of HuggingFace training pipeline with existing data
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleStockDataset(Dataset):
    """Simplified dataset for testing"""
    
    def __init__(self, data_dir: str, symbols: list, seq_len: int = 30):
        self.seq_len = seq_len
        self.samples = []
        
        data_path = Path(data_dir)
        for symbol in symbols[:3]:  # Limit to 3 symbols for quick test
            file_path = data_path / f"{symbol}.csv"
            if file_path.exists():
                logger.info(f"Loading {symbol} from {file_path}")
                df = pd.read_csv(file_path, index_col=0)
                
                # Extract OHLC data (handle both upper and lowercase)
                cols = df.columns.tolist()
                ohlc_cols = []
                for target_col in ['open', 'high', 'low', 'close']:
                    for col in cols:
                        if col.lower() == target_col:
                            ohlc_cols.append(col)
                            break
                
                if len(ohlc_cols) != 4:
                    logger.warning(f"Skipping {symbol}: missing OHLC columns")
                    continue
                    
                ohlc = df[ohlc_cols].values
                
                # Normalize
                ohlc = (ohlc - ohlc.mean(axis=0)) / (ohlc.std(axis=0) + 1e-8)
                
                # Create sequences
                for i in range(len(ohlc) - seq_len - 5):
                    seq = ohlc[i:i+seq_len]
                    target = ohlc[i+seq_len:i+seq_len+5]
                    
                    # Simple action label based on price change
                    price_change = (target[0, 3] - seq[-1, 3]) / (abs(seq[-1, 3]) + 1e-8)
                    if price_change > 0.01:
                        action = 0  # Buy
                    elif price_change < -0.01:
                        action = 2  # Sell  
                    else:
                        action = 1  # Hold
                    
                    self.samples.append((seq, target, action))
        
        logger.info(f"Created {len(self.samples)} samples from {len(symbols)} symbols")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        seq, target, action = self.samples[idx]
        return {
            'input_ids': torch.FloatTensor(seq),
            'labels': torch.FloatTensor(target),
            'action_labels': torch.tensor(action, dtype=torch.long)
        }


class SimpleTransformer(nn.Module):
    """Simplified transformer model"""
    
    def __init__(self, input_dim=4, hidden_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.price_head = nn.Linear(hidden_dim, 5 * input_dim)  # 5 timesteps * 4 features
        self.action_head = nn.Linear(hidden_dim, 3)  # 3 actions
    
    def forward(self, input_ids=None, labels=None, action_labels=None, **kwargs):
        # Project input
        x = self.input_proj(input_ids)
        
        # Transformer
        x = self.transformer(x)
        
        # Pool (use mean)
        x = x.mean(dim=1)
        
        # Predictions
        price_pred = self.price_head(x)
        action_logits = self.action_head(x)
        
        # Calculate loss
        loss = None
        if labels is not None:
            price_loss = nn.functional.mse_loss(
                price_pred.view(labels.shape), 
                labels
            )
            loss = price_loss
            
        if action_labels is not None:
            action_loss = nn.functional.cross_entropy(
                action_logits,
                action_labels
            )
            loss = (loss + action_loss) if loss is not None else action_loss
        
        return {'loss': loss, 'logits': action_logits}


def main():
    logger.info("Starting quick HuggingFace test")
    
    # Create datasets
    train_dataset = SimpleStockDataset(
        data_dir="../trainingdata/train",
        symbols=['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA'],
        seq_len=30
    )
    
    # For now, use train data for validation (test has too few samples)
    eval_dataset = SimpleStockDataset(
        data_dir="../trainingdata/train",
        symbols=['SPY', 'QQQ'],  # Different symbols for eval
        seq_len=30
    )
    
    # Create model
    model = SimpleTransformer()
    
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./quick_hf_output",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=1e-4,
        warmup_steps=100,
        logging_steps=10,
        eval_steps=50,
        eval_strategy="steps",  # Changed from evaluation_strategy
        save_steps=100,
        save_total_limit=2,
        report_to=[],  # Disable wandb/tensorboard for quick test
        disable_tqdm=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Evaluate
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")
    
    logger.info("Quick test complete!")


if __name__ == "__main__":
    main()