#!/usr/bin/env python3
"""
Test script to verify the training pipeline works
"""

import sys
import os
import torch

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from config import create_config
from data_utils import generate_synthetic_data, split_data
from train_hf import StockDataset, HFTrainer
from hf_trainer import TransformerTradingModel, HFTrainingConfig


def test_pipeline():
    """Test the complete training pipeline"""
    
    print("Testing HuggingFace-style training pipeline...")
    print("=" * 50)
    
    # Create minimal test config
    config = create_config('quick_test')
    config.training.max_steps = 10  # Very short test
    config.evaluation.eval_steps = 5
    config.evaluation.save_steps = 8
    config.evaluation.logging_steps = 2
    config.training.batch_size = 4
    config.data.sequence_length = 20
    config.data.prediction_horizon = 3
    config.model.hidden_size = 64
    config.model.num_layers = 2
    config.model.num_heads = 4
    
    print("✓ Configuration created")
    
    # Generate test data
    print("Generating test data...")
    data = generate_synthetic_data(length=200, n_features=8)
    print(f"✓ Generated data: {data.shape}")
    
    # Split data
    train_data, val_data, test_data = split_data(data, 0.7, 0.2, 0.1)
    print(f"✓ Split data - Train: {train_data.shape}, Val: {val_data.shape}")
    
    # Create datasets
    print("Creating datasets...")
    try:
        train_dataset = StockDataset(
            train_data,
            sequence_length=config.data.sequence_length,
            prediction_horizon=config.data.prediction_horizon
        )
        
        val_dataset = StockDataset(
            val_data,
            sequence_length=config.data.sequence_length,
            prediction_horizon=config.data.prediction_horizon
        ) if len(val_data) > config.data.sequence_length + config.data.prediction_horizon else None
        
        print(f"✓ Created datasets - Train: {len(train_dataset)}, Val: {len(val_dataset) if val_dataset else 0}")
        
        # Test dataset access
        sample = train_dataset[0]
        print(f"✓ Sample shapes - Input: {sample['input_ids'].shape}, Labels: {sample['labels'].shape}")
        
    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        return False
    
    # Create model
    print("Creating model...")
    try:
        # Convert config to HFTrainingConfig
        hf_config = HFTrainingConfig(
            hidden_size=config.model.hidden_size,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout,
            
            learning_rate=config.training.learning_rate,
            warmup_steps=config.training.warmup_steps,
            max_steps=config.training.max_steps,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            max_grad_norm=config.training.max_grad_norm,
            
            optimizer_name=config.training.optimizer,
            weight_decay=config.training.weight_decay,
            adam_beta1=config.training.adam_beta1,
            adam_beta2=config.training.adam_beta2,
            adam_epsilon=config.training.adam_epsilon,
            
            batch_size=config.training.batch_size,
            eval_steps=config.evaluation.eval_steps,
            save_steps=config.evaluation.save_steps,
            logging_steps=config.evaluation.logging_steps,
            
            sequence_length=config.data.sequence_length,
            prediction_horizon=config.data.prediction_horizon,
            
            use_mixed_precision=False,  # Disable for testing
            use_gradient_checkpointing=False,
            use_data_parallel=False,
            
            output_dir="test_output",
            logging_dir="test_logs",
            cache_dir="test_cache"
        )
        
        model = TransformerTradingModel(hf_config, input_dim=data.shape[1])
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model created with {total_params:,} parameters")
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False
    
    # Test forward pass
    print("Testing forward pass...")
    try:
        model.eval()
        with torch.no_grad():
            batch_input = sample['input_ids'].unsqueeze(0)  # Add batch dimension
            attention_mask = sample['attention_mask'].unsqueeze(0)
            
            outputs = model(batch_input, attention_mask=attention_mask)
            
            print(f"✓ Forward pass successful")
            print(f"  Action logits shape: {outputs['action_logits'].shape}")
            print(f"  Value shape: {outputs['value'].shape}")
            print(f"  Price predictions shape: {outputs['price_predictions'].shape}")
    
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    # Test training setup (without actual training)
    print("Testing trainer setup...")
    try:
        trainer = HFTrainer(
            model=model,
            config=hf_config,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        
        print("✓ Trainer created successfully")
        
        # Test single training step
        from torch.utils.data import DataLoader
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=2,
            shuffle=False
        )
        
        batch = next(iter(train_loader))
        
        # Move to device
        device = next(model.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Set model to training mode
        trainer.model.train()
        
        # Ensure gradients are enabled
        for param in trainer.model.parameters():
            param.requires_grad = True
        
        loss = trainer.training_step(batch)
        
        print(f"✓ Training step successful, loss: {loss:.4f}")
        
    except Exception as e:
        print(f"✗ Trainer setup failed: {e}")
        return False
    
    print("=" * 50)
    print("✓ All pipeline tests passed!")
    print("The HuggingFace-style training system is ready to use.")
    
    return True


if __name__ == "__main__":
    success = test_pipeline()
    exit(0 if success else 1)