#!/usr/bin/env python3
"""
Distributed Multi-GPU Training Script
Scales training across multiple GPUs using PyTorch DDP
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
import json
import time
import os
import sys
import logging
from datetime import datetime
from typing import Dict, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from hftraining.robust_data_pipeline import (
    create_robust_dataloader, 
    download_and_process_stocks,
    EnhancedStockDataset,
    RobustCollator
)
from hftraining.train_production import ScaledTransformerModel
from hftraining.modern_optimizers import get_optimizer

def setup_distributed(rank: int, world_size: int):
    """Setup distributed training environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set device
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()

class DistributedTrainer:
    """Distributed trainer for multi-GPU training"""
    
    def __init__(self, rank: int, world_size: int, config: Dict):
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.device = torch.device(f'cuda:{rank}')
        
        # Only rank 0 logs
        self.is_main = rank == 0
        
        if self.is_main:
            self.setup_logging()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
    def setup_logging(self):
        """Setup logging for main process"""
        log_dir = Path(self.config.get('log_dir', 'hftraining/logs/distributed'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log(self, message: str, level=logging.INFO):
        """Log message only on main process"""
        if self.is_main:
            self.logger.log(level, message)
    
    def setup_model(self, input_features: int):
        """Setup distributed model"""
        model_config = {
            'hidden_size': self.config.get('hidden_size', 768),
            'num_heads': self.config.get('num_heads', 12),
            'num_layers': self.config.get('num_layers', 12),
            'intermediate_size': self.config.get('intermediate_size', 3072),
            'dropout': self.config.get('dropout', 0.1),
            'sequence_length': self.config.get('sequence_length', 60),
            'prediction_horizon': self.config.get('prediction_horizon', 5),
            'input_features': input_features
        }
        
        # Create model
        self.model = ScaledTransformerModel(model_config).to(self.device)
        
        # Wrap in DDP
        self.model = DDP(
            self.model, 
            device_ids=[self.rank],
            output_device=self.rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True  # Memory optimization
        )
        
        if self.is_main:
            total_params = sum(p.numel() for p in self.model.parameters())
            self.log(f"Model initialized with {total_params:,} parameters")
    
    def setup_optimizer(self):
        """Setup optimizer with per-GPU learning rate scaling"""
        # Scale learning rate by world size
        base_lr = self.config.get('learning_rate', 1e-4)
        scaled_lr = base_lr * np.sqrt(self.world_size)
        
        self.optimizer = get_optimizer(
            self.config.get('optimizer', 'adamw'),
            self.model.parameters(),
            lr=scaled_lr,
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        # Scheduler
        total_steps = self.config.get('max_steps', 10000)
        warmup_steps = self.config.get('warmup_steps', 1000)
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=scaled_lr,
            total_steps=total_steps,
            pct_start=warmup_steps/total_steps,
            anneal_strategy='cos'
        )
        
        # Mixed precision
        self.scaler = GradScaler()
        
        if self.is_main:
            self.log(f"Optimizer setup: LR={scaled_lr:.2e} (scaled from {base_lr:.2e})")
    
    def train_step(self, batch: Dict) -> Dict:
        """Single distributed training step"""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # Mixed precision forward
        with autocast():
            outputs = self.model(
                batch['input_ids'],
                batch.get('attention_mask'),
                use_checkpointing=True
            )
            
            # Compute losses
            price_predictions = outputs['price_predictions']
            batch_size = price_predictions.size(0)
            pred_horizon = self.config['prediction_horizon']
            num_features = batch['labels'].size(-1)
            
            price_predictions = price_predictions.view(batch_size, pred_horizon, num_features)
            price_loss = torch.nn.functional.mse_loss(price_predictions, batch['labels'])
            
            # Action loss
            if 'action_labels' in batch:
                action_logits = outputs['action_logits']
                action_labels = batch['action_labels'].squeeze(-1)
                action_loss = torch.nn.functional.cross_entropy(action_logits, action_labels)
            else:
                action_loss = torch.tensor(0.0).to(self.device)
            
            # Total loss
            total_loss = price_loss + self.config.get('action_loss_weight', 0.5) * action_loss
            
            # Scale for gradient accumulation
            total_loss = total_loss / self.config.get('gradient_accumulation_steps', 1)
        
        # Backward with gradient scaling
        self.scaler.scale(total_loss).backward()
        
        return {
            'total_loss': total_loss.item() * self.config.get('gradient_accumulation_steps', 1),
            'price_loss': price_loss.item(),
            'action_loss': action_loss.item() if action_loss != 0 else 0
        }
    
    def train_epoch(self, train_loader, val_loader=None):
        """Train one epoch with distributed data parallel"""
        epoch_losses = []
        accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        
        # Set epoch for distributed sampler
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(self.epoch)
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Training step
                loss_dict = self.train_step(batch)
                epoch_losses.append(loss_dict)
                
                # Gradient accumulation
                if (batch_idx + 1) % accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.get('max_grad_norm', 1.0)
                    )
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    
                    # Logging (only on main)
                    if self.is_main and self.global_step % self.config.get('log_interval', 50) == 0:
                        avg_loss = np.mean([l['total_loss'] for l in epoch_losses[-10:]])
                        lr = self.scheduler.get_last_lr()[0]
                        
                        self.log(
                            f"Step {self.global_step} | Loss: {avg_loss:.4f} | "
                            f"LR: {lr:.2e} | GPU: {self.rank}/{self.world_size}"
                        )
                    
                    # Validation
                    if val_loader and self.global_step % self.config.get('eval_interval', 500) == 0:
                        val_loss = self.validate(val_loader)
                        
                        if self.is_main and val_loss < self.best_loss:
                            self.best_loss = val_loss
                            self.save_checkpoint('best')
                    
                    # Checkpoint
                    if self.is_main and self.global_step % self.config.get('checkpoint_interval', 1000) == 0:
                        self.save_checkpoint(f'step_{self.global_step}')
                    
                    # Check max steps
                    if self.global_step >= self.config.get('max_steps', 10000):
                        return False
                        
            except Exception as e:
                self.log(f"Error in batch {batch_idx}: {e}", logging.ERROR)
                continue
        
        return True
    
    def validate(self, val_loader):
        """Distributed validation"""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    with autocast():
                        outputs = self.model(batch['input_ids'], batch.get('attention_mask'))
                        
                        # Compute validation loss
                        price_predictions = outputs['price_predictions']
                        batch_size = price_predictions.size(0)
                        pred_horizon = self.config['prediction_horizon']
                        num_features = batch['labels'].size(-1)
                        
                        price_predictions = price_predictions.view(batch_size, pred_horizon, num_features)
                        loss = torch.nn.functional.mse_loss(price_predictions, batch['labels'])
                    
                    val_losses.append(loss.item())
                    
                except Exception as e:
                    continue
        
        # Gather losses from all processes
        if val_losses:
            avg_loss = np.mean(val_losses)
            
            # All-reduce to get global average
            loss_tensor = torch.tensor(avg_loss).to(self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            global_avg_loss = loss_tensor.item() / self.world_size
            
            if self.is_main:
                self.log(f"Validation Loss: {global_avg_loss:.4f}")
            
            return global_avg_loss
        
        return float('inf')
    
    def save_checkpoint(self, name: str):
        """Save checkpoint (only on main process)"""
        if not self.is_main:
            return
        
        checkpoint_dir = Path(self.config.get('checkpoint_dir', 'hftraining/checkpoints/distributed'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.module.state_dict(),  # Unwrap DDP
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        path = checkpoint_dir / f'{name}.pt'
        torch.save(checkpoint, path)
        self.log(f"Checkpoint saved: {path}")
    
    def train(self, train_loader, val_loader=None):
        """Main distributed training loop"""
        self.log(f"Starting distributed training on GPU {self.rank}/{self.world_size}")
        
        max_epochs = self.config.get('max_epochs', 100)
        
        try:
            for epoch in range(max_epochs):
                self.epoch = epoch
                
                if self.is_main:
                    self.log(f"Epoch {epoch+1}/{max_epochs}")
                
                # Train epoch
                should_continue = self.train_epoch(train_loader, val_loader)
                
                if not should_continue:
                    break
                
                # Synchronize processes
                dist.barrier()
                
                # Save epoch checkpoint
                if self.is_main:
                    self.save_checkpoint(f'epoch_{epoch+1}')
            
            # Final checkpoint
            if self.is_main:
                self.save_checkpoint('final')
                
        except KeyboardInterrupt:
            self.log("Training interrupted")
            if self.is_main:
                self.save_checkpoint('interrupted')
        
        except Exception as e:
            self.log(f"Training failed: {e}", logging.ERROR)
            if self.is_main:
                self.save_checkpoint('error')
            raise
        
        finally:
            self.log("Training completed")


def train_worker(rank: int, world_size: int, config: Dict, data_info: Dict):
    """Worker function for distributed training"""
    
    # Setup distributed
    setup_distributed(rank, world_size)
    
    try:
        # Load data
        train_data = np.load(data_info['train_path'])
        val_data = np.load(data_info['val_path'])
        
        # Create datasets
        train_dataset = EnhancedStockDataset(
            train_data,
            sequence_length=config['sequence_length'],
            prediction_horizon=config['prediction_horizon'],
            augment=True
        )
        
        val_dataset = EnhancedStockDataset(
            val_data,
            sequence_length=config['sequence_length'],
            prediction_horizon=config['prediction_horizon'],
            augment=False
        )
        
        # Create distributed samplers
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        
        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['batch_size'] // world_size,  # Per-GPU batch size
            sampler=train_sampler,
            num_workers=2,
            collate_fn=RobustCollator(),
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config['batch_size'] // world_size,
            sampler=val_sampler,
            num_workers=2,
            collate_fn=RobustCollator(),
            pin_memory=True,
            drop_last=False
        )
        
        # Initialize trainer
        trainer = DistributedTrainer(rank, world_size, config)
        
        # Setup model and optimizer
        input_features = train_data.shape[1]
        trainer.setup_model(input_features)
        trainer.setup_optimizer()
        
        # Train
        trainer.train(train_loader, val_loader)
        
    finally:
        cleanup_distributed()


def main():
    """Main function for distributed training"""
    
    # Configuration for distributed training
    config = {
        # Model (larger for multi-GPU)
        'hidden_size': 768,
        'num_heads': 12,
        'num_layers': 12,
        'intermediate_size': 3072,
        'dropout': 0.1,
        
        # Data
        'sequence_length': 60,
        'prediction_horizon': 5,
        'batch_size': 128,  # Total batch size across all GPUs
        
        # Training
        'optimizer': 'adamw',
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'max_steps': 20000,
        'max_epochs': 100,
        'warmup_steps': 1000,
        'gradient_accumulation_steps': 2,
        'max_grad_norm': 1.0,
        
        # Logging
        'log_interval': 50,
        'eval_interval': 500,
        'checkpoint_interval': 1000,
        
        # Other
        'action_loss_weight': 0.5,
        'log_dir': 'hftraining/logs/distributed',
        'checkpoint_dir': 'hftraining/checkpoints/distributed'
    }
    
    # Check GPU availability
    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        print(f"Distributed training requires at least 2 GPUs, found {n_gpus}")
        print("Use train_production.py for single GPU training")
        return
    
    print(f"Found {n_gpus} GPUs for distributed training")
    
    # Download and prepare data
    print("Preparing data...")
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX', 'ORCL', 'CRM']
    
    try:
        data, feature_names = download_and_process_stocks(symbols, start_date='2015-01-01')
        print(f"Data shape: {data.shape}")
    except:
        print("Using synthetic data...")
        np.random.seed(42)
        data = np.random.randn(50000, 20)
    
    # Split and save data
    train_size = int(0.8 * len(data))
    val_size = int(0.1 * len(data))
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    
    # Save to temporary files for workers
    temp_dir = Path('hftraining/temp')
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = temp_dir / 'train_data.npy'
    val_path = temp_dir / 'val_data.npy'
    
    np.save(train_path, train_data)
    np.save(val_path, val_data)
    
    data_info = {
        'train_path': str(train_path),
        'val_path': str(val_path)
    }
    
    print(f"Data prepared: Train {train_data.shape}, Val {val_data.shape}")
    
    # Launch distributed training
    print(f"Launching distributed training on {n_gpus} GPUs...")
    
    mp.spawn(
        train_worker,
        args=(n_gpus, config, data_info),
        nprocs=n_gpus,
        join=True
    )
    
    # Cleanup
    train_path.unlink()
    val_path.unlink()
    
    print("Distributed training completed!")


if __name__ == "__main__":
    main()