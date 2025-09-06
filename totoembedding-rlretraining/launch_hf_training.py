#!/usr/bin/env python3
"""
Launch script for HuggingFace-style RL training with Toto embeddings
Includes distributed training support and advanced monitoring
"""

import argparse
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime
import numpy as np
from typing import Dict, Any, Optional

from hf_rl_trainer import HFRLConfig, TotoTransformerRL, PPOTrainer
from multi_asset_env import MultiAssetTradingEnv

# Import HF utilities if available
import sys
import logging
sys.path.append('../hftraining')
try:
    from logging_utils import setup_logger
except ImportError:
    # Fallback to basic logging
    def setup_logger(name, log_file=None):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler if specified
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        
        return logger


class HFRLLauncher:
    """
    Advanced launcher for HuggingFace-style RL training
    """
    
    def __init__(self, args):
        self.args = args
        self.config = self._load_config()
        self.logger = setup_logger(
            name="hf_rl_training",
            log_file=f"{self.config.logging_dir}/training_{datetime.now():%Y%m%d_%H%M%S}.log"
        )
        
        # TensorBoard logging is handled inside PPOTrainer via SummaryWriter
        # (No external experiment tracker required.)
    
    def _load_config(self) -> HFRLConfig:
        """Load and merge configuration"""
        # Start with default config
        config = HFRLConfig()
        
        # Load from file if provided
        if self.args.config_file and Path(self.args.config_file).exists():
            with open(self.args.config_file, 'r') as f:
                config_dict = json.load(f)
                for key, value in config_dict.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        
        # Override with command line arguments
        if self.args.learning_rate:
            config.learning_rate = self.args.learning_rate
        if self.args.batch_size:
            config.batch_size = self.args.batch_size
        if self.args.num_epochs:
            config.num_train_epochs = self.args.num_epochs
        if self.args.optimizer:
            config.optimizer_type = self.args.optimizer
        if self.args.no_mixed_precision:
            config.use_mixed_precision = False
        if self.args.gradient_checkpointing:
            config.gradient_checkpointing = True
        if self.args.unfreeze_embeddings:
            config.freeze_toto_embeddings = False
        
        # Create directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.logging_dir).mkdir(parents=True, exist_ok=True)
        
        return config
    
    # Removed W&B setup; using TensorBoard via SummaryWriter in PPOTrainer
    
    def create_environments(self) -> tuple:
        """Create training and evaluation environments"""
        # Load data configuration
        data_config = {
            'data_dir': self.args.train_dir or "../trainingdata/train",
            'symbols': self.args.symbols if self.args.symbols else None,
            'initial_balance': self.args.initial_balance,
            'max_positions': self.args.max_positions,
            'window_size': 30
        }
        
        # Training environment
        train_env = MultiAssetTradingEnv(**data_config)
        
        # Evaluation environment (using test data)
        eval_config = data_config.copy()
        eval_config['data_dir'] = self.args.test_dir or "../trainingdata/test"
        eval_env = MultiAssetTradingEnv(**eval_config)
        
        return train_env, eval_env
    
    def create_model(self, env: MultiAssetTradingEnv) -> TotoTransformerRL:
        """Create the model with proper initialization"""
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        self.logger.info(f"Creating model with obs_dim={obs_dim}, action_dim={action_dim}")
        
        model = TotoTransformerRL(self.config, obs_dim, action_dim)
        
        # Load pretrained weights if specified
        if self.args.pretrained_model:
            self.logger.info(f"Loading pretrained model from {self.args.pretrained_model}")
            checkpoint = torch.load(self.args.pretrained_model, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
        
        # Log model statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        self.logger.info(f"Model Statistics:")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")
        self.logger.info(f"  Frozen parameters: {frozen_params:,}")
        self.logger.info(f"  Frozen ratio: {frozen_params/total_params:.1%}")
        
        return model
    
    def train_single_gpu(self):
        """Single GPU training"""
        self.logger.info("Starting single GPU training")
        
        # Create environments
        train_env, eval_env = self.create_environments()
        
        # Create model
        model = self.create_model(train_env)
        
        # Create trainer
        trainer = PPOTrainer(
            config=self.config,
            model=model,
            env=train_env,
            eval_env=eval_env
        )
        
        # No-op: Trainer internally logs to TensorBoard (SummaryWriter)
        
        # Train
        final_metrics = trainer.train()
        
        # Save final results
        self._save_results(final_metrics)
        
        return final_metrics
    
    def train_distributed(self):
        """Multi-GPU distributed training"""
        world_size = torch.cuda.device_count()
        if world_size < 2:
            self.logger.warning("Less than 2 GPUs available, falling back to single GPU training")
            return self.train_single_gpu()
        
        self.logger.info(f"Starting distributed training on {world_size} GPUs")
        mp.spawn(
            self._train_distributed_worker,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
    
    def _train_distributed_worker(self, rank: int, world_size: int):
        """Worker function for distributed training"""
        # Setup distributed environment
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        
        # Set device
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        
        # Create environments
        train_env, eval_env = self.create_environments()
        
        # Create model
        model = self.create_model(train_env).to(device)
        model = DDP(model, device_ids=[rank])
        
        # Adjust config for distributed training
        self.config.batch_size = self.config.batch_size // world_size
        
        # Create trainer
        trainer = PPOTrainer(
            config=self.config,
            model=model,
            env=train_env,
            eval_env=eval_env
        )
        
        # Train
        if rank == 0:
            # Only main process logs
            final_metrics = trainer.train()
            self._save_results(final_metrics)
        else:
            trainer.train()
        
        dist.destroy_process_group()
    
    def _save_results(self, metrics: Dict[str, Any]):
        """Save training results"""
        results = {
            'config': self.config.__dict__,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'args': vars(self.args)
        }
        
        results_path = f"{self.config.output_dir}/training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {results_path}")
        
        # Results are written to disk; TensorBoard reads from logging_dir
    
    def run(self):
        """Main entry point"""
        self.logger.info("="*60)
        self.logger.info("HuggingFace-style RL Training with Toto Embeddings")
        self.logger.info("="*60)
        
        # Log configuration
        self.logger.info("Configuration:")
        for key, value in self.config.__dict__.items():
            self.logger.info(f"  {key}: {value}")
        
        try:
            if self.args.distributed:
                final_metrics = self.train_distributed()
            else:
                final_metrics = self.train_single_gpu()
            
            self.logger.info("Training completed successfully!")
            
            # Log final metrics
            if final_metrics:
                self.logger.info("Final Metrics:")
                for key, value in final_metrics.items():
                    if isinstance(value, (int, float)):
                        self.logger.info(f"  {key}: {value:.4f}")
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}", exc_info=True)
            raise
        
        finally:
            # Nothing to finalize for TensorBoard SummaryWriter here
            pass


def main():
    parser = argparse.ArgumentParser(description='HuggingFace-style RL Training')
    
    # Configuration
    parser.add_argument('--config-file', type=str, help='Path to configuration JSON file')
    
    # Model configuration
    parser.add_argument('--pretrained-model', type=str, help='Path to pretrained model checkpoint')
    parser.add_argument('--unfreeze-embeddings', action='store_true', help='Unfreeze Toto embeddings for training')
    
    # Training configuration
    parser.add_argument('--num-epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--optimizer', choices=['gpro', 'adamw', 'lion', 'adafactor'], help='Optimizer to use')
    
    # Data configuration
    parser.add_argument('--train-dir', type=str, default='../trainingdata/train', help='Training data directory')
    parser.add_argument('--test-dir', type=str, default='../trainingdata/test', help='Test data directory')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to trade')
    
    # Environment configuration
    parser.add_argument('--initial-balance', type=float, default=100000, help='Initial portfolio balance')
    parser.add_argument('--max-positions', type=int, default=10, help='Maximum number of positions')
    
    # Training options
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')
    parser.add_argument('--no-mixed-precision', action='store_true', help='Disable mixed precision training')
    parser.add_argument('--gradient-checkpointing', action='store_true', help='Enable gradient checkpointing')
    
    # Logging options
    # TensorBoard is enabled by default via PPOTrainer SummaryWriter
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Create and run launcher
    launcher = HFRLLauncher(args)
    launcher.run()


if __name__ == "__main__":
    main()
