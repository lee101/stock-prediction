#!/usr/bin/env python3
"""
Quick launcher for base model training with optimized parameters
"""

import argparse
from base_model_trainer import BaseModelTrainer

def main():
    parser = argparse.ArgumentParser(description='Train Universal Base Model')
    parser.add_argument('--config', default='config/base_model_config.json', help='Config file path')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--cross-validation', action='store_true', help='Use cross-validation')
    parser.add_argument('--profit-tracking', action='store_true', default=True, help='Enable profit tracking')
    parser.add_argument('--fine-tune', action='store_true', default=True, help='Run fine-tuning after base training')
    
    args = parser.parse_args()
    
    print("🚀 Starting Universal Base Model Training")
    print(f"Configuration: {args.config}")
    print(f"Epochs: {args.epochs}")
    print(f"Cross-validation: {args.cross_validation}")
    print(f"Profit tracking: {args.profit_tracking}")
    
    # Create trainer
    trainer = BaseModelTrainer(args.config)
    
    # Update configuration based on args
    if hasattr(trainer.config, 'num_train_epochs'):
        trainer.config.num_train_epochs = args.epochs
    
    trainer.base_config.generalization_test = args.cross_validation
    trainer.base_config.profit_tracking_enabled = args.profit_tracking
    trainer.base_config.fine_tune_enabled = args.fine_tune
    
    # Train base model
    print("\n📈 Training base model...")
    base_model_path = trainer.train_base_model()
    
    # Evaluate generalization
    print("\n🔍 Evaluating generalization...")
    generalization_results = trainer.evaluate_generalization(base_model_path)
    
    print("\n📊 Generalization Results:")
    for category, metrics in generalization_results.items():
        print(f"  {category}:")
        print(f"    Mean Return: {metrics['mean_return']:.4f}")
        print(f"    Sharpe Ratio: {metrics['mean_sharpe']:.2f}")
        print(f"    Consistency: {metrics['consistency']:.2%}")
    
    # Fine-tune for strategies if enabled
    if args.fine_tune:
        print("\n🎯 Fine-tuning for specific strategies...")
        
        strategies = [
            {
                'name': 'high_growth', 
                'symbols': ['TSLA', 'NVDA', 'NFLX', 'MSFT', 'U'],
                'description': 'High growth tech stocks'
            },
            {
                'name': 'crypto_focus', 
                'symbols': ['BTCUSD', 'ETHUSD', 'LTCUSD', 'UNIUSD'],
                'description': 'Cryptocurrency trading'
            },
            {
                'name': 'blue_chip', 
                'symbols': ['AAPL', 'MSFT', 'GOOG', 'ADBE'],
                'description': 'Stable blue chip stocks'
            },
            {
                'name': 'balanced_portfolio',
                'symbols': ['AAPL', 'BTCUSD', 'TSLA', 'MSFT', 'ETHUSD', 'NVDA'],
                'description': 'Balanced multi-asset portfolio'
            }
        ]
        
        finetuned_models = {}
        for strategy in strategies:
            print(f"\n  🔧 Fine-tuning: {strategy['name']} ({strategy['description']})")
            model_path = trainer.fine_tune_for_strategy(
                base_model_path=base_model_path,
                target_symbols=strategy['symbols'],
                strategy_name=strategy['name'],
                num_epochs=25  # Fewer epochs for fine-tuning
            )
            finetuned_models[strategy['name']] = model_path
    
    # Summary
    print("\n" + "="*80)
    print("✅ BASE MODEL TRAINING COMPLETED")
    print("="*80)
    print(f"🎯 Base Model: {base_model_path}")
    print(f"📊 Generalization Report: {trainer.output_dir}/generalization_results.json")
    
    if args.fine_tune and 'finetuned_models' in locals():
        print("\n🎯 Fine-tuned Models:")
        for name, path in finetuned_models.items():
            print(f"  {name}: {path}")
    
    print(f"\n📁 All outputs saved to: {trainer.output_dir}")
    print("🔥 Ready for production deployment!")

if __name__ == "__main__":
    main()
