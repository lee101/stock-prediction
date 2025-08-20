#!/usr/bin/env python3
"""
Analyze and compare different model checkpoints
Find the best model based on various metrics
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import json


def analyze_checkpoint(model_path):
    """Analyze a single checkpoint file"""
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    info = {
        'file': model_path.name,
        'episode': checkpoint.get('episode', -1),
        'metric_type': checkpoint.get('metric_type', 'unknown'),
        'metric_value': checkpoint.get('metric_value', 0),
        'run_name': checkpoint.get('run_name', 'unknown'),
        'timestamp': checkpoint.get('timestamp', 'unknown'),
        'global_step': checkpoint.get('global_step', 0)
    }
    
    # Extract metrics if available
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        
        # Get last values
        if 'episode_rewards' in metrics and len(metrics['episode_rewards']) > 0:
            info['last_reward'] = metrics['episode_rewards'][-1]
            info['avg_reward_last_10'] = np.mean(metrics['episode_rewards'][-10:]) if len(metrics['episode_rewards']) >= 10 else info['last_reward']
        
        if 'episode_sharpes' in metrics and len(metrics['episode_sharpes']) > 0:
            info['last_sharpe'] = metrics['episode_sharpes'][-1]
            info['avg_sharpe_last_10'] = np.mean(metrics['episode_sharpes'][-10:]) if len(metrics['episode_sharpes']) >= 10 else info['last_sharpe']
            info['max_sharpe'] = max(metrics['episode_sharpes'])
        
        if 'episode_profits' in metrics and len(metrics['episode_profits']) > 0:
            info['last_profit'] = metrics['episode_profits'][-1]
            info['avg_profit_last_10'] = np.mean(metrics['episode_profits'][-10:]) if len(metrics['episode_profits']) >= 10 else info['last_profit']
            info['max_profit'] = max(metrics['episode_profits'])
        
        if 'actor_losses' in metrics and len(metrics['actor_losses']) > 0:
            info['last_actor_loss'] = metrics['actor_losses'][-1]
            info['avg_actor_loss'] = np.mean(metrics['actor_losses'][-100:]) if len(metrics['actor_losses']) >= 100 else np.mean(metrics['actor_losses'])
        
        if 'critic_losses' in metrics and len(metrics['critic_losses']) > 0:
            info['last_critic_loss'] = metrics['critic_losses'][-1]
            info['avg_critic_loss'] = np.mean(metrics['critic_losses'][-100:]) if len(metrics['critic_losses']) >= 100 else np.mean(metrics['critic_losses'])
    
    return info


def find_best_checkpoint(models_dir='models'):
    """Find the best checkpoint based on different criteria"""
    
    models_path = Path(models_dir)
    if not models_path.exists():
        print(f"❌ Models directory not found: {models_dir}")
        return None
    
    # Find all checkpoint files
    checkpoint_files = list(models_path.glob('*.pth'))
    
    if not checkpoint_files:
        print(f"❌ No checkpoint files found in {models_dir}")
        return None
    
    print(f"\n📊 Analyzing {len(checkpoint_files)} checkpoints...")
    print("-" * 80)
    
    # Analyze all checkpoints
    all_info = []
    for checkpoint_file in checkpoint_files:
        try:
            info = analyze_checkpoint(checkpoint_file)
            all_info.append(info)
            print(f"✓ {checkpoint_file.name}: Episode {info['episode']}, "
                  f"{info['metric_type']}={info['metric_value']:.4f}")
        except Exception as e:
            print(f"✗ Failed to load {checkpoint_file.name}: {e}")
    
    if not all_info:
        print("❌ No valid checkpoints found")
        return None
    
    # Convert to DataFrame for easy analysis
    df = pd.DataFrame(all_info)
    
    print("\n" + "="*80)
    print("🏆 BEST MODELS BY DIFFERENT CRITERIA")
    print("="*80)
    
    results = {}
    
    # Best by stored metric value (what the training thought was best)
    if 'metric_value' in df.columns:
        best_idx = df['metric_value'].idxmax()
        best = df.loc[best_idx]
        print(f"\n📈 Best by Training Metric ({best['metric_type']}):")
        print(f"  File: {best['file']}")
        print(f"  Episode: {best['episode']}")
        print(f"  {best['metric_type']}: {best['metric_value']:.4f}")
        results['best_training_metric'] = best['file']
    
    # Best by Sharpe ratio
    if 'max_sharpe' in df.columns:
        best_idx = df['max_sharpe'].idxmax()
        best = df.loc[best_idx]
        print(f"\n📊 Best by Sharpe Ratio:")
        print(f"  File: {best['file']}")
        print(f"  Episode: {best['episode']}")
        print(f"  Max Sharpe: {best['max_sharpe']:.4f}")
        print(f"  Avg Sharpe (last 10): {best.get('avg_sharpe_last_10', 0):.4f}")
        results['best_sharpe'] = best['file']
    
    # Best by profit
    if 'max_profit' in df.columns:
        best_idx = df['max_profit'].idxmax()
        best = df.loc[best_idx]
        print(f"\n💰 Best by Profit:")
        print(f"  File: {best['file']}")
        print(f"  Episode: {best['episode']}")
        print(f"  Max Profit: {best['max_profit']:.2%}")
        print(f"  Avg Profit (last 10): {best.get('avg_profit_last_10', 0):.2%}")
        results['best_profit'] = best['file']
    
    # Best by lowest loss
    if 'avg_actor_loss' in df.columns:
        best_idx = df['avg_actor_loss'].idxmin()
        best = df.loc[best_idx]
        print(f"\n📉 Best by Lowest Actor Loss:")
        print(f"  File: {best['file']}")
        print(f"  Episode: {best['episode']}")
        print(f"  Avg Actor Loss: {best['avg_actor_loss']:.6f}")
        results['best_loss'] = best['file']
    
    # Find the sweet spot around episode 600
    df_filtered = df[(df['episode'] >= 550) & (df['episode'] <= 650)]
    if not df_filtered.empty and 'max_sharpe' in df_filtered.columns:
        best_idx = df_filtered['max_sharpe'].idxmax()
        best = df_filtered.loc[best_idx]
        print(f"\n🎯 Best Around Episode 600 (Sweet Spot):")
        print(f"  File: {best['file']}")
        print(f"  Episode: {best['episode']}")
        print(f"  Max Sharpe: {best.get('max_sharpe', 0):.4f}")
        print(f"  Max Profit: {best.get('max_profit', 0):.2%}")
        results['best_episode_600'] = best['file']
    
    # Create comparison plot
    if len(df) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Episode vs Metric Value
        if 'episode' in df.columns and 'metric_value' in df.columns:
            ax = axes[0, 0]
            ax.scatter(df['episode'], df['metric_value'], alpha=0.6)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Metric Value')
            ax.set_title('Training Progress')
            ax.grid(True, alpha=0.3)
            
            # Mark episode 600 region
            ax.axvspan(550, 650, alpha=0.2, color='red', label='Sweet Spot')
            ax.legend()
        
        # Plot 2: Max Sharpe by Episode
        if 'episode' in df.columns and 'max_sharpe' in df.columns:
            ax = axes[0, 1]
            ax.scatter(df['episode'], df['max_sharpe'], alpha=0.6, color='green')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Max Sharpe Ratio')
            ax.set_title('Sharpe Ratio Progress')
            ax.grid(True, alpha=0.3)
            ax.axvspan(550, 650, alpha=0.2, color='red')
        
        # Plot 3: Max Profit by Episode
        if 'episode' in df.columns and 'max_profit' in df.columns:
            ax = axes[1, 0]
            ax.scatter(df['episode'], df['max_profit'], alpha=0.6, color='blue')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Max Profit (%)')
            ax.set_title('Profit Progress')
            ax.grid(True, alpha=0.3)
            ax.axvspan(550, 650, alpha=0.2, color='red')
        
        # Plot 4: Loss Progress
        if 'episode' in df.columns and 'avg_actor_loss' in df.columns:
            ax = axes[1, 1]
            ax.scatter(df['episode'], df['avg_actor_loss'], alpha=0.6, color='orange')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Avg Actor Loss')
            ax.set_title('Loss Progress')
            ax.grid(True, alpha=0.3)
            ax.axvspan(550, 650, alpha=0.2, color='red')
        
        plt.suptitle('Checkpoint Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plt.savefig('checkpoint_analysis.png', dpi=100, bbox_inches='tight')
        print(f"\n📊 Analysis plot saved to checkpoint_analysis.png")
        plt.show()
    
    # Save results to JSON
    with open('best_checkpoints.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📁 Best checkpoints saved to best_checkpoints.json")
    
    # Create summary CSV
    df.to_csv('checkpoint_summary.csv', index=False)
    print(f"📁 Full summary saved to checkpoint_summary.csv")
    
    return results


def compare_models_on_stock(model_files, stock='AAPL', start='2023-01-01', end='2024-01-01'):
    """Compare multiple models on the same stock"""
    
    from visualize_trades import TradeVisualizer
    
    results = []
    
    for model_file in model_files:
        if not Path(model_file).exists():
            print(f"❌ Model not found: {model_file}")
            continue
        
        print(f"\n📊 Testing {model_file} on {stock}...")
        
        visualizer = TradeVisualizer(
            model_path=model_file,
            stock_symbol=stock,
            start_date=start,
            end_date=end
        )
        
        visualizer.run_backtest()
        
        results.append({
            'model': Path(model_file).name,
            'stock': stock,
            'total_return': visualizer.final_metrics.get('total_return', 0),
            'sharpe_ratio': visualizer.final_metrics.get('sharpe_ratio', 0),
            'max_drawdown': visualizer.final_metrics.get('max_drawdown', 0),
            'win_rate': visualizer.final_metrics.get('win_rate', 0),
            'num_trades': visualizer.final_metrics.get('num_trades', 0)
        })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    
    if not comparison_df.empty:
        print("\n" + "="*80)
        print(f"📊 MODEL COMPARISON ON {stock}")
        print("="*80)
        print(comparison_df.to_string())
        
        # Save to CSV
        comparison_df.to_csv(f'model_comparison_{stock}.csv', index=False)
        print(f"\n📁 Comparison saved to model_comparison_{stock}.csv")
    
    return comparison_df


def main():
    """Main function"""
    
    print("\n" + "="*80)
    print("🔍 CHECKPOINT ANALYSIS SYSTEM")
    print("="*80)
    
    # Find best checkpoints
    best_models = find_best_checkpoint('models')
    
    if best_models:
        print("\n" + "="*80)
        print("🎯 RECOMMENDATIONS")
        print("="*80)
        
        print("\n1. For maximum profit potential:")
        print(f"   Use: {best_models.get('best_profit', 'N/A')}")
        
        print("\n2. For best risk-adjusted returns:")
        print(f"   Use: {best_models.get('best_sharpe', 'N/A')}")
        
        print("\n3. For the sweet spot (episode ~600):")
        print(f"   Use: {best_models.get('best_episode_600', 'N/A')}")
        
        print("\n4. For lowest prediction error:")
        print(f"   Use: {best_models.get('best_loss', 'N/A')}")
        
        # Test on unseen stock
        if best_models.get('best_episode_600'):
            print("\n" + "="*80)
            print("🧪 TESTING BEST MODEL ON UNSEEN STOCK (AAPL)")
            print("="*80)
            
            model_path = f"models/{best_models['best_episode_600']}"
            
            # Compare different models
            models_to_test = []
            if best_models.get('best_episode_600'):
                models_to_test.append(f"models/{best_models['best_episode_600']}")
            if best_models.get('best_profit') and best_models.get('best_profit') != best_models.get('best_episode_600'):
                models_to_test.append(f"models/{best_models['best_profit']}")
            if best_models.get('best_sharpe') and best_models.get('best_sharpe') != best_models.get('best_episode_600'):
                models_to_test.append(f"models/{best_models['best_sharpe']}")
            
            if models_to_test:
                compare_models_on_stock(models_to_test, stock='AAPL')
    
    print("\n✅ Analysis complete!")


if __name__ == '__main__':
    main()