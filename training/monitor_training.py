#!/usr/bin/env python3
"""
Monitor training progress from checkpoint files
"""

import json
import torch
from pathlib import Path
import time
from datetime import datetime


def monitor_checkpoints():
    """Monitor training progress from saved checkpoints"""
    
    models_dir = Path('models')
    results_dir = Path('results')
    
    print("\n" + "="*80)
    print("📊 TRAINING MONITOR")
    print("="*80)
    
    while True:
        print(f"\n🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 40)
        
        # Check for best models
        best_models = list(models_dir.glob('best_*.pth'))
        if best_models:
            print("\n🏆 Best Models Found:")
            for model_path in best_models:
                try:
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                    if 'metrics' in checkpoint:
                        metrics = checkpoint['metrics']
                        if metrics.get('episode_sharpes'):
                            best_sharpe = max(metrics['episode_sharpes'][-10:]) if len(metrics['episode_sharpes']) > 0 else 0
                            print(f"  {model_path.name}: Best Sharpe = {best_sharpe:.3f}")
                        if metrics.get('episode_profits'):
                            best_return = max(metrics['episode_profits'][-10:]) if len(metrics['episode_profits']) > 0 else 0
                            print(f"    Best Return = {best_return:.2%}")
                except Exception as e:
                    print(f"  Could not load {model_path.name}: {e}")
        
        # Check for recent checkpoints
        checkpoints = sorted(models_dir.glob('checkpoint_ep*.pth'), key=lambda x: x.stat().st_mtime, reverse=True)[:3]
        if checkpoints:
            print("\n📁 Recent Checkpoints:")
            for cp_path in checkpoints:
                try:
                    checkpoint = torch.load(cp_path, map_location='cpu', weights_only=False)
                    episode = cp_path.stem.split('ep')[-1]
                    print(f"  Episode {episode}")
                    
                    if 'metrics' in checkpoint:
                        metrics = checkpoint['metrics']
                        if metrics.get('episode_rewards') and len(metrics['episode_rewards']) > 0:
                            recent_reward = metrics['episode_rewards'][-1]
                            print(f"    Last Reward: {recent_reward:.3f}")
                        if metrics.get('episode_sharpes') and len(metrics['episode_sharpes']) > 0:
                            recent_sharpe = metrics['episode_sharpes'][-1]
                            print(f"    Last Sharpe: {recent_sharpe:.3f}")
                        if metrics.get('episode_profits') and len(metrics['episode_profits']) > 0:
                            recent_return = metrics['episode_profits'][-1]
                            print(f"    Last Return: {recent_return:.2%}")
                except Exception as e:
                    print(f"  Could not load {cp_path.name}")
        
        # Check for result files
        result_files = list(results_dir.glob('*.json'))
        if result_files:
            print("\n📈 Latest Results:")
            latest_result = max(result_files, key=lambda x: x.stat().st_mtime)
            try:
                with open(latest_result, 'r') as f:
                    results = json.load(f)
                    if 'test_metrics' in results:
                        test_metrics = results['test_metrics']
                        print(f"  {latest_result.name}:")
                        print(f"    Test Return: {test_metrics.get('total_return', 0):.2%}")
                        print(f"    Test Sharpe: {test_metrics.get('sharpe_ratio', 0):.3f}")
                        print(f"    Win Rate: {test_metrics.get('win_rate', 0):.2%}")
                        
                        # Check if profitable
                        if test_metrics.get('total_return', 0) > 0.05 and test_metrics.get('sharpe_ratio', 0) > 1.0:
                            print("\n🎉 *** PROFITABLE MODEL ACHIEVED! ***")
                            print(f"  Return: {test_metrics.get('total_return', 0):.2%}")
                            print(f"  Sharpe: {test_metrics.get('sharpe_ratio', 0):.3f}")
                            return True
            except Exception as e:
                print(f"  Could not load {latest_result.name}")
        
        # Wait before next check
        time.sleep(30)


if __name__ == '__main__':
    try:
        monitor_checkpoints()
    except KeyboardInterrupt:
        print("\n\n✋ Monitoring stopped")