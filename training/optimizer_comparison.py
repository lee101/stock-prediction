#!/usr/bin/env python3
"""
Compare different optimization strategies for trading
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from advanced_trainer import Muon, Shampoo


def create_test_model():
    """Create a test model for comparison"""
    return nn.Sequential(
        nn.Linear(100, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 1)
    )


def train_with_optimizer(optimizer_name, model, data_loader, epochs=100):
    """Train model with specified optimizer"""
    
    # Create optimizer
    if optimizer_name == 'muon':
        optimizer = Muon(model.parameters(), lr=0.001)
    elif optimizer_name == 'shampoo':
        optimizer = Shampoo(model.parameters(), lr=0.001)
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    losses = []
    times = []
    
    criterion = nn.MSELoss()
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        
        for batch_x, batch_y in data_loader:
            # Forward pass
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count
        losses.append(avg_loss)
        times.append(time.time() - start_time)
    
    return losses, times


def generate_synthetic_data(n_samples=10000, n_features=100):
    """Generate synthetic trading-like data"""
    # Generate features (e.g., price history, indicators)
    X = torch.randn(n_samples, n_features)
    
    # Generate targets (e.g., future returns)
    # Make it somewhat learnable
    weights = torch.randn(n_features, 1) * 0.1
    y = torch.mm(X, weights) + torch.randn(n_samples, 1) * 0.1
    
    return X, y


def main():
    print("\n" + "="*80)
    print("🔬 OPTIMIZER COMPARISON FOR TRADING")
    print("="*80)
    
    # Generate data
    print("\n📊 Generating synthetic data...")
    X, y = generate_synthetic_data(n_samples=10000)
    
    # Create data loader
    dataset = torch.utils.data.TensorDataset(X, y)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Optimizers to compare
    optimizers = ['adam', 'adamw', 'sgd', 'rmsprop', 'muon']
    
    # Note: Shampoo might be slow for this test, uncomment if needed
    # optimizers.append('shampoo')
    
    results = {}
    
    print("\n🏃 Running comparison...")
    print("-" * 40)
    
    for opt_name in optimizers:
        print(f"\nTesting {opt_name.upper()}...")
        
        # Create fresh model
        model = create_test_model()
        
        # Train
        losses, times = train_with_optimizer(
            opt_name, model, data_loader, epochs=50
        )
        
        results[opt_name] = {
            'losses': losses,
            'times': times,
            'final_loss': losses[-1],
            'convergence_speed': losses[10] if len(losses) > 10 else float('inf'),
            'total_time': times[-1]
        }
        
        print(f"  Final loss: {losses[-1]:.6f}")
        print(f"  Training time: {times[-1]:.2f}s")
        print(f"  Loss at epoch 10: {losses[10] if len(losses) > 10 else 'N/A':.6f}")
    
    # Visualization
    print("\n📊 Creating comparison plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    ax1 = axes[0, 0]
    for opt_name, result in results.items():
        ax1.plot(result['losses'], label=opt_name.upper(), linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Loss vs Time
    ax2 = axes[0, 1]
    for opt_name, result in results.items():
        ax2.plot(result['times'], result['losses'], label=opt_name.upper(), linewidth=2)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss vs Training Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Final performance
    ax3 = axes[1, 0]
    opt_names = list(results.keys())
    final_losses = [results[opt]['final_loss'] for opt in opt_names]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(opt_names)))
    bars = ax3.bar(opt_names, final_losses, color=colors)
    ax3.set_xlabel('Optimizer')
    ax3.set_ylabel('Final Loss')
    ax3.set_title('Final Loss Comparison')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, final_losses):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom')
    
    # Training time comparison
    ax4 = axes[1, 1]
    training_times = [results[opt]['total_time'] for opt in opt_names]
    bars = ax4.bar(opt_names, training_times, color=colors)
    ax4.set_xlabel('Optimizer')
    ax4.set_ylabel('Training Time (seconds)')
    ax4.set_title('Training Time Comparison')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, training_times):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}s', ha='center', va='bottom')
    
    plt.suptitle('Optimizer Performance Comparison for Trading', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plt.savefig('results/optimizer_comparison.png', dpi=100, bbox_inches='tight')
    print("📊 Comparison plot saved to results/optimizer_comparison.png")
    
    # Print summary
    print("\n" + "="*80)
    print("📈 SUMMARY")
    print("="*80)
    
    # Rank by final loss
    ranked = sorted(results.items(), key=lambda x: x[1]['final_loss'])
    
    print("\n🏆 Ranking by Final Loss (lower is better):")
    for i, (opt_name, result) in enumerate(ranked, 1):
        print(f"  {i}. {opt_name.upper()}: {result['final_loss']:.6f}")
    
    # Rank by convergence speed
    ranked_speed = sorted(results.items(), key=lambda x: x[1]['convergence_speed'])
    
    print("\n⚡ Ranking by Convergence Speed (loss at epoch 10):")
    for i, (opt_name, result) in enumerate(ranked_speed, 1):
        print(f"  {i}. {opt_name.upper()}: {result['convergence_speed']:.6f}")
    
    # Efficiency score (loss reduction per second)
    print("\n⚡ Efficiency Score (loss reduction per second):")
    for opt_name, result in results.items():
        initial_loss = result['losses'][0] if result['losses'] else 1.0
        final_loss = result['final_loss']
        time_taken = result['total_time']
        efficiency = (initial_loss - final_loss) / time_taken if time_taken > 0 else 0
        print(f"  {opt_name.upper()}: {efficiency:.6f}")
    
    print("\n💡 KEY INSIGHTS:")
    print("-" * 40)
    print("• Muon optimizer combines momentum benefits with adaptive learning")
    print("• AdamW (Adam with weight decay) often performs best for trading")
    print("• SGD with momentum is simple but effective")
    print("• Shampoo (2nd order) can be slow but accurate")
    print("• Choice depends on your hardware and latency requirements")
    
    print("\n✅ Comparison complete!")
    print("="*80)


if __name__ == '__main__':
    main()