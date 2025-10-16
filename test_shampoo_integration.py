#!/usr/bin/env python3
"""Test Shampoo optimizer integration in training scripts"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "hftraining"))

def test_shampoo_import():
    """Test that Shampoo can be imported"""
    try:
        from hftraining.modern_optimizers import Shampoo
        print("✓ Shampoo import successful")
        return True
    except ImportError as e:
        print(f"✗ Failed to import Shampoo: {e}")
        return False

def test_shampoo_basic():
    """Test basic Shampoo functionality"""
    try:
        from hftraining.modern_optimizers import Shampoo
        
        # Create simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 1)
        )
        
        # Create optimizer
        optimizer = Shampoo(
            model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-10,
            weight_decay=0.01
        )
        
        # Test training step
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)
        
        # Forward pass
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print("✓ Shampoo basic training step successful")
        return True
        
    except Exception as e:
        print(f"✗ Shampoo basic test failed: {e}")
        return False

def test_training_scripts():
    """Test that training scripts can use Shampoo"""
    scripts_to_test = [
        "hftraining/train_production_v2.py",
        "hftraining/train_optimized.py", 
        "hftraining/train_fixed.py"
    ]
    
    results = []
    for script in scripts_to_test:
        script_path = Path(script)
        if not script_path.exists():
            print(f"✗ Script not found: {script}")
            results.append(False)
            continue
            
        # Check if Shampoo import is present
        content = script_path.read_text()
        if "from modern_optimizers import Shampoo" in content:
            print(f"✓ {script} has Shampoo import")
            results.append(True)
        else:
            print(f"✗ {script} missing Shampoo import")
            results.append(False)
    
    return all(results)

def test_optimizer_creation():
    """Test creating Shampoo optimizer with different configurations"""
    try:
        from hftraining.modern_optimizers import Shampoo
        
        configs = [
            {"lr": 0.001, "betas": (0.9, 0.999)},
            {"lr": 0.0001, "betas": (0.95, 0.999), "weight_decay": 0.01},
            {"lr": 0.003, "eps": 1e-8}
        ]
        
        model = torch.nn.Linear(10, 10)
        
        for i, config in enumerate(configs):
            optimizer = Shampoo(model.parameters(), **config)
            print(f"✓ Config {i+1} created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Optimizer creation failed: {e}")
        return False

def run_quick_training_test():
    """Run a quick training test with Shampoo"""
    try:
        from hftraining.modern_optimizers import Shampoo
        
        # Simple dataset
        X = torch.randn(100, 10)
        y = torch.randn(100, 1)
        
        # Simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
        
        optimizer = Shampoo(model.parameters(), lr=0.001)  # Lower LR for Shampoo
        
        # Train for a few steps
        initial_loss = None
        for epoch in range(10):
            output = model(X)
            loss = torch.nn.functional.mse_loss(output, y)
            
            if initial_loss is None:
                initial_loss = loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        final_loss = loss.item()
        
        if final_loss < initial_loss:
            print(f"✓ Training converged: {initial_loss:.4f} -> {final_loss:.4f}")
            return True
        else:
            print(f"✗ Training did not converge: {initial_loss:.4f} -> {final_loss:.4f}")
            return False
            
    except Exception as e:
        print(f"✗ Quick training test failed: {e}")
        return False

def main():
    print("=" * 60)
    print("Testing Shampoo Optimizer Integration")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_shampoo_import),
        ("Basic Functionality", test_shampoo_basic),
        ("Training Scripts", test_training_scripts),
        ("Optimizer Creation", test_optimizer_creation),
        ("Quick Training", run_quick_training_test)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{name}:")
        results.append(test_func())
    
    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    
    if all(results):
        print("✓ All tests passed! Shampoo is ready to use.")
    else:
        print("✗ Some tests failed. Check the output above.")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)