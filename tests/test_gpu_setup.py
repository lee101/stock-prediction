#!/usr/bin/env python3
"""
GPU Setup Test Script
Tests GPU availability and functionality for training and inference.
"""

import torch
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.gpu_utils import GPUManager, GPUMonitor, log_gpu_info, get_device


def test_cuda_availability():
    """Test basic CUDA availability"""
    print("=" * 60)
    print("CUDA Availability Test")
    print("=" * 60)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Multi-processor count: {props.multi_processor_count}")
    else:
        print("\n⚠️  No CUDA-capable GPU detected!")
        print("Training and inference will run on CPU (slower)")
    
    print()


def test_gpu_operations():
    """Test basic GPU tensor operations"""
    print("=" * 60)
    print("GPU Operations Test")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("Skipping GPU operations test (no GPU available)")
        return
    
    device = torch.device('cuda')
    
    try:
        # Test tensor creation
        print("Creating tensors on GPU...")
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        # Test computation
        print("Testing matrix multiplication...")
        z = torch.matmul(x, y)
        
        # Test memory
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"Memory allocated: {allocated:.1f} MB")
        print(f"Memory reserved: {reserved:.1f} MB")
        
        # Test mixed precision
        print("\nTesting mixed precision...")
        with torch.cuda.amp.autocast():
            z_amp = torch.matmul(x, y)
        
        print("✓ GPU operations successful!")
        
    except Exception as e:
        print(f"✗ GPU operations failed: {e}")
    
    finally:
        # Clean up
        torch.cuda.empty_cache()
    
    print()


def test_gpu_utils():
    """Test GPU utility functions"""
    print("=" * 60)
    print("GPU Utils Test")
    print("=" * 60)
    
    # Test GPUManager
    manager = GPUManager()
    print(f"CUDA available: {manager.cuda_available}")
    print(f"Device count: {manager.device_count}")
    
    if manager.cuda_available:
        # Get best GPU
        best_gpu = manager.get_best_gpu()
        print(f"Best GPU selected: {best_gpu}")
        
        # Get GPU info
        info = manager.get_gpu_info(0)
        if info:
            print(f"\nGPU 0 Info:")
            print(f"  Name: {info.name}")
            print(f"  Memory: {info.memory_used:.1f}/{info.memory_total:.1f} GB")
            print(f"  Compute capability: {info.compute_capability}")
            if info.temperature:
                print(f"  Temperature: {info.temperature}°C")
            if info.power:
                print(f"  Power: {info.power:.1f}W")
        
        # Test memory optimization
        print("\nOptimizing memory...")
        manager.optimize_memory()
        
        # Test optimization flags
        print("Setting optimization flags...")
        manager.setup_optimization_flags(allow_tf32=True, benchmark_cudnn=True)
    
    print()


def test_model_on_gpu():
    """Test loading a simple model on GPU"""
    print("=" * 60)
    print("Model GPU Test")
    print("=" * 60)
    
    device = get_device("auto")
    print(f"Using device: {device}")
    
    # Create a simple model
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(100, 256)
            self.relu = torch.nn.ReLU()
            self.linear2 = torch.nn.Linear(256, 10)
        
        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            return x
    
    try:
        # Create and move model to device
        model = SimpleModel().to(device)
        print(f"Model moved to {device}")
        
        # Test forward pass
        batch_size = 32
        input_data = torch.randn(batch_size, 100).to(device)
        
        with torch.no_grad():
            output = model(input_data)
        
        print(f"Forward pass successful: input {input_data.shape} -> output {output.shape}")
        
        # Test backward pass
        model.train()
        output = model(input_data)
        loss = output.mean()
        loss.backward()
        
        print("Backward pass successful")
        
        # Test mixed precision if GPU
        if device.type == 'cuda':
            print("\nTesting mixed precision training...")
            scaler = torch.cuda.amp.GradScaler()
            optimizer = torch.optim.Adam(model.parameters())
            
            with torch.cuda.amp.autocast():
                output = model(input_data)
                loss = output.mean()
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            print("Mixed precision training successful")
        
        print("\n✓ Model GPU test passed!")
        
    except Exception as e:
        print(f"\n✗ Model GPU test failed: {e}")
    
    print()


def test_multi_gpu():
    """Test multi-GPU setup if available"""
    print("=" * 60)
    print("Multi-GPU Test")
    print("=" * 60)
    
    if torch.cuda.device_count() < 2:
        print(f"Only {torch.cuda.device_count()} GPU(s) available, skipping multi-GPU test")
        return
    
    print(f"Found {torch.cuda.device_count()} GPUs")
    
    try:
        # Create a simple model
        model = torch.nn.Linear(100, 10)
        
        # Test DataParallel
        model_dp = torch.nn.DataParallel(model)
        print("DataParallel wrapper created")
        
        # Test forward pass
        input_data = torch.randn(64, 100).cuda()
        output = model_dp(input_data)
        
        print(f"Multi-GPU forward pass successful: {output.shape}")
        print("✓ Multi-GPU test passed!")
        
    except Exception as e:
        print(f"✗ Multi-GPU test failed: {e}")
    
    print()


def main():
    """Run all GPU tests"""
    print("\n" + "=" * 60)
    print("GPU SETUP TEST SUITE")
    print("=" * 60 + "\n")
    
    # Run tests
    test_cuda_availability()
    test_gpu_operations()
    test_gpu_utils()
    test_model_on_gpu()
    test_multi_gpu()
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print("✓ GPU is available and functional")
        print("✓ Ready for GPU-accelerated training and inference")
        
        # Log detailed GPU info
        print("\nDetailed GPU Information:")
        log_gpu_info()
    else:
        print("⚠️  No GPU detected - will use CPU")
        print("   For better performance, consider:")
        print("   1. Installing CUDA and cuDNN")
        print("   2. Installing PyTorch with CUDA support")
        print("   3. Using a machine with NVIDIA GPU")
    
    print("\nFor full GPU setup instructions, see: GPU_SETUP_GUIDE.md")


if __name__ == "__main__":
    main()