#!/usr/bin/env python3
"""GPU Training Wrapper - Sets up environment and runs training"""
import os
import sys
import subprocess

def setup_cuda_env():
    """Setup CUDA environment variables"""
    # CUDA paths
    os.environ['CUDA_HOME'] = '/usr/local/cuda-12.0'
    os.environ['PATH'] = f"/usr/local/cuda-12.0/bin:{os.environ.get('PATH', '')}"
    
    # Library paths
    venv_path = '/media/lee/crucial2/code/stock/.venv'
    lib_paths = [
        f"{venv_path}/lib/python3.12/site-packages/nvidia/nvjitlink/lib",
        f"{venv_path}/lib/python3.12/site-packages/nvidia/cublas/lib", 
        f"{venv_path}/lib/python3.12/site-packages/nvidia/cudnn/lib",
        f"{venv_path}/lib/python3.12/site-packages/nvidia/nccl/lib",
        "/usr/local/cuda-12.0/lib64",
    ]
    
    ld_path = ":".join(lib_paths)
    os.environ['LD_LIBRARY_PATH'] = f"{ld_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    
    # Preload nvjitlink to avoid symbol errors
    os.environ['LD_PRELOAD'] = f"{venv_path}/lib/python3.12/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12"
    
    # CUDA settings
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For debugging
    
    # PyTorch settings
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'  # RTX 3080
    
    print("🔧 CUDA Environment Setup:")
    print(f"  CUDA_HOME: {os.environ['CUDA_HOME']}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    # Check GPU with nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                              capture_output=True, text=True, check=True)
        print(f"  GPU: {result.stdout.strip()}")
    except Exception as e:
        print(f"  ⚠️ nvidia-smi check failed: {e}")

def test_pytorch():
    """Test PyTorch GPU access"""
    print("\n🧪 Testing PyTorch GPU Access...")
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA compiled: {torch.version.cuda}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  ✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # Quick test
            x = torch.randn(100, 100).cuda()
            y = x @ x.T
            print(f"  ✅ GPU compute test successful!")
            return True
        else:
            print("  ❌ CUDA not available - will use CPU")
            return False
    except Exception as e:
        print(f"  ❌ PyTorch test failed: {e}")
        return False

def main():
    """Main entry point"""
    print("="*60)
    print("🚀 GPU Training Launcher")
    print("="*60)
    
    # Setup environment
    setup_cuda_env()
    
    # Test GPU
    gpu_available = test_pytorch()
    
    # Run training
    print("\n📊 Starting Training...")
    print("-"*60)
    
    # Import and run training
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_hf import main as train_main
    
    try:
        train_main()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()