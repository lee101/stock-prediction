#!/usr/bin/env python3
"""Simple CUDA test without torch"""
import subprocess
import sys

# Test nvidia-smi first
print("Testing nvidia-smi...")
result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                       capture_output=True, text=True)
if result.returncode == 0:
    print(f"✅ GPU detected: {result.stdout.strip()}")
else:
    print(f"❌ nvidia-smi failed: {result.stderr}")
    
# Try importing torch with better error handling
print("\nTrying to import torch...")
try:
    import os
    # Set environment to help CUDA initialization
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'  # RTX 3080 is compute 8.6
    
    # Preload the library
    os.environ['LD_PRELOAD'] = '/media/lee/crucial2/code/stock/.venv/lib/python3.12/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12'
    
    import torch
    print(f"✅ PyTorch imported: {torch.__version__}")
    print(f"CUDA compiled: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        # Try to get more info about why CUDA isn't available
        print("\nDebug info:")
        print(f"_CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        print(f"torch._C._cuda_getDeviceCount: {torch._C._cuda_getDeviceCount()}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()