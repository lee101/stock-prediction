# GPU Setup and Usage Guide

## Table of Contents
1. [System Requirements](#system-requirements)
2. [CUDA Installation](#cuda-installation)
3. [PyTorch GPU Setup](#pytorch-gpu-setup)
4. [Environment Configuration](#environment-configuration)
5. [GPU Usage in HFTraining](#gpu-usage-in-hftraining)
6. [GPU Usage in HFInference](#gpu-usage-in-hfinference)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)
9. [Monitoring GPU Usage](#monitoring-gpu-usage)

## System Requirements

### Hardware Requirements
- **NVIDIA GPU**: CUDA Compute Capability 3.5 or higher
  - Recommended: RTX 3060 or better for training
  - Minimum: GTX 1050 Ti (4GB VRAM) for inference
- **VRAM Requirements**:
  - Training: 8GB+ recommended (16GB+ for large models)
  - Inference: 4GB minimum
- **System RAM**: 16GB+ recommended

### Software Requirements
- **Operating System**: Linux (Ubuntu 20.04/22.04) or Windows 10/11
- **NVIDIA Driver**: Version 470.0 or newer
- **CUDA Toolkit**: 11.8 or 12.1+ (matching PyTorch requirements)
- **Python**: 3.8-3.11

## CUDA Installation

### Ubuntu/Linux

```bash
# 1. Check current GPU and driver
nvidia-smi

# 2. Install NVIDIA driver (if not installed)
sudo apt update
sudo apt install nvidia-driver-535  # or latest stable version

# 3. Install CUDA Toolkit 12.1
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-1

# 4. Add CUDA to PATH (add to ~/.bashrc)
export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# 5. Verify installation
nvcc --version
nvidia-smi
```

### Windows

1. Download and install [NVIDIA Driver](https://www.nvidia.com/Download/index.aspx)
2. Download and install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
3. Verify installation:
   ```cmd
   nvidia-smi
   nvcc --version
   ```

## PyTorch GPU Setup

### Installation with uv (Recommended)

```bash
# Install PyTorch with CUDA 12.1 support
uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu121

# Or for CUDA 11.8
uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu118

# Install project requirements
uv pip install -r requirements.txt
```

### Verify GPU Access

```python
# tests/test_gpu_setup.py
import torch

def test_gpu_availability():
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
            
        # Test tensor operations
        device = torch.device('cuda')
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        z = torch.matmul(x, y)
        print(f"\nTensor multiplication successful on {device}")
    else:
        print("GPU not available. Check CUDA installation.")

if __name__ == "__main__":
    test_gpu_availability()
```

Run test:
```bash
python tests/test_gpu_setup.py
```

## Environment Configuration

### Environment Variables

Create a `.env` file in project root:
```bash
# GPU Configuration
export CUDA_VISIBLE_DEVICES=0  # Use first GPU (set to 0,1 for multi-GPU)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Mixed Precision
export TORCH_ALLOW_TF32=1  # Enable TF32 for Ampere GPUs (RTX 30xx+)

# Debugging (optional)
export CUDA_LAUNCH_BLOCKING=0  # Set to 1 for debugging
export TORCH_USE_CUDA_DSA=1   # Enable for better error messages
```

### Docker Setup (Optional)

```dockerfile
# Dockerfile.gpu
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git wget && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support
RUN pip3 install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu121

# Copy project files
WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .

# Set environment
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONPATH=/app

CMD ["python3", "hftraining/run_training.py"]
```

Run with Docker:
```bash
docker build -f Dockerfile.gpu -t stock-gpu .
docker run --gpus all -v $(pwd)/data:/app/data stock-gpu
```

## GPU Usage in HFTraining

### Basic GPU Configuration

```python
# hftraining/config.py additions
@dataclass
class GPUConfig:
    """GPU-specific configuration"""
    enabled: bool = True
    device: str = "auto"  # "auto", "cuda", "cuda:0", "cpu"
    mixed_precision: bool = True
    mixed_precision_dtype: str = "float16"  # "float16", "bfloat16"
    allow_tf32: bool = True  # For Ampere GPUs
    gradient_checkpointing: bool = False  # Memory vs speed tradeoff
    multi_gpu_strategy: str = "ddp"  # "dp", "ddp", "none"
    
    def get_device(self) -> torch.device:
        """Get the configured device"""
        if self.device == "auto":
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.device)
```

### Training with GPU

```python
# hftraining/train_hf.py modifications
class HFStockTrainer:
    def __init__(self, config, train_dataset, val_dataset):
        self.gpu_config = config.gpu
        self.device = self.gpu_config.get_device()
        
        # Enable TF32 for Ampere GPUs
        if self.gpu_config.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Initialize model on GPU
        self.model = TransformerTradingModel(config).to(self.device)
        
        # Setup mixed precision
        self.scaler = None
        if self.gpu_config.mixed_precision and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
            self.amp_dtype = (torch.bfloat16 if self.gpu_config.mixed_precision_dtype == "bfloat16" 
                            else torch.float16)
        
        # Multi-GPU setup
        if torch.cuda.device_count() > 1 and self.gpu_config.multi_gpu_strategy != "none":
            self._setup_multi_gpu()
    
    def _setup_multi_gpu(self):
        """Setup multi-GPU training"""
        if self.gpu_config.multi_gpu_strategy == "dp":
            self.model = nn.DataParallel(self.model)
            self.logger.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        elif self.gpu_config.multi_gpu_strategy == "ddp":
            # Requires proper initialization with torch.distributed
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.model = DDP(self.model, device_ids=[self.device])
            self.logger.info(f"Using DistributedDataParallel")
    
    def train_step(self, batch):
        """Single training step with GPU optimization"""
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Mixed precision training
        if self.scaler is not None:
            with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                outputs = self.model(**batch)
                loss = outputs['loss']
            
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.config.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(**batch)
            loss = outputs['loss']
            loss.backward()
            
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            self.optimizer.step()
        
        return loss.item()
```

### Command Line Usage

```bash
# Single GPU training
python hftraining/run_training.py --gpu_device cuda:0 --mixed_precision

# Multi-GPU training
CUDA_VISIBLE_DEVICES=0,1 python hftraining/run_training.py --multi_gpu ddp

# CPU-only training
python hftraining/run_training.py --gpu_device cpu

# With gradient checkpointing (saves memory)
python hftraining/run_training.py --gradient_checkpointing
```

## GPU Usage in HFInference

### Inference Engine GPU Setup

```python
# hfinference/hf_trading_engine.py modifications
class HFTradingEngine:
    def __init__(self, model_path=None, config=None, device='auto', optimize_for_inference=True):
        """
        Initialize trading engine with GPU support
        
        Args:
            device: 'auto', 'cuda', 'cuda:0', 'cpu'
            optimize_for_inference: Enable inference optimizations
        """
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path, config)
        self.model.to(self.device)
        self.model.eval()
        
        # Inference optimizations
        if optimize_for_inference and self.device.type == 'cuda':
            self._optimize_for_inference()
    
    def _optimize_for_inference(self):
        """Apply GPU optimizations for inference"""
        # Enable cudnn benchmarking for consistent input sizes
        torch.backends.cudnn.benchmark = True
        
        # Compile model with torch.compile (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            self.model = torch.compile(self.model, mode="reduce-overhead")
            self.logger.info("Model compiled with torch.compile")
        
        # Use half precision for faster inference
        if self.config.get('use_half_precision', True):
            self.model.half()
            self.logger.info("Using FP16 for inference")
    
    @torch.no_grad()
    def predict(self, data):
        """Run inference with GPU optimization"""
        # Prepare data
        data_tensor = self._prepare_data(data).to(self.device)
        
        # Use autocast for mixed precision
        if self.device.type == 'cuda':
            with torch.cuda.amp.autocast():
                outputs = self.model(data_tensor)
        else:
            outputs = self.model(data_tensor)
        
        return self._process_outputs(outputs)
    
    def batch_predict(self, data_list, batch_size=32):
        """Efficient batch prediction on GPU"""
        predictions = []
        
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i+batch_size]
            batch_tensor = torch.stack([self._prepare_data(d) for d in batch])
            batch_tensor = batch_tensor.to(self.device)
            
            with torch.no_grad():
                if self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_tensor)
                else:
                    outputs = self.model(batch_tensor)
                
                predictions.extend(self._process_outputs(outputs))
        
        return predictions
```

### Production Engine GPU Configuration

```python
# hfinference/production_engine.py modifications
class ProductionTradingEngine:
    def __init__(self, config_path='config/production.yaml'):
        self.config = self._load_config(config_path)
        
        # GPU configuration
        self.gpu_config = self.config.get('gpu', {})
        self.device = self._setup_device()
        
        # Model ensemble on GPU
        self.models = self._load_model_ensemble()
        
        # Warm up GPU
        if self.device.type == 'cuda':
            self._warmup_gpu()
    
    def _setup_device(self):
        """Setup GPU device with fallback"""
        device_str = self.gpu_config.get('device', 'auto')
        
        if device_str == 'auto':
            if torch.cuda.is_available():
                # Select GPU with most free memory
                device_id = self._get_best_gpu()
                return torch.device(f'cuda:{device_id}')
            return torch.device('cpu')
        
        return torch.device(device_str)
    
    def _get_best_gpu(self):
        """Select GPU with most free memory"""
        if torch.cuda.device_count() == 1:
            return 0
        
        max_free = 0
        best_device = 0
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free = props.total_memory - torch.cuda.memory_allocated(i)
            if free > max_free:
                max_free = free
                best_device = i
        
        return best_device
    
    def _warmup_gpu(self):
        """Warm up GPU with dummy forward passes"""
        self.logger.info("Warming up GPU...")
        dummy_input = torch.randn(1, 60, self.config['input_size']).to(self.device)
        
        for model in self.models:
            with torch.no_grad():
                for _ in range(3):
                    _ = model(dummy_input)
        
        torch.cuda.synchronize()
        self.logger.info("GPU warmup complete")
```

## Performance Optimization

### Memory Optimization

```python
# utils/gpu_utils.py
import torch
import gc

def optimize_gpu_memory():
    """Optimize GPU memory usage"""
    if torch.cuda.is_available():
        # Clear cache
        torch.cuda.empty_cache()
        
        # Garbage collection
        gc.collect()
        
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of VRAM
        
        # Enable memory efficient attention (if available)
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)

def profile_gpu_memory(func):
    """Decorator to profile GPU memory usage"""
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_memory = torch.cuda.memory_allocated()
            
        result = func(*args, **kwargs)
        
        if torch.cuda.is_available():
            end_memory = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
            
            print(f"GPU Memory Usage for {func.__name__}:")
            print(f"  Start: {start_memory / 1024**2:.1f} MB")
            print(f"  End: {end_memory / 1024**2:.1f} MB")
            print(f"  Peak: {peak_memory / 1024**2:.1f} MB")
            print(f"  Delta: {(end_memory - start_memory) / 1024**2:.1f} MB")
        
        return result
    return wrapper
```

### Batch Size Optimization

```python
# hftraining/auto_tune.py modifications
class AutoBatchTuner:
    """Automatically find optimal batch size for GPU"""
    
    def find_optimal_batch_size(self, model, dataset, device, max_batch_size=128):
        """Find largest batch size that fits in GPU memory"""
        model.to(device)
        model.eval()
        
        batch_size = max_batch_size
        while batch_size > 0:
            try:
                # Create dummy batch
                dummy_batch = self._create_dummy_batch(batch_size, dataset)
                dummy_batch = {k: v.to(device) for k, v in dummy_batch.items()}
                
                # Try forward pass
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        _ = model(**dummy_batch)
                
                # Try backward pass
                model.train()
                with torch.cuda.amp.autocast():
                    outputs = model(**dummy_batch)
                    loss = outputs['loss']
                
                scaler = torch.cuda.amp.GradScaler()
                scaler.scale(loss).backward()
                
                # Clear gradients
                model.zero_grad()
                torch.cuda.empty_cache()
                
                print(f"Optimal batch size: {batch_size}")
                return batch_size
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    batch_size = int(batch_size * 0.8)  # Reduce by 20%
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    raise e
        
        return 1  # Fallback to batch size of 1
```

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory

```python
# Solutions:
# a) Reduce batch size
config.batch_size = config.batch_size // 2

# b) Enable gradient checkpointing
model.gradient_checkpointing_enable()

# c) Use gradient accumulation
config.gradient_accumulation_steps = 4

# d) Clear cache periodically
if step % 100 == 0:
    torch.cuda.empty_cache()
```

#### 2. CUDA Version Mismatch

```bash
# Check versions
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
nvcc --version

# Reinstall with correct CUDA version
uv pip uninstall torch
uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu121
```

#### 3. Slow GPU Performance

```python
# Enable optimizations
torch.backends.cudnn.benchmark = True  # For consistent input sizes
torch.backends.cuda.matmul.allow_tf32 = True  # For Ampere GPUs
torch.set_float32_matmul_precision('high')  # Balance speed/precision
```

#### 4. Multi-GPU Issues

```bash
# Debug multi-GPU setup
export NCCL_DEBUG=INFO  # Show NCCL communication details
export CUDA_LAUNCH_BLOCKING=1  # Synchronous execution for debugging

# Test multi-GPU
python -m torch.distributed.launch --nproc_per_node=2 hftraining/train_hf.py
```

## Monitoring GPU Usage

### Real-time Monitoring

```bash
# Basic monitoring
watch -n 1 nvidia-smi

# Detailed monitoring
nvidia-smi dmon -s pucvmet -i 0

# Continuous logging
nvidia-smi --query-gpu=timestamp,gpu_name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu --format=csv -l 1 > gpu_log.csv
```

### In-Code Monitoring

```python
# utils/gpu_monitor.py
import torch
import pynvml

class GPUMonitor:
    def __init__(self):
        if torch.cuda.is_available():
            pynvml.nvmlInit()
            self.device_count = torch.cuda.device_count()
    
    def get_gpu_stats(self, device_id=0):
        """Get current GPU statistics"""
        if not torch.cuda.is_available():
            return None
        
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        
        # Memory info
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memory_used = mem_info.used / 1024**3  # GB
        memory_total = mem_info.total / 1024**3  # GB
        
        # Utilization
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        
        # Temperature
        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        
        # Power
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Watts
        
        return {
            'memory_used_gb': memory_used,
            'memory_total_gb': memory_total,
            'memory_percent': (memory_used / memory_total) * 100,
            'gpu_utilization': utilization.gpu,
            'memory_utilization': utilization.memory,
            'temperature': temperature,
            'power_watts': power
        }
    
    def log_gpu_stats(self, logger, step=None):
        """Log GPU stats to logger"""
        for i in range(self.device_count):
            stats = self.get_gpu_stats(i)
            if stats:
                prefix = f"GPU_{i}"
                logger.log({
                    f"{prefix}/memory_gb": stats['memory_used_gb'],
                    f"{prefix}/memory_percent": stats['memory_percent'],
                    f"{prefix}/utilization": stats['gpu_utilization'],
                    f"{prefix}/temperature": stats['temperature'],
                    f"{prefix}/power": stats['power_watts']
                }, step=step)
```

### TensorBoard GPU Metrics

```python
# Add to training loop
from torch.utils.tensorboard import SummaryWriter
from utils.gpu_monitor import GPUMonitor

writer = SummaryWriter('logs/gpu_metrics')
gpu_monitor = GPUMonitor()

for step, batch in enumerate(train_loader):
    # Training step
    loss = train_step(batch)
    
    # Log GPU metrics
    if step % 10 == 0:
        stats = gpu_monitor.get_gpu_stats()
        if stats:
            writer.add_scalar('GPU/Memory_GB', stats['memory_used_gb'], step)
            writer.add_scalar('GPU/Utilization', stats['gpu_utilization'], step)
            writer.add_scalar('GPU/Temperature', stats['temperature'], step)
```

## Best Practices

1. **Always check GPU availability** before assuming CUDA operations
2. **Use mixed precision training** for 2x speedup with minimal accuracy loss
3. **Profile your code** to identify bottlenecks
4. **Monitor temperature** to prevent thermal throttling
5. **Use gradient checkpointing** for large models with limited VRAM
6. **Batch operations** to maximize GPU utilization
7. **Clear cache** periodically to prevent memory fragmentation
8. **Use torch.compile** for inference optimization (PyTorch 2.0+)
9. **Pin memory** for faster CPU-GPU transfers
10. **Use persistent workers** in DataLoader for GPU training

## Additional Resources

- [PyTorch CUDA Documentation](https://pytorch.org/docs/stable/cuda.html)
- [NVIDIA Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/index.html)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
- [Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)