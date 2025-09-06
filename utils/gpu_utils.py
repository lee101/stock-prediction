#!/usr/bin/env python3
"""
GPU Utilities for Training and Inference
Provides common GPU operations, monitoring, and optimization utilities.
"""

import torch
import gc
import os
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

# Optional dependencies
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """GPU information and statistics"""
    device_id: int
    name: str
    memory_total: float  # GB
    memory_used: float  # GB
    memory_free: float  # GB
    utilization: float  # %
    temperature: Optional[float] = None  # Celsius
    power: Optional[float] = None  # Watts
    compute_capability: Optional[Tuple[int, int]] = None


class GPUManager:
    """Manages GPU device selection and configuration"""
    
    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.cuda_available else 0
        
        if PYNVML_AVAILABLE and self.cuda_available:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
            except Exception as e:
                logger.warning(f"Failed to initialize NVML: {e}")
                self.nvml_initialized = False
        else:
            self.nvml_initialized = False
    
    def get_device(self, device: str = "auto") -> torch.device:
        """
        Get the appropriate device based on configuration.
        
        Args:
            device: Device specification ('auto', 'cuda', 'cuda:0', 'cpu')
        
        Returns:
            torch.device: The selected device
        """
        if device == "auto":
            if self.cuda_available:
                # Select GPU with most free memory
                best_device = self.get_best_gpu()
                return torch.device(f'cuda:{best_device}')
            return torch.device('cpu')
        
        return torch.device(device)
    
    def get_best_gpu(self) -> int:
        """Select GPU with most free memory"""
        if not self.cuda_available:
            return 0
        
        if self.device_count == 1:
            return 0
        
        max_free = 0
        best_device = 0
        
        for i in range(self.device_count):
            free = self.get_gpu_memory_info(i)['free']
            if free > max_free:
                max_free = free
                best_device = i
        
        logger.info(f"Selected GPU {best_device} with {max_free:.1f}GB free memory")
        return best_device
    
    def get_gpu_info(self, device_id: int = 0) -> Optional[GPUInfo]:
        """Get comprehensive GPU information"""
        if not self.cuda_available or device_id >= self.device_count:
            return None
        
        # Basic PyTorch info
        props = torch.cuda.get_device_properties(device_id)
        memory_info = self.get_gpu_memory_info(device_id)
        
        info = GPUInfo(
            device_id=device_id,
            name=props.name,
            memory_total=props.total_memory / 1024**3,
            memory_used=memory_info['used'],
            memory_free=memory_info['free'],
            utilization=0.0,
            compute_capability=(props.major, props.minor)
        )
        
        # Extended info from NVML if available
        if self.nvml_initialized:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                
                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                info.utilization = util.gpu
                
                # Temperature
                info.temperature = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                
                # Power
                info.power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Watts
                
            except Exception as e:
                logger.debug(f"Failed to get extended GPU info: {e}")
        
        return info
    
    def get_gpu_memory_info(self, device_id: int = 0) -> Dict[str, float]:
        """Get GPU memory information in GB"""
        if not self.cuda_available or device_id >= self.device_count:
            return {'total': 0, 'used': 0, 'free': 0}
        
        torch.cuda.set_device(device_id)
        total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(device_id) / 1024**3
        reserved = torch.cuda.memory_reserved(device_id) / 1024**3
        free = total - reserved
        
        return {
            'total': total,
            'allocated': allocated,
            'reserved': reserved,
            'used': reserved,
            'free': free
        }
    
    def optimize_memory(self, device_id: Optional[int] = None):
        """Optimize GPU memory usage"""
        if not self.cuda_available:
            return
        
        if device_id is not None:
            torch.cuda.set_device(device_id)
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Garbage collection
        gc.collect()
        
        # Log memory stats
        if device_id is not None:
            mem_info = self.get_gpu_memory_info(device_id)
            logger.info(f"GPU {device_id} memory after optimization: "
                       f"{mem_info['used']:.1f}/{mem_info['total']:.1f} GB used")
    
    def setup_optimization_flags(self, allow_tf32: bool = True, 
                                benchmark_cudnn: bool = True,
                                deterministic: bool = False):
        """Setup GPU optimization flags"""
        if not self.cuda_available:
            return
        
        # TF32 for Ampere GPUs (RTX 30xx/40xx)
        if allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("Enabled TF32 for matrix operations")
        
        # CuDNN benchmarking
        if benchmark_cudnn and not deterministic:
            torch.backends.cudnn.benchmark = True
            logger.info("Enabled CuDNN benchmarking")
        
        # Deterministic mode (slower but reproducible)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info("Enabled deterministic mode")


class GPUMonitor:
    """Monitor GPU usage during training/inference"""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.manager = GPUManager()
        self.history = []
    
    def get_current_stats(self) -> Optional[Dict[str, float]]:
        """Get current GPU statistics"""
        info = self.manager.get_gpu_info(self.device_id)
        if info is None:
            return None
        
        stats = {
            'memory_used_gb': info.memory_used,
            'memory_total_gb': info.memory_total,
            'memory_percent': (info.memory_used / info.memory_total) * 100,
            'utilization': info.utilization,
            'temperature': info.temperature,
            'power': info.power
        }
        
        self.history.append(stats)
        return stats
    
    def log_stats(self, logger_func=None, prefix: str = "GPU"):
        """Log current GPU statistics"""
        stats = self.get_current_stats()
        if stats is None:
            return
        
        if logger_func is None:
            logger_func = logger.info
        
        logger_func(f"{prefix} Stats - "
                   f"Memory: {stats['memory_used_gb']:.1f}/{stats['memory_total_gb']:.1f}GB "
                   f"({stats['memory_percent']:.1f}%), "
                   f"Utilization: {stats['utilization']:.1f}%, "
                   f"Temp: {stats['temperature']:.0f}°C" if stats['temperature'] else "")
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics from history"""
        if not self.history:
            return {}
        
        import numpy as np
        
        summary = {}
        for key in self.history[0].keys():
            if key and self.history[0][key] is not None:
                values = [h[key] for h in self.history if h[key] is not None]
                if values:
                    summary[f"{key}_mean"] = np.mean(values)
                    summary[f"{key}_max"] = np.max(values)
                    summary[f"{key}_min"] = np.min(values)
        
        return summary


class AutoBatchSizer:
    """Automatically find optimal batch size for GPU"""
    
    def __init__(self, model, device, max_batch_size: int = 128):
        self.model = model
        self.device = device
        self.max_batch_size = max_batch_size
        self.manager = GPUManager()
    
    def find_optimal_batch_size(self, sample_input: torch.Tensor,
                               use_mixed_precision: bool = True) -> int:
        """
        Find the largest batch size that fits in GPU memory.
        
        Args:
            sample_input: Sample input tensor (single item)
            use_mixed_precision: Whether to use mixed precision
        
        Returns:
            Optimal batch size
        """
        self.model.to(self.device)
        self.model.eval()
        
        batch_size = self.max_batch_size
        
        while batch_size > 0:
            try:
                # Clear memory
                self.manager.optimize_memory()
                
                # Create batch
                batch = sample_input.unsqueeze(0).repeat(batch_size, *[1]*sample_input.ndim)
                batch = batch.to(self.device)
                
                # Forward pass
                with torch.no_grad():
                    if use_mixed_precision and self.device.type == 'cuda':
                        with torch.cuda.amp.autocast():
                            _ = self.model(batch)
                    else:
                        _ = self.model(batch)
                
                # Backward pass test
                self.model.train()
                if use_mixed_precision and self.device.type == 'cuda':
                    scaler = torch.cuda.amp.GradScaler()
                    with torch.cuda.amp.autocast():
                        output = self.model(batch)
                        loss = output.mean()  # Dummy loss
                    scaler.scale(loss).backward()
                else:
                    output = self.model(batch)
                    loss = output.mean()
                    loss.backward()
                
                # Clear gradients
                self.model.zero_grad()
                
                logger.info(f"Optimal batch size found: {batch_size}")
                return batch_size
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    batch_size = int(batch_size * 0.8)  # Reduce by 20%
                    logger.debug(f"OOM with batch size {batch_size}, trying smaller")
                    self.manager.optimize_memory()
                else:
                    raise e
            
            finally:
                # Clean up
                if 'batch' in locals():
                    del batch
                if 'output' in locals():
                    del output
                if 'loss' in locals():
                    del loss
                self.manager.optimize_memory()
        
        logger.warning("Could not find suitable batch size, defaulting to 1")
        return 1


def profile_gpu_memory(func):
    """Decorator to profile GPU memory usage of a function"""
    def wrapper(*args, **kwargs):
        manager = GPUManager()
        
        if manager.cuda_available:
            torch.cuda.reset_peak_memory_stats()
            start_memory = torch.cuda.memory_allocated() / 1024**3
        
        result = func(*args, **kwargs)
        
        if manager.cuda_available:
            end_memory = torch.cuda.memory_allocated() / 1024**3
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            
            logger.info(f"GPU Memory Profile for {func.__name__}:")
            logger.info(f"  Start: {start_memory:.2f} GB")
            logger.info(f"  End: {end_memory:.2f} GB")
            logger.info(f"  Peak: {peak_memory:.2f} GB")
            logger.info(f"  Delta: {(end_memory - start_memory):.2f} GB")
        
        return result
    
    return wrapper


def warmup_gpu(model, input_shape: Tuple[int, ...], device: torch.device, 
              num_iterations: int = 3):
    """
    Warm up GPU with dummy forward passes.
    
    Args:
        model: The model to warm up
        input_shape: Shape of input tensor
        device: Device to use
        num_iterations: Number of warmup iterations
    """
    if device.type != 'cuda':
        return
    
    logger.info("Warming up GPU...")
    model.eval()
    
    with torch.no_grad():
        dummy_input = torch.randn(*input_shape, device=device)
        for _ in range(num_iterations):
            _ = model(dummy_input)
    
    torch.cuda.synchronize()
    logger.info("GPU warmup complete")


# Convenience functions
def get_device(device_spec: str = "auto") -> torch.device:
    """Get the appropriate device"""
    manager = GPUManager()
    return manager.get_device(device_spec)


def setup_gpu_optimizations(**kwargs):
    """Setup GPU optimizations"""
    manager = GPUManager()
    manager.setup_optimization_flags(**kwargs)


def log_gpu_info():
    """Log information about available GPUs"""
    manager = GPUManager()
    
    if not manager.cuda_available:
        logger.info("No CUDA-capable GPU detected")
        return
    
    logger.info(f"Found {manager.device_count} GPU(s):")
    for i in range(manager.device_count):
        info = manager.get_gpu_info(i)
        if info:
            logger.info(f"  GPU {i}: {info.name} "
                       f"({info.memory_total:.1f}GB, "
                       f"Compute {info.compute_capability[0]}.{info.compute_capability[1]})")