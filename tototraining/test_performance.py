#!/usr/bin/env python3
"""
Performance tests for the Toto retraining system.
Tests training efficiency, memory usage, and computational performance.
"""

import pytest
import torch
import numpy as np
import pandas as pd
import time
import gc
import psutil
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, List, Tuple, Optional
import warnings
from dataclasses import dataclass
from contextlib import contextmanager

# Import modules under test
from toto_ohlc_trainer import TotoOHLCConfig, TotoOHLCTrainer
from toto_ohlc_dataloader import DataLoaderConfig, TotoOHLCDataLoader, OHLCPreprocessor
from enhanced_trainer import EnhancedTotoTrainer

# Suppress warnings during testing
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    execution_time: float
    peak_memory_mb: float
    average_memory_mb: float
    cpu_percent: float
    gpu_memory_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None


class MemoryProfiler:
    """Memory profiling utility"""
    
    def __init__(self):
        self.start_memory = 0
        self.peak_memory = 0
        self.memory_samples = []
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self, sample_interval: float = 0.1):
        """Start memory monitoring in background thread"""
        self.start_memory = self._get_memory_usage()
        self.peak_memory = self.start_memory
        self.memory_samples = [self.start_memory]
        self.monitoring = True
        
        def monitor():
            while self.monitoring:
                memory = self._get_memory_usage()
                self.memory_samples.append(memory)
                self.peak_memory = max(self.peak_memory, memory)
                time.sleep(sample_interval)
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> PerformanceMetrics:
        """Stop monitoring and return metrics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        final_memory = self._get_memory_usage()
        
        return PerformanceMetrics(
            execution_time=0,  # Will be set by caller
            peak_memory_mb=self.peak_memory,
            average_memory_mb=np.mean(self.memory_samples) if self.memory_samples else 0,
            cpu_percent=psutil.cpu_percent(),
            gpu_memory_mb=self._get_gpu_memory() if torch.cuda.is_available() else None,
            gpu_utilization=self._get_gpu_utilization() if torch.cuda.is_available() else None
        )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    def _get_gpu_memory(self) -> Optional[float]:
        """Get GPU memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return None
    
    def _get_gpu_utilization(self) -> Optional[float]:
        """Get GPU utilization percentage"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except:
            return None


@contextmanager
def performance_monitor(sample_interval: float = 0.1):
    """Context manager for performance monitoring"""
    profiler = MemoryProfiler()
    start_time = time.time()
    
    profiler.start_monitoring(sample_interval)
    
    try:
        yield profiler
    finally:
        execution_time = time.time() - start_time
        metrics = profiler.stop_monitoring()
        metrics.execution_time = execution_time
        profiler.final_metrics = metrics


def create_performance_test_data(n_samples: int, n_symbols: int = 3) -> Dict[str, pd.DataFrame]:
    """Create test data for performance testing"""
    np.random.seed(42)
    data = {}
    
    symbols = [f'PERF_{i:03d}' for i in range(n_symbols)]
    
    for symbol in symbols:
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='15T')
        
        # Generate realistic price series
        base_price = 100 + np.random.uniform(-20, 20)
        prices = [base_price]
        
        for _ in range(n_samples - 1):
            change = np.random.normal(0, 0.01)
            new_price = max(prices[-1] * (1 + change), 1.0)
            prices.append(new_price)
        
        closes = np.array(prices)
        opens = np.concatenate([[closes[0]], closes[:-1]]) + np.random.normal(0, 0.002, n_samples)
        highs = np.maximum(np.maximum(opens, closes), 
                          np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.005, n_samples))))
        lows = np.minimum(np.minimum(opens, closes),
                         np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.005, n_samples))))
        volumes = np.random.randint(1000, 100000, n_samples)
        
        data[symbol] = pd.DataFrame({
            'timestamp': dates,
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': closes,
            'Volume': volumes
        })
    
    return data


@pytest.fixture
def performance_test_data_small():
    """Small dataset for quick performance tests"""
    return create_performance_test_data(n_samples=500, n_symbols=2)


@pytest.fixture
def performance_test_data_medium():
    """Medium dataset for comprehensive performance tests"""
    return create_performance_test_data(n_samples=2000, n_symbols=5)


@pytest.fixture
def performance_test_data_large():
    """Large dataset for stress testing"""
    return create_performance_test_data(n_samples=10000, n_symbols=10)


class TestDataLoadingPerformance:
    """Test data loading performance"""
    
    def test_small_dataset_loading_speed(self, performance_test_data_small):
        """Test loading speed for small datasets"""
        config = DataLoaderConfig(
            sequence_length=50,
            prediction_length=10,
            batch_size=16,
            normalization_method="robust",
            add_technical_indicators=True,
            min_sequence_length=60
        )
        
        with performance_monitor() as profiler:
            preprocessor = OHLCPreprocessor(config)
            preprocessor.fit_scalers(performance_test_data_small)
            
            for symbol, data in performance_test_data_small.items():
                transformed = preprocessor.transform(data, symbol)
                features = preprocessor.prepare_features(transformed)
        
        metrics = profiler.final_metrics
        
        # Performance assertions for small dataset
        assert metrics.execution_time < 5.0, f"Small dataset loading took too long: {metrics.execution_time:.2f}s"
        assert metrics.peak_memory_mb < 500, f"Small dataset used too much memory: {metrics.peak_memory_mb:.1f}MB"
    
    def test_medium_dataset_loading_speed(self, performance_test_data_medium):
        """Test loading speed for medium datasets"""
        config = DataLoaderConfig(
            sequence_length=100,
            prediction_length=20,
            batch_size=32,
            normalization_method="robust",
            add_technical_indicators=True,
            min_sequence_length=120
        )
        
        with performance_monitor() as profiler:
            preprocessor = OHLCPreprocessor(config)
            preprocessor.fit_scalers(performance_test_data_medium)
            
            # Create dataset
            from toto_ohlc_dataloader import OHLCDataset as DataLoaderOHLCDataset
            dataset = DataLoaderOHLCDataset(performance_test_data_medium, config, preprocessor, 'train')
            
            # Create dataloader
            dataloader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=config.batch_size,
                num_workers=0  # Single thread for consistent testing
            )
            
            # Process several batches
            batch_count = 0
            for batch in dataloader:
                batch_count += 1
                if batch_count >= 10:  # Process 10 batches
                    break
        
        metrics = profiler.final_metrics
        
        # Performance assertions for medium dataset
        assert metrics.execution_time < 20.0, f"Medium dataset processing took too long: {metrics.execution_time:.2f}s"
        assert metrics.peak_memory_mb < 1500, f"Medium dataset used too much memory: {metrics.peak_memory_mb:.1f}MB"
    
    @pytest.mark.slow
    def test_large_dataset_loading_stress(self, performance_test_data_large):
        """Stress test with large dataset"""
        config = DataLoaderConfig(
            sequence_length=200,
            prediction_length=50,
            batch_size=64,
            normalization_method="robust",
            add_technical_indicators=True,
            min_sequence_length=250,
            max_symbols=5  # Limit to avoid excessive memory usage
        )
        
        # Use only first 5 symbols for stress test
        limited_data = dict(list(performance_test_data_large.items())[:5])
        
        with performance_monitor() as profiler:
            preprocessor = OHLCPreprocessor(config)
            preprocessor.fit_scalers(limited_data)
            
            from toto_ohlc_dataloader import OHLCDataset as DataLoaderOHLCDataset
            dataset = DataLoaderOHLCDataset(limited_data, config, preprocessor, 'train')
            
            if len(dataset) > 0:
                dataloader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=config.batch_size,
                    num_workers=0
                )
                
                # Process limited number of batches to avoid test timeout
                batch_count = 0
                for batch in dataloader:
                    batch_count += 1
                    if batch_count >= 5:  # Process only 5 batches for stress test
                        break
        
        metrics = profiler.final_metrics
        
        # Stress test assertions - more lenient
        assert metrics.execution_time < 60.0, f"Large dataset stress test took too long: {metrics.execution_time:.2f}s"
        assert metrics.peak_memory_mb < 4000, f"Large dataset used excessive memory: {metrics.peak_memory_mb:.1f}MB"
    
    def test_memory_efficiency_batch_processing(self, performance_test_data_medium):
        """Test memory efficiency of batch processing"""
        config = DataLoaderConfig(
            sequence_length=50,
            prediction_length=10,
            batch_size=8,
            normalization_method="robust",
            add_technical_indicators=False,  # Disable for simpler memory profile
            min_sequence_length=60
        )
        
        preprocessor = OHLCPreprocessor(config)
        preprocessor.fit_scalers(performance_test_data_medium)
        
        from toto_ohlc_dataloader import OHLCDataset as DataLoaderOHLCDataset
        dataset = DataLoaderOHLCDataset(performance_test_data_medium, config, preprocessor, 'train')
        
        if len(dataset) > 0:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, num_workers=0)
            
            # Measure memory usage across multiple batches
            memory_measurements = []
            
            for i, batch in enumerate(dataloader):
                if i >= 10:  # Test 10 batches
                    break
                
                # Force garbage collection and measure memory
                gc.collect()
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                memory_measurements.append(memory_mb)
                
                # Process batch to simulate actual usage
                _ = batch.series.mean()
            
            # Memory should remain relatively stable across batches
            memory_std = np.std(memory_measurements)
            memory_growth = memory_measurements[-1] - memory_measurements[0] if len(memory_measurements) > 1 else 0
            
            # Memory should not grow excessively between batches
            assert memory_growth < 100, f"Excessive memory growth: {memory_growth:.1f}MB"
            assert memory_std < 50, f"Unstable memory usage: {memory_std:.1f}MB std"


class TestTrainingPerformance:
    """Test training performance characteristics"""
    
    @pytest.fixture
    def minimal_trainer_config(self):
        """Create minimal configuration for performance testing"""
        return TotoOHLCConfig(
            patch_size=4,
            stride=2,
            embed_dim=32,  # Small for faster testing
            num_layers=2,
            num_heads=4,
            mlp_hidden_dim=64,
            dropout=0.1,
            sequence_length=20,
            prediction_length=5,
            validation_days=5
        )
    
    @patch('toto_ohlc_trainer.Toto')
    def test_model_initialization_speed(self, mock_toto, minimal_trainer_config):
        """Test model initialization performance"""
        # Mock Toto model
        mock_model = Mock()
        mock_model.parameters.return_value = [torch.randn(100, requires_grad=True)]
        mock_toto.return_value = mock_model
        
        with performance_monitor() as profiler:
            trainer = TotoOHLCTrainer(minimal_trainer_config)
            trainer.initialize_model(input_dim=5)
        
        metrics = profiler.final_metrics
        
        # Model initialization should be fast
        assert metrics.execution_time < 2.0, f"Model initialization too slow: {metrics.execution_time:.2f}s"
        assert metrics.peak_memory_mb < 200, f"Model initialization used too much memory: {metrics.peak_memory_mb:.1f}MB"
    
    @patch('toto_ohlc_trainer.Toto')
    def test_forward_pass_performance(self, mock_toto, minimal_trainer_config):
        """Test forward pass performance"""
        # Create mock model with predictable output
        mock_model = Mock()
        mock_model.parameters.return_value = [torch.randn(100, requires_grad=True)]
        mock_model.model = Mock()
        
        # Mock output
        batch_size = 8
        mock_output = Mock()
        mock_output.loc = torch.randn(batch_size, minimal_trainer_config.prediction_length)
        mock_model.model.return_value = mock_output
        
        mock_toto.return_value = mock_model
        
        trainer = TotoOHLCTrainer(minimal_trainer_config)
        trainer.initialize_model(input_dim=5)
        
        # Create test batch
        seq_len = minimal_trainer_config.sequence_length
        x = torch.randn(batch_size, seq_len, 5)
        y = torch.randn(batch_size, minimal_trainer_config.prediction_length)
        
        with performance_monitor() as profiler:
            # Simulate multiple forward passes
            for _ in range(10):
                # Simulate forward pass logic from trainer
                x_reshaped = x.transpose(1, 2).contiguous()
                input_padding_mask = torch.zeros(batch_size, 1, seq_len, dtype=torch.bool)
                id_mask = torch.ones(batch_size, 1, seq_len, dtype=torch.float32)
                
                output = trainer.model.model(x_reshaped, input_padding_mask, id_mask)
                predictions = output.loc
                loss = torch.nn.functional.mse_loss(predictions, y)
        
        metrics = profiler.final_metrics
        
        # Forward passes should be efficient
        assert metrics.execution_time < 1.0, f"Forward passes too slow: {metrics.execution_time:.2f}s"
    
    @patch('toto_ohlc_trainer.Toto')
    def test_training_epoch_performance(self, mock_toto, minimal_trainer_config, performance_test_data_small):
        """Test training epoch performance"""
        # Mock model setup
        mock_model = Mock()
        mock_model.parameters.return_value = [torch.randn(100, requires_grad=True)]
        mock_model.train = Mock()
        mock_model.model = Mock()
        
        batch_size = 4
        mock_output = Mock()
        mock_output.loc = torch.randn(batch_size, minimal_trainer_config.prediction_length)
        mock_model.model.return_value = mock_output
        
        mock_toto.return_value = mock_model
        
        trainer = TotoOHLCTrainer(minimal_trainer_config)
        trainer.initialize_model(input_dim=5)
        
        # Create mock dataloader
        mock_batches = []
        for _ in range(5):  # 5 batches
            x = torch.randn(batch_size, minimal_trainer_config.sequence_length, 5)
            y = torch.randn(batch_size, minimal_trainer_config.prediction_length)
            mock_batches.append((x, y))
        
        with performance_monitor() as profiler:
            # Mock training epoch
            trainer.model.train()
            total_loss = 0.0
            
            for batch_idx, (x, y) in enumerate(mock_batches):
                trainer.optimizer.zero_grad()
                
                # Forward pass
                batch_size, seq_len, features = x.shape
                x_reshaped = x.transpose(1, 2).contiguous()
                input_padding_mask = torch.zeros(batch_size, 1, seq_len, dtype=torch.bool)
                id_mask = torch.ones(batch_size, 1, seq_len, dtype=torch.float32)
                
                output = trainer.model.model(x_reshaped, input_padding_mask, id_mask)
                predictions = output.loc
                loss = torch.nn.functional.mse_loss(predictions, y)
                
                # Backward pass (simulated)
                total_loss += loss.item()
                trainer.optimizer.step()
        
        metrics = profiler.final_metrics
        
        # Training epoch should complete within reasonable time
        assert metrics.execution_time < 5.0, f"Training epoch too slow: {metrics.execution_time:.2f}s"
        assert total_loss >= 0, "Loss should be non-negative"


class TestScalabilityCharacteristics:
    """Test scalability with different data sizes"""
    
    def test_linear_scaling_batch_size(self):
        """Test that processing time scales approximately linearly with batch size"""
        config = DataLoaderConfig(
            sequence_length=30,
            prediction_length=5,
            normalization_method="robust",
            add_technical_indicators=False,
            min_sequence_length=35
        )
        
        # Test data
        test_data = create_performance_test_data(n_samples=200, n_symbols=2)
        preprocessor = OHLCPreprocessor(config)
        preprocessor.fit_scalers(test_data)
        
        from toto_ohlc_dataloader import OHLCDataset as DataLoaderOHLCDataset
        dataset = DataLoaderOHLCDataset(test_data, config, preprocessor, 'train')
        
        if len(dataset) == 0:
            pytest.skip("Insufficient data for scalability test")
        
        batch_sizes = [4, 8, 16, 32]
        processing_times = []
        
        for batch_size in batch_sizes:
            dataloader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=batch_size,
                num_workers=0,
                drop_last=True
            )
            
            start_time = time.time()
            
            # Process fixed number of samples
            samples_processed = 0
            target_samples = 64  # Process same number of samples each time
            
            for batch in dataloader:
                samples_processed += batch.series.shape[0]
                
                # Simulate processing
                _ = batch.series.mean()
                
                if samples_processed >= target_samples:
                    break
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
        
        # Processing time should not grow excessively with batch size
        # (some growth expected due to batch processing overhead)
        time_ratio = processing_times[-1] / processing_times[0] if processing_times[0] > 0 else 1
        assert time_ratio < 3.0, f"Processing time grew too much with batch size: {time_ratio:.2f}x"
    
    def test_memory_scaling_sequence_length(self):
        """Test memory usage scaling with sequence length"""
        base_config = DataLoaderConfig(
            prediction_length=5,
            batch_size=8,
            normalization_method="robust",
            add_technical_indicators=False,
            min_sequence_length=20
        )
        
        test_data = create_performance_test_data(n_samples=500, n_symbols=2)
        
        sequence_lengths = [20, 40, 80]
        memory_usages = []
        
        for seq_len in sequence_lengths:
            config = base_config
            config.sequence_length = seq_len
            config.min_sequence_length = seq_len + 5
            
            # Force garbage collection before test
            gc.collect()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            preprocessor = OHLCPreprocessor(config)
            preprocessor.fit_scalers(test_data)
            
            from toto_ohlc_dataloader import OHLCDataset as DataLoaderOHLCDataset
            dataset = DataLoaderOHLCDataset(test_data, config, preprocessor, 'train')
            
            if len(dataset) > 0:
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size)
                
                # Process a few batches
                for i, batch in enumerate(dataloader):
                    _ = batch.series.sum()  # Force tensor computation
                    if i >= 3:  # Process 3 batches
                        break
            
            peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = peak_memory - start_memory
            memory_usages.append(memory_usage)
            
            # Clean up
            del dataset, dataloader, preprocessor
            gc.collect()
        
        # Memory should scale reasonably with sequence length
        # Expect roughly quadratic growth due to attention mechanism
        if len(memory_usages) >= 2:
            memory_growth_ratio = memory_usages[-1] / memory_usages[0] if memory_usages[0] > 0 else 1
            seq_growth_ratio = sequence_lengths[-1] / sequence_lengths[0]
            
            # Memory growth should not be worse than cubic scaling
            assert memory_growth_ratio < seq_growth_ratio ** 3, f"Memory scaling too poor: {memory_growth_ratio:.2f}x for {seq_growth_ratio:.2f}x sequence length"


class TestResourceUtilization:
    """Test system resource utilization"""
    
    def test_cpu_utilization_during_processing(self, performance_test_data_medium):
        """Test CPU utilization during data processing"""
        config = DataLoaderConfig(
            sequence_length=50,
            prediction_length=10,
            batch_size=16,
            normalization_method="robust",
            add_technical_indicators=True,
            min_sequence_length=60,
            num_workers=0  # Single threaded for predictable CPU usage
        )
        
        cpu_before = psutil.cpu_percent(interval=1)
        
        with performance_monitor(sample_interval=0.5) as profiler:
            preprocessor = OHLCPreprocessor(config)
            preprocessor.fit_scalers(performance_test_data_medium)
            
            from toto_ohlc_dataloader import OHLCDataset as DataLoaderOHLCDataset
            dataset = DataLoaderOHLCDataset(performance_test_data_medium, config, preprocessor, 'train')
            
            if len(dataset) > 0:
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size)
                
                # Process batches to generate CPU load
                for i, batch in enumerate(dataloader):
                    # Simulate CPU-intensive operations
                    _ = batch.series.std(dim=-1)
                    _ = batch.series.mean(dim=-1)
                    
                    if i >= 10:  # Process 10 batches
                        break
        
        metrics = profiler.final_metrics
        
        # Should utilize CPU but not excessively
        assert metrics.cpu_percent < 90, f"Excessive CPU usage: {metrics.cpu_percent:.1f}%"
        assert metrics.cpu_percent > cpu_before, "Should show increased CPU usage during processing"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_utilization(self):
        """Test GPU memory utilization if available"""
        device = torch.device('cuda')
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        # Create tensors on GPU
        large_tensors = []
        for _ in range(5):
            tensor = torch.randn(1000, 1000, device=device)
            large_tensors.append(tensor)
        
        peak_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        memory_used = peak_memory - initial_memory
        
        # Clean up
        del large_tensors
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        # Should have used GPU memory and cleaned up
        assert memory_used > 10, f"Should have used significant GPU memory: {memory_used:.1f}MB"
        assert abs(final_memory - initial_memory) < 5, f"Memory leak detected: {final_memory - initial_memory:.1f}MB difference"
    
    def test_memory_leak_detection(self, performance_test_data_small):
        """Test for memory leaks in repeated operations"""
        config = DataLoaderConfig(
            sequence_length=20,
            prediction_length=5,
            batch_size=4,
            normalization_method="robust",
            add_technical_indicators=False,
            min_sequence_length=25
        )
        
        memory_measurements = []
        
        # Perform repeated operations
        for iteration in range(5):
            gc.collect()  # Force garbage collection
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Create and destroy objects
            preprocessor = OHLCPreprocessor(config)
            preprocessor.fit_scalers(performance_test_data_small)
            
            from toto_ohlc_dataloader import OHLCDataset as DataLoaderOHLCDataset
            dataset = DataLoaderOHLCDataset(performance_test_data_small, config, preprocessor, 'train')
            
            if len(dataset) > 0:
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size)
                
                # Process one batch
                for batch in dataloader:
                    _ = batch.series.mean()
                    break
            
            # Clean up
            del dataset, dataloader, preprocessor
            
            gc.collect()
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024
            memory_measurements.append(memory_after)
        
        # Memory should not grow significantly across iterations
        if len(memory_measurements) >= 2:
            memory_growth = memory_measurements[-1] - memory_measurements[0]
            assert memory_growth < 50, f"Potential memory leak detected: {memory_growth:.1f}MB growth"


class TestPerformanceBenchmarks:
    """Benchmark tests for performance comparison"""
    
    def test_data_loading_benchmark(self, performance_test_data_medium):
        """Benchmark data loading performance"""
        config = DataLoaderConfig(
            sequence_length=100,
            prediction_length=20,
            batch_size=32,
            normalization_method="robust",
            add_technical_indicators=True,
            min_sequence_length=120
        )
        
        # Benchmark different aspects
        benchmarks = {}
        
        # 1. Preprocessor fitting
        start_time = time.time()
        preprocessor = OHLCPreprocessor(config)
        preprocessor.fit_scalers(performance_test_data_medium)
        benchmarks['preprocessor_fit'] = time.time() - start_time
        
        # 2. Data transformation
        start_time = time.time()
        transformed_data = {}
        for symbol, data in performance_test_data_medium.items():
            transformed_data[symbol] = preprocessor.transform(data, symbol)
        benchmarks['data_transformation'] = time.time() - start_time
        
        # 3. Dataset creation
        start_time = time.time()
        from toto_ohlc_dataloader import OHLCDataset as DataLoaderOHLCDataset
        dataset = DataLoaderOHLCDataset(performance_test_data_medium, config, preprocessor, 'train')
        benchmarks['dataset_creation'] = time.time() - start_time
        
        # 4. DataLoader iteration
        if len(dataset) > 0:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size)
            
            start_time = time.time()
            batch_count = 0
            for batch in dataloader:
                batch_count += 1
                if batch_count >= 10:
                    break
            benchmarks['dataloader_iteration'] = time.time() - start_time
        
        # Print benchmarks for reference
        print("\nData Loading Benchmarks:")
        for operation, duration in benchmarks.items():
            print(f"  {operation}: {duration:.3f}s")
        
        # Benchmark assertions (these are guidelines, not strict requirements)
        assert benchmarks['preprocessor_fit'] < 10.0, "Preprocessor fitting too slow"
        assert benchmarks['data_transformation'] < 15.0, "Data transformation too slow"
        assert benchmarks['dataset_creation'] < 5.0, "Dataset creation too slow"
        
        if 'dataloader_iteration' in benchmarks:
            assert benchmarks['dataloader_iteration'] < 10.0, "DataLoader iteration too slow"


if __name__ == "__main__":
    # Run performance tests with appropriate markers
    pytest.main([
        __file__, 
        "-v", 
        "--tb=short",
        "-m", "not slow",  # Skip slow tests by default
        "--disable-warnings"
    ])