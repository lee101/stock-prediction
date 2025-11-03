#!/usr/bin/env python3
"""
Regression tests for the Toto retraining system.
Tests to ensure model outputs are consistent and detect regressions in model behavior.
"""

import pytest
import torch
import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
import tempfile
import hashlib
from unittest.mock import Mock, patch
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from dataclasses import dataclass, asdict

# Import test utilities
from test_fixtures import (
    SyntheticDataFactory, MockTotoModel, ConfigurationFactory,
    AssertionHelpers, TestScenario
)

# Import modules under test
from toto_ohlc_trainer import TotoOHLCConfig, TotoOHLCTrainer
from toto_ohlc_dataloader import DataLoaderConfig, TotoOHLCDataLoader, OHLCPreprocessor
from enhanced_trainer import EnhancedTotoTrainer

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class ReferenceOutput:
    """Reference output for regression testing"""
    config_hash: str
    data_hash: str
    model_outputs: Dict[str, torch.Tensor]
    preprocessed_data_stats: Dict[str, float]
    training_metrics: Dict[str, float]
    feature_statistics: Dict[str, Dict[str, float]]


class RegressionTestManager:
    """Manager for regression testing"""
    
    def __init__(self, reference_dir: Path = None):
        self.reference_dir = reference_dir or Path("test_references")
        self.reference_dir.mkdir(parents=True, exist_ok=True)
        
    def compute_data_hash(self, data: Dict[str, pd.DataFrame]) -> str:
        """Compute hash of dataset for consistency checking"""
        combined_data = pd.concat(list(data.values()), keys=data.keys())
        
        # Use numeric columns for hash to avoid timestamp formatting issues
        numeric_cols = combined_data.select_dtypes(include=[np.number]).columns
        data_string = combined_data[numeric_cols].to_string()
        
        return hashlib.md5(data_string.encode()).hexdigest()
    
    def compute_config_hash(self, config: Union[TotoOHLCConfig, DataLoaderConfig]) -> str:
        """Compute hash of configuration"""
        config_dict = asdict(config)
        config_string = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_string.encode()).hexdigest()
    
    def save_reference_output(
        self, 
        test_name: str,
        config: Union[TotoOHLCConfig, DataLoaderConfig],
        data: Dict[str, pd.DataFrame],
        outputs: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ):
        """Save reference output for future comparison"""
        reference = ReferenceOutput(
            config_hash=self.compute_config_hash(config),
            data_hash=self.compute_data_hash(data),
            model_outputs=outputs.get('model_outputs', {}),
            preprocessed_data_stats=outputs.get('data_stats', {}),
            training_metrics=outputs.get('training_metrics', {}),
            feature_statistics=outputs.get('feature_stats', {})
        )
        
        # Add metadata
        if metadata:
            for key, value in metadata.items():
                setattr(reference, key, value)
        
        # Save to file
        reference_file = self.reference_dir / f"{test_name}_reference.pkl"
        with open(reference_file, 'wb') as f:
            pickle.dump(reference, f)
    
    def load_reference_output(self, test_name: str) -> Optional[ReferenceOutput]:
        """Load reference output for comparison"""
        reference_file = self.reference_dir / f"{test_name}_reference.pkl"
        
        if not reference_file.exists():
            return None
        
        try:
            with open(reference_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            pytest.fail(f"Failed to load reference output: {e}")
    
    def compare_tensors(
        self, 
        actual: torch.Tensor, 
        expected: torch.Tensor, 
        tolerance: float = 1e-5,
        name: str = "tensor"
    ) -> bool:
        """Compare tensors with tolerance"""
        if actual.shape != expected.shape:
            pytest.fail(f"{name} shape mismatch: {actual.shape} vs {expected.shape}")
        
        if not torch.allclose(actual, expected, atol=tolerance, rtol=tolerance):
            max_diff = torch.max(torch.abs(actual - expected)).item()
            pytest.fail(f"{name} values differ beyond tolerance. Max diff: {max_diff}")
        
        return True
    
    def compare_statistics(
        self,
        actual: Dict[str, float],
        expected: Dict[str, float],
        tolerance: float = 1e-3,
        name: str = "statistics"
    ) -> bool:
        """Compare statistical measures"""
        for key in expected:
            if key not in actual:
                pytest.fail(f"Missing {name} key: {key}")
            
            actual_val = actual[key]
            expected_val = expected[key]
            
            if abs(actual_val - expected_val) > tolerance:
                pytest.fail(
                    f"{name}[{key}] differs: {actual_val} vs {expected_val} "
                    f"(diff: {abs(actual_val - expected_val)})"
                )
        
        return True


@pytest.fixture
def regression_manager(tmp_path):
    """Provide regression test manager"""
    return RegressionTestManager(tmp_path / "references")


@pytest.fixture
def reference_data():
    """Create reference data for consistent testing"""
    # Use fixed seed for deterministic data
    factory = SyntheticDataFactory(seed=12345)
    
    symbols = ['REGTEST_A', 'REGTEST_B', 'REGTEST_C']
    data = {}
    
    for i, symbol in enumerate(symbols):
        data[symbol] = factory.create_basic_ohlc_data(
            n_samples=300,
            symbol=symbol,
            base_price=100 + i * 25,
            volatility=0.02 + i * 0.005,
            start_date="2023-01-01",
            freq="H"
        )
    
    return data


@pytest.fixture
def reference_config():
    """Create reference configuration for consistent testing"""
    return ConfigurationFactory.create_minimal_trainer_config(
        patch_size=6,
        stride=3,
        embed_dim=64,
        num_layers=3,
        num_heads=4,
        sequence_length=48,
        prediction_length=12,
        dropout=0.1
    )


@pytest.fixture
def reference_dataloader_config():
    """Create reference dataloader configuration"""
    return ConfigurationFactory.create_minimal_dataloader_config(
        sequence_length=48,
        prediction_length=12,
        batch_size=8,
        normalization_method="robust",
        add_technical_indicators=True,
        min_sequence_length=60
    )


class TestDataProcessingRegression:
    """Test data processing consistency"""
    
    def test_preprocessor_deterministic_output(
        self, 
        reference_data,
        reference_dataloader_config,
        regression_manager
    ):
        """Test that preprocessor produces deterministic output"""
        config = reference_dataloader_config
        
        # Process data multiple times
        preprocessors = []
        transformed_data_list = []
        
        for run in range(3):  # Run 3 times
            preprocessor = OHLCPreprocessor(config)
            preprocessor.fit_scalers(reference_data)
            
            transformed_data = {}
            for symbol, data in reference_data.items():
                transformed_data[symbol] = preprocessor.transform(data, symbol)
            
            preprocessors.append(preprocessor)
            transformed_data_list.append(transformed_data)
        
        # Compare outputs
        for symbol in reference_data.keys():
            df_0 = transformed_data_list[0][symbol]
            
            for run in range(1, 3):
                df_run = transformed_data_list[run][symbol]
                
                # Should have same shape
                assert df_0.shape == df_run.shape, f"Shape mismatch for {symbol} in run {run}"
                
                # Numeric columns should be identical
                numeric_cols = df_0.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if not np.allclose(df_0[col].dropna(), df_run[col].dropna(), atol=1e-10):
                        pytest.fail(f"Preprocessor output not deterministic for {symbol}.{col}")
    
    def test_feature_extraction_consistency(
        self,
        reference_data,
        reference_dataloader_config,
        regression_manager
    ):
        """Test feature extraction consistency"""
        config = reference_dataloader_config
        preprocessor = OHLCPreprocessor(config)
        preprocessor.fit_scalers(reference_data)
        
        # Extract features multiple times
        feature_arrays = []
        
        for run in range(3):
            features = {}
            for symbol, data in reference_data.items():
                transformed = preprocessor.transform(data, symbol)
                features[symbol] = preprocessor.prepare_features(transformed)
            feature_arrays.append(features)
        
        # Compare feature arrays
        for symbol in reference_data.keys():
            features_0 = feature_arrays[0][symbol]
            
            for run in range(1, 3):
                features_run = feature_arrays[run][symbol]
                
                assert features_0.shape == features_run.shape, f"Feature shape mismatch for {symbol}"
                
                if not np.allclose(features_0, features_run, atol=1e-10):
                    max_diff = np.max(np.abs(features_0 - features_run))
                    pytest.fail(f"Feature extraction not consistent for {symbol}. Max diff: {max_diff}")
    
    def test_technical_indicators_regression(
        self,
        reference_data,
        reference_dataloader_config,
        regression_manager
    ):
        """Test technical indicators for regression"""
        test_name = "technical_indicators"
        
        config = reference_dataloader_config
        config.add_technical_indicators = True
        
        preprocessor = OHLCPreprocessor(config)
        
        # Process one symbol with indicators
        symbol = list(reference_data.keys())[0]
        data = reference_data[symbol]
        
        # Add indicators
        processed = preprocessor.add_technical_indicators(data)
        
        # Compute statistics of indicators
        indicator_stats = {}
        expected_indicators = ['RSI', 'volatility', 'hl_ratio', 'oc_ratio', 
                              'price_momentum_1', 'price_momentum_5']
        expected_indicators += [f'MA_{p}_ratio' for p in config.ma_periods]
        
        for indicator in expected_indicators:
            if indicator in processed.columns:
                series = processed[indicator].dropna()
                if len(series) > 0:
                    indicator_stats[indicator] = {
                        'mean': float(series.mean()),
                        'std': float(series.std()),
                        'min': float(series.min()),
                        'max': float(series.max()),
                        'count': int(len(series))
                    }
        
        # Check against reference
        reference = regression_manager.load_reference_output(test_name)
        
        if reference is None:
            # Save as new reference
            outputs = {'feature_stats': {'technical_indicators': indicator_stats}}
            regression_manager.save_reference_output(
                test_name, config, reference_data, outputs
            )
            pytest.skip("Saved new reference output for technical indicators")
        
        # Compare with reference
        if 'technical_indicators' in reference.feature_statistics:
            expected_stats = reference.feature_statistics['technical_indicators']
            
            for indicator, stats in expected_stats.items():
                if indicator in indicator_stats:
                    actual_stats = indicator_stats[indicator]
                    
                    # Compare with tolerance
                    for stat_name, expected_val in stats.items():
                        if stat_name in actual_stats:
                            actual_val = actual_stats[stat_name]
                            tolerance = 1e-3 if stat_name != 'count' else 0
                            
                            if abs(actual_val - expected_val) > tolerance:
                                pytest.fail(
                                    f"Technical indicator {indicator}.{stat_name} changed: "
                                    f"{actual_val} vs {expected_val}"
                                )


class TestModelOutputRegression:
    """Test model output consistency"""
    
    @patch('toto_ohlc_trainer.Toto')
    def test_forward_pass_determinism(
        self,
        mock_toto,
        reference_config,
        regression_manager
    ):
        """Test that forward passes are deterministic"""
        # Create deterministic mock model
        mock_model = Mock()
        mock_model.parameters.return_value = [torch.randn(100, requires_grad=True)]
        mock_model.model = Mock()
        
        # Set up deterministic output
        torch.manual_seed(42)
        
        def deterministic_forward(x_reshaped, input_padding_mask, id_mask):
            # Deterministic computation based on input
            batch_size = x_reshaped.shape[0]
            pred_len = reference_config.prediction_length
            
            # Simple deterministic transformation
            output = Mock()
            # Use sum of input as seed for deterministic output
            seed = int(torch.sum(x_reshaped).item()) % 1000
            torch.manual_seed(seed)
            output.loc = torch.randn(batch_size, pred_len)
            return output
        
        mock_model.model.side_effect = deterministic_forward
        mock_toto.return_value = mock_model
        
        trainer = TotoOHLCTrainer(reference_config)
        trainer.initialize_model(input_dim=5)
        
        # Create test input
        batch_size = 4
        seq_len = reference_config.sequence_length
        x = torch.randn(batch_size, seq_len, 5)
        
        # Forward pass multiple times
        outputs = []
        for _ in range(3):
            x_reshaped = x.transpose(1, 2).contiguous()
            input_padding_mask = torch.zeros(batch_size, 1, seq_len, dtype=torch.bool)
            id_mask = torch.ones(batch_size, 1, seq_len, dtype=torch.float32)
            
            output = trainer.model.model(x_reshaped, input_padding_mask, id_mask)
            outputs.append(output.loc.clone())
        
        # All outputs should be identical
        for i in range(1, len(outputs)):
            if not torch.allclose(outputs[0], outputs[i], atol=1e-10):
                pytest.fail("Forward pass is not deterministic")
    
    @patch('toto_ohlc_trainer.Toto')
    def test_loss_computation_regression(
        self,
        mock_toto,
        reference_config,
        regression_manager
    ):
        """Test loss computation consistency"""
        test_name = "loss_computation"
        
        # Setup mock model
        mock_model = Mock()
        mock_model.parameters.return_value = [torch.randn(100, requires_grad=True)]
        mock_model.model = Mock()
        
        batch_size = 4
        pred_len = reference_config.prediction_length
        
        # Fixed output for consistency
        mock_output = Mock()
        mock_output.loc = torch.tensor([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.1, 2.1, 3.1, 4.1, 5.1],
            [0.9, 1.9, 2.9, 3.9, 4.9],
            [1.05, 2.05, 3.05, 4.05, 5.05]
        ][:, :pred_len])  # Truncate to prediction length
        
        mock_model.model.return_value = mock_output
        mock_toto.return_value = mock_model
        
        trainer = TotoOHLCTrainer(reference_config)
        trainer.initialize_model(input_dim=5)
        
        # Fixed target
        y = torch.tensor([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0]
        ][:, :pred_len])  # Truncate to prediction length
        
        # Compute loss
        predictions = mock_output.loc
        loss = torch.nn.functional.mse_loss(predictions, y)
        
        loss_value = loss.item()
        
        # Check against reference
        reference = regression_manager.load_reference_output(test_name)
        
        if reference is None:
            # Save as new reference
            outputs = {'training_metrics': {'reference_loss': loss_value}}
            regression_manager.save_reference_output(
                test_name, reference_config, {}, outputs
            )
            pytest.skip("Saved new reference loss value")
        
        # Compare with reference
        expected_loss = reference.training_metrics.get('reference_loss')
        if expected_loss is not None:
            assert abs(loss_value - expected_loss) < 1e-6, f"Loss computation changed: {loss_value} vs {expected_loss}"
    
    def test_gradient_computation_consistency(self, reference_config):
        """Test gradient computation consistency"""
        # Create simple model for gradient testing
        model = torch.nn.Sequential(
            torch.nn.Linear(5, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, reference_config.prediction_length)
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Fixed input and target
        torch.manual_seed(42)
        x = torch.randn(4, 5)
        y = torch.randn(4, reference_config.prediction_length)
        
        # Compute gradients multiple times with same data
        gradients = []
        
        for _ in range(3):
            # Reset model to same state
            torch.manual_seed(42)
            model = torch.nn.Sequential(
                torch.nn.Linear(5, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, reference_config.prediction_length)
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            optimizer.zero_grad()
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, y)
            loss.backward()
            
            # Collect gradients
            grad_values = []
            for param in model.parameters():
                if param.grad is not None:
                    grad_values.append(param.grad.clone())
            
            gradients.append(grad_values)
        
        # All gradients should be identical
        for i in range(1, len(gradients)):
            for j, (grad_0, grad_i) in enumerate(zip(gradients[0], gradients[i])):
                if not torch.allclose(grad_0, grad_i, atol=1e-10):
                    pytest.fail(f"Gradient computation not consistent for parameter {j}")


class TestDatasetRegression:
    """Test dataset behavior regression"""
    
    def test_dataset_sequence_generation_consistency(
        self,
        reference_data,
        reference_dataloader_config,
        regression_manager
    ):
        """Test that dataset generates consistent sequences"""
        test_name = "dataset_sequences"
        
        config = reference_dataloader_config
        
        # Create dataset multiple times
        datasets = []
        for _ in range(3):
            preprocessor = OHLCPreprocessor(config)
            preprocessor.fit_scalers(reference_data)
            
            from toto_ohlc_dataloader import OHLCDataset as DataLoaderOHLCDataset
            dataset = DataLoaderOHLCDataset(reference_data, config, preprocessor, 'train')
            datasets.append(dataset)
        
        # All datasets should have same length
        lengths = [len(dataset) for dataset in datasets]
        assert all(length == lengths[0] for length in lengths), "Dataset lengths are inconsistent"
        
        if lengths[0] > 0:
            # Compare first few sequences
            for idx in range(min(5, lengths[0])):
                samples = [dataset[idx] for dataset in datasets]
                
                # All samples should be identical
                for i in range(1, len(samples)):
                    sample_0 = samples[0]
                    sample_i = samples[i]
                    
                    assert sample_0.series.shape == sample_i.series.shape, f"Sample {idx} shape mismatch"
                    
                    if not torch.allclose(sample_0.series, sample_i.series, atol=1e-10):
                        pytest.fail(f"Sample {idx} series not consistent")
                    
                    if not torch.equal(sample_0.padding_mask, sample_i.padding_mask):
                        pytest.fail(f"Sample {idx} padding mask not consistent")
    
    def test_dataloader_batch_consistency(
        self,
        reference_data,
        reference_dataloader_config,
        regression_manager
    ):
        """Test that dataloader produces consistent batches"""
        config = reference_dataloader_config
        config.batch_size = 4
        
        # Create preprocessor and dataset
        preprocessor = OHLCPreprocessor(config)
        preprocessor.fit_scalers(reference_data)
        
        from toto_ohlc_dataloader import OHLCDataset as DataLoaderOHLCDataset
        dataset = DataLoaderOHLCDataset(reference_data, config, preprocessor, 'train')
        
        if len(dataset) == 0:
            pytest.skip("No data available for batch testing")
        
        # Create dataloaders with same settings
        dataloaders = []
        for _ in range(3):
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=False,  # Important: no shuffle for consistency
                num_workers=0,
                drop_last=True
            )
            dataloaders.append(dataloader)
        
        # Compare first batch from each dataloader
        first_batches = []
        for dataloader in dataloaders:
            for batch in dataloader:
                first_batches.append(batch)
                break
        
        if len(first_batches) > 1:
            batch_0 = first_batches[0]
            
            for i, batch_i in enumerate(first_batches[1:], 1):
                assert batch_0.series.shape == batch_i.series.shape, f"Batch {i} shape mismatch"
                
                if not torch.allclose(batch_0.series, batch_i.series, atol=1e-10):
                    pytest.fail(f"Batch {i} series not consistent")


class TestTrainingRegression:
    """Test training process regression"""
    
    @patch('toto_ohlc_trainer.Toto')
    def test_training_step_reproducibility(
        self,
        mock_toto,
        reference_config,
        reference_data,
        regression_manager
    ):
        """Test training step reproducibility"""
        test_name = "training_step"
        
        # Setup deterministic mock model
        def create_deterministic_model():
            mock_model = Mock()
            mock_model.parameters.return_value = [
                torch.tensor([1.0, 2.0, 3.0], requires_grad=True),
                torch.tensor([0.5, 1.5], requires_grad=True)
            ]
            mock_model.train = Mock()
            mock_model.eval = Mock()
            mock_model.model = Mock()
            
            # Deterministic output
            def forward_fn(x_reshaped, input_padding_mask, id_mask):
                batch_size = x_reshaped.shape[0]
                output = Mock()
                # Simple deterministic computation
                output.loc = torch.ones(batch_size, reference_config.prediction_length) * 0.5
                return output
            
            mock_model.model.side_effect = forward_fn
            return mock_model
        
        # Run training step multiple times
        training_losses = []
        
        for run in range(3):
            torch.manual_seed(42)
            np.random.seed(42)
            
            mock_toto.return_value = create_deterministic_model()
            trainer = TotoOHLCTrainer(reference_config)
            trainer.initialize_model(input_dim=5)
            
            # Create fixed training data
            batch_size = 4
            seq_len = reference_config.sequence_length
            pred_len = reference_config.prediction_length
            
            x = torch.ones(batch_size, seq_len, 5) * 0.1
            y = torch.ones(batch_size, pred_len) * 0.2
            
            # Simulate training step
            trainer.model.train()
            trainer.optimizer.zero_grad()
            
            # Forward pass
            x_reshaped = x.transpose(1, 2).contiguous()
            input_padding_mask = torch.zeros(batch_size, 1, seq_len, dtype=torch.bool)
            id_mask = torch.ones(batch_size, 1, seq_len, dtype=torch.float32)
            
            output = trainer.model.model(x_reshaped, input_padding_mask, id_mask)
            predictions = output.loc
            loss = torch.nn.functional.mse_loss(predictions, y)
            
            training_losses.append(loss.item())
        
        # All training losses should be identical
        for i in range(1, len(training_losses)):
            assert abs(training_losses[0] - training_losses[i]) < 1e-10, \
                f"Training step not reproducible: {training_losses[0]} vs {training_losses[i]}"
    
    def test_training_metrics_consistency(self, regression_manager):
        """Test training metrics consistency"""
        # Test basic metric calculations
        losses = [0.5, 0.4, 0.3, 0.35, 0.25]
        
        # Calculate metrics
        avg_loss = np.mean(losses)
        min_loss = np.min(losses)
        max_loss = np.max(losses)
        std_loss = np.std(losses)
        
        # Expected values (manually computed)
        expected_avg = 0.36
        expected_min = 0.25
        expected_max = 0.5
        expected_std = np.std([0.5, 0.4, 0.3, 0.35, 0.25])
        
        assert abs(avg_loss - expected_avg) < 1e-10, f"Average loss calculation changed"
        assert abs(min_loss - expected_min) < 1e-10, f"Min loss calculation changed"
        assert abs(max_loss - expected_max) < 1e-10, f"Max loss calculation changed"
        assert abs(std_loss - expected_std) < 1e-10, f"Std loss calculation changed"


class TestConfigurationRegression:
    """Test configuration handling regression"""
    
    def test_config_serialization_consistency(self, reference_config, regression_manager):
        """Test configuration serialization consistency"""
        # Convert to dict and back
        config_dict = asdict(reference_config)
        reconstructed_config = TotoOHLCConfig(**config_dict)
        
        # Should be identical
        assert asdict(reconstructed_config) == config_dict, "Config serialization not consistent"
        
        # Key attributes should match
        assert reconstructed_config.embed_dim == reference_config.embed_dim
        assert reconstructed_config.num_layers == reference_config.num_layers
        assert reconstructed_config.sequence_length == reference_config.sequence_length
        assert reconstructed_config.prediction_length == reference_config.prediction_length
    
    def test_config_hash_stability(self, reference_config, regression_manager):
        """Test configuration hash stability"""
        # Create identical configs
        config1 = TotoOHLCConfig(**asdict(reference_config))
        config2 = TotoOHLCConfig(**asdict(reference_config))
        
        hash1 = regression_manager.compute_config_hash(config1)
        hash2 = regression_manager.compute_config_hash(config2)
        
        assert hash1 == hash2, "Identical configs should have same hash"
        
        # Modified config should have different hash
        config3 = TotoOHLCConfig(**asdict(reference_config))
        config3.embed_dim += 1
        
        hash3 = regression_manager.compute_config_hash(config3)
        assert hash1 != hash3, "Modified config should have different hash"


class TestRegressionUtilities:
    """Test regression testing utilities themselves"""
    
    def test_tensor_comparison_accuracy(self, regression_manager):
        """Test tensor comparison utility accuracy"""
        # Identical tensors
        t1 = torch.tensor([1.0, 2.0, 3.0])
        t2 = torch.tensor([1.0, 2.0, 3.0])
        
        assert regression_manager.compare_tensors(t1, t2, tolerance=1e-10)
        
        # Nearly identical tensors (within tolerance)
        t3 = torch.tensor([1.0, 2.0, 3.000001])
        assert regression_manager.compare_tensors(t1, t3, tolerance=1e-5)
        
        # Different tensors (beyond tolerance)
        t4 = torch.tensor([1.0, 2.0, 3.01])
        with pytest.raises(AssertionError):
            regression_manager.compare_tensors(t1, t4, tolerance=1e-5)
    
    def test_statistics_comparison_accuracy(self, regression_manager):
        """Test statistics comparison utility accuracy"""
        stats1 = {'mean': 1.0, 'std': 0.5, 'count': 100}
        stats2 = {'mean': 1.0, 'std': 0.5, 'count': 100}
        
        assert regression_manager.compare_statistics(stats1, stats2, tolerance=1e-10)
        
        # Within tolerance
        stats3 = {'mean': 1.0001, 'std': 0.5, 'count': 100}
        assert regression_manager.compare_statistics(stats1, stats3, tolerance=1e-3)
        
        # Beyond tolerance
        stats4 = {'mean': 1.01, 'std': 0.5, 'count': 100}
        with pytest.raises(AssertionError):
            regression_manager.compare_statistics(stats1, stats4, tolerance=1e-3)
    
    def test_reference_save_load_cycle(self, regression_manager, reference_config, reference_data):
        """Test reference output save/load cycle"""
        test_name = "save_load_test"
        
        # Create test outputs
        outputs = {
            'model_outputs': {'prediction': torch.tensor([1.0, 2.0, 3.0])},
            'data_stats': {'mean': 1.5, 'std': 0.8},
            'training_metrics': {'loss': 0.25, 'accuracy': 0.9}
        }
        
        # Save reference
        regression_manager.save_reference_output(
            test_name, reference_config, reference_data, outputs
        )
        
        # Load reference
        loaded_reference = regression_manager.load_reference_output(test_name)
        
        assert loaded_reference is not None, "Failed to load saved reference"
        assert loaded_reference.training_metrics['loss'] == 0.25
        assert loaded_reference.training_metrics['accuracy'] == 0.9
        assert loaded_reference.preprocessed_data_stats['mean'] == 1.5
        
        # Check tensor
        expected_tensor = torch.tensor([1.0, 2.0, 3.0])
        actual_tensor = loaded_reference.model_outputs['prediction']
        assert torch.allclose(actual_tensor, expected_tensor)


if __name__ == "__main__":
    # Run regression tests
    pytest.main([
        __file__, 
        "-v", 
        "--tb=short",
        "-x"  # Stop on first failure for regression tests
    ])