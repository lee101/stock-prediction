#!/usr/bin/env python3
"""
Realistic integration tests for totoembedding/ directory.
Tests embedding models, pretrained loaders, and auditing without mocks.
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import json
import pickle

# Add paths
TEST_DIR = Path(__file__).parent.parent
REPO_ROOT = TEST_DIR.parent
sys.path.extend([str(REPO_ROOT), str(REPO_ROOT / 'totoembedding')])

import pytest


class TestEmbeddingModel:
    """Test embedding model with real data."""
    
    @pytest.fixture
    def sample_sequences(self):
        """Generate sample sequences for embedding."""
        n_samples = 200
        seq_len = 50
        n_features = 15
        
        # Create sequences with patterns
        sequences = []
        for i in range(n_samples):
            # Add some structure to make embeddings meaningful
            base_pattern = np.sin(np.linspace(0, 2*np.pi, seq_len))
            noise = np.random.randn(seq_len, n_features) * 0.1
            pattern = base_pattern.reshape(-1, 1) * (1 + i/n_samples)
            sequence = pattern + noise
            sequences.append(sequence)
        
        return torch.tensor(np.array(sequences), dtype=torch.float32)
    
    def test_embedding_model_training(self, sample_sequences):
        """Test that embedding model learns meaningful representations."""
        from totoembedding.embedding_model import (
            TotoEmbeddingModel, 
            EmbeddingConfig,
            ContrastiveLoss
        )
        
        config = EmbeddingConfig(
            input_dim=15,
            embedding_dim=64,
            hidden_dims=[128, 256, 128],
            sequence_length=50,
            dropout=0.1,
            use_attention=True,
            num_heads=4
        )
        
        model = TotoEmbeddingModel(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = ContrastiveLoss(temperature=0.1)
        
        # Training loop
        model.train()
        initial_embeddings = model(sample_sequences[:10]).detach()
        
        for epoch in range(10):
            # Create positive pairs (augmented versions)
            batch_size = 32
            for i in range(0, len(sample_sequences) - batch_size, batch_size):
                batch = sample_sequences[i:i+batch_size]
                
                # Simple augmentation - add noise
                augmented = batch + torch.randn_like(batch) * 0.01
                
                optimizer.zero_grad()
                embeddings1 = model(batch)
                embeddings2 = model(augmented)
                
                loss = criterion(embeddings1, embeddings2)
                loss.backward()
                optimizer.step()
        
        # Test that embeddings changed and are meaningful
        final_embeddings = model(sample_sequences[:10])
        
        # Embeddings should have changed
        assert not torch.allclose(initial_embeddings, final_embeddings)
        
        # Similar inputs should have similar embeddings
        emb1 = model(sample_sequences[0:1])
        emb2 = model(sample_sequences[0:1] + torch.randn(1, 50, 15) * 0.001)
        similarity = torch.cosine_similarity(emb1, emb2)
        assert similarity > 0.9, "Similar inputs should have similar embeddings"
        
        # Different inputs should have different embeddings
        emb3 = model(sample_sequences[100:101])
        similarity_diff = torch.cosine_similarity(emb1, emb3)
        assert similarity_diff < similarity, "Different inputs should be less similar"
    
    def test_embedding_model_inference_speed(self, sample_sequences):
        """Test that embedding model has reasonable inference speed."""
        from totoembedding.embedding_model import TotoEmbeddingModel, EmbeddingConfig
        import time
        
        config = EmbeddingConfig(
            input_dim=15,
            embedding_dim=32,
            hidden_dims=[64, 64],
            sequence_length=50
        )
        
        model = TotoEmbeddingModel(config)
        model.eval()
        
        # Warmup
        with torch.no_grad():
            _ = model(sample_sequences[:10])
        
        # Time batch inference
        batch_sizes = [1, 16, 64]
        for batch_size in batch_sizes:
            batch = sample_sequences[:batch_size]
            
            start_time = time.time()
            with torch.no_grad():
                embeddings = model(batch)
            inference_time = time.time() - start_time
            
            # Should be fast enough (< 100ms for batch of 64)
            if batch_size == 64:
                assert inference_time < 0.1, f"Inference too slow: {inference_time:.3f}s"
            
            assert embeddings.shape == (batch_size, config.embedding_dim)


class TestPretrainedLoader:
    """Test loading and using pretrained models."""
    
    def test_pretrained_model_save_load(self):
        """Test saving and loading pretrained models."""
        from totoembedding.pretrained_loader import (
            PretrainedModelManager,
            ModelRegistry
        )
        from totoembedding.embedding_model import TotoEmbeddingModel, EmbeddingConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = PretrainedModelManager(cache_dir=tmpdir)
            
            # Create and save a model
            config = EmbeddingConfig(
                input_dim=10,
                embedding_dim=32,
                hidden_dims=[64],
                model_name="test_model_v1"
            )
            
            model = TotoEmbeddingModel(config)
            
            # Train slightly to change weights
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            data = torch.randn(10, 20, 10)
            for _ in range(5):
                optimizer.zero_grad()
                loss = model(data).sum()
                loss.backward()
                optimizer.step()
            
            # Save model
            model_path = manager.save_model(
                model, 
                config,
                metadata={'version': '1.0', 'trained_on': 'test_data'}
            )
            
            # Load model
            loaded_model, loaded_config, metadata = manager.load_model(model_path)
            
            # Verify loaded correctly
            assert loaded_config.embedding_dim == config.embedding_dim
            assert metadata['version'] == '1.0'
            
            # Verify weights are same
            for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
                assert torch.allclose(p1, p2)
    
    def test_model_registry(self):
        """Test model registry for managing multiple models."""
        from totoembedding.pretrained_loader import ModelRegistry
        
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_path=tmpdir)
            
            # Register models
            registry.register_model(
                name="small_embed",
                path=f"{tmpdir}/small.pt",
                config={'embedding_dim': 32},
                performance_metrics={'loss': 0.5, 'accuracy': 0.85}
            )
            
            registry.register_model(
                name="large_embed",
                path=f"{tmpdir}/large.pt",
                config={'embedding_dim': 128},
                performance_metrics={'loss': 0.3, 'accuracy': 0.92}
            )
            
            # Query registry
            all_models = registry.list_models()
            assert len(all_models) == 2
            
            # Get best model by metric
            best_model = registry.get_best_model(metric='accuracy')
            assert best_model['name'] == "large_embed"
            assert best_model['performance_metrics']['accuracy'] == 0.92
            
            # Filter models
            small_models = registry.filter_models(
                lambda m: m['config']['embedding_dim'] < 64
            )
            assert len(small_models) == 1
            assert small_models[0]['name'] == "small_embed"


class TestEmbeddingAudit:
    """Test embedding auditing and analysis."""
    
    def test_embedding_quality_audit(self):
        """Test auditing embedding quality."""
        from totoembedding.audit_embeddings import (
            EmbeddingAuditor,
            QualityMetrics
        )
        
        # Create sample embeddings with known properties
        n_samples = 500
        embedding_dim = 64
        
        # Create embeddings with clusters
        embeddings = []
        labels = []
        for cluster_id in range(5):
            cluster_center = np.random.randn(embedding_dim)
            for _ in range(100):
                # Add samples around cluster center
                sample = cluster_center + np.random.randn(embedding_dim) * 0.1
                embeddings.append(sample)
                labels.append(cluster_id)
        
        embeddings = torch.tensor(np.array(embeddings), dtype=torch.float32)
        labels = torch.tensor(labels)
        
        auditor = EmbeddingAuditor()
        metrics = auditor.audit_embeddings(embeddings, labels)
        
        # Check quality metrics
        assert 'silhouette_score' in metrics
        assert metrics['silhouette_score'] > 0.5  # Should have good clustering
        
        assert 'calinski_harabasz_score' in metrics
        assert metrics['calinski_harabasz_score'] > 100  # Good separation
        
        assert 'embedding_variance' in metrics
        assert metrics['embedding_variance'] > 0.5  # Not collapsed
        
        assert 'intrinsic_dimension' in metrics
        assert 10 < metrics['intrinsic_dimension'] < 50  # Reasonable dimension
    
    def test_embedding_visualization(self):
        """Test embedding visualization generation."""
        from totoembedding.audit_embeddings import visualize_embeddings
        
        # Create sample embeddings
        embeddings = torch.randn(200, 128)
        labels = torch.randint(0, 4, (200,))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = Path(tmpdir) / 'embeddings.png'
            
            visualize_embeddings(
                embeddings,
                labels=labels,
                method='tsne',
                save_path=plot_path,
                show_plot=False
            )
            
            assert plot_path.exists()
            assert plot_path.stat().st_size > 0
    
    def test_embedding_distance_analysis(self):
        """Test analyzing distances in embedding space."""
        from totoembedding.audit_embeddings import analyze_distances
        
        # Create embeddings with known structure
        n_samples = 100
        dim = 32
        
        # Two distinct groups
        group1 = torch.randn(n_samples // 2, dim) * 0.1
        group2 = torch.randn(n_samples // 2, dim) * 0.1 + 5  # Offset
        embeddings = torch.cat([group1, group2])
        
        analysis = analyze_distances(embeddings)
        
        assert 'mean_distance' in analysis
        assert 'std_distance' in analysis
        assert 'min_distance' in analysis
        assert 'max_distance' in analysis
        
        # Should detect the separation
        assert analysis['max_distance'] > analysis['mean_distance'] * 1.5
        
        # Check nearest neighbor analysis
        assert 'mean_nn_distance' in analysis
        assert analysis['mean_nn_distance'] < analysis['mean_distance']


class TestEmbeddingIntegration:
    """Test integration between embedding components."""
    
    def test_end_to_end_embedding_pipeline(self):
        """Test complete embedding pipeline from data to evaluation."""
        from totoembedding.embedding_model import TotoEmbeddingModel, EmbeddingConfig
        from totoembedding.pretrained_loader import PretrainedModelManager
        from totoembedding.audit_embeddings import EmbeddingAuditor
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Create and train model
            config = EmbeddingConfig(
                input_dim=20,
                embedding_dim=48,
                hidden_dims=[96, 96],
                sequence_length=30
            )
            
            model = TotoEmbeddingModel(config)
            
            # Generate training data
            train_data = torch.randn(500, 30, 20)
            
            # Simple training
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            model.train()
            
            for epoch in range(5):
                for i in range(0, len(train_data), 32):
                    batch = train_data[i:i+32]
                    optimizer.zero_grad()
                    embeddings = model(batch)
                    # Simple loss - maximize variance
                    loss = -embeddings.var()
                    loss.backward()
                    optimizer.step()
            
            # 2. Save model
            manager = PretrainedModelManager(cache_dir=tmpdir)
            model_path = manager.save_model(model, config)
            
            # 3. Load and use model
            loaded_model, _, _ = manager.load_model(model_path)
            loaded_model.eval()
            
            # 4. Generate embeddings
            test_data = torch.randn(100, 30, 20)
            with torch.no_grad():
                test_embeddings = loaded_model(test_data)
            
            # 5. Audit embeddings
            auditor = EmbeddingAuditor()
            metrics = auditor.audit_embeddings(test_embeddings)
            
            # Verify pipeline worked
            assert test_embeddings.shape == (100, 48)
            assert 'embedding_variance' in metrics
            assert metrics['embedding_variance'] > 0.1
    
    def test_embedding_fine_tuning(self):
        """Test fine-tuning pretrained embeddings."""
        from totoembedding.embedding_model import TotoEmbeddingModel, EmbeddingConfig
        
        # Create base model
        config = EmbeddingConfig(
            input_dim=10,
            embedding_dim=32,
            hidden_dims=[64]
        )
        
        base_model = TotoEmbeddingModel(config)
        
        # Get initial embeddings
        test_data = torch.randn(50, 25, 10)
        with torch.no_grad():
            initial_embeddings = base_model(test_data).clone()
        
        # Fine-tune on specific task
        base_model.train()
        optimizer = torch.optim.Adam(base_model.parameters(), lr=1e-4)
        
        # Simulate task-specific training
        task_data = torch.randn(200, 25, 10)
        task_labels = torch.randint(0, 3, (200,))
        
        # Add classification head for fine-tuning
        classifier = nn.Linear(32, 3)
        
        for epoch in range(10):
            for i in range(0, len(task_data), 16):
                batch = task_data[i:i+16]
                batch_labels = task_labels[i:i+16]
                
                optimizer.zero_grad()
                embeddings = base_model(batch)
                logits = classifier(embeddings.mean(dim=1))
                loss = nn.CrossEntropyLoss()(logits, batch_labels)
                loss.backward()
                optimizer.step()
        
        # Check embeddings changed but not drastically
        with torch.no_grad():
            final_embeddings = base_model(test_data)
        
        # Should have changed
        assert not torch.allclose(initial_embeddings, final_embeddings)
        
        # But not too much (fine-tuning preserves structure)
        cosine_sim = torch.cosine_similarity(
            initial_embeddings.flatten(),
            final_embeddings.flatten(),
            dim=0
        )
        assert cosine_sim > 0.7, "Fine-tuning should preserve embedding structure"


class TestEmbeddingRobustness:
    """Test robustness of embedding models."""
    
    def test_embedding_noise_robustness(self):
        """Test that embeddings are robust to input noise."""
        from totoembedding.embedding_model import TotoEmbeddingModel, EmbeddingConfig
        
        config = EmbeddingConfig(
            input_dim=15,
            embedding_dim=64,
            hidden_dims=[128, 128],
            dropout=0.2
        )
        
        model = TotoEmbeddingModel(config)
        model.eval()
        
        # Original data
        data = torch.randn(20, 40, 15)
        
        with torch.no_grad():
            original_embeddings = model(data)
            
            # Test with different noise levels
            noise_levels = [0.01, 0.05, 0.1]
            for noise_level in noise_levels:
                noisy_data = data + torch.randn_like(data) * noise_level
                noisy_embeddings = model(noisy_data)
                
                # Calculate similarity
                similarities = []
                for i in range(len(data)):
                    sim = torch.cosine_similarity(
                        original_embeddings[i],
                        noisy_embeddings[i],
                        dim=0
                    )
                    similarities.append(sim.item())
                
                mean_similarity = np.mean(similarities)
                
                # Should maintain high similarity even with noise
                if noise_level <= 0.05:
                    assert mean_similarity > 0.9, f"Not robust to {noise_level} noise"
                else:
                    assert mean_similarity > 0.7, f"Too sensitive to {noise_level} noise"
    
    def test_embedding_missing_data_handling(self):
        """Test handling of missing data in embeddings."""
        from totoembedding.embedding_model import TotoEmbeddingModel, EmbeddingConfig
        
        config = EmbeddingConfig(
            input_dim=10,
            embedding_dim=32,
            handle_missing=True,
            missing_value_strategy='zero'
        )
        
        model = TotoEmbeddingModel(config)
        model.eval()
        
        # Create data with missing values (represented as NaN)
        data = torch.randn(30, 20, 10)
        data_with_missing = data.clone()
        
        # Randomly mask some values
        mask = torch.rand_like(data) < 0.1  # 10% missing
        data_with_missing[mask] = float('nan')
        
        with torch.no_grad():
            # Model should handle NaN values
            embeddings = model(data_with_missing)
            
            # Should produce valid embeddings
            assert not torch.isnan(embeddings).any()
            assert not torch.isinf(embeddings).any()
            
            # Should be somewhat similar to complete data embeddings
            complete_embeddings = model(data)
            
            similarities = []
            for i in range(len(data)):
                sim = torch.cosine_similarity(
                    embeddings[i],
                    complete_embeddings[i],
                    dim=0
                )
                similarities.append(sim.item())
            
            assert np.mean(similarities) > 0.8, "Missing data handling too disruptive"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])