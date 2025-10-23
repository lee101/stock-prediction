#!/usr/bin/env python3
"""Unit tests for modern optimizers."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add hftraining to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../hftraining'))

from hftraining.modern_optimizers import get_optimizer, Lion, AdaFactor, LAMB, Sophia, Adan
from hftraining.hf_trainer import GPro


class TestOptimizerFactory:
    """Test optimizer factory function."""
    
    def test_get_optimizer_gpro(self):
        """Test GPro optimizer creation."""
        model = nn.Linear(10, 1)
        optimizer = get_optimizer("gpro", model.parameters(), lr=0.001)
        
        assert isinstance(optimizer, GPro)
        assert optimizer.defaults['lr'] == 0.001
    
    def test_get_optimizer_lion(self):
        """Test Lion optimizer creation."""
        model = nn.Linear(10, 1)
        optimizer = get_optimizer("lion", model.parameters(), lr=0.001)
        
        assert isinstance(optimizer, Lion)
        assert optimizer.defaults['lr'] == 0.001
    
    def test_get_optimizer_adafactor(self):
        """Test AdaFactor optimizer creation."""
        model = nn.Linear(10, 1)
        optimizer = get_optimizer("adafactor", model.parameters(), lr=0.001)
        
        assert isinstance(optimizer, AdaFactor)
        assert optimizer.defaults['lr'] == 0.001
    
    def test_get_optimizer_lamb(self):
        """Test LAMB optimizer creation."""
        model = nn.Linear(10, 1)
        optimizer = get_optimizer("lamb", model.parameters(), lr=0.001)
        
        assert isinstance(optimizer, LAMB)
        assert optimizer.defaults['lr'] == 0.001
    
    def test_get_optimizer_sophia(self):
        """Test Sophia optimizer creation."""
        model = nn.Linear(10, 1)
        optimizer = get_optimizer("sophia", model.parameters(), lr=0.001)
        
        assert isinstance(optimizer, Sophia)
        assert optimizer.defaults['lr'] == 0.001
    
    def test_get_optimizer_adan(self):
        """Test Adan optimizer creation."""
        model = nn.Linear(10, 1)
        optimizer = get_optimizer("adan", model.parameters(), lr=0.001)
        
        assert isinstance(optimizer, Adan)
        assert optimizer.defaults['lr'] == 0.001
    
    def test_get_optimizer_adamw(self):
        """Test AdamW optimizer creation (fallback to torch)."""
        model = nn.Linear(10, 1)
        optimizer = get_optimizer("adamw", model.parameters(), lr=0.001)
        
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.defaults['lr'] == 0.001
    
    def test_get_optimizer_unknown(self):
        """Test unknown optimizer fallback."""
        model = nn.Linear(10, 1)
        optimizer = get_optimizer("unknown_optimizer", model.parameters(), lr=0.001)
        
        # Should fallback to AdamW
        assert isinstance(optimizer, torch.optim.AdamW)


class TestGProOptimizer:
    """Test GPro optimizer functionality."""
    
    def test_gpro_init_default(self):
        """Test GPro initialization with defaults."""
        model = nn.Linear(5, 1)
        optimizer = GPro(model.parameters())
        
        assert optimizer.defaults['lr'] == 0.001
        assert optimizer.defaults['betas'] == (0.9, 0.999)
        assert optimizer.defaults['eps'] == 1e-8
        assert optimizer.defaults['weight_decay'] == 0.01
        assert optimizer.defaults['projection_factor'] == 0.5
    
    def test_gpro_init_custom(self):
        """Test GPro initialization with custom parameters."""
        model = nn.Linear(5, 1)
        optimizer = GPro(
            model.parameters(),
            lr=0.01,
            betas=(0.95, 0.99),
            eps=1e-6,
            weight_decay=0.001,
            projection_factor=0.3
        )
        
        assert optimizer.defaults['lr'] == 0.01
        assert optimizer.defaults['betas'] == (0.95, 0.99)
        assert optimizer.defaults['eps'] == 1e-6
        assert optimizer.defaults['weight_decay'] == 0.001
        assert optimizer.defaults['projection_factor'] == 0.3
    
    def test_gpro_invalid_params(self):
        """Test GPro with invalid parameters."""
        model = nn.Linear(5, 1)
        
        # Invalid learning rate
        with pytest.raises(ValueError, match="Invalid learning rate"):
            GPro(model.parameters(), lr=-0.01)
        
        # Invalid epsilon
        with pytest.raises(ValueError, match="Invalid epsilon"):
            GPro(model.parameters(), eps=-1e-8)
        
        # Invalid beta1
        with pytest.raises(ValueError, match="Invalid beta parameter"):
            GPro(model.parameters(), betas=(1.5, 0.999))
        
        # Invalid beta2
        with pytest.raises(ValueError, match="Invalid beta parameter"):
            GPro(model.parameters(), betas=(0.9, 1.5))
        
        # Invalid weight decay
        with pytest.raises(ValueError, match="Invalid weight_decay"):
            GPro(model.parameters(), weight_decay=-0.01)
    
    def test_gpro_optimization_step(self):
        """Test GPro optimization step."""
        model = nn.Linear(10, 1)
        optimizer = GPro(model.parameters(), lr=0.01)
        
        # Store initial parameters
        initial_params = [p.clone() for p in model.parameters()]
        
        # Create sample data and compute loss
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)
        loss = nn.MSELoss()(model(x), y)
        
        # Backward pass
        loss.backward()
        
        # Optimization step
        optimizer.step()
        optimizer.zero_grad()
        
        # Check that parameters changed
        final_params = list(model.parameters())
        for initial, final in zip(initial_params, final_params):
            assert not torch.equal(initial, final)
    
    def test_gpro_projection_mechanism(self):
        """Test GPro projection mechanism with large gradients."""
        model = nn.Linear(5, 1)
        optimizer = GPro(model.parameters(), lr=0.1, projection_factor=0.1)
        
        # Create artificially large gradients
        with torch.no_grad():
            for param in model.parameters():
                param.grad = torch.randn_like(param) * 100  # Large gradients
        
        # Should handle large gradients without exploding
        optimizer.step()
        optimizer.zero_grad()
        
        # Check parameters are still finite
        for param in model.parameters():
            assert torch.all(torch.isfinite(param))


class TestLionOptimizer:
    """Test Lion optimizer functionality."""
    
    def test_lion_init_default(self):
        """Test Lion initialization with defaults."""
        model = nn.Linear(5, 1)
        optimizer = Lion(model.parameters())
        
        assert optimizer.defaults['lr'] == 0.0001
        assert optimizer.defaults['betas'] == (0.9, 0.99)
        assert optimizer.defaults['weight_decay'] == 0.01
    
    def test_lion_optimization_step(self):
        """Test Lion optimization step."""
        model = nn.Linear(8, 1)
        optimizer = Lion(model.parameters(), lr=0.001)
        
        initial_params = [p.clone() for p in model.parameters()]
        
        x = torch.randn(16, 8)
        y = torch.randn(16, 1)
        loss = nn.MSELoss()(model(x), y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Parameters should change
        final_params = list(model.parameters())
        for initial, final in zip(initial_params, final_params):
            assert not torch.equal(initial, final)
    
    def test_lion_sign_based_updates(self):
        """Test Lion's sign-based update mechanism."""
        model = nn.Linear(3, 1)
        optimizer = Lion(model.parameters(), lr=0.1)
        
        # Set known gradients
        with torch.no_grad():
            for param in model.parameters():
                param.grad = torch.ones_like(param) * 0.5  # Positive gradients
        
        initial_params = [p.clone() for p in model.parameters()]
        optimizer.step()
        
        # With positive gradients, parameters should decrease (sign-based)
        final_params = list(model.parameters())
        for initial, final in zip(initial_params, final_params):
            assert torch.all(final < initial)


class TestAdaFactorOptimizer:
    """Test AdaFactor optimizer functionality."""
    
    def test_adafactor_init_default(self):
        """Test AdaFactor initialization."""
        model = nn.Linear(5, 1)
        optimizer = AdaFactor(model.parameters())
        
        assert optimizer.defaults['lr'] == 0.001
        assert optimizer.defaults['beta2'] == 0.999
        assert optimizer.defaults['eps'] == 1e-8
        assert optimizer.defaults['weight_decay'] == 0.0
    
    def test_adafactor_optimization_step(self):
        """Test AdaFactor optimization step."""
        model = nn.Linear(6, 1)
        optimizer = AdaFactor(model.parameters(), lr=0.01)
        
        initial_params = [p.clone() for p in model.parameters()]
        
        x = torch.randn(20, 6)
        y = torch.randn(20, 1)
        loss = nn.MSELoss()(model(x), y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Parameters should change
        final_params = list(model.parameters())
        for initial, final in zip(initial_params, final_params):
            assert not torch.equal(initial, final)


class TestLAMBOptimizer:
    """Test LAMB optimizer functionality."""
    
    def test_lamb_init_default(self):
        """Test LAMB initialization."""
        model = nn.Linear(5, 1)
        optimizer = LAMB(model.parameters())
        
        assert optimizer.defaults['lr'] == 0.001
        assert optimizer.defaults['betas'] == (0.9, 0.999)
        assert optimizer.defaults['eps'] == 1e-8
        assert optimizer.defaults['weight_decay'] == 0.01
    
    def test_lamb_optimization_step(self):
        """Test LAMB optimization step."""
        model = nn.Linear(12, 1)
        optimizer = LAMB(model.parameters(), lr=0.01)
        
        initial_params = [p.clone() for p in model.parameters()]
        
        x = torch.randn(24, 12)
        y = torch.randn(24, 1)
        loss = nn.MSELoss()(model(x), y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Parameters should change
        final_params = list(model.parameters())
        for initial, final in zip(initial_params, final_params):
            assert not torch.equal(initial, final)
    
    def test_lamb_layer_adaptation(self):
        """Test LAMB's layer-wise adaptation."""
        # Create model with different layer sizes
        model = nn.Sequential(
            nn.Linear(10, 50),
            nn.Linear(50, 20),
            nn.Linear(20, 1)
        )
        optimizer = LAMB(model.parameters(), lr=0.01)
        
        # Run optimization step
        x = torch.randn(16, 10)
        y = torch.randn(16, 1)
        loss = nn.MSELoss()(model(x), y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Should handle different layer sizes without issues
        for param in model.parameters():
            assert torch.all(torch.isfinite(param))


class TestSophiaOptimizer:
    """Test Sophia optimizer functionality."""
    
    def test_sophia_init_default(self):
        """Test Sophia initialization."""
        model = nn.Linear(5, 1)
        optimizer = Sophia(model.parameters())
        
        assert optimizer.defaults['lr'] == 0.001
        assert optimizer.defaults['betas'] == (0.9, 0.999)
        assert optimizer.defaults['eps'] == 1e-8
        assert optimizer.defaults['weight_decay'] == 0.0
    
    def test_sophia_optimization_step(self):
        """Test Sophia optimization step."""
        model = nn.Linear(7, 1)
        optimizer = Sophia(model.parameters(), lr=0.01)
        
        initial_params = [p.clone() for p in model.parameters()]
        
        x = torch.randn(14, 7)
        y = torch.randn(14, 1)
        loss = nn.MSELoss()(model(x), y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Parameters should change
        final_params = list(model.parameters())
        for initial, final in zip(initial_params, final_params):
            assert not torch.equal(initial, final)


class TestAdanOptimizer:
    """Test Adan optimizer functionality."""
    
    def test_adan_init_default(self):
        """Test Adan initialization."""
        model = nn.Linear(5, 1)
        optimizer = Adan(model.parameters())
        
        assert optimizer.defaults['lr'] == 0.001
        assert optimizer.defaults['betas'] == (0.98, 0.92, 0.99)
        assert optimizer.defaults['eps'] == 1e-8
        assert optimizer.defaults['weight_decay'] == 0.02
    
    def test_adan_optimization_step(self):
        """Test Adan optimization step."""
        model = nn.Linear(9, 1)
        optimizer = Adan(model.parameters(), lr=0.01)
        
        initial_params = [p.clone() for p in model.parameters()]
        
        x = torch.randn(18, 9)
        y = torch.randn(18, 1)
        loss = nn.MSELoss()(model(x), y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Parameters should change
        final_params = list(model.parameters())
        for initial, final in zip(initial_params, final_params):
            assert not torch.equal(initial, final)
    
    def test_adan_triple_momentum(self):
        """Test Adan's triple momentum mechanism."""
        model = nn.Linear(4, 1)
        optimizer = Adan(model.parameters(), lr=0.1, betas=(0.9, 0.8, 0.95))
        
        # Run several optimization steps to build up momentum
        for i in range(5):
            x = torch.randn(8, 4)
            y = torch.randn(8, 1)
            loss = nn.MSELoss()(model(x), y)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Check that state contains momentum terms
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if len(state) > 0:  # State is initialized after first step
                    assert 'exp_avg' in state
                    assert 'exp_avg_diff' in state
                    assert 'exp_avg_sq' in state


class TestOptimizerIntegration:
    """Test optimizer integration and comparative behavior."""
    
    def test_optimizer_convergence_comparison(self):
        """Test that different optimizers can optimize a simple problem."""
        # Simple quadratic function: f(x) = (x - 2)^2
        target = 2.0
        
        optimizers_to_test = [
            ("gpro", GPro),
            ("lion", Lion), 
            ("lamb", LAMB),
            ("adafactor", AdaFactor)
        ]
        
        for name, optimizer_class in optimizers_to_test:
            # Create parameter to optimize
            param = torch.tensor([0.0], requires_grad=True)
            optimizer = optimizer_class([param], lr=0.1)
            
            # Optimize for several steps
            for _ in range(50):
                loss = (param - target) ** 2
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            # Should converge close to target
            assert abs(param.item() - target) < 0.5, f"{name} failed to converge"
    
    def test_optimizer_with_different_model_sizes(self):
        """Test optimizers with different model architectures."""
        model_configs = [
            (5, 1),      # Small model
            (50, 10),    # Medium model
            (100, 50)    # Larger model
        ]
        
        for input_size, output_size in model_configs:
            model = nn.Linear(input_size, output_size)
            
            # Test with GPro optimizer
            optimizer = GPro(model.parameters(), lr=0.01)
            
            x = torch.randn(32, input_size)
            y = torch.randn(32, output_size)
            loss = nn.MSELoss()(model(x), y)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Should handle without errors
            for param in model.parameters():
                assert torch.all(torch.isfinite(param))
    
    def test_mixed_precision_compatibility(self):
        """Test optimizer compatibility with mixed precision."""
        model = nn.Linear(10, 1)
        optimizer = GPro(model.parameters(), lr=0.01)
        
        # Simulate mixed precision with gradient scaling
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        x = torch.randn(16, 10)
        y = torch.randn(16, 1)
        
        if scaler:
            with torch.cuda.amp.autocast():
                loss = nn.MSELoss()(model(x), y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # CPU fallback
            loss = nn.MSELoss()(model(x), y)
            loss.backward()
            optimizer.step()
        
        optimizer.zero_grad()
        
        # Should work without issues
        for param in model.parameters():
            assert torch.all(torch.isfinite(param))