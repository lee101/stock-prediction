#!/usr/bin/env python3
"""Mock torch module for testing when torch is not properly installed."""

import numpy as np
from unittest.mock import MagicMock
from contextlib import contextmanager


class TensorMock:
    """Mock tensor class."""
    def __init__(self, data=None):
        if data is None:
            data = np.random.randn(1, 1)
        self.data = np.array(data)
        self.shape = self.data.shape
        self.device = 'cpu'
        self.grad = None
        
    def to(self, device):
        self.device = device
        return self
    
    def unsqueeze(self, dim):
        return TensorMock(np.expand_dims(self.data, dim))
    
    def float(self):
        return TensorMock(self.data.astype(float))
    
    def long(self):
        return TensorMock(self.data.astype(int))
    
    def mean(self):
        return TensorMock(np.mean(self.data))
    
    def sum(self):
        return TensorMock(np.sum(self.data))
    
    def backward(self):
        pass
    
    def clone(self):
        return TensorMock(self.data.copy())
    
    def __getitem__(self, idx):
        return TensorMock(self.data[idx])
    
    def __len__(self):
        return len(self.data)
    
    def item(self):
        """Return scalar value from tensor."""
        if self.data.size == 1:
            return float(self.data.flatten()[0])
        else:
            raise ValueError("Can only convert size-1 tensors to scalars")


def randn(*shape):
    """Mock randn function."""
    return TensorMock(np.random.randn(*shape))


def tensor(data, dtype=None):
    """Mock tensor function."""
    return TensorMock(data)


def FloatTensor(data):
    """Mock FloatTensor function."""
    return TensorMock(data)


def softmax(input, dim=None):
    """Mock softmax function."""
    data = input.data if hasattr(input, 'data') else input
    exp_x = np.exp(data - np.max(data, axis=dim, keepdims=True))
    return TensorMock(exp_x / np.sum(exp_x, axis=dim, keepdims=True))


@contextmanager
def no_grad():
    """Mock no_grad context manager."""
    yield


def zeros(*shape):
    """Mock zeros function."""
    return TensorMock(np.zeros(shape))


def ones(*shape):
    """Mock ones function."""
    return TensorMock(np.ones(shape))


def from_numpy(arr):
    """Mock from_numpy function."""
    return TensorMock(arr)


def load(path, map_location=None):
    """Mock torch.load function."""
    return {
        'model_state_dict': {},
        'config': {
            'hidden_size': 256,
            'num_heads': 8, 
            'num_layers': 4,
            'input_features': 21,
            'sequence_length': 60,
            'prediction_horizon': 5
        },
        'global_step': 1000
    }


def save(obj, path):
    """Mock torch.save function."""
    pass


def device(name):
    """Mock device function."""
    return name


def isnan(tensor):
    """Mock isnan function."""
    data = tensor.data if hasattr(tensor, 'data') else tensor
    return TensorMock(np.isnan(data))


def allclose(a, b):
    """Mock allclose function."""
    data_a = a.data if hasattr(a, 'data') else a
    data_b = b.data if hasattr(b, 'data') else b
    return np.allclose(data_a, data_b)


def argmax(tensor, dim=None):
    """Mock argmax function."""
    data = tensor.data if hasattr(tensor, 'data') else tensor
    # Handle empty arrays
    if isinstance(data, np.ndarray) and data.size == 0:
        return TensorMock(0)
    if dim is None:
        return TensorMock(np.argmax(data))
    else:
        return TensorMock(np.argmax(data, axis=dim))


def argmin(tensor, dim=None):
    """Mock argmin function."""
    data = tensor.data if hasattr(tensor, 'data') else tensor
    if dim is None:
        return TensorMock(np.argmin(data))
    else:
        return TensorMock(np.argmin(data, axis=dim))


def cat(tensors, dim=0):
    """Mock cat function."""
    data_list = [t.data if hasattr(t, 'data') else t for t in tensors]
    return TensorMock(np.concatenate(data_list, axis=dim))


def stack(tensors, dim=0):
    """Mock stack function."""
    data_list = [t.data if hasattr(t, 'data') else t for t in tensors]
    return TensorMock(np.stack(data_list, axis=dim))


# Create a functional module
class functional:
    relu = lambda x: relu(x)
    sigmoid = lambda x: sigmoid(x)
    tanh = lambda x: tanh(x)
    dropout = lambda x, p=0.5, training=True: dropout(x, p, training)
    layer_norm = lambda x, shape, weight=None, bias=None: layer_norm(x, shape, weight, bias)
    linear = lambda input, weight, bias=None: linear(input, weight, bias)
    cross_entropy = lambda input, target, weight=None, reduction='mean': cross_entropy(input, target, weight, reduction)
    mse_loss = lambda input, target, reduction='mean': mse_loss(input, target, reduction)
    softmax = lambda x, dim=None: softmax(x, dim)

# Mock nn module
class nn:
    # Add functional as class attribute
    functional = functional
    
    class Module:
        def __init__(self):
            self.training = True
            self._parameters = {}
            self._modules = {}
            
        def train(self, mode=True):
            self.training = mode
            return self
            
        def eval(self):
            return self.train(False)
            
        def to(self, device):
            return self
            
        def parameters(self):
            return [TensorMock() for _ in range(5)]
            
        def state_dict(self):
            return {}
            
        def load_state_dict(self, state_dict):
            pass
            
        def forward(self, x):
            return x
            
        def __call__(self, x):
            return self.forward(x)
    
    class Linear(Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = TensorMock(np.random.randn(out_features, in_features))
            self.bias = TensorMock(np.random.randn(out_features))
            
        def forward(self, x):
            return x
    
    class TransformerEncoder(Module):
        def __init__(self, encoder_layer, num_layers):
            super().__init__()
            self.num_layers = num_layers
            
        def forward(self, x):
            return x
    
    class TransformerEncoderLayer(Module):
        def __init__(self, d_model, nhead, batch_first=True):
            super().__init__()
            self.d_model = d_model
            self.nhead = nhead
            
        def forward(self, x):
            return x
    
    class LayerNorm(Module):
        def __init__(self, normalized_shape):
            super().__init__()
            self.normalized_shape = normalized_shape
            
        def forward(self, x):
            return x
    
    class Dropout(Module):
        def __init__(self, p=0.1):
            super().__init__()
            self.p = p
            
        def forward(self, x):
            return x
    
    class ReLU(Module):
        def forward(self, x):
            return x
    
    class Sequential(Module):
        def __init__(self, *modules):
            super().__init__()
            self.modules = modules
            
        def forward(self, x):
            for module in self.modules:
                x = module(x)
            return x
    
    class MSELoss(Module):
        def __call__(self, pred, target):
            return TensorMock(np.mean((pred.data - target.data) ** 2))
    
    class CrossEntropyLoss(Module):
        def __call__(self, pred, target):
            return TensorMock(0.5)


# Mock optim module
class optim:
    class Optimizer:
        def __init__(self, params, lr=1e-3):
            self.param_groups = [{'lr': lr}]
            self.state = {}
            
        def step(self):
            pass
            
        def zero_grad(self):
            pass
    
    class Adam(Optimizer):
        pass
    
    class AdamW(Optimizer):
        pass
    
    class SGD(Optimizer):
        pass


# Mock cuda module
class cuda:
    @staticmethod
    def is_available():
        return False
    
    @staticmethod
    def device_count():
        return 0


# Functional API (F)
def relu(x):
    """Mock relu function."""
    data = x.data if hasattr(x, 'data') else x
    return TensorMock(np.maximum(data, 0))


def sigmoid(x):
    """Mock sigmoid function."""
    data = x.data if hasattr(x, 'data') else x
    return TensorMock(1 / (1 + np.exp(-data)))


def tanh(x):
    """Mock tanh function."""
    data = x.data if hasattr(x, 'data') else x
    return TensorMock(np.tanh(data))


def dropout(x, p=0.5, training=True):
    """Mock dropout function."""
    return x  # No dropout in testing


def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    """Mock layer norm function."""
    return x  # Simplified for testing


def linear(input, weight, bias=None):
    """Mock linear function."""
    return input  # Simplified for testing


def cross_entropy(input, target, weight=None, reduction='mean'):
    """Mock cross entropy loss."""
    return TensorMock(0.5)


def mse_loss(input, target, reduction='mean'):
    """Mock MSE loss."""
    data_input = input.data if hasattr(input, 'data') else input
    data_target = target.data if hasattr(target, 'data') else target
    loss = np.mean((data_input - data_target) ** 2)
    return TensorMock(loss)


# Mock utils module
class utils:
    class data:
        class DataLoader:
            def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
                self.dataset = dataset
                self.batch_size = batch_size
                self.shuffle = shuffle
                
            def __iter__(self):
                # Simple batching
                for i in range(0, len(self.dataset), self.batch_size):
                    yield self.dataset[i:i+self.batch_size]
                    
            def __len__(self):
                return (len(self.dataset) + self.batch_size - 1) // self.batch_size
        
        class Dataset:
            def __init__(self):
                pass
                
            def __len__(self):
                return 0
                
            def __getitem__(self, idx):
                return None
    
    class tensorboard:
        class SummaryWriter:
            def __init__(self, *args, **kwargs):
                pass
                
            def add_scalar(self, tag, scalar_value, global_step=None):
                pass
                
            def add_histogram(self, tag, values, global_step=None):
                pass
                
            def close(self):
                pass
                
            def flush(self):
                pass


__version__ = "2.2.2+cu121"