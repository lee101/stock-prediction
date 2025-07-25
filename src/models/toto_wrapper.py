"""
Toto forecasting wrapper to replace Chronos
"""
import torch
import numpy as np
from pathlib import Path
from typing import Union, List, Optional, Tuple
from dataclasses import dataclass
import sys
# need to uv pip install -e . in here after dl toto
sys.path.insert(0, '/mnt/fast/code/chronos-forecasting/toto')

from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster
from toto.model.toto import Toto


@dataclass
class TotoForecast:
    """Container for Toto forecast results - compatible with Chronos format"""
    samples: np.ndarray
    
    def numpy(self):
        """Return numpy array of samples in Chronos-compatible format"""
        # Toto returns shape (1, num_variables, prediction_length, num_samples)
        # We need to reshape to match Chronos format
        samples = self.samples
        
        # Remove batch dimension if present (first dim = 1)
        if samples.ndim == 4 and samples.shape[0] == 1:
            samples = samples.squeeze(0)  # Now (num_variables, prediction_length, num_samples)
        
        # Remove variable dimension if single variable (first dim = 1)
        if samples.ndim == 3 and samples.shape[0] == 1:
            samples = samples.squeeze(0)  # Now (prediction_length, num_samples)
        
        # For single prediction step, return 1D array of samples
        if samples.ndim == 2 and samples.shape[0] == 1:
            return samples.squeeze(0)  # Shape (num_samples,)
        
        # For multiple prediction steps, transpose to (num_samples, prediction_length)
        if samples.ndim == 2:
            return samples.T
        
        return samples


class TotoPipeline:
    """
    Wrapper class that mimics ChronosPipeline interface for Toto model
    """
    
    def __init__(self, model: Toto, device: str = 'cuda'):
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        
        # Optionally compile for speed
        try:
            self.model.compile()
        except Exception as e:
            print(f"Could not compile model: {e}")
            
        self.forecaster = TotoForecaster(self.model.model)
        
    @classmethod
    def from_pretrained(
        cls, 
        model_id: str = "Datadog/Toto-Open-Base-1.0",
        device_map: str = "cuda",
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs
    ):
        """
        Load pretrained Toto model
        
        Args:
            model_id: Model identifier (default: Datadog/Toto-Open-Base-1.0)
            device_map: Device to load model on
            torch_dtype: Data type for model (ignored for compatibility)
            **kwargs: Additional arguments (ignored)
            
        Returns:
            TotoPipeline instance
        """
        device = device_map if device_map != "mps" else "cpu"  # MPS not fully supported
        
        # Load pre-trained Toto model
        model = Toto.from_pretrained(model_id)
        
        return cls(model, device)
    
    def predict(
        self,
        context: Union[torch.Tensor, np.ndarray, List[float]],
        prediction_length: int,
        num_samples: int = 2048,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> List[TotoForecast]:
        """
        Generate forecasts using Toto model
        
        Args:
            context: Historical time series data
            prediction_length: Number of steps to forecast
            num_samples: Number of sample paths to generate
            temperature: Sampling temperature (ignored, for compatibility)
            top_k: Top-k sampling (ignored, for compatibility)
            top_p: Top-p sampling (ignored, for compatibility)
            
        Returns:
            List containing single TotoForecast object
        """
        # Convert context to tensor if needed
        if isinstance(context, (list, np.ndarray)):
            context = torch.tensor(context, dtype=torch.float32)
            
        # Move to device
        context = context.to(self.device)
        
        # Ensure 2D shape (variables x timesteps)
        if context.dim() == 1:
            context = context.unsqueeze(0)  # Add variable dimension
            
        # Get context length
        seq_len = context.shape[-1]
        
        # Create timestamps (assuming regular intervals)
        # Using 15-minute intervals as default (can be adjusted)
        time_interval_seconds = 60 * 15  # 15 minutes
        timestamp_seconds = torch.zeros(1, seq_len).to(self.device)
        time_interval_tensor = torch.full((1,), time_interval_seconds).to(self.device)
        
        # Create MaskedTimeseries input
        inputs = MaskedTimeseries(
            series=context,
            padding_mask=torch.full_like(context, True, dtype=torch.bool),
            id_mask=torch.zeros_like(context),
            timestamp_seconds=timestamp_seconds,
            time_interval_seconds=time_interval_tensor,
        )
        
        # Generate forecasts
        # Note: Toto generates multiple samples at once
        with torch.inference_mode():
            forecast = self.forecaster.forecast(
                inputs,
                prediction_length=prediction_length,
                num_samples=num_samples,
                samples_per_batch=256,  # Batch size for memory efficiency
            )
        
        # Convert to numpy and reshape to match Chronos output format
        # Chronos returns shape: (num_samples, prediction_length)
        samples = forecast.samples.cpu().numpy()
        
        # If samples has shape (1, num_samples, prediction_length), squeeze first dim
        if samples.ndim == 3 and samples.shape[0] == 1:
            samples = samples.squeeze(0)
            
        # Return as list with single forecast (matching Chronos interface)
        return [TotoForecast(samples=samples)]