from __future__ import annotations

from typing import Optional

from .dependency_injection import (
    register_observer,
    resolve_numpy,
    resolve_torch,
)
from chronos import BaseChronosPipeline

torch = resolve_torch()
np = resolve_numpy()


def _refresh_torch(module):
    global torch
    torch = module


def _refresh_numpy(module):
    global np
    np = module


register_observer("torch", _refresh_torch)
register_observer("numpy", _refresh_numpy)

class ForecastingBoltWrapper:
    def __init__(self, model_name="amazon/chronos-bolt-base", device="cuda"):
        self.model_name = model_name
        self.device = device
        self.pipeline: Optional[BaseChronosPipeline] = None
    
    def load_pipeline(self):
        if self.pipeline is None:
            self.pipeline = BaseChronosPipeline.from_pretrained(
                self.model_name,
                device_map=self.device,
            )
            model_attr = getattr(self.pipeline, "model", None)
            if model_attr is not None and hasattr(model_attr, "eval"):
                evaluated_model = model_attr.eval()
                try:
                    setattr(self.pipeline, "model", evaluated_model)
                except AttributeError:
                    pass
    
    def predict_sequence(self, context_data, prediction_length=7):
        """
        Make predictions for a sequence of steps
        
        Args:
            context_data: torch.Tensor or array-like data for context
            prediction_length: int, number of predictions to make
            
        Returns:
            list of predictions
        """
        self.load_pipeline()

        pipeline = self.pipeline
        if pipeline is None:
            raise RuntimeError("Chronos pipeline failed to load before prediction.")

        if not isinstance(context_data, torch.Tensor):
            context_data = torch.tensor(context_data, dtype=torch.float)
        
        predictions = []
        
        for pred_idx in reversed(range(1, prediction_length + 1)):
            current_context = context_data[:-pred_idx] if pred_idx > 1 else context_data
            
            forecast = pipeline.predict(
                current_context,
                prediction_length=1,
            )
            
            tensor = forecast[0]
            if hasattr(tensor, "detach"):
                tensor = tensor.detach().cpu().numpy()
            else:
                tensor = np.asarray(tensor)
            low, median, high = np.quantile(tensor, [0.1, 0.5, 0.9], axis=0)
            predictions.append(median.item())
        
        return predictions
    
    def predict_single(self, context_data, prediction_length=1):
        """
        Make a single prediction
        
        Args:
            context_data: torch.Tensor or array-like data for context
            prediction_length: int, prediction horizon
            
        Returns:
            median prediction value
        """
        self.load_pipeline()

        pipeline = self.pipeline
        if pipeline is None:
            raise RuntimeError("Chronos pipeline failed to load before prediction.")

        if not isinstance(context_data, torch.Tensor):
            context_data = torch.tensor(context_data, dtype=torch.float)

        forecast = pipeline.predict(
            context_data,
            prediction_length,
        )
        
        tensor = forecast[0]
        if hasattr(tensor, "detach"):
            tensor = tensor.detach().cpu().numpy()
        else:
            tensor = np.asarray(tensor)
        low, median, high = np.quantile(tensor, [0.1, 0.5, 0.9], axis=0)
        return median.item() if prediction_length == 1 else median
