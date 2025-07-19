import torch
import numpy as np
from chronos import BaseChronosPipeline

class ForecastingBoltWrapper:
    def __init__(self, model_name="amazon/chronos-bolt-base", device="cuda"):
        self.model_name = model_name
        self.device = device
        self.pipeline = None
    
    def load_pipeline(self):
        if self.pipeline is None:
            self.pipeline = BaseChronosPipeline.from_pretrained(
                self.model_name,
                device_map=self.device,
            )
            self.pipeline.model = self.pipeline.model.eval()
    
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
        
        if not isinstance(context_data, torch.Tensor):
            context_data = torch.tensor(context_data, dtype=torch.float)
        
        predictions = []
        
        for pred_idx in reversed(range(1, prediction_length + 1)):
            current_context = context_data[:-pred_idx] if pred_idx > 1 else context_data
            
            forecast = self.pipeline.predict(
                current_context,
                prediction_length=1,
            )
            
            low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
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
        
        if not isinstance(context_data, torch.Tensor):
            context_data = torch.tensor(context_data, dtype=torch.float)
        
        forecast = self.pipeline.predict(
            context_data,
            prediction_length,
        )
        
        low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
        return median.item() if prediction_length == 1 else median