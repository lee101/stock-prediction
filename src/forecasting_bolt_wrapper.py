from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any, Optional

from chronos import BaseChronosPipeline


def _optional_import(module_name: str) -> ModuleType | None:
    try:
        return import_module(module_name)
    except ModuleNotFoundError:
        return None


torch: ModuleType | None = _optional_import("torch")
np: ModuleType | None = _optional_import("numpy")


def setup_forecasting_bolt_imports(
    *,
    torch_module: ModuleType | None = None,
    numpy_module: ModuleType | None = None,
    **_: Any,
) -> None:
    global torch, np
    if torch_module is not None:
        torch = torch_module
    if numpy_module is not None:
        np = numpy_module


def _require_torch() -> ModuleType:
    global torch
    if torch is not None:
        return torch
    try:
        module = import_module("torch")
    except ModuleNotFoundError as exc:
        raise RuntimeError("Torch is unavailable. Call setup_forecasting_bolt_imports before use.") from exc
    torch = module
    return module


def _require_numpy() -> ModuleType:
    global np
    if np is not None:
        return np
    try:
        module = import_module("numpy")
    except ModuleNotFoundError as exc:
        raise RuntimeError("NumPy is unavailable. Call setup_forecasting_bolt_imports before use.") from exc
    np = module
    return module


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

        torch_mod = _require_torch()
        numpy_mod = _require_numpy()

        if not isinstance(context_data, torch_mod.Tensor):
            context_data = torch_mod.tensor(context_data, dtype=torch_mod.float)

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
                tensor = numpy_mod.asarray(tensor)
            _, median, _ = numpy_mod.quantile(tensor, [0.1, 0.5, 0.9], axis=0)
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

        torch_mod = _require_torch()
        numpy_mod = _require_numpy()

        if not isinstance(context_data, torch_mod.Tensor):
            context_data = torch_mod.tensor(context_data, dtype=torch_mod.float)

        forecast = pipeline.predict(
            context_data,
            prediction_length,
        )

        tensor = forecast[0]
        if hasattr(tensor, "detach"):
            tensor = tensor.detach().cpu().numpy()
        else:
            tensor = numpy_mod.asarray(tensor)
        _, median, _ = numpy_mod.quantile(tensor, [0.1, 0.5, 0.9], axis=0)
        return median.item() if prediction_length == 1 else median
