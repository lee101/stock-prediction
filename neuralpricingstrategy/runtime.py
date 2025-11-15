from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import pandas as pd
import torch

from .feature_builder import FeatureBuilder, FeatureSpec
from .models import PricingAdjustmentModel, PricingModelConfig


@dataclass
class PricingAdjustment:
    low_price: float
    high_price: float
    base_low_price: float
    base_high_price: float
    low_delta: float
    high_delta: float
    pnl_gain: float


class NeuralPricingAdjuster:
    """Load and execute a trained neural pricing model."""

    def __init__(
        self,
        *,
        run_dir: str | Path,
        device: Optional[str] = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._spec = self._load_spec()
        self._builder = FeatureBuilder(
            numeric_columns=list(self._spec.numeric_stats.keys()),
            categorical_columns=list(self._spec.categorical_levels.keys()),
        )
        self._builder._spec = self._spec
        self.clamp_pct = self._load_clamp_pct()
        self._model = self._load_model()
        self.last_error: Optional[str] = None

    def _load_spec(self) -> FeatureSpec:
        spec_path = self.run_dir / "feature_spec.json"
        if not spec_path.exists():
            raise FileNotFoundError(f"Missing feature_spec.json in {self.run_dir}")
        data = json.loads(spec_path.read_text())
        return FeatureSpec.from_dict(data)

    def _load_clamp_pct(self) -> float:
        config_path = self.run_dir / "run_config.json"
        if not config_path.exists():
            return 0.08
        data = json.loads(config_path.read_text())
        return float(data.get("clamp_pct", 0.08))

    def _load_model(self) -> PricingAdjustmentModel:
        state_path = self.run_dir / "pricing_model.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"Missing pricing_model.pt in {self.run_dir}")
        config = PricingModelConfig(
            input_dim=len(self._spec.feature_names),
            max_delta_pct=self.clamp_pct,
        )
        model = PricingAdjustmentModel(config)
        state = torch.load(state_path, map_location=self.device)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def _coerce_mapping(payload: Mapping[str, object]) -> Mapping[str, object]:
        if hasattr(payload, "to_dict"):
            return payload.to_dict()  # type: ignore[return-value]
        return payload

    def _prepare_row(
        self,
        payload: Mapping[str, object],
        *,
        symbol: Optional[str],
    ) -> pd.DataFrame:
        row = dict(payload)
        if symbol:
            row["symbol"] = symbol.upper()
        else:
            existing = row.get("symbol")
            row["symbol"] = str(existing).upper() if existing else "UNKNOWN"
        row.setdefault("close_prediction_source", "UNKNOWN")
        return pd.DataFrame([row])

    def adjust(
        self,
        payload: Mapping[str, object],
        *,
        symbol: Optional[str] = None,
    ) -> Optional[PricingAdjustment]:
        data = self._coerce_mapping(payload)
        try:
            base_low = float(data.get("maxdiffalwayson_low_price", 0.0))
            base_high = float(data.get("maxdiffalwayson_high_price", 0.0))
        except (TypeError, ValueError):
            self.last_error = "Invalid base prices."
            return None
        if base_low <= 0 or base_high <= 0:
            self.last_error = "Base prices missing or non-positive."
            return None
        frame = self._prepare_row(data, symbol=symbol)
        features = self._builder.transform(frame).astype("float32")
        tensor = torch.from_numpy(features).to(self.device)
        try:
            with torch.inference_mode():
                output = self._model(tensor)[0].detach().cpu().numpy()
        except Exception as exc:  # pragma: no cover - defensive
            self.last_error = f"Neural pricing inference failed: {exc}"
            return None
        low_delta, high_delta, pnl_gain = map(float, output.tolist())
        adjusted_low = max(0.0, base_low * (1.0 + low_delta))
        adjusted_high = max(0.0, base_high * (1.0 + high_delta))
        self.last_error = None
        return PricingAdjustment(
            low_price=adjusted_low,
            high_price=adjusted_high,
            base_low_price=base_low,
            base_high_price=base_high,
            low_delta=low_delta,
            high_delta=high_delta,
            pnl_gain=pnl_gain,
        )


__all__ = ["NeuralPricingAdjuster", "PricingAdjustment"]
