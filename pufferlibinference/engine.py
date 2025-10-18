from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.serialization import add_safe_globals

import hftraining.portfolio_rl_trainer as portfolio_rl_trainer
from hftraining.data_utils import StockDataProcessor
from hftraining.portfolio_rl_trainer import PortfolioAllocationModel, PortfolioRLConfig

from .config import InferenceDataConfig, PufferInferenceConfig
from .data import RollingWindowSet, prepare_inference_windows

LOGGER = logging.getLogger(__name__)

# Ensure torch can resolve the training module path embedded in checkpoints.
sys.modules.setdefault("portfolio_rl_trainer", portfolio_rl_trainer)
add_safe_globals([PortfolioRLConfig])


@dataclass(slots=True)
class AllocationDecision:
    """
    Single-step allocation snapshot emitted by the inference engine.
    """

    timestamp: np.datetime64
    weights: Dict[str, float]
    turnover: float
    trading_cost: float
    financing_cost: float
    net_return: float
    gross_exposure: float
    portfolio_value: float


@dataclass(slots=True)
class InferenceResult:
    """
    Aggregated output after rolling the allocator over a dataset.
    """

    decisions: Sequence[AllocationDecision]
    equity_curve: np.ndarray
    step_returns: np.ndarray
    summary: Dict[str, float]


class PortfolioRLInferenceEngine:
    """
    End-to-end inference pipeline for PufferLib portfolio allocators.
    """

    def __init__(self, inference_cfg: PufferInferenceConfig, data_cfg: InferenceDataConfig):
        self.config = inference_cfg
        self.data_cfg = data_cfg
        self.device = self._resolve_device(inference_cfg.device)
        self._checkpoint_payload = self._load_checkpoint()
        self._model = self._initialise_model()
        self._processor = self._initialise_processor()

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def simulate(self, *, initial_value: float = 1.0) -> InferenceResult:
        """
        Roll the allocator over the configured dataset and return portfolio telemetry.
        """
        rolling_windows = prepare_inference_windows(self.data_cfg, self._processor)
        self._validate_rollout(rolling_windows)
        decisions, equity_curve, step_returns = self._run_rollout(rolling_windows, initial_value=initial_value)
        summary = self._summarise(equity_curve, step_returns, decisions)
        return InferenceResult(
            decisions=decisions,
            equity_curve=equity_curve,
            step_returns=step_returns,
            summary=summary,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _resolve_device(self, requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested)

    def _load_checkpoint(self) -> Dict[str, object]:
        checkpoint_path = self.config.resolved_checkpoint()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        LOGGER.info("Loading PufferLib checkpoint from %s", checkpoint_path)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        expected_keys = {"model_state_dict", "config", "symbols"}
        missing = expected_keys - payload.keys()
        if missing:
            raise KeyError(f"Checkpoint missing required keys: {missing}")
        return payload

    def _initialise_model(self) -> PortfolioAllocationModel:
        payload = self._checkpoint_payload
        rl_config = payload["config"]
        if not isinstance(rl_config, PortfolioRLConfig):
            raise TypeError(f"Checkpoint config must be PortfolioRLConfig; received {type(rl_config)!r}")

        state_dict: Dict[str, torch.Tensor] = payload["model_state_dict"]
        # Torch compile can prefix parameters with _orig_mod.; normalise to match eager names.
        cleaned_state = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        input_proj_key = "input_proj.weight"
        head_weight_key = "head.3.weight"
        if input_proj_key not in cleaned_state or head_weight_key not in cleaned_state:
            raise KeyError("Unexpected checkpoint layout – cannot infer input dimensions.")

        input_dim = cleaned_state[input_proj_key].shape[1]
        num_assets = cleaned_state[head_weight_key].shape[0]

        model = PortfolioAllocationModel(
            input_dim=input_dim,
            config=rl_config,
            num_assets=num_assets,
        )
        model.load_state_dict(cleaned_state, strict=True)
        model.to(self.device)
        model.eval()
        LOGGER.info(
            "Loaded allocator (assets=%d, hidden_size=%d, seq_len inferred post-processor).",
            num_assets,
            rl_config.hidden_size,
        )
        return model

    def _initialise_processor(self) -> StockDataProcessor:
        processor_path = self.config.resolved_processor()
        if processor_path is None:
            processor_path = self._infer_processor_path()
        if processor_path is None or not processor_path.exists():
            raise FileNotFoundError(
                "Unable to locate StockDataProcessor scalers. Provide processor_path in the inference config."
            )
        LOGGER.info("Loading data processor scalers from %s", processor_path)
        processor = StockDataProcessor(
            use_toto_forecasts=self.data_cfg.use_toto_forecasts,
            toto_options=self.data_cfg.toto_options,
            toto_prediction_features=self.data_cfg.toto_prediction_frames,
        )
        processor.load_scalers(str(processor_path))
        return processor

    def _infer_processor_path(self) -> Optional[Path]:
        checkpoint = self.config.resolved_checkpoint()
        candidate_names = ("data_processor.pkl", "processor.pkl")
        for parent in checkpoint.parents:
            for name in candidate_names:
                candidate = parent / name
                if candidate.exists():
                    return candidate
        return None

    def _validate_rollout(self, rollout: RollingWindowSet) -> None:
        expected_assets = self._checkpoint_payload["symbols"]
        if list(rollout.symbols) != list(expected_assets):
            raise ValueError(
                f"Symbol mismatch: checkpoint expects {expected_assets}, received {rollout.symbols}."
            )
        model_input_dim = self._model.input_proj.in_features
        if rollout.input_dim != model_input_dim:
            raise ValueError(
                f"Input dimension mismatch: model expects {model_input_dim}, window set provides {rollout.input_dim}."
            )

    def _run_rollout(
        self,
        rollout: RollingWindowSet,
        *,
        initial_value: float,
    ) -> tuple[List[AllocationDecision], np.ndarray, np.ndarray]:
        model = self._model
        device = self.device
        per_asset_fees = np.asarray(
            rollout.per_asset_fees + self.config.transaction_cost_decimal,
            dtype=np.float32,
        )

        previous_weights = np.zeros(rollout.num_assets, dtype=np.float32)
        portfolio_value = float(initial_value)
        portfolio_values = [portfolio_value]
        decisions: List[AllocationDecision] = []
        step_returns: List[float] = []

        for idx in range(rollout.num_samples):
            window = rollout.inputs[idx]
            input_tensor = torch.from_numpy(window).unsqueeze(0).to(device)
            with torch.no_grad():
                raw_weights = model(input_tensor).squeeze(0).detach().cpu().numpy()

            weights = self._post_process_weights(raw_weights)
            gross_exposure = float(np.abs(weights).sum())
            turnover = float(np.abs(weights - previous_weights).sum())

            trading_cost = float(np.sum(np.abs(weights - previous_weights) * per_asset_fees))
            financing_cost = float(
                max(0.0, gross_exposure - 1.0) * self.config.borrowing_cost / self.config.trading_days_per_year
            )

            asset_returns = rollout.future_returns[idx]
            net_return = float(np.dot(weights, asset_returns) - trading_cost - financing_cost)

            portfolio_value *= (1.0 + net_return)
            portfolio_values.append(portfolio_value)
            step_returns.append(net_return)

            decision = AllocationDecision(
                timestamp=rollout.timestamps[idx],
                weights={symbol: float(weight) for symbol, weight in zip(rollout.symbols, weights)},
                turnover=turnover,
                trading_cost=trading_cost,
                financing_cost=financing_cost,
                net_return=net_return,
                gross_exposure=gross_exposure,
                portfolio_value=portfolio_value,
            )
            decisions.append(decision)
            previous_weights = weights

        return decisions, np.asarray(portfolio_values, dtype=np.float32), np.asarray(step_returns, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Weight handling                                                    #
    # ------------------------------------------------------------------ #

    def _post_process_weights(self, weights: np.ndarray) -> np.ndarray:
        adjusted = weights.astype(np.float32, copy=True)
        if self.config.clamp_action is not None:
            limit = float(self.config.clamp_action)
            adjusted = np.clip(adjusted, -limit, limit)

        if not self.config.allow_short:
            adjusted = np.clip(adjusted, 0.0, None)

        gross = float(np.abs(adjusted).sum())
        leverage_cap = max(1e-6, self.config.leverage_limit - max(0.0, self.config.min_cash_buffer))
        if self.config.enforce_leverage_limit and gross > leverage_cap:
            adjusted = adjusted / gross * leverage_cap
            gross = leverage_cap

        if self.config.min_cash_buffer > 0.0 and gross > 0.0:
            shrink_factor = max(0.0, leverage_cap) / max(gross, 1e-6)
            shrink_factor = min(shrink_factor, 1.0)
            adjusted = adjusted * shrink_factor

        return adjusted

    # ------------------------------------------------------------------ #
    # Metrics                                                            #
    # ------------------------------------------------------------------ #

    def _summarise(
        self,
        equity_curve: np.ndarray,
        step_returns: np.ndarray,
        decisions: Sequence[AllocationDecision],
    ) -> Dict[str, float]:
        initial_value = float(equity_curve[0])
        final_value = float(equity_curve[-1])
        total_return = final_value / max(initial_value, 1e-8) - 1.0
        average_turnover = float(np.mean([d.turnover for d in decisions])) if decisions else 0.0

        annualisation = self.config.trading_days_per_year
        per_step_mean = float(step_returns.mean()) if step_returns.size else 0.0
        per_step_std = float(step_returns.std(ddof=0)) if step_returns.size else 0.0
        if per_step_std > 0:
            sharpe = (per_step_mean * math.sqrt(annualisation)) / per_step_std
        else:
            sharpe = 0.0

        cummax = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - cummax) / np.clip(cummax, a_min=1e-8, a_max=None)
        max_drawdown = float(drawdowns.min())

        return {
            "initial_value": initial_value,
            "final_value": final_value,
            "cumulative_return": total_return,
            "average_turnover": average_turnover,
            "annualised_sharpe": sharpe,
            "max_drawdown": max_drawdown,
        }
