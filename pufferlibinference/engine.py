from __future__ import annotations

import json
import logging
import os
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from collections import OrderedDict

import numpy as np
import torch
from torch.serialization import add_safe_globals

import hftraining.portfolio_rl_trainer as portfolio_rl_trainer
from hftraining.data_utils import StockDataProcessor
from hftraining.portfolio_rl_trainer import PortfolioAllocationModel, PortfolioRLConfig

from .config import InferenceDataConfig, PufferInferenceConfig
from .data import RollingWindowSet, prepare_inference_windows

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from pufferlibtraining3.envs.market_env import MarketEnv, MarketEnvConfig
    from pufferlibtraining3.models import MarketPolicy

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
        self._checkpoint_payload, self._checkpoint_kind = self._load_checkpoint()
        self._training3_summary: Optional[Dict[str, Any]] = None
        if self._checkpoint_kind == "portfolio_rl":
            self._model = self._initialise_model()
            self._processor = self._initialise_processor()
        else:
            self._training3_summary = self._load_training3_summary()
            self._model = None
            self._processor = None

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def simulate(self, *, initial_value: float = 1.0) -> InferenceResult:
        """
        Roll the allocator over the configured dataset and return portfolio telemetry.
        """
        if self._checkpoint_kind == "market_policy":
            return self._simulate_market_policy(initial_value=initial_value)
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

    def _load_checkpoint(self) -> Tuple[Dict[str, object], str]:
        checkpoint_path = self.config.resolved_checkpoint()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        LOGGER.info("Loading PufferLib checkpoint from %s", checkpoint_path)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        expected_keys = {"model_state_dict", "config", "symbols"}
        if isinstance(payload, dict) and expected_keys.issubset(payload.keys()):
            return payload, "portfolio_rl"

        if self._is_market_policy_state_dict(payload):
            wrapped = {"state_dict": dict(payload)}
            return wrapped, "market_policy"

        raise KeyError("Checkpoint structure not recognised – expected PortfolioRL or PufferLib policy weights.")

    def _is_market_policy_state_dict(self, payload: object) -> bool:
        if isinstance(payload, (dict, OrderedDict)):
            keys = set(payload.keys())
            encoder_key = "encoder.0.weight"
            actor_keys = {"actor_mean.weight", "actor_head.weight"}
            return encoder_key in keys and any(key in keys for key in actor_keys)
        return False

    def _extract_run_id(self) -> Optional[str]:
        checkpoint = self.config.resolved_checkpoint()
        stem = checkpoint.stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        return digits or None

    def _load_training3_summary(self) -> Optional[Dict[str, Any]]:
        summary_env = os.getenv("PUFFERLIBTRAINING3_SUMMARY")
        if summary_env:
            path = Path(summary_env).expanduser()
            if path.exists():
                try:
                    return json.loads(path.read_text())
                except Exception as error:  # pragma: no cover - defensive
                    LOGGER.warning("Failed to parse summary from %s: %s", path, error)

        run_id = self._extract_run_id()
        if not run_id:
            return None

        runs_root = Path("pufferlibtraining3") / "runs"
        if not runs_root.exists():
            return None

        for summary_path in runs_root.rglob("summary.json"):
            try:
                data = json.loads(summary_path.read_text())
            except Exception:  # pragma: no cover - defensive
                continue
            if str(data.get("run_id")) == run_id:
                return data
        return None

    def _device_string(self) -> str:
        if self.device.type == "cuda" and self.device.index is not None:
            return f"cuda:{self.device.index}"
        return self.device.type

    def _build_market_env_config(self) -> "MarketEnvConfig":
        from pufferlibtraining3.envs.market_env import MarketEnvConfig

        env_config_data: Dict[str, Any] = {}
        if self._training3_summary:
            env_config_data.update(self._training3_summary.get("env_config", {}))
        cfg = MarketEnvConfig(**env_config_data)
        cfg.data_root = str(self.data_cfg.resolved_data_dir())
        symbols = list(self.data_cfg.normalised_symbols())
        if symbols:
            cfg.symbol = symbols[0]
        cfg.device = self._device_string()
        return cfg

    def _instantiate_market_env(self) -> Tuple["MarketEnv", "MarketEnvConfig"]:
        from pufferlibtraining3.envs.market_env import MarketEnv

        cfg = self._build_market_env_config()
        env = MarketEnv(cfg=cfg)
        return env, cfg

    def _instantiate_market_policy(self, env: "MarketEnv") -> "MarketPolicy":
        from pufferlibtraining3.models import MarketPolicy, PolicyConfig

        state_dict: Dict[str, torch.Tensor] = self._checkpoint_payload["state_dict"]
        policy_cfg = PolicyConfig()

        class _PolicyEnvAdapter:
            def __init__(self, base_env: "MarketEnv"):
                self.single_observation_space = base_env.observation_space
                self.single_action_space = base_env.action_space

        adapter = _PolicyEnvAdapter(env)
        model = MarketPolicy(adapter, policy_cfg)
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()
        return model

    def _simulate_market_policy(self, *, initial_value: float) -> InferenceResult:
        import numpy as np

        env, env_cfg = self._instantiate_market_env()
        try:
            model = self._instantiate_market_policy(env)
            obs, _ = env.reset()
            portfolio_value = float(initial_value)
            equity_curve: List[float] = [portfolio_value]
            step_returns: List[float] = []
            decisions: List[AllocationDecision] = []

            previous_env_equity = float(env.equity.detach().cpu().item())
            previous_position = float(env.position.detach().cpu().item())

            start_limit = self._normalise_limit(self.data_cfg.start_date)
            end_limit = self._normalise_limit(self.data_cfg.end_date)
            recorded_any = False

            symbols = list(self.data_cfg.normalised_symbols())
            if not symbols:
                symbol = env_cfg.symbol
                if symbol:
                    symbols = [str(symbol).upper()]
            if not symbols:
                symbols = ["ASSET"]

            while True:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = model.forward_eval(obs_tensor)
                action = self._deterministic_market_action(logits, env)
                obs, _, terminated, truncated, info = env.step(action)

                timestamp = self._normalise_timestamp(info.get("timestamp"))
                current_env_equity = float(env.equity.detach().cpu().item())
                prev_env_equity = max(previous_env_equity, 1e-8)
                return_ratio = (current_env_equity - previous_env_equity) / prev_env_equity
                previous_env_equity = current_env_equity

                previous_portfolio = portfolio_value
                portfolio_value = previous_portfolio * (1.0 + return_ratio)

                scale = previous_portfolio / prev_env_equity
                trading_cost = float(info.get("trading_cost", 0.0)) * scale
                financing_cost = float(info.get("financing_cost", 0.0)) * scale

                intraday_position = float(info.get("intraday_position", 0.0))
                overnight_position = float(info.get("overnight_position", intraday_position))
                gross_exposure = abs(intraday_position)
                turnover = abs(intraday_position - previous_position)
                previous_position = overnight_position

                include = self._timestamp_in_range(timestamp, start_limit, end_limit)
                if include:
                    recorded_any = True
                    equity_curve.append(portfolio_value)
                    step_returns.append(return_ratio)
                    weights = {symbols[0]: intraday_position}
                    decision = AllocationDecision(
                        timestamp=timestamp if timestamp is not None else np.datetime64("NaT"),
                        weights=weights,
                        turnover=turnover,
                        trading_cost=trading_cost,
                        financing_cost=financing_cost,
                        net_return=return_ratio,
                        gross_exposure=gross_exposure,
                        portfolio_value=portfolio_value,
                    )
                    decisions.append(decision)

                if terminated or truncated:
                    break
                if end_limit is not None and timestamp is not None and timestamp > end_limit and recorded_any:
                    break

            if not recorded_any:
                raise RuntimeError(
                    "No inference timesteps fell within the requested date range; "
                    "ensure start/end dates align with available market data."
                )

            equity_array = np.asarray(equity_curve, dtype=np.float32)
            returns_array = np.asarray(step_returns, dtype=np.float32)
            summary = self._summarise(equity_array, returns_array, decisions)
            return InferenceResult(
                decisions=decisions,
                equity_curve=equity_array,
                step_returns=returns_array,
                summary=summary,
            )
        finally:
            try:
                env.close()
            except Exception:  # pragma: no cover - defensive cleanup
                pass

    def _deterministic_market_action(self, logits: Any, env: "MarketEnv") -> torch.Tensor:
        if hasattr(logits, "mean"):  # Continuous actions -> Normal distribution
            action_tensor = logits.mean
        elif isinstance(logits, (tuple, list)):
            stacked = torch.stack([torch.as_tensor(x) for x in logits], dim=-1)
            action_tensor = torch.softmax(stacked, dim=-1)
            action_tensor = torch.argmax(action_tensor, dim=-1, keepdim=False).float()
        else:
            action_tensor = torch.as_tensor(logits)
            if action_tensor.ndim > 1 and env.action_space.__class__.__name__ == "Discrete":
                action_tensor = torch.argmax(action_tensor, dim=-1, keepdim=False).float()

        if action_tensor.ndim > 1:
            action_tensor = action_tensor.squeeze(0)
        return action_tensor.detach().cpu()

    def _normalise_limit(self, value: Optional[str]) -> Optional["np.datetime64"]:
        import numpy as np

        if value is None:
            return None
        try:
            return np.datetime64(value)
        except Exception:
            return None

    def _normalise_timestamp(self, value: Any) -> Optional["np.datetime64"]:
        import numpy as np
        import pandas as pd  # type: ignore

        if value is None:
            return None
        if isinstance(value, np.datetime64):
            return value
        if hasattr(value, "to_datetime64"):
            try:
                return value.to_datetime64()
            except Exception:
                pass
        if isinstance(value, (int, float)):
            return None
        try:
            return np.datetime64(value)
        except Exception:
            try:
                return np.datetime64(pd.to_datetime(value))
            except Exception:
                return None

    def _timestamp_in_range(
        self,
        timestamp: Optional["np.datetime64"],
        start: Optional["np.datetime64"],
        end: Optional["np.datetime64"],
    ) -> bool:
        if timestamp is None:
            return start is None and end is None
        if start is not None and timestamp < start:
            return False
        if end is not None and timestamp > end:
            return False
        return True

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
