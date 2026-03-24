"""Reward computation wrapping BinanceMarketSimulator for GRPO training."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from binanceneural.marketsimulator import BinanceMarketSimulator, SimulationConfig
from src.robust_trading_metrics import compute_pnl_smoothness

from .data_prompt import MarketSnapshot
from .schema import validate_plan, plan_to_sim_actions

log = logging.getLogger(__name__)

PENALTY_INVALID = -2.0
PENALTY_FLAT = -1.0
PENALTY_SIM_ERROR = -1.5
DEFAULT_REWARD_TYPE = "sortino_only"
SUPPORTED_REWARD_TYPES = frozenset({"sortino_only", "sortino_drawdown", "sortino_smoothness"})

REWARD_SIM_CONFIG = SimulationConfig(
    maker_fee=0.001,
    initial_cash=10_000.0,
    max_hold_hours=6,
    fill_buffer_bps=5.0,
    decision_lag_bars=2,
)


def compute_sortino(returns: np.ndarray, periods_per_year: float = 8760.0) -> float:
    """Compute Sortino ratio from hourly returns array."""
    if len(returns) < 2:
        return 0.0
    mean_ret = float(np.mean(returns))
    neg = returns[returns < 0.0]
    if neg.size == 0:
        return max(mean_ret * 100, 0.0)  # all positive = good
    downside_dev = float(np.sqrt(np.mean(neg * neg)))
    if downside_dev <= 0:
        return 0.0
    return float(mean_ret / downside_dev * np.sqrt(periods_per_year))


def _normalize_reward_type(reward_type: str | None) -> str:
    value = (reward_type or DEFAULT_REWARD_TYPE).strip().lower()
    if value not in SUPPORTED_REWARD_TYPES:
        supported = ", ".join(sorted(SUPPORTED_REWARD_TYPES))
        raise ValueError(f"unsupported reward_type {value!r}; expected one of: {supported}")
    return value


def score_reward_metrics(
    *,
    sortino: float,
    total_return: float,
    max_drawdown: float,
    pnl_smoothness: float,
    reward_type: str = DEFAULT_REWARD_TYPE,
) -> float:
    """Convert simulated trading metrics into the scalar RL reward."""
    normalized_type = _normalize_reward_type(reward_type)
    reward = 0.5 * float(sortino) + 50.0 * float(total_return)
    if normalized_type == "sortino_drawdown":
        reward -= 12.0 * abs(float(max_drawdown))
    elif normalized_type == "sortino_smoothness":
        reward -= 120.0 * max(float(pnl_smoothness), 0.0)
    return float(np.clip(reward, -5.0, 50.0))


def compute_reward(
    plan_text: str,
    snapshot: MarketSnapshot,
    config: Optional[SimulationConfig] = None,
    initial_cash: float = 10_000.0,
    reward_type: str = DEFAULT_REWARD_TYPE,
) -> float:
    """Parse plan, simulate, return Sortino as reward.

    Returns PENALTY_INVALID for unparseable output, PENALTY_FLAT for all-FLAT plans.
    """
    cfg = config or REWARD_SIM_CONFIG
    plan = validate_plan(plan_text)
    if plan is None:
        return PENALTY_INVALID

    actions_df = plan_to_sim_actions(plan, snapshot.forward_bars, initial_cash=initial_cash)
    if actions_df.empty:
        return PENALTY_FLAT

    try:
        sim = BinanceMarketSimulator(cfg)
        result = sim.run(bars=snapshot.forward_bars, actions=actions_df)
        equity = result.combined_equity
        if equity is None or len(equity) < 2:
            return 0.0
        returns = equity.pct_change().dropna().values
        sortino = compute_sortino(returns)
        total_ret = float((equity.iloc[-1] / equity.iloc[0]) - 1) if equity.iloc[0] > 0 else 0.0
        max_drawdown = _max_drawdown(equity.values)
        pnl_smoothness = compute_pnl_smoothness(returns)
        return score_reward_metrics(
            sortino=sortino,
            total_return=total_ret,
            max_drawdown=max_drawdown,
            pnl_smoothness=pnl_smoothness,
            reward_type=reward_type,
        )
    except Exception as e:
        log.warning("sim error: %s", e)
        return PENALTY_SIM_ERROR


def compute_reward_detailed(
    plan_text: str,
    snapshot: MarketSnapshot,
    config: Optional[SimulationConfig] = None,
    reward_type: str = DEFAULT_REWARD_TYPE,
) -> dict:
    """Like compute_reward but returns full metrics dict."""
    cfg = config or REWARD_SIM_CONFIG
    plan = validate_plan(plan_text)
    if plan is None:
        return {"reward": PENALTY_INVALID, "error": "invalid_json", "n_trades": 0}

    actions_df = plan_to_sim_actions(plan, snapshot.forward_bars)
    if actions_df.empty:
        return {"reward": PENALTY_FLAT, "error": "all_flat", "n_trades": 0}

    try:
        sim = BinanceMarketSimulator(cfg)
        result = sim.run(bars=snapshot.forward_bars, actions=actions_df)
        equity = result.combined_equity
        if equity is None or len(equity) < 2:
            return {"reward": 0.0, "sortino": 0.0, "total_return": 0.0, "n_trades": 0}
        returns = equity.pct_change().dropna().values
        sortino = compute_sortino(returns)
        total_ret = float((equity.iloc[-1] / equity.iloc[0]) - 1)
        max_dd = _max_drawdown(equity.values)
        pnl_smoothness = compute_pnl_smoothness(returns)
        n_trades = sum(len(sr.trades) for sr in result.per_symbol.values())
        reward = score_reward_metrics(
            sortino=sortino,
            total_return=total_ret,
            max_drawdown=max_dd,
            pnl_smoothness=pnl_smoothness,
            reward_type=reward_type,
        )
        return {
            "reward": reward,
            "sortino": sortino,
            "total_return": total_ret,
            "max_drawdown": max_dd,
            "pnl_smoothness": pnl_smoothness,
            "n_trades": n_trades,
        }
    except Exception as e:
        return {"reward": PENALTY_SIM_ERROR, "error": str(e), "n_trades": 0}


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1.0)
    return float(np.min(dd))


class GRPORewardFn:
    """Callable reward function compatible with TRL GRPOTrainer.

    TRL calls reward_fn(completions=...) -> list[float].
    We map window_ids embedded in prompts back to MarketSnapshots.
    """

    def __init__(
        self,
        snapshot_map: dict[str, MarketSnapshot],
        config: Optional[SimulationConfig] = None,
        reward_type: str = DEFAULT_REWARD_TYPE,
    ):
        self.snapshot_map = snapshot_map
        self.config = config or REWARD_SIM_CONFIG
        self.reward_type = _normalize_reward_type(reward_type)
        self.__name__ = f"grpo_reward_{self.reward_type}"

    def __call__(self, completions: list[str], **kwargs) -> list[float]:
        """Compute rewards for a batch of completions.

        The prompts contain [window:XXXX] tags that map to snapshots.
        """
        prompts = kwargs.get("prompts", [])
        rewards = []
        for i, completion in enumerate(completions):
            prompt = prompts[i] if i < len(prompts) else ""
            snapshot = self._find_snapshot(prompt)
            if snapshot is None:
                rewards.append(PENALTY_INVALID)
                continue
            completion_text = self._completion_to_text(completion)
            r = compute_reward(completion_text, snapshot, self.config, reward_type=self.reward_type)
            rewards.append(r)
        return rewards

    def _find_snapshot(self, prompt: object) -> Optional[MarketSnapshot]:
        import re

        m = re.search(r"\[window:([a-f0-9]+)\]", self._prompt_to_text(prompt))
        if m:
            wid = m.group(1)
            return self.snapshot_map.get(wid)
        return None

    def _prompt_to_text(self, prompt: object) -> str:
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, dict):
            content = prompt.get("content")
            if content is not None:
                return self._message_content_to_text(content)
            return " ".join(self._prompt_to_text(v) for v in prompt.values())
        if isinstance(prompt, (list, tuple)):
            return " ".join(self._prompt_to_text(v) for v in prompt)
        return str(prompt)

    def _completion_to_text(self, completion: object) -> str:
        if isinstance(completion, str):
            return completion
        if isinstance(completion, dict):
            content = completion.get("content")
            if content is not None:
                return self._message_content_to_text(content)
            return " ".join(self._completion_to_text(v) for v in completion.values())
        if isinstance(completion, (list, tuple)):
            message_bits = []
            for item in completion:
                if isinstance(item, dict) and "content" in item:
                    role = item.get("role")
                    if role in (None, "assistant"):
                        message_bits.append(self._message_content_to_text(item["content"]))
                else:
                    message_bits.append(self._completion_to_text(item))
            return " ".join(bit for bit in message_bits if bit)
        return str(completion)

    def _message_content_to_text(self, content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            for key in ("text", "content"):
                if key in content:
                    return self._message_content_to_text(content[key])
            return " ".join(self._message_content_to_text(v) for v in content.values())
        if isinstance(content, (list, tuple)):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and "text" in item:
                        parts.append(str(item["text"]))
                    elif "content" in item:
                        parts.append(self._message_content_to_text(item["content"]))
                    else:
                        parts.append(" ".join(self._message_content_to_text(v) for v in item.values()))
                else:
                    parts.append(self._message_content_to_text(item))
            return " ".join(part for part in parts if part)
        return str(content)
